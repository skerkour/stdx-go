package chacha20

import (
	"crypto/subtle"
	"simd/archsimd"
	"unsafe"
)

// rotl16Idx and rotl8Idx are the VPSHUFB index vectors that rotate each 32-bit
// element of a Uint8x32 by 16/8 bits. They are kept as scalar data (not vector
// values, per the archsimd guidance against storing SIMD types in aggregates)
// and materialized as loop-invariant vector locals in xorKeyStreamAVX2.
var (
	rotl16Idx = [32]int8{
		2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
		2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
	}
	rotl8Idx = [32]int8{
		3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14,
		3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14,
	}

	// counterOffsets[k] is added to the counter word (element 0 of each 128-bit
	// lane of a column's words-12-15 vector) to yield the block counters 2k and
	// 2k+1 within a batch of 8. The base column 0 vector already holds c in the
	// low lane and c+1 in the high lane, so the other columns add 2k to both
	// lanes (mirroring two_AVX2 in chachaAVX2_amd64.s).
	counterOffsets = [4][8]uint32{
		{0, 0, 0, 0, 1, 0, 0, 0},
		{2, 0, 0, 0, 2, 0, 0, 0},
		{4, 0, 0, 0, 4, 0, 0, 0},
		{6, 0, 0, 0, 6, 0, 0, 0},
	}

	// counterIncrement advances the column counters (word 12, element 0 of each
	// 128-bit lane) by one batch of 4 blocks.
	counterIncrement = [8]uint32{4, 0, 0, 0, 4, 0, 0, 0}
)

// xorKeyStream is the amd64 SIMD backend hook. It XORs src with the key
// stream generated from state and maintains leftover key stream state.
//
// The 512-bit AVX-512 path is used when the CPU supports it, then the 256-bit
// AVX2 path when the CPU supports AVX2; otherwise the portable scalar backend
// runs.
func (cipher *CipherIetf) xorKeyStream(dst, src []byte) {
	switch {
	case archsimd.X86.AVX512() && len(src) > 256:
		cipher.xorKeyStreamAVX512(dst, src)
	case archsimd.X86.AVX2() && len(src) > 64:
		cipher.xorKeyStreamAVX2(dst, src)
	default:
		cipher.xorKeyStreamScalar(dst, src)
	}
}

// xorKeyStreamAVX2 is the 8-block SIMD core. It XORs src with the key stream
// generated from state (words 0-11 and 13-15 fixed, word 12 is the block
// counter, advanced as blocks are produced) and maintains leftover key stream
// state.
//
// It processes 8 blocks (512 bytes) per iteration using the column-major
// layout of chachaAVX2_amd64.s: each column k holds the 16 words of two
// blocks (block 2k in the low 128-bit lanes, block 2k+1 in the high lanes)
// across four Uint32x8 vectors, so one vectorized quarter round operates on
// four ChaCha columns at once. The 16/8-bit rotations are single VPSHUFB
// shuffles and the output transpose is fused into the load/xor/store. If the
// input is not a multiple of 256 bytes, the final iteration produces the tail
// and any unused key stream of the last partial block is retained.
//
// Two columns (4 blocks) are kept in flight rather than four: the two VPSHUFB
// rotation masks are loop-invariant register values, so holding all 16 state
// words at once (as chachaAVX2_amd64.s does with memory-operand masks) would
// exceed the 16 available YMM registers and force spills in the round loop.
// With two columns the round loop needs only 8 state vectors + 2 masks.
func (cipher *CipherIetf) xorKeyStreamAVX2(dst, src []byte) {
	// initial state in column-major layout. s01/s23 are the two halves of the
	// 64-byte state; each column duplicates one 4-word group across its two
	// 128-bit lanes.
	s01 := archsimd.LoadUint32x8Array(&[8]uint32{
		cipher.state[0], cipher.state[1], cipher.state[2], cipher.state[3],
		cipher.state[4], cipher.state[5], cipher.state[6], cipher.state[7],
	})
	s23 := archsimd.LoadUint32x8Array(&[8]uint32{
		cipher.state[8], cipher.state[9], cipher.state[10], cipher.state[11],
		cipher.state[12], cipher.state[13], cipher.state[14], cipher.state[15],
	})

	c0w0 := s01.ConcatPermute128Scalars(0, 0, s01)
	c0w1 := s01.ConcatPermute128Scalars(1, 1, s01)
	c0w2 := s23.ConcatPermute128Scalars(0, 0, s23)
	c0w3 := s23.ConcatPermute128Scalars(1, 1, s23).Add(archsimd.LoadUint32x8Array(&counterOffsets[0]))

	c1w0, c1w1, c1w2 := c0w0, c0w1, c0w2
	c1w3 := c0w3.Add(archsimd.LoadUint32x8Array(&counterOffsets[1]))

	// VPSHUFB rotation masks and the per-iteration counter increment, all
	// loop-invariant.
	rotl16 := archsimd.LoadInt8x32Array(&rotl16Idx)
	rotl8 := archsimd.LoadInt8x32Array(&rotl8Idx)
	counterInc := archsimd.LoadUint32x8Array(&counterIncrement)

	// len(dst) >= 256 is not needed but it removes bound checks
	for len(src) >= 256 && len(dst) >= 256 {
		// 20 rounds, no add-back: keeps the round loop to 10 live vectors.
		r0w0, r0w1, r0w2, r0w3, r1w0, r1w1, r1w2, r1w3 :=
			chacha4ColumnsRoundsAVX2(c0w0, c0w1, c0w2, c0w3, c1w0, c1w1, c1w2, c1w3, rotl16, rotl8)

		// add-back against the initial columns
		o0w0 := r0w0.Add(c0w0)
		o0w1 := r0w1.Add(c0w1)
		o0w2 := r0w2.Add(c0w2)
		o0w3 := r0w3.Add(c0w3)
		o1w0 := r1w0.Add(c1w0)
		o1w1 := r1w1.Add(c1w1)
		o1w2 := r1w2.Add(c1w2)
		o1w3 := r1w3.Add(c1w3)

		srcPointer := unsafe.Pointer(&src[0])
		dstPointer := unsafe.Pointer(&dst[0])

		// each column outputs two 64-byte blocks: block 2k and 2k+1
		xorStoreCol(dstPointer, srcPointer, 0, o0w0, o0w1, o0w2, o0w3)
		xorStoreCol(dstPointer, srcPointer, 128, o1w0, o1w1, o1w2, o1w3)

		src = src[256:]
		dst = dst[256:]
		c0w3 = c0w3.Add(counterInc)
		c1w3 = c1w3.Add(counterInc)
	}

	cipher.state[12] = c0w3.GetLo().GetElem(0)

	// Tail: < 256 bytes. Generate 4 blocks of key stream into a local
	// buffer, XOR the consumed part, and keep the unused bytes of the last
	// partial block as leftover (<= 63).
	if len(src) > 0 {
		var keystream [256]byte
		r0w0, r0w1, r0w2, r0w3, r1w0, r1w1, r1w2, r1w3 :=
			chacha4ColumnsRoundsAVX2(c0w0, c0w1, c0w2, c0w3, c1w0, c1w1, c1w2, c1w3, rotl16, rotl8)

		ks := unsafe.Pointer(&keystream[0])
		storeCol(ks, 0, r0w0.Add(c0w0), r0w1.Add(c0w1), r0w2.Add(c0w2), r0w3.Add(c0w3))
		storeCol(ks, 128, r1w0.Add(c1w0), r1w1.Add(c1w1), r1w2.Add(c1w2), r1w3.Add(c1w3))

		// min() removes bound checks
		n := min(len(src), 256)
		subtle.XORBytes(dst, src, keystream[:n])
		cipher.state[12] += uint32(n / 64)

		if n%64 != 0 {
			leftoverLen := min(64-n%64, 63)
			copy(cipher.leftover[:leftoverLen], keystream[n:])
			cipher.state[12] += 1
			cipher.leftoverLen = uint8(leftoverLen)
			return
		}
	}

	cipher.leftoverLen = 0
}

// chacha4ColumnsRoundsAVX2 runs the 20 ChaCha20 rounds on two column groups
// (blocks 0-3). The eight column vectors are passed in as the working state
// and returned after the rounds; the caller performs the add-back so that only
// the 8 state vectors and the 2 rotation masks (10 values) are live inside the
// round loop.
func chacha4ColumnsRoundsAVX2(i0w0, i0w1, i0w2, i0w3, i1w0, i1w1, i1w2, i1w3 archsimd.Uint32x8, rotl16, rotl8 archsimd.Int8x32) (o0w0, o0w1, o0w2, o0w3, o1w0, o1w1, o1w2, o1w3 archsimd.Uint32x8) {
	// working state
	w0w0, w0w1, w0w2, w0w3, w1w0, w1w1, w1w2, w1w3 :=
		i0w0, i0w1, i0w2, i0w3, i1w0, i1w1, i1w2, i1w3

	for r := 0; r < 10; r++ {
		// column quarter rounds
		w0w0, w0w1, w0w2, w0w3 = quarterRoundCol(w0w0, w0w1, w0w2, w0w3, rotl16, rotl8)
		w1w0, w1w1, w1w2, w1w3 = quarterRoundCol(w1w0, w1w1, w1w2, w1w3, rotl16, rotl8)

		// rotate the rows so the next quarter rounds hit the diagonals
		w0w1 = w0w1.ConcatPermuteScalarsGrouped(1, 2, 3, 0, w0w1)
		w0w2 = w0w2.ConcatPermuteScalarsGrouped(2, 3, 0, 1, w0w2)
		w0w3 = w0w3.ConcatPermuteScalarsGrouped(3, 0, 1, 2, w0w3)
		w1w1 = w1w1.ConcatPermuteScalarsGrouped(1, 2, 3, 0, w1w1)
		w1w2 = w1w2.ConcatPermuteScalarsGrouped(2, 3, 0, 1, w1w2)
		w1w3 = w1w3.ConcatPermuteScalarsGrouped(3, 0, 1, 2, w1w3)

		// diagonal quarter rounds
		w0w0, w0w1, w0w2, w0w3 = quarterRoundCol(w0w0, w0w1, w0w2, w0w3, rotl16, rotl8)
		w1w0, w1w1, w1w2, w1w3 = quarterRoundCol(w1w0, w1w1, w1w2, w1w3, rotl16, rotl8)

		// inverse row rotation
		w0w1 = w0w1.ConcatPermuteScalarsGrouped(3, 0, 1, 2, w0w1)
		w0w2 = w0w2.ConcatPermuteScalarsGrouped(2, 3, 0, 1, w0w2)
		w0w3 = w0w3.ConcatPermuteScalarsGrouped(1, 2, 3, 0, w0w3)
		w1w1 = w1w1.ConcatPermuteScalarsGrouped(3, 0, 1, 2, w1w1)
		w1w2 = w1w2.ConcatPermuteScalarsGrouped(2, 3, 0, 1, w1w2)
		w1w3 = w1w3.ConcatPermuteScalarsGrouped(1, 2, 3, 0, w1w3)
	}

	return w0w0, w0w1, w0w2, w0w3, w1w0, w1w1, w1w2, w1w3
}

// quarterRoundCol is the ChaCha20 quarter round on the column-major layout.
// Each 32-bit lane of the four vectors is one ChaCha column, so one call
// performs four quarter rounds. The 16/8-bit rotations are single VPSHUFB
// byte shuffles (the same trick as chachaAVX2_amd64.s); the 12/7-bit rotations
// use two shifts and an or, as AVX2 has no variable rotate.
func quarterRoundCol(a, b, c, d archsimd.Uint32x8, rotl16, rotl8 archsimd.Int8x32) (archsimd.Uint32x8, archsimd.Uint32x8, archsimd.Uint32x8, archsimd.Uint32x8) {
	// a += b; d ^= a; d <<<= 16
	a = a.Add(b)
	d = d.Xor(a)
	d = d.ReshapeToUint8s().PermuteOrZeroGrouped(rotl16).ReshapeToUint32s()

	// c += d; b ^= c; b <<<= 12
	c = c.Add(d)
	b = b.Xor(c)
	b = b.ShiftAllLeft(12).Or(b.ShiftAllRight(20))

	// a += b; d ^= a; d <<<= 8
	a = a.Add(b)
	d = d.Xor(a)
	d = d.ReshapeToUint8s().PermuteOrZeroGrouped(rotl8).ReshapeToUint32s()

	// c += d; b ^= c; b <<<= 7
	c = c.Add(d)
	b = b.Xor(c)
	b = b.ShiftAllLeft(7).Or(b.ShiftAllRight(25))

	return a, b, c, d
}

// xorStoreCol XORs 128 bytes of key stream held in the column vectors w0..w3
// (block 2k in the low 128-bit lanes, block 2k+1 in the high lanes) with
// src[off:off+128] and stores the result into dst[off:off+128].
//
// The ConcatPermute128Scalars call is the fused output transpose: it selects
// the low/high 128-bit lanes of the word-group vectors into block-major order
// before the XOR, so no separate transpose step is needed.
func xorStoreCol(dst, src unsafe.Pointer, off int, w0, w1, w2, w3 archsimd.Uint32x8) {
	t := w0.ConcatPermute128Scalars(0, 2, w1)
	t = t.Xor(archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(src, off))))
	t.StoreArray((*[8]uint32)(unsafe.Add(dst, off)))

	t = w2.ConcatPermute128Scalars(0, 2, w3)
	t = t.Xor(archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(src, off+32))))
	t.StoreArray((*[8]uint32)(unsafe.Add(dst, off+32)))

	t = w0.ConcatPermute128Scalars(1, 3, w1)
	t = t.Xor(archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(src, off+64))))
	t.StoreArray((*[8]uint32)(unsafe.Add(dst, off+64)))

	t = w2.ConcatPermute128Scalars(1, 3, w3)
	t = t.Xor(archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(src, off+96))))
	t.StoreArray((*[8]uint32)(unsafe.Add(dst, off+96)))
}

// storeCol stores 128 bytes of key stream held in the column vectors w0..w3
// (block 2k in the low 128-bit lanes, block 2k+1 in the high lanes) into
// ks[off:off+128].
func storeCol(ks unsafe.Pointer, off int, w0, w1, w2, w3 archsimd.Uint32x8) {
	w0.ConcatPermute128Scalars(0, 2, w1).StoreArray((*[8]uint32)(unsafe.Add(ks, off)))
	w2.ConcatPermute128Scalars(0, 2, w3).StoreArray((*[8]uint32)(unsafe.Add(ks, off+32)))
	w0.ConcatPermute128Scalars(1, 3, w1).StoreArray((*[8]uint32)(unsafe.Add(ks, off+64)))
	w2.ConcatPermute128Scalars(1, 3, w3).StoreArray((*[8]uint32)(unsafe.Add(ks, off+96)))
}
