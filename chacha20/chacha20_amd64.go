//go:build amd64 && goexperiment.simd

package chacha20

import (
	"crypto/subtle"
	"simd/archsimd"
	"unsafe"
)

// rotl8Idx/rotl16Idx are the VPSHUFB index vectors that rotate each 32-bit
// element of a Uint32x8 by 8/16 bits. On AVX2 VPSHUFB operates within each
// 128-bit lane, so the 16-byte pattern is repeated in both lanes.
var (
	rotl8Idx = archsimd.LoadInt8x32Array(&[32]int8{
		3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14,
		3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14,
	})
	rotl16Idx = archsimd.LoadInt8x32Array(&[32]int8{
		2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
		2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
	})
)

// xorKeyStream is the amd64 SIMD backend hook. It XORs src with the key
// stream generated from state and maintains leftover key stream state.
//
// The 256-bit AVX2 path is used when the CPU supports AVX2; otherwise the
// portable scalar backend runs.
func (cipher *CipherIetf) xorKeyStream(dst, src []byte) {
	if archsimd.X86.AVX2() && len(src) >= 64 {
		cipher.xorKeyStreamAVX2(dst, src)
	} else {
		cipher.xorKeyStreamScalar(dst, src)
	}
}

// xorKeyStreamAVX2 is the 8-block SIMD core. It XORs src with the key stream
// generated from state (words 0-11 and 13-15 fixed, word 12 is the block
// counter, advanced as blocks are produced) and maintains leftover key stream
// state.
//
// It processes 8 blocks (512 bytes) per iteration. If the input is not a
// multiple of 512 bytes, the final iteration produces the tail and any unused
// key stream of the last partial block is retained.
func (cipher *CipherIetf) xorKeyStreamAVX2(dst, src []byte) {
	// counter is tracked as a scalar; chacha8BlocksAVX2 rebuilds the vector
	// layout from cipher.state on demand.
	counter := cipher.state[12]

	for len(src) >= 512 {
		s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15 :=
			chacha8BlocksAVX2(&cipher.state, counter)

		srcPointer := unsafe.Pointer(&src[0])
		dstPointer := unsafe.Pointer(&dst[0])
		xorStore8(dstPointer, srcPointer, 0, s0, s1, s2, s3)
		xorStore8(dstPointer, srcPointer, 128, s4, s5, s6, s7)
		xorStore8(dstPointer, srcPointer, 256, s8, s9, s10, s11)
		xorStore8(dstPointer, srcPointer, 384, s12, s13, s14, s15)

		src = src[512:]
		dst = dst[512:]
		counter += 8
	}

	cipher.state[12] = counter

	// Tail: < 512 bytes. Generate 8 blocks of key stream into a local
	// buffer, XOR the consumed part, and keep the unused bytes of the last
	// partial block as leftover (<= 63).
	if len(src) > 0 {
		var keystream [512]byte
		s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15 :=
			chacha8BlocksAVX2(&cipher.state, counter)

		ks := unsafe.Pointer(&keystream[0])
		store8(ks, 0, s0, s1, s2, s3)
		store8(ks, 128, s4, s5, s6, s7)
		store8(ks, 256, s8, s9, s10, s11)
		store8(ks, 384, s12, s13, s14, s15)

		n := len(src)
		subtle.XORBytes(dst, src, keystream[:n])
		cipher.state[12] += uint32(n / 64)

		if n%64 != 0 {
			end := 64 * (n/64 + 1)
			leftoverLen := end - n
			copy(cipher.leftover[:], keystream[n:end])
			cipher.state[12] += 1
			cipher.leftoverLen = uint8(leftoverLen)
			return
		}
	}
	cipher.leftoverLen = 0
}

// chacha8BlocksAVX2 computes 8 blocks of ChaCha20 key stream for the given
// counter (20 rounds + add-back) and returns them in the AVX2 row layout:
// register v[4g+j] holds row j (words 4j..4j+3) of blocks 2g (low 128-bit
// half) and 2g+1 (high 128-bit half), for g in 0..3.
//
// The layout matches crypto/chacha/chachaAVX2_amd64.s. The 16 initial-state
// vectors are not kept live through the rounds (16 working registers plus the
// two rotation constants already exceed the 16 YMM registers); instead the
// initial rows are reloaded from state right at the add-back.
func chacha8BlocksAVX2(state *[16]uint32, counter uint32) (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 archsimd.Uint32x8) {
	lo := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&state[0])))
	hi := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&state[8])))
	row0 := lo.ConcatPermute128Scalars(0, 0, lo)
	row1 := lo.ConcatPermute128Scalars(1, 1, lo)
	row2 := hi.ConcatPermute128Scalars(0, 0, hi)
	n0, n1, n2 := state[13], state[14], state[15]

	row3a := archsimd.LoadUint32x8Array(&[8]uint32{counter, n0, n1, n2, counter + 1, n0, n1, n2})
	row3b := archsimd.LoadUint32x8Array(&[8]uint32{counter + 2, n0, n1, n2, counter + 3, n0, n1, n2})
	row3c := archsimd.LoadUint32x8Array(&[8]uint32{counter + 4, n0, n1, n2, counter + 5, n0, n1, n2})
	row3d := archsimd.LoadUint32x8Array(&[8]uint32{counter + 6, n0, n1, n2, counter + 7, n0, n1, n2})

	v0, v1, v2, v3 = row0, row1, row2, row3a
	v4, v5, v6, v7 = row0, row1, row2, row3b
	v8, v9, v10, v11 = row0, row1, row2, row3c
	v12, v13, v14, v15 = row0, row1, row2, row3d

	for r := 0; r < 10; r++ {
		// column round
		v0, v1, v2, v3 = quarterRoundAVX2(v0, v1, v2, v3)
		v4, v5, v6, v7 = quarterRoundAVX2(v4, v5, v6, v7)
		v8, v9, v10, v11 = quarterRoundAVX2(v8, v9, v10, v11)
		v12, v13, v14, v15 = quarterRoundAVX2(v12, v13, v14, v15)

		// rotate rows so the columns become the diagonals
		v1 = v1.PermuteScalarsGrouped(1, 2, 3, 0)
		v2 = v2.PermuteScalarsGrouped(2, 3, 0, 1)
		v3 = v3.PermuteScalarsGrouped(3, 0, 1, 2)
		v5 = v5.PermuteScalarsGrouped(1, 2, 3, 0)
		v6 = v6.PermuteScalarsGrouped(2, 3, 0, 1)
		v7 = v7.PermuteScalarsGrouped(3, 0, 1, 2)
		v9 = v9.PermuteScalarsGrouped(1, 2, 3, 0)
		v10 = v10.PermuteScalarsGrouped(2, 3, 0, 1)
		v11 = v11.PermuteScalarsGrouped(3, 0, 1, 2)
		v13 = v13.PermuteScalarsGrouped(1, 2, 3, 0)
		v14 = v14.PermuteScalarsGrouped(2, 3, 0, 1)
		v15 = v15.PermuteScalarsGrouped(3, 0, 1, 2)

		// diagonal round
		v0, v1, v2, v3 = quarterRoundAVX2(v0, v1, v2, v3)
		v4, v5, v6, v7 = quarterRoundAVX2(v4, v5, v6, v7)
		v8, v9, v10, v11 = quarterRoundAVX2(v8, v9, v10, v11)
		v12, v13, v14, v15 = quarterRoundAVX2(v12, v13, v14, v15)

		// inverse row rotation
		v1 = v1.PermuteScalarsGrouped(3, 0, 1, 2)
		v2 = v2.PermuteScalarsGrouped(2, 3, 0, 1)
		v3 = v3.PermuteScalarsGrouped(1, 2, 3, 0)
		v5 = v5.PermuteScalarsGrouped(3, 0, 1, 2)
		v6 = v6.PermuteScalarsGrouped(2, 3, 0, 1)
		v7 = v7.PermuteScalarsGrouped(1, 2, 3, 0)
		v9 = v9.PermuteScalarsGrouped(3, 0, 1, 2)
		v10 = v10.PermuteScalarsGrouped(2, 3, 0, 1)
		v11 = v11.PermuteScalarsGrouped(1, 2, 3, 0)
		v13 = v13.PermuteScalarsGrouped(3, 0, 1, 2)
		v14 = v14.PermuteScalarsGrouped(2, 3, 0, 1)
		v15 = v15.PermuteScalarsGrouped(1, 2, 3, 0)
	}

	// add-back: reload the initial rows here (constant reload) so the round
	// loop only keeps the 16 working registers and the two rotation constants
	// live.
	lo = archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&state[0])))
	hi = archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&state[8])))
	row0 = lo.ConcatPermute128Scalars(0, 0, lo)
	row1 = lo.ConcatPermute128Scalars(1, 1, lo)
	row2 = hi.ConcatPermute128Scalars(0, 0, hi)
	row3a = archsimd.LoadUint32x8Array(&[8]uint32{counter, n0, n1, n2, counter + 1, n0, n1, n2})
	row3b = archsimd.LoadUint32x8Array(&[8]uint32{counter + 2, n0, n1, n2, counter + 3, n0, n1, n2})
	row3c = archsimd.LoadUint32x8Array(&[8]uint32{counter + 4, n0, n1, n2, counter + 5, n0, n1, n2})
	row3d = archsimd.LoadUint32x8Array(&[8]uint32{counter + 6, n0, n1, n2, counter + 7, n0, n1, n2})

	v0 = v0.Add(row0)
	v1 = v1.Add(row1)
	v2 = v2.Add(row2)
	v3 = v3.Add(row3a)
	v4 = v4.Add(row0)
	v5 = v5.Add(row1)
	v6 = v6.Add(row2)
	v7 = v7.Add(row3b)
	v8 = v8.Add(row0)
	v9 = v9.Add(row1)
	v10 = v10.Add(row2)
	v11 = v11.Add(row3c)
	v12 = v12.Add(row0)
	v13 = v13.Add(row1)
	v14 = v14.Add(row2)
	v15 = v15.Add(row3d)
	return
}

// quarterRoundAVX2 is the ChaCha20 quarter round on the AVX2 row layout.
// The 16- and 8-bit rotations use VPSHUFB (single instruction); the 12- and
// 7-bit rotations use two shifts and an or, the fastest AVX2 option (there is
// no single-instruction 32-bit rotate before AVX512).
func quarterRoundAVX2(a, b, c, d archsimd.Uint32x8) (archsimd.Uint32x8, archsimd.Uint32x8, archsimd.Uint32x8, archsimd.Uint32x8) {
	// a += b; d ^= a; d <<<= 16
	a = a.Add(b)
	d = d.Xor(a)
	d = d.ReshapeToUint8s().PermuteOrZeroGrouped(rotl16Idx).ReshapeToUint32s()

	// c += d; b ^= c; b <<<= 12
	c = c.Add(d)
	b = b.Xor(c)
	b = b.ShiftAllLeft(12).Or(b.ShiftAllRight(20))

	// a += b; d ^= a; d <<<= 8
	a = a.Add(b)
	d = d.Xor(a)
	d = d.ReshapeToUint8s().PermuteOrZeroGrouped(rotl8Idx).ReshapeToUint32s()

	// c += d; b ^= c; b <<<= 7
	c = c.Add(d)
	b = b.Xor(c)
	b = b.ShiftAllLeft(7).Or(b.ShiftAllRight(25))

	return a, b, c, d
}

// xorStore8 XORs 128 bytes of key stream (2 blocks held in the row vectors of
// one group) with src[off:off+128] and stores the result into
// dst[off:off+128]. The two blocks of the group are de-interleaved from the
// 128-bit halves with VPERM2I128.
func xorStore8(dst, src unsafe.Pointer, off int, v0, v1, v2, v3 archsimd.Uint32x8) {
	// block 2g, bytes 0..31 (rows 0-1)
	t := v0.ConcatPermute128Scalars(0, 2, v1)
	t = t.Xor(archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(src, off))))
	t.StoreArray((*[8]uint32)(unsafe.Add(dst, off)))

	// block 2g, bytes 32..63 (rows 2-3)
	t = v2.ConcatPermute128Scalars(0, 2, v3)
	t = t.Xor(archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(src, off+32))))
	t.StoreArray((*[8]uint32)(unsafe.Add(dst, off+32)))

	// block 2g+1, bytes 0..31
	t = v0.ConcatPermute128Scalars(1, 3, v1)
	t = t.Xor(archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(src, off+64))))
	t.StoreArray((*[8]uint32)(unsafe.Add(dst, off+64)))

	// block 2g+1, bytes 32..63
	t = v2.ConcatPermute128Scalars(1, 3, v3)
	t = t.Xor(archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(src, off+96))))
	t.StoreArray((*[8]uint32)(unsafe.Add(dst, off+96)))
}

// store8 stores 128 bytes of key stream held in the row vectors of one group
// into ks[off:off+128] in block-major order.
func store8(ks unsafe.Pointer, off int, v0, v1, v2, v3 archsimd.Uint32x8) {
	t := v0.ConcatPermute128Scalars(0, 2, v1)
	t.StoreArray((*[8]uint32)(unsafe.Add(ks, off)))
	t = v2.ConcatPermute128Scalars(0, 2, v3)
	t.StoreArray((*[8]uint32)(unsafe.Add(ks, off+32)))
	t = v0.ConcatPermute128Scalars(1, 3, v1)
	t.StoreArray((*[8]uint32)(unsafe.Add(ks, off+64)))
	t = v2.ConcatPermute128Scalars(1, 3, v3)
	t.StoreArray((*[8]uint32)(unsafe.Add(ks, off+96)))
}
