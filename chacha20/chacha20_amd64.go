//go:build amd64 && goexperiment.simd

package chacha20

import (
	"crypto/subtle"
	"simd/archsimd"
	"unsafe"
)

// xorKeyStream is the amd64 SIMD backend hook. It XORs src with the key
// stream generated from state and maintains leftover key stream state.
//
// The 256-bit AVX2 path is used when the CPU supports AVX2; otherwise the
// portable scalar backend runs.
func (cipher *CipherIetf) xorKeyStream(dst, src []byte) {
	if archsimd.X86.AVX2() && len(src) > 64 {
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
	// counter is kept as an 8-lane SIMD vector [c..c+7] and advanced in SIMD
	// per iteration, mirroring the arm64 backend.'
	counterIncrement := archsimd.BroadcastUint32x8(8)
	counter := archsimd.BroadcastUint32x8(cipher.state[12]).Add(archsimd.LoadUint32x8Array(&[8]uint32{0, 1, 2, 3, 4, 5, 6, 7}))

	// initial state
	i0 := archsimd.BroadcastUint32x8(cipher.state[0])
	i1 := archsimd.BroadcastUint32x8(cipher.state[1])
	i2 := archsimd.BroadcastUint32x8(cipher.state[2])
	i3 := archsimd.BroadcastUint32x8(cipher.state[3])
	i4 := archsimd.BroadcastUint32x8(cipher.state[4])
	i5 := archsimd.BroadcastUint32x8(cipher.state[5])
	i6 := archsimd.BroadcastUint32x8(cipher.state[6])
	i7 := archsimd.BroadcastUint32x8(cipher.state[7])
	i8 := archsimd.BroadcastUint32x8(cipher.state[8])
	i9 := archsimd.BroadcastUint32x8(cipher.state[9])
	i10 := archsimd.BroadcastUint32x8(cipher.state[10])
	i11 := archsimd.BroadcastUint32x8(cipher.state[11])
	i13 := archsimd.BroadcastUint32x8(cipher.state[13])
	i14 := archsimd.BroadcastUint32x8(cipher.state[14])
	i15 := archsimd.BroadcastUint32x8(cipher.state[15])

	for len(src) >= 512 {
		w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15 :=
			chacha8BlocksAVX2(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, counter, i13, i14, i15)

		srcPointer := unsafe.Pointer(&src[0])
		dstPointer := unsafe.Pointer(&dst[0])

		// each transpose4AVX2 result r[b] holds words 4g..4g+3 of block b
		// (low half) and block b+4 (high half)
		a0, a1, a2, a3 := transpose4AVX2(w0, w1, w2, w3)     // words 0-3
		b0, b1, b2, b3 := transpose4AVX2(w4, w5, w6, w7)     // words 4-7
		c0, c1, c2, c3 := transpose4AVX2(w8, w9, w10, w11)   // words 8-11
		d0, d1, d2, d3 := transpose4AVX2(w12, w13, w14, w15) // words 12-15

		xorStoreBlock(dstPointer, srcPointer, 0, 0, 2, a0, b0, c0, d0)
		xorStoreBlock(dstPointer, srcPointer, 64, 0, 2, a1, b1, c1, d1)
		xorStoreBlock(dstPointer, srcPointer, 128, 0, 2, a2, b2, c2, d2)
		xorStoreBlock(dstPointer, srcPointer, 192, 0, 2, a3, b3, c3, d3)
		xorStoreBlock(dstPointer, srcPointer, 256, 1, 3, a0, b0, c0, d0)
		xorStoreBlock(dstPointer, srcPointer, 320, 1, 3, a1, b1, c1, d1)
		xorStoreBlock(dstPointer, srcPointer, 384, 1, 3, a2, b2, c2, d2)
		xorStoreBlock(dstPointer, srcPointer, 448, 1, 3, a3, b3, c3, d3)

		src = src[512:]
		dst = dst[512:]
		counter = counter.Add(counterIncrement)
	}

	cipher.state[12] = counter.GetLo().GetElem(0)

	// Tail: < 512 bytes. Generate 8 blocks of key stream into a local
	// buffer, XOR the consumed part, and keep the unused bytes of the last
	// partial block as leftover (<= 63).
	if len(src) > 0 {
		var keystream [512]byte
		w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15 :=
			chacha8BlocksAVX2(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, counter, i13, i14, i15)

		a0, a1, a2, a3 := transpose4AVX2(w0, w1, w2, w3)     // words 0-3
		b0, b1, b2, b3 := transpose4AVX2(w4, w5, w6, w7)     // words 4-7
		c0, c1, c2, c3 := transpose4AVX2(w8, w9, w10, w11)   // words 8-11
		d0, d1, d2, d3 := transpose4AVX2(w12, w13, w14, w15) // words 12-15

		ks := unsafe.Pointer(&keystream[0])
		storeBlock(ks, 0, 0, 2, a0, b0, c0, d0)
		storeBlock(ks, 64, 0, 2, a1, b1, c1, d1)
		storeBlock(ks, 128, 0, 2, a2, b2, c2, d2)
		storeBlock(ks, 192, 0, 2, a3, b3, c3, d3)
		storeBlock(ks, 256, 1, 3, a0, b0, c0, d0)
		storeBlock(ks, 320, 1, 3, a1, b1, c1, d1)
		storeBlock(ks, 384, 1, 3, a2, b2, c2, d2)
		storeBlock(ks, 448, 1, 3, a3, b3, c3, d3)

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

// chacha8BlocksAVX2 computes 8 blocks of ChaCha20 key stream (20 rounds +
// add-back) and returns them in the word-major layout: register wi holds word
// i of the 8 blocks, one block per lane.
//
// The 16 initial-state vectors are passed in and the compiler keeps them in the stack frame across the round
// loop and reloads them at the add-back.
func chacha8BlocksAVX2(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15 archsimd.Uint32x8) (w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15 archsimd.Uint32x8) {
	// working state: the initial state, to be transformed by the rounds
	w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15 =
		i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15

	for r := 0; r < 10; r++ {
		w0, w4, w8, w12 = quarterRoundAVX2(w0, w4, w8, w12)
		w1, w5, w9, w13 = quarterRoundAVX2(w1, w5, w9, w13)
		w2, w6, w10, w14 = quarterRoundAVX2(w2, w6, w10, w14)
		w3, w7, w11, w15 = quarterRoundAVX2(w3, w7, w11, w15)

		w0, w5, w10, w15 = quarterRoundAVX2(w0, w5, w10, w15)
		w1, w6, w11, w12 = quarterRoundAVX2(w1, w6, w11, w12)
		w2, w7, w8, w13 = quarterRoundAVX2(w2, w7, w8, w13)
		w3, w4, w9, w14 = quarterRoundAVX2(w3, w4, w9, w14)
	}

	// add-back
	w0 = w0.Add(i0)
	w1 = w1.Add(i1)
	w2 = w2.Add(i2)
	w3 = w3.Add(i3)
	w4 = w4.Add(i4)
	w5 = w5.Add(i5)
	w6 = w6.Add(i6)
	w7 = w7.Add(i7)
	w8 = w8.Add(i8)
	w9 = w9.Add(i9)
	w10 = w10.Add(i10)
	w11 = w11.Add(i11)
	w12 = w12.Add(i12)
	w13 = w13.Add(i13)
	w14 = w14.Add(i14)
	w15 = w15.Add(i15)
	return
}

// quarterRoundAVX2 is the ChaCha20 quarter round on the word-major layout.
// All rotations use two shifts and an or with immediate counts, which on AVX2
// is the fastest option that needs no extra vector register (VPSHUFB would
// require shuffle-index registers or per-use loads, both of which measurably
// increase round-loop register pressure; there is no single-instruction
// 32-bit rotate before AVX512).
func quarterRoundAVX2(a, b, c, d archsimd.Uint32x8) (archsimd.Uint32x8, archsimd.Uint32x8, archsimd.Uint32x8, archsimd.Uint32x8) {
	// a += b; d ^= a; d <<<= 16
	a = a.Add(b)
	d = d.Xor(a)
	d = d.ShiftAllLeft(16).Or(d.ShiftAllRight(16))

	// c += d; b ^= c; b <<<= 12
	c = c.Add(d)
	b = b.Xor(c)
	b = b.ShiftAllLeft(12).Or(b.ShiftAllRight(20))

	// a += b; d ^= a; d <<<= 8
	a = a.Add(b)
	d = d.Xor(a)
	d = d.ShiftAllLeft(8).Or(d.ShiftAllRight(24))

	// c += d; b ^= c; b <<<= 7
	c = c.Add(d)
	b = b.Xor(c)
	b = b.ShiftAllLeft(7).Or(b.ShiftAllRight(25))

	return a, b, c, d
}

// transpose4AVX2 transposes one word group of 4 word-major vectors (each
// holding word i of the 8 blocks, one block per lane) into 4 vectors holding
// 4 consecutive words of each block. r[b] holds words of block b (low 128-bit
// half) and block b+4 (high 128-bit half), for b in 0..3. The transpose is
// performed within each 128-bit lane, so it handles blocks 0-3 and 4-7 at
// once.
func transpose4AVX2(a, b, c, d archsimd.Uint32x8) (archsimd.Uint32x8, archsimd.Uint32x8, archsimd.Uint32x8, archsimd.Uint32x8) {
	t0 := a.InterleaveLoGrouped(b) // [a0 b0 a1 b1 | a4 b4 a5 b5]
	t1 := a.InterleaveHiGrouped(b) // [a2 b2 a3 b3 | a6 b6 a7 b7]
	t2 := c.InterleaveLoGrouped(d) // [c0 d0 c1 d1 | c4 d4 c5 d5]
	t3 := c.InterleaveHiGrouped(d) // [c2 d2 c3 d3 | c6 d6 c7 d7]
	return t0.ConcatPermuteScalarsGrouped(0, 1, 4, 5, t2),
		t0.ConcatPermuteScalarsGrouped(2, 3, 6, 7, t2),
		t1.ConcatPermuteScalarsGrouped(0, 1, 4, 5, t3),
		t1.ConcatPermuteScalarsGrouped(2, 3, 6, 7, t3)
}

// xorStoreBlock XORs 64 bytes of key stream (one block, whose words are held
// in the four transposed vectors r0..r3, selecting the low or high 128-bit
// halves with sel0/sel1 = 0,2 or 1,3) with src[off:off+64] and stores the
// result into dst[off:off+64].
func xorStoreBlock(dst, src unsafe.Pointer, off int, sel0, sel1 uint8, r0, r1, r2, r3 archsimd.Uint32x8) {
	t := r0.ConcatPermute128Scalars(sel0, sel1, r1)
	t = t.Xor(archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(src, off))))
	t.StoreArray((*[8]uint32)(unsafe.Add(dst, off)))

	t = r2.ConcatPermute128Scalars(sel0, sel1, r3)
	t = t.Xor(archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(src, off+32))))
	t.StoreArray((*[8]uint32)(unsafe.Add(dst, off+32)))
}

// storeBlock stores 64 bytes of key stream (one block, held in the four
// transposed vectors r0..r3, selecting the low or high 128-bit halves with
// sel0/sel1 = 0,2 or 1,3) into ks[off:off+64].
func storeBlock(ks unsafe.Pointer, off int, sel0, sel1 uint8, r0, r1, r2, r3 archsimd.Uint32x8) {
	t := r0.ConcatPermute128Scalars(sel0, sel1, r1)
	t.StoreArray((*[8]uint32)(unsafe.Add(ks, off)))

	t = r2.ConcatPermute128Scalars(sel0, sel1, r3)
	t.StoreArray((*[8]uint32)(unsafe.Add(ks, off+32)))
}
