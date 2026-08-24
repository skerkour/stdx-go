//go:build amd64

package blake3

import (
	"encoding/binary"
	"simd/archsimd"
)

// This file is the experimental, cgo-free, assembly-free SIMD acceleration of
// BLAKE3's per-chunk compression for amd64 (AVX2), written with Go's
// experimental simd/archsimd intrinsics. It only builds on amd64 under
// GOEXPERIMENT=simd; the generic build (blake3_generic.go) keeps every other
// configuration scalar, and a runtime AVX2 check keeps CPUs without AVX2 on
// the scalar path.
//
// 256-bit registers hold 8 uint32 lanes, so we hash 8 full 1024-byte chunks at
// once — one chunk per lane. The 16 compression state words become 16 vectors;
// the quarter-round mixing is pure elementwise Add/Xor, with the byte-aligned
// 16/8-bit rotations done as a single VPSHUFB (PermuteOrZeroGrouped) and the
// 12/7-bit rotations as a shift-or (VPSRLD/VPSLLD). The message and state
// vectors are named locals and the 7 rounds are fully unrolled, so the compiler
// can keep the hot working set in registers and schedule around the 16-register
// pressure of AVX2. The per-block message is gathered with a scalar transpose
// into the lane-major layout (m[i] lane j = word i of chunk j). Results are
// bit-identical to the scalar path.

// simdLanes is the number of chunks (or output blocks) per batch.
const simdLanes = 8

// fillChunkCVs hashes chunks per SIMD batch, synchronously on the calling
// goroutine. All work is stack-scoped, so nothing allocates. The AVX-512
// kernel (16 chunks/batch) is used when available, else the AVX2 kernel
// (8 chunks/batch), else the scalar path on CPUs without AVX2.
func fillChunkCVs(data []byte, cvs [][8]uint32, base uint64, key [8]uint32, flags uint32) {
	if !archsimd.X86.AVX2() {
		fillChunkCVsScalar(data, cvs, base, key, flags)
		return
	}
	n := len(cvs)
	batch := 0
	if archsimd.X86.AVX512() {
		for ; batch+simdLanesAvx512 <= n; batch += simdLanesAvx512 {
			compressChunksAvx512(data, batch, base, cvs, key, flags)
		}
	}
	for ; batch+simdLanes <= n; batch += simdLanes {
		compressChunksAvx2(data, batch, base, cvs, key, flags)
	}
	for ; batch < n; batch++ {
		cvs[batch] = compressChunkCV(data, batch, base, key, flags)
	}
}

// compressOutputs fills out with the extendable root output, simdLanesAvx512
// blocks per AVX-512 batch (or simdLanes per AVX2 batch). Each output block
// uses a distinct counter (the per-lane counter vectors) against the same
// message block and input CV.
func compressOutputs(out []byte, cv *[8]uint32, block *[16]uint32, blkLen, flags uint32, start uint64) {
	if !archsimd.X86.AVX2() {
		compressOutputsScalar(out, cv, block, blkLen, flags, start)
		return
	}
	if archsimd.X86.AVX512() {
		for len(out) >= simdLanesAvx512*blockLen {
			compressOutputsAvx512(out, cv, block, blkLen, flags, start)
			out = out[simdLanesAvx512*blockLen:]
			start += simdLanesAvx512
		}
	}
	for len(out) >= simdLanes*blockLen {
		compressOutputsAvx2(out, cv, block, blkLen, flags, start)
		out = out[simdLanes*blockLen:]
		start += simdLanes
	}
	compressOutputsScalar(out, cv, block, blkLen, flags, start)
}

// rotr16Idx and rotr8Idx are the VPSHUFB index vectors that rotate each 32-bit
// element of a Uint8x32 right by 16/8 bits. They are kept as plain byte arrays
// (no SIMD instructions at init) and materialized into vectors by the compiler
// inside the kernels. rotr8Idx is {1,2,3,0,...}: rotr8 of a 32-bit word moves
// its least-significant byte to the most-significant position.
var (
	rotr16Idx = [32]int8{
		2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
		2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
	}
	rotr8Idx = [32]int8{
		1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12,
		1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12,
	}
)

// rotr16 and rotr8 rotate each 32-bit lane right by 16/8 bits with a single
// VPSHUFB (byte permute within each 4-byte group) instead of shift+or.
func rotr16(x archsimd.Uint32x8) archsimd.Uint32x8 {
	return x.ReshapeToUint8s().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&rotr16Idx)).ReshapeToUint32s()
}

func rotr8(x archsimd.Uint32x8) archsimd.Uint32x8 {
	return x.ReshapeToUint8s().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&rotr8Idx)).ReshapeToUint32s()
}

// rotr rotates each 32-bit lane right by k via shift+or for the non-byte
// aligned rotations (12 and 7), which have no single-instruction VPSHUFB.
func rotr(x archsimd.Uint32x8, k uint64) archsimd.Uint32x8 {
	return x.ShiftAllRight(k).Or(x.ShiftAllLeft(32 - k))
}

// gVec is the BLAKE3 quarter round on simdLanes lanes at once. The rotations
// are right by 16/12/8/7, matching the scalar RotateLeft32(x, -k).
func gVec(va, vb, vc, vd, mx, my archsimd.Uint32x8) (archsimd.Uint32x8, archsimd.Uint32x8, archsimd.Uint32x8, archsimd.Uint32x8) {
	va = va.Add(vb).Add(mx)
	vd = rotr16(vd.Xor(va))
	vc = vc.Add(vd)
	vb = rotr(vb.Xor(vc), 12)
	va = va.Add(vb).Add(my)
	vd = rotr8(vd.Xor(va))
	vc = vc.Add(vd)
	vb = rotr(vb.Xor(vc), 7)
	return va, vb, vc, vd
}

// bcast returns a vector with x in every lane.
func bcast(x uint32) archsimd.Uint32x8 {
	a := [8]uint32{x, x, x, x, x, x, x, x}
	return archsimd.LoadUint32x8(a[:])
}

// compressChunksAvx2 hashes the simdLanes full 1024-byte chunks at indices
// base..base+simdLanes-1 (lane j = chunk base+j) and writes their chaining
// values into cvs. counterBase is the BLAKE3 chunk counter of data[0].
//
// The 16 message vectors and 16 state vectors are held as distinct named
// locals (no indexed arrays) and the 7 rounds are fully unrolled with static
// message indices, so the Go compiler can schedule around AVX2's 16-register
// pressure instead of forcing every quarter round through memory.
func compressChunksAvx2(data []byte, base int, counterBase uint64, cvs [][8]uint32, key [8]uint32, flags uint32) {
	// Per-lane chunk counters (chunk index), split into low/high 32 bits.
	var ctrLo, ctrHi [simdLanes]uint32
	for j := 0; j < simdLanes; j++ {
		c := counterBase + uint64(base+j)
		ctrLo[j] = uint32(c)
		ctrHi[j] = uint32(c >> 32)
	}
	vCtrLo := archsimd.LoadUint32x8(ctrLo[:])
	vCtrHi := archsimd.LoadUint32x8(ctrHi[:])
	vBlockLen := bcast(blockLen)

	// Chaining value and IV, one broadcast vector per word.
	cv0 := bcast(key[0])
	cv1 := bcast(key[1])
	cv2 := bcast(key[2])
	cv3 := bcast(key[3])
	cv4 := bcast(key[4])
	cv5 := bcast(key[5])
	cv6 := bcast(key[6])
	cv7 := bcast(key[7])
	iv0 := bcast(iv[0])
	iv1 := bcast(iv[1])
	iv2 := bcast(iv[2])
	iv3 := bcast(iv[3])

	// A full chunk is 16 blocks of 64 bytes; block 0 carries CHUNK_START and
	// block 15 carries CHUNK_END, mirroring chunkState.update + output().
	for b := 0; b < 16; b++ {
		// Gather each chunk's 16 message words and transpose 8 chunks at a
		// time so m[i] lane j = word i of chunk j.
		var scratch [16][simdLanes]uint32
		for j := 0; j < simdLanes; j++ {
			blk := data[(base+j)*chunkLen+b*blockLen:]
			for i := 0; i < 16; i++ {
				scratch[i][j] = binary.LittleEndian.Uint32(blk[i*4:])
			}
		}
		m0 := archsimd.LoadUint32x8(scratch[0][:])
		m1 := archsimd.LoadUint32x8(scratch[1][:])
		m2 := archsimd.LoadUint32x8(scratch[2][:])
		m3 := archsimd.LoadUint32x8(scratch[3][:])
		m4 := archsimd.LoadUint32x8(scratch[4][:])
		m5 := archsimd.LoadUint32x8(scratch[5][:])
		m6 := archsimd.LoadUint32x8(scratch[6][:])
		m7 := archsimd.LoadUint32x8(scratch[7][:])
		m8 := archsimd.LoadUint32x8(scratch[8][:])
		m9 := archsimd.LoadUint32x8(scratch[9][:])
		m10 := archsimd.LoadUint32x8(scratch[10][:])
		m11 := archsimd.LoadUint32x8(scratch[11][:])
		m12 := archsimd.LoadUint32x8(scratch[12][:])
		m13 := archsimd.LoadUint32x8(scratch[13][:])
		m14 := archsimd.LoadUint32x8(scratch[14][:])
		m15 := archsimd.LoadUint32x8(scratch[15][:])

		fl := flags
		if b == 0 {
			fl |= flagChunkStart
		}
		if b == 15 {
			fl |= flagChunkEnd
		}

		v0, v1, v2, v3 := cv0, cv1, cv2, cv3
		v4, v5, v6, v7 := cv4, cv5, cv6, cv7
		v8, v9, v10, v11 := iv0, iv1, iv2, iv3
		v12, v13, v14, v15 := vCtrLo, vCtrHi, vBlockLen, bcast(fl)

		// 7 unrolled rounds; each line is one gVec with the static message
		// schedule index.
		// round 0
		v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m0, m1)
		v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m2, m3)
		v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m4, m5)
		v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m6, m7)
		v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m8, m9)
		v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m10, m11)
		v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m12, m13)
		v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m14, m15)
		// round 1
		v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m2, m6)
		v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m3, m10)
		v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m7, m0)
		v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m4, m13)
		v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m1, m11)
		v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m12, m5)
		v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m9, m14)
		v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m15, m8)
		// round 2
		v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m3, m4)
		v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m10, m12)
		v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m13, m2)
		v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m7, m14)
		v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m6, m5)
		v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m9, m0)
		v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m11, m15)
		v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m8, m1)
		// round 3
		v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m10, m7)
		v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m12, m9)
		v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m14, m3)
		v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m13, m15)
		v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m4, m0)
		v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m11, m2)
		v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m5, m8)
		v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m1, m6)
		// round 4
		v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m12, m13)
		v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m9, m11)
		v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m15, m10)
		v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m14, m8)
		v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m7, m2)
		v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m5, m3)
		v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m0, m1)
		v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m6, m4)
		// round 5
		v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m9, m14)
		v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m11, m5)
		v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m8, m12)
		v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m15, m1)
		v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m13, m3)
		v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m0, m10)
		v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m2, m6)
		v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m4, m7)
		// round 6
		v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m11, m15)
		v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m5, m0)
		v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m1, m9)
		v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m8, m6)
		v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m14, m10)
		v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m2, m12)
		v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m3, m4)
		v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m7, m13)

		cv0 = v0.Xor(v8)
		cv1 = v1.Xor(v9)
		cv2 = v2.Xor(v10)
		cv3 = v3.Xor(v11)
		cv4 = v4.Xor(v12)
		cv5 = v5.Xor(v13)
		cv6 = v6.Xor(v14)
		cv7 = v7.Xor(v15)
	}

	// Scatter lane j of each CV word back to chunk base+j.
	var lane [simdLanes]uint32
	cv0.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[base+j][0] = lane[j]
	}
	cv1.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[base+j][1] = lane[j]
	}
	cv2.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[base+j][2] = lane[j]
	}
	cv3.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[base+j][3] = lane[j]
	}
	cv4.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[base+j][4] = lane[j]
	}
	cv5.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[base+j][5] = lane[j]
	}
	cv6.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[base+j][6] = lane[j]
	}
	cv7.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[base+j][7] = lane[j]
	}
}

// compressOutputsAvx2 computes simdLanes 64-byte root output blocks starting
// at block start. The message block and input CV are constant across lanes;
// only the counter varies. As with compressChunksLanes, the message and state
// vectors are named locals and the rounds are fully unrolled.
func compressOutputsAvx2(out []byte, cv *[8]uint32, block *[16]uint32, blkLen, flags uint32, start uint64) {
	var ctrLo, ctrHi [simdLanes]uint32
	for j := 0; j < simdLanes; j++ {
		c := start + uint64(j)
		ctrLo[j] = uint32(c)
		ctrHi[j] = uint32(c >> 32)
	}
	vCtrLo := archsimd.LoadUint32x8(ctrLo[:])
	vCtrHi := archsimd.LoadUint32x8(ctrHi[:])
	vBlockLen := bcast(blkLen)
	vFlags := bcast(flags)

	cv0 := bcast(cv[0])
	cv1 := bcast(cv[1])
	cv2 := bcast(cv[2])
	cv3 := bcast(cv[3])
	cv4 := bcast(cv[4])
	cv5 := bcast(cv[5])
	cv6 := bcast(cv[6])
	cv7 := bcast(cv[7])
	iv0 := bcast(iv[0])
	iv1 := bcast(iv[1])
	iv2 := bcast(iv[2])
	iv3 := bcast(iv[3])

	m0 := bcast(block[0])
	m1 := bcast(block[1])
	m2 := bcast(block[2])
	m3 := bcast(block[3])
	m4 := bcast(block[4])
	m5 := bcast(block[5])
	m6 := bcast(block[6])
	m7 := bcast(block[7])
	m8 := bcast(block[8])
	m9 := bcast(block[9])
	m10 := bcast(block[10])
	m11 := bcast(block[11])
	m12 := bcast(block[12])
	m13 := bcast(block[13])
	m14 := bcast(block[14])
	m15 := bcast(block[15])

	v0, v1, v2, v3 := cv0, cv1, cv2, cv3
	v4, v5, v6, v7 := cv4, cv5, cv6, cv7
	v8, v9, v10, v11 := iv0, iv1, iv2, iv3
	v12, v13, v14, v15 := vCtrLo, vCtrHi, vBlockLen, vFlags

	// 7 unrolled rounds; the message schedule is the same as the chunk kernel.
	// round 0
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m0, m1)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m2, m3)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m4, m5)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m6, m7)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m8, m9)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m10, m11)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m12, m13)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m14, m15)
	// round 1
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m2, m6)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m3, m10)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m7, m0)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m4, m13)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m1, m11)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m12, m5)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m9, m14)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m15, m8)
	// round 2
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m3, m4)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m10, m12)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m13, m2)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m7, m14)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m6, m5)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m9, m0)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m11, m15)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m8, m1)
	// round 3
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m10, m7)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m12, m9)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m14, m3)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m13, m15)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m4, m0)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m11, m2)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m5, m8)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m1, m6)
	// round 4
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m12, m13)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m9, m11)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m15, m10)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m14, m8)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m7, m2)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m5, m3)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m0, m1)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m6, m4)
	// round 5
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m9, m14)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m11, m5)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m8, m12)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m15, m1)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m13, m3)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m0, m10)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m2, m6)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m4, m7)
	// round 6
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m11, m15)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m5, m0)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m1, m9)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m8, m6)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m14, m10)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m2, m12)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m3, m4)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m7, m13)

	// Final output XOR: state[i] ^= state[i+8], state[i+8] ^= cv[i].
	v0 = v0.Xor(v8)
	v1 = v1.Xor(v9)
	v2 = v2.Xor(v10)
	v3 = v3.Xor(v11)
	v4 = v4.Xor(v12)
	v5 = v5.Xor(v13)
	v6 = v6.Xor(v14)
	v7 = v7.Xor(v15)
	v8 = v8.Xor(cv0)
	v9 = v9.Xor(cv1)
	v10 = v10.Xor(cv2)
	v11 = v11.Xor(cv3)
	v12 = v12.Xor(cv4)
	v13 = v13.Xor(cv5)
	v14 = v14.Xor(cv6)
	v15 = v15.Xor(cv7)

	// Scatter the word-major output vectors into the block-major byte layout:
	// word i of block j lands at out[j*64+i*4].
	var lane [simdLanes]uint32
	v0.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen:], lane[j])
	}
	v1.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+4:], lane[j])
	}
	v2.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+8:], lane[j])
	}
	v3.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+12:], lane[j])
	}
	v4.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+16:], lane[j])
	}
	v5.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+20:], lane[j])
	}
	v6.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+24:], lane[j])
	}
	v7.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+28:], lane[j])
	}
	v8.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+32:], lane[j])
	}
	v9.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+36:], lane[j])
	}
	v10.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+40:], lane[j])
	}
	v11.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+44:], lane[j])
	}
	v12.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+48:], lane[j])
	}
	v13.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+52:], lane[j])
	}
	v14.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+56:], lane[j])
	}
	v15.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+60:], lane[j])
	}
}

// mergeParentCVs computes the parent chaining value of each pair (src[2i],
// src[2i+1]) into out[i], using the 8-lane AVX2 parent kernel in batches.
func mergeParentCVs(out, src [][8]uint32, key [8]uint32, flags uint32) {
	if !archsimd.X86.AVX2() {
		for i := 0; i < len(out); i++ {
			out[i] = parentCV(src[2*i], src[2*i+1], key, flags)
		}
		return
	}
	// Prefer the 16-lane AVX-512 parent kernel when available.
	if archsimd.X86.AVX512() {
		n := len(out)
		i := 0
		for ; i+simdLanesAvx512 <= n; i += simdLanesAvx512 {
			var left, right [simdLanesAvx512][8]uint32
			for j := 0; j < simdLanesAvx512; j++ {
				left[j] = src[2*(i+j)]
				right[j] = src[2*(i+j)+1]
			}
			cvs := compressParentsAvx512(&left, &right, key, flags)
			for j := 0; j < simdLanesAvx512; j++ {
				out[i+j] = cvs[j]
			}
		}
		for ; i < n; i++ {
			out[i] = parentCV(src[2*i], src[2*i+1], key, flags)
		}
		return
	}
	n := len(out)
	i := 0
	for ; i+simdLanes <= n; i += simdLanes {
		var left, right [simdLanes][8]uint32
		for j := 0; j < simdLanes; j++ {
			left[j] = src[2*(i+j)]
			right[j] = src[2*(i+j)+1]
		}
		cvs := compressParentsLanes(&left, &right, key, flags)
		for j := 0; j < simdLanes; j++ {
			out[i+j] = cvs[j]
		}
	}
	for ; i < n; i++ {
		out[i] = parentCV(src[2*i], src[2*i+1], key, flags)
	}
}

// compressParentsLanes computes the chaining value of 8 parent nodes in one
// AVX2 batch. Parent j's block is [left[j], right[j]]; the message is already
// lane-major (no transpose needed), the counter is 0, and the flags carry
// flagParent. Only the first 8 output words (the CV) are needed.
func compressParentsLanes(left, right *[simdLanes][8]uint32, key [8]uint32, flags uint32) [simdLanes][8]uint32 {
	// Lane-major message vectors: m[i] lane j = word i of parent j.
	var scratch [16][simdLanes]uint32
	for j := 0; j < simdLanes; j++ {
		for w := 0; w < 8; w++ {
			scratch[w][j] = left[j][w]
			scratch[w+8][j] = right[j][w]
		}
	}
	m0 := archsimd.LoadUint32x8(scratch[0][:])
	m1 := archsimd.LoadUint32x8(scratch[1][:])
	m2 := archsimd.LoadUint32x8(scratch[2][:])
	m3 := archsimd.LoadUint32x8(scratch[3][:])
	m4 := archsimd.LoadUint32x8(scratch[4][:])
	m5 := archsimd.LoadUint32x8(scratch[5][:])
	m6 := archsimd.LoadUint32x8(scratch[6][:])
	m7 := archsimd.LoadUint32x8(scratch[7][:])
	m8 := archsimd.LoadUint32x8(scratch[8][:])
	m9 := archsimd.LoadUint32x8(scratch[9][:])
	m10 := archsimd.LoadUint32x8(scratch[10][:])
	m11 := archsimd.LoadUint32x8(scratch[11][:])
	m12 := archsimd.LoadUint32x8(scratch[12][:])
	m13 := archsimd.LoadUint32x8(scratch[13][:])
	m14 := archsimd.LoadUint32x8(scratch[14][:])
	m15 := archsimd.LoadUint32x8(scratch[15][:])

	cv0 := bcast(key[0])
	cv1 := bcast(key[1])
	cv2 := bcast(key[2])
	cv3 := bcast(key[3])
	cv4 := bcast(key[4])
	cv5 := bcast(key[5])
	cv6 := bcast(key[6])
	cv7 := bcast(key[7])
	iv0 := bcast(iv[0])
	iv1 := bcast(iv[1])
	iv2 := bcast(iv[2])
	iv3 := bcast(iv[3])

	v0, v1, v2, v3 := cv0, cv1, cv2, cv3
	v4, v5, v6, v7 := cv4, cv5, cv6, cv7
	v8, v9, v10, v11 := iv0, iv1, iv2, iv3
	v12 := bcast(0)
	v13 := bcast(0)
	v14 := bcast(blockLen)
	v15 := bcast(flags | flagParent)

	// round 0
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m0, m1)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m2, m3)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m4, m5)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m6, m7)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m8, m9)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m10, m11)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m12, m13)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m14, m15)
	// round 1
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m2, m6)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m3, m10)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m7, m0)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m4, m13)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m1, m11)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m12, m5)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m9, m14)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m15, m8)
	// round 2
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m3, m4)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m10, m12)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m13, m2)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m7, m14)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m6, m5)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m9, m0)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m11, m15)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m8, m1)
	// round 3
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m10, m7)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m12, m9)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m14, m3)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m13, m15)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m4, m0)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m11, m2)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m5, m8)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m1, m6)
	// round 4
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m12, m13)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m9, m11)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m15, m10)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m14, m8)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m7, m2)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m5, m3)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m0, m1)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m6, m4)
	// round 5
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m9, m14)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m11, m5)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m8, m12)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m15, m1)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m13, m3)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m0, m10)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m2, m6)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m4, m7)
	// round 6
	v0, v4, v8, v12 = gVec(v0, v4, v8, v12, m11, m15)
	v1, v5, v9, v13 = gVec(v1, v5, v9, v13, m5, m0)
	v2, v6, v10, v14 = gVec(v2, v6, v10, v14, m1, m9)
	v3, v7, v11, v15 = gVec(v3, v7, v11, v15, m8, m6)
	v0, v5, v10, v15 = gVec(v0, v5, v10, v15, m14, m10)
	v1, v6, v11, v12 = gVec(v1, v6, v11, v12, m2, m12)
	v2, v7, v8, v13 = gVec(v2, v7, v8, v13, m3, m4)
	v3, v4, v9, v14 = gVec(v3, v4, v9, v14, m7, m13)

	// CV update: word i = v[i]^v[i+8], then scatter across the 8 parents.
	v0 = v0.Xor(v8)
	v1 = v1.Xor(v9)
	v2 = v2.Xor(v10)
	v3 = v3.Xor(v11)
	v4 = v4.Xor(v12)
	v5 = v5.Xor(v13)
	v6 = v6.Xor(v14)
	v7 = v7.Xor(v15)

	var lane [simdLanes]uint32
	var cvs [simdLanes][8]uint32
	v0.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[j][0] = lane[j]
	}
	v1.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[j][1] = lane[j]
	}
	v2.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[j][2] = lane[j]
	}
	v3.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[j][3] = lane[j]
	}
	v4.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[j][4] = lane[j]
	}
	v5.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[j][5] = lane[j]
	}
	v6.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[j][6] = lane[j]
	}
	v7.Store(lane[:])
	for j := 0; j < simdLanes; j++ {
		cvs[j][7] = lane[j]
	}
	return cvs
}
