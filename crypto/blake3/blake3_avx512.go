//go:build amd64

package blake3

import (
	"encoding/binary"
	"simd/archsimd"
)

// This file is the experimental, cgo-free, assembly-free SIMD acceleration of
// BLAKE3's per-chunk compression for amd64 with AVX-512, written with Go's
// experimental simd/archsimd intrinsics. It only builds on amd64 under
// GOEXPERIMENT=simd, and the dispatch in blake3_simd_amd64.go gates it at
// runtime on archsimd.X86.AVX512(); CPUs without AVX-512 fall back to the AVX2
// or scalar paths.
//
// 512-bit registers hold 16 uint32 lanes, so we hash 16 full 1024-byte chunks
// at once — one chunk per lane. The 16 compression state words become 16
// vectors; the quarter-round mixing is pure elementwise Add/Xor, and each
// 16/12/8/7-bit right rotation is a single VPRORVD (Uint32x16.RotateRight
// against a broadcast count vector). The message and state vectors are named
// locals and the 7 rounds are fully unrolled. Like chacha20's AVX-512 core, the
// 16 state vectors and 4 rotation-count vectors occupy 20 of the 32 ZMM
// registers, leaving room for the message vectors and temporaries so the round
// loop spills as little as possible. The per-block message is gathered with a
// scalar transpose into the lane-major layout (m[i] lane j = word i of chunk
// j). Results are bit-identical to the scalar path.

// simdLanesAvx512 is the number of chunks (or output blocks) per AVX-512 batch.
const simdLanesAvx512 = 16

// rotCounts holds the broadcast rotation counts backing the four VPRORVD
// rotations of each quarter round. They are plain data (no SIMD instructions at
// init, so the package initializes on CPUs without AVX-512); the archsimd
// vectors are materialized lazily inside the AVX-512 kernels.
var rotCounts = [4][16]uint32{
	{16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},
	{12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
	{8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8},
	{7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7},
}

// bcast512 returns a vector with x in every lane.
func bcast512(x uint32) archsimd.Uint32x16 {
	a := [16]uint32{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x}
	return archsimd.LoadUint32x16(a[:])
}

// gVec512 is the BLAKE3 quarter round on 16 lanes at once. The rotations are
// right by 16/12/8/7, matching the scalar RotateLeft32(x, -k), each a single
// VPRORVD against the loop-invariant count vectors r16/r12/r8/r7.
func gVec512(va, vb, vc, vd, mx, my, r16, r12, r8, r7 archsimd.Uint32x16) (archsimd.Uint32x16, archsimd.Uint32x16, archsimd.Uint32x16, archsimd.Uint32x16) {
	va = va.Add(vb).Add(mx)
	vd = vd.Xor(va).RotateRight(r16)
	vc = vc.Add(vd)
	vb = vb.Xor(vc).RotateRight(r12)
	va = va.Add(vb).Add(my)
	vd = vd.Xor(va).RotateRight(r8)
	vc = vc.Add(vd)
	vb = vb.Xor(vc).RotateRight(r7)
	return va, vb, vc, vd
}

// compressChunksAvx512 hashes the simdLanes512 full 1024-byte chunks at
// indices base..base+simdLanes512-1 (lane j = chunk base+j) and writes their
// chaining values into cvs. counterBase is the BLAKE3 chunk counter of data[0].
//
// The 16 message vectors and 16 state vectors are held as distinct named
// locals (no indexed arrays) and the 7 rounds are fully unrolled with static
// message indices; 16 state + 4 count vectors fit alongside the messages in
// the 32 ZMM registers.
func compressChunksAvx512(data []byte, base int, counterBase uint64, cvs [][8]uint32, key [8]uint32, flags uint32) {
	// Per-lane chunk counters (chunk index), split into low/high 32 bits.
	var ctrLo, ctrHi [simdLanesAvx512]uint32
	for j := 0; j < simdLanesAvx512; j++ {
		c := counterBase + uint64(base+j)
		ctrLo[j] = uint32(c)
		ctrHi[j] = uint32(c >> 32)
	}
	vCtrLo := archsimd.LoadUint32x16(ctrLo[:])
	vCtrHi := archsimd.LoadUint32x16(ctrHi[:])
	vBlockLen := bcast512(blockLen)

	// Chaining value and IV, one broadcast vector per word.
	cv0 := bcast512(key[0])
	cv1 := bcast512(key[1])
	cv2 := bcast512(key[2])
	cv3 := bcast512(key[3])
	cv4 := bcast512(key[4])
	cv5 := bcast512(key[5])
	cv6 := bcast512(key[6])
	cv7 := bcast512(key[7])
	iv0 := bcast512(iv[0])
	iv1 := bcast512(iv[1])
	iv2 := bcast512(iv[2])
	iv3 := bcast512(iv[3])

	// VPRORVD count vectors, loop-invariant.
	r16 := archsimd.LoadUint32x16Array(&rotCounts[0])
	r12 := archsimd.LoadUint32x16Array(&rotCounts[1])
	r8 := archsimd.LoadUint32x16Array(&rotCounts[2])
	r7 := archsimd.LoadUint32x16Array(&rotCounts[3])

	// A full chunk is 16 blocks of 64 bytes; block 0 carries CHUNK_START and
	// block 15 carries CHUNK_END, mirroring chunkState.update + output().
	for b := 0; b < 16; b++ {
		// Gather each chunk's 16 message words and transpose 16 chunks at a
		// time so m[i] lane j = word i of chunk j.
		var scratch [16][simdLanesAvx512]uint32
		for j := 0; j < simdLanesAvx512; j++ {
			blk := data[(base+j)*chunkLen+b*blockLen:]
			for i := 0; i < 16; i++ {
				scratch[i][j] = binary.LittleEndian.Uint32(blk[i*4:])
			}
		}
		m0 := archsimd.LoadUint32x16(scratch[0][:])
		m1 := archsimd.LoadUint32x16(scratch[1][:])
		m2 := archsimd.LoadUint32x16(scratch[2][:])
		m3 := archsimd.LoadUint32x16(scratch[3][:])
		m4 := archsimd.LoadUint32x16(scratch[4][:])
		m5 := archsimd.LoadUint32x16(scratch[5][:])
		m6 := archsimd.LoadUint32x16(scratch[6][:])
		m7 := archsimd.LoadUint32x16(scratch[7][:])
		m8 := archsimd.LoadUint32x16(scratch[8][:])
		m9 := archsimd.LoadUint32x16(scratch[9][:])
		m10 := archsimd.LoadUint32x16(scratch[10][:])
		m11 := archsimd.LoadUint32x16(scratch[11][:])
		m12 := archsimd.LoadUint32x16(scratch[12][:])
		m13 := archsimd.LoadUint32x16(scratch[13][:])
		m14 := archsimd.LoadUint32x16(scratch[14][:])
		m15 := archsimd.LoadUint32x16(scratch[15][:])

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
		v12, v13, v14, v15 := vCtrLo, vCtrHi, vBlockLen, bcast512(fl)

		// 7 unrolled rounds; each line is one gVec512 with the static message
		// schedule index.
		// round 0
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m0, m1, r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m2, m3, r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m4, m5, r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m6, m7, r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m8, m9, r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m10, m11, r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m12, m13, r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m14, m15, r16, r12, r8, r7)
		// round 1
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m2, m6, r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m3, m10, r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m7, m0, r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m4, m13, r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m1, m11, r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m12, m5, r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m9, m14, r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m15, m8, r16, r12, r8, r7)
		// round 2
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m3, m4, r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m10, m12, r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m13, m2, r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m7, m14, r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m6, m5, r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m9, m0, r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m11, m15, r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m8, m1, r16, r12, r8, r7)
		// round 3
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m10, m7, r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m12, m9, r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m14, m3, r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m13, m15, r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m4, m0, r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m11, m2, r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m5, m8, r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m1, m6, r16, r12, r8, r7)
		// round 4
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m12, m13, r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m9, m11, r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m15, m10, r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m14, m8, r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m7, m2, r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m5, m3, r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m0, m1, r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m6, m4, r16, r12, r8, r7)
		// round 5
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m9, m14, r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m11, m5, r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m8, m12, r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m15, m1, r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m13, m3, r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m0, m10, r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m2, m6, r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m4, m7, r16, r12, r8, r7)
		// round 6
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m11, m15, r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m5, m0, r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m1, m9, r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m8, m6, r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m14, m10, r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m2, m12, r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m3, m4, r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m7, m13, r16, r12, r8, r7)

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
	var lane [simdLanesAvx512]uint32
	cv0.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[base+j][0] = lane[j]
	}
	cv1.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[base+j][1] = lane[j]
	}
	cv2.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[base+j][2] = lane[j]
	}
	cv3.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[base+j][3] = lane[j]
	}
	cv4.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[base+j][4] = lane[j]
	}
	cv5.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[base+j][5] = lane[j]
	}
	cv6.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[base+j][6] = lane[j]
	}
	cv7.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[base+j][7] = lane[j]
	}
}

// compressOutputsAvx512 computes simdLanes512 64-byte root output blocks
// starting at block start. The message block and input CV are constant across
// lanes; only the counter varies. As with compressChunksLanes512, the message
// and state vectors are named locals and the rounds are fully unrolled.
func compressOutputsAvx512(out []byte, cv *[8]uint32, block *[16]uint32, blkLen, flags uint32, start uint64) {
	var ctrLo, ctrHi [simdLanesAvx512]uint32
	for j := 0; j < simdLanesAvx512; j++ {
		c := start + uint64(j)
		ctrLo[j] = uint32(c)
		ctrHi[j] = uint32(c >> 32)
	}
	vCtrLo := archsimd.LoadUint32x16(ctrLo[:])
	vCtrHi := archsimd.LoadUint32x16(ctrHi[:])
	vBlockLen := bcast512(blkLen)
	vFlags := bcast512(flags)

	cv0 := bcast512(cv[0])
	cv1 := bcast512(cv[1])
	cv2 := bcast512(cv[2])
	cv3 := bcast512(cv[3])
	cv4 := bcast512(cv[4])
	cv5 := bcast512(cv[5])
	cv6 := bcast512(cv[6])
	cv7 := bcast512(cv[7])
	iv0 := bcast512(iv[0])
	iv1 := bcast512(iv[1])
	iv2 := bcast512(iv[2])
	iv3 := bcast512(iv[3])

	m0 := bcast512(block[0])
	m1 := bcast512(block[1])
	m2 := bcast512(block[2])
	m3 := bcast512(block[3])
	m4 := bcast512(block[4])
	m5 := bcast512(block[5])
	m6 := bcast512(block[6])
	m7 := bcast512(block[7])
	m8 := bcast512(block[8])
	m9 := bcast512(block[9])
	m10 := bcast512(block[10])
	m11 := bcast512(block[11])
	m12 := bcast512(block[12])
	m13 := bcast512(block[13])
	m14 := bcast512(block[14])
	m15 := bcast512(block[15])

	r16 := archsimd.LoadUint32x16Array(&rotCounts[0])
	r12 := archsimd.LoadUint32x16Array(&rotCounts[1])
	r8 := archsimd.LoadUint32x16Array(&rotCounts[2])
	r7 := archsimd.LoadUint32x16Array(&rotCounts[3])

	v0, v1, v2, v3 := cv0, cv1, cv2, cv3
	v4, v5, v6, v7 := cv4, cv5, cv6, cv7
	v8, v9, v10, v11 := iv0, iv1, iv2, iv3
	v12, v13, v14, v15 := vCtrLo, vCtrHi, vBlockLen, vFlags

	// 7 unrolled rounds; the message schedule is the same as the chunk kernel.
	// round 0
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m0, m1, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m2, m3, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m4, m5, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m6, m7, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m8, m9, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m10, m11, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m12, m13, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m14, m15, r16, r12, r8, r7)
	// round 1
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m2, m6, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m3, m10, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m7, m0, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m4, m13, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m1, m11, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m12, m5, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m9, m14, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m15, m8, r16, r12, r8, r7)
	// round 2
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m3, m4, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m10, m12, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m13, m2, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m7, m14, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m6, m5, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m9, m0, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m11, m15, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m8, m1, r16, r12, r8, r7)
	// round 3
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m10, m7, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m12, m9, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m14, m3, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m13, m15, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m4, m0, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m11, m2, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m5, m8, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m1, m6, r16, r12, r8, r7)
	// round 4
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m12, m13, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m9, m11, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m15, m10, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m14, m8, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m7, m2, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m5, m3, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m0, m1, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m6, m4, r16, r12, r8, r7)
	// round 5
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m9, m14, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m11, m5, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m8, m12, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m15, m1, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m13, m3, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m0, m10, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m2, m6, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m4, m7, r16, r12, r8, r7)
	// round 6
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m11, m15, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m5, m0, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m1, m9, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m8, m6, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m14, m10, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m2, m12, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m3, m4, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m7, m13, r16, r12, r8, r7)

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
	var lane [simdLanesAvx512]uint32
	v0.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen:], lane[j])
	}
	v1.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+4:], lane[j])
	}
	v2.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+8:], lane[j])
	}
	v3.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+12:], lane[j])
	}
	v4.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+16:], lane[j])
	}
	v5.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+20:], lane[j])
	}
	v6.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+24:], lane[j])
	}
	v7.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+28:], lane[j])
	}
	v8.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+32:], lane[j])
	}
	v9.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+36:], lane[j])
	}
	v10.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+40:], lane[j])
	}
	v11.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+44:], lane[j])
	}
	v12.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+48:], lane[j])
	}
	v13.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+52:], lane[j])
	}
	v14.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+56:], lane[j])
	}
	v15.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		binary.LittleEndian.PutUint32(out[j*blockLen+60:], lane[j])
	}
}

// compressParentsAvx512 computes the chaining value of 16 parent nodes in one
// AVX-512 batch. Parent j's block is [left[j], right[j]]; the message is
// already lane-major (no transpose needed), the counter is 0, and the flags
// carry flagParent. Only the first 8 output words (the CV) are needed.
func compressParentsAvx512(left, right *[simdLanesAvx512][8]uint32, key [8]uint32, flags uint32) [simdLanesAvx512][8]uint32 {
	// Lane-major message vectors: m[i] lane j = word i of parent j.
	var scratch [16][simdLanesAvx512]uint32
	for j := 0; j < simdLanesAvx512; j++ {
		for w := 0; w < 8; w++ {
			scratch[w][j] = left[j][w]
			scratch[w+8][j] = right[j][w]
		}
	}
	m0 := archsimd.LoadUint32x16(scratch[0][:])
	m1 := archsimd.LoadUint32x16(scratch[1][:])
	m2 := archsimd.LoadUint32x16(scratch[2][:])
	m3 := archsimd.LoadUint32x16(scratch[3][:])
	m4 := archsimd.LoadUint32x16(scratch[4][:])
	m5 := archsimd.LoadUint32x16(scratch[5][:])
	m6 := archsimd.LoadUint32x16(scratch[6][:])
	m7 := archsimd.LoadUint32x16(scratch[7][:])
	m8 := archsimd.LoadUint32x16(scratch[8][:])
	m9 := archsimd.LoadUint32x16(scratch[9][:])
	m10 := archsimd.LoadUint32x16(scratch[10][:])
	m11 := archsimd.LoadUint32x16(scratch[11][:])
	m12 := archsimd.LoadUint32x16(scratch[12][:])
	m13 := archsimd.LoadUint32x16(scratch[13][:])
	m14 := archsimd.LoadUint32x16(scratch[14][:])
	m15 := archsimd.LoadUint32x16(scratch[15][:])

	cv0 := bcast512(key[0])
	cv1 := bcast512(key[1])
	cv2 := bcast512(key[2])
	cv3 := bcast512(key[3])
	cv4 := bcast512(key[4])
	cv5 := bcast512(key[5])
	cv6 := bcast512(key[6])
	cv7 := bcast512(key[7])
	iv0 := bcast512(iv[0])
	iv1 := bcast512(iv[1])
	iv2 := bcast512(iv[2])
	iv3 := bcast512(iv[3])

	// Loop-invariant VPRORVD count vectors.
	r16 := archsimd.LoadUint32x16Array(&rotCounts[0])
	r12 := archsimd.LoadUint32x16Array(&rotCounts[1])
	r8 := archsimd.LoadUint32x16Array(&rotCounts[2])
	r7 := archsimd.LoadUint32x16Array(&rotCounts[3])

	v0, v1, v2, v3 := cv0, cv1, cv2, cv3
	v4, v5, v6, v7 := cv4, cv5, cv6, cv7
	v8, v9, v10, v11 := iv0, iv1, iv2, iv3
	v12 := bcast512(0)
	v13 := bcast512(0)
	v14 := bcast512(blockLen)
	v15 := bcast512(flags | flagParent)

	// round 0
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m0, m1, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m2, m3, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m4, m5, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m6, m7, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m8, m9, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m10, m11, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m12, m13, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m14, m15, r16, r12, r8, r7)
	// round 1
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m2, m6, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m3, m10, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m7, m0, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m4, m13, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m1, m11, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m12, m5, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m9, m14, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m15, m8, r16, r12, r8, r7)
	// round 2
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m3, m4, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m10, m12, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m13, m2, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m7, m14, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m6, m5, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m9, m0, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m11, m15, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m8, m1, r16, r12, r8, r7)
	// round 3
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m10, m7, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m12, m9, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m14, m3, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m13, m15, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m4, m0, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m11, m2, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m5, m8, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m1, m6, r16, r12, r8, r7)
	// round 4
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m12, m13, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m9, m11, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m15, m10, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m14, m8, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m7, m2, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m5, m3, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m0, m1, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m6, m4, r16, r12, r8, r7)
	// round 5
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m9, m14, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m11, m5, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m8, m12, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m15, m1, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m13, m3, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m0, m10, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m2, m6, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m4, m7, r16, r12, r8, r7)
	// round 6
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, m11, m15, r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, m5, m0, r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, m1, m9, r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, m8, m6, r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, m14, m10, r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, m2, m12, r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, m3, m4, r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, m7, m13, r16, r12, r8, r7)

	// CV update: word i = v[i]^v[i+8], then scatter across the 16 parents.
	v0 = v0.Xor(v8)
	v1 = v1.Xor(v9)
	v2 = v2.Xor(v10)
	v3 = v3.Xor(v11)
	v4 = v4.Xor(v12)
	v5 = v5.Xor(v13)
	v6 = v6.Xor(v14)
	v7 = v7.Xor(v15)

	var lane [simdLanesAvx512]uint32
	var cvs [simdLanesAvx512][8]uint32
	v0.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[j][0] = lane[j]
	}
	v1.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[j][1] = lane[j]
	}
	v2.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[j][2] = lane[j]
	}
	v3.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[j][3] = lane[j]
	}
	v4.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[j][4] = lane[j]
	}
	v5.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[j][5] = lane[j]
	}
	v6.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[j][6] = lane[j]
	}
	v7.Store(lane[:])
	for j := 0; j < simdLanesAvx512; j++ {
		cvs[j][7] = lane[j]
	}
	return cvs
}
