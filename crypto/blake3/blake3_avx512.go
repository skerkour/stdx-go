//go:build amd64 && goexperiment.simd

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
//
// mx and my point at the lane-major message rows for the two message words of
// this column; the message is loaded from memory here rather than passed as
// vectors. The 16 state vectors plus the 4 rotation-count vectors already fill
// most of the 32 ZMM registers, so keeping the 16 message words as memory
// operands (folding into VPADDD where the compiler can) avoids spilling state,
// matching the C reference kernel.
func gVec512(va, vb, vc, vd archsimd.Uint32x16, mx, my *[16]uint32, r16, r12, r8, r7 archsimd.Uint32x16) (archsimd.Uint32x16, archsimd.Uint32x16, archsimd.Uint32x16, archsimd.Uint32x16) {
	va = va.Add(vb).Add(archsimd.LoadUint32x16(mx[:]))
	vd = vd.Xor(va).RotateRight(r16)
	vc = vc.Add(vd)
	vb = vb.Xor(vc).RotateRight(r12)
	va = va.Add(vb).Add(archsimd.LoadUint32x16(my[:]))
	vd = vd.Xor(va).RotateRight(r8)
	vc = vc.Add(vd)
	vb = vb.Xor(vc).RotateRight(r7)
	return va, vb, vc, vd
}

// compressChunksAvx512 hashes the simdLanes512 full 1024-byte chunks at
// indices base..base+simdLanes512-1 (lane j = chunk base+j) and writes their
// chaining values into cvs. counterBase is the BLAKE3 chunk counter of data[0].
//
// The message data for all 16 blocks of the batch is transposed into the
// lane-major layout (t[b][i] lane j = word i of block b of chunk base+j) once,
// before the block loop, so the hot loop does no scalar gather. Within the
// loop the 16 state vectors plus the 4 rotation-count vectors are the only
// long-lived vector values and the messages are read through gVec512's
// memory-operand loads, avoiding the state spills of materializing all 16
// message vectors as registers.
func compressChunksAvx512(data []byte, base int, counterBase uint64, cvs [][8]uint32, key [8]uint32, flags uint32) {
	// Transpose all 16 blocks once: t[b][i][j] = little-endian word i of the
	// 64-byte block b of chunk base+j.
	var t [16][16][simdLanesAvx512]uint32
	for b := 0; b < 16; b++ {
		for j := 0; j < simdLanesAvx512; j++ {
			blk := data[(base+j)*chunkLen+b*blockLen:]
			for i := 0; i < 16; i++ {
				t[b][i][j] = binary.LittleEndian.Uint32(blk[i*4:])
			}
		}
	}

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
		// schedule index. The message operands are pointers into the
		// transposed buffer, read fresh per round.
		// round 0
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &t[b][0], &t[b][1], r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &t[b][2], &t[b][3], r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &t[b][4], &t[b][5], r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &t[b][6], &t[b][7], r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &t[b][8], &t[b][9], r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &t[b][10], &t[b][11], r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &t[b][12], &t[b][13], r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &t[b][14], &t[b][15], r16, r12, r8, r7)
		// round 1
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &t[b][2], &t[b][6], r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &t[b][3], &t[b][10], r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &t[b][7], &t[b][0], r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &t[b][4], &t[b][13], r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &t[b][1], &t[b][11], r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &t[b][12], &t[b][5], r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &t[b][9], &t[b][14], r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &t[b][15], &t[b][8], r16, r12, r8, r7)
		// round 2
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &t[b][3], &t[b][4], r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &t[b][10], &t[b][12], r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &t[b][13], &t[b][2], r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &t[b][7], &t[b][14], r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &t[b][6], &t[b][5], r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &t[b][9], &t[b][0], r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &t[b][11], &t[b][15], r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &t[b][8], &t[b][1], r16, r12, r8, r7)
		// round 3
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &t[b][10], &t[b][7], r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &t[b][12], &t[b][9], r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &t[b][14], &t[b][3], r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &t[b][13], &t[b][15], r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &t[b][4], &t[b][0], r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &t[b][11], &t[b][2], r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &t[b][5], &t[b][8], r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &t[b][1], &t[b][6], r16, r12, r8, r7)
		// round 4
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &t[b][12], &t[b][13], r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &t[b][9], &t[b][11], r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &t[b][15], &t[b][10], r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &t[b][14], &t[b][8], r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &t[b][7], &t[b][2], r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &t[b][5], &t[b][3], r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &t[b][0], &t[b][1], r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &t[b][6], &t[b][4], r16, r12, r8, r7)
		// round 5
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &t[b][9], &t[b][14], r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &t[b][11], &t[b][5], r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &t[b][8], &t[b][12], r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &t[b][15], &t[b][1], r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &t[b][13], &t[b][3], r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &t[b][0], &t[b][10], r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &t[b][2], &t[b][6], r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &t[b][4], &t[b][7], r16, r12, r8, r7)
		// round 6
		v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &t[b][11], &t[b][15], r16, r12, r8, r7)
		v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &t[b][5], &t[b][0], r16, r12, r8, r7)
		v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &t[b][1], &t[b][9], r16, r12, r8, r7)
		v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &t[b][8], &t[b][6], r16, r12, r8, r7)
		v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &t[b][14], &t[b][10], r16, r12, r8, r7)
		v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &t[b][2], &t[b][12], r16, r12, r8, r7)
		v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &t[b][3], &t[b][4], r16, r12, r8, r7)
		v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &t[b][7], &t[b][13], r16, r12, r8, r7)

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

	// The message block is the same in every output block, so broadcast each
	// word into its own 16-lane row and pass pointers to gVec512.
	var m [16][simdLanesAvx512]uint32
	for i := 0; i < 16; i++ {
		for j := 0; j < simdLanesAvx512; j++ {
			m[i][j] = block[i]
		}
	}

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
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &m[0], &m[1], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &m[2], &m[3], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &m[4], &m[5], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &m[6], &m[7], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &m[8], &m[9], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &m[10], &m[11], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &m[12], &m[13], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &m[14], &m[15], r16, r12, r8, r7)
	// round 1
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &m[2], &m[6], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &m[3], &m[10], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &m[7], &m[0], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &m[4], &m[13], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &m[1], &m[11], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &m[12], &m[5], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &m[9], &m[14], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &m[15], &m[8], r16, r12, r8, r7)
	// round 2
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &m[3], &m[4], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &m[10], &m[12], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &m[13], &m[2], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &m[7], &m[14], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &m[6], &m[5], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &m[9], &m[0], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &m[11], &m[15], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &m[8], &m[1], r16, r12, r8, r7)
	// round 3
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &m[10], &m[7], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &m[12], &m[9], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &m[14], &m[3], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &m[13], &m[15], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &m[4], &m[0], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &m[11], &m[2], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &m[5], &m[8], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &m[1], &m[6], r16, r12, r8, r7)
	// round 4
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &m[12], &m[13], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &m[9], &m[11], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &m[15], &m[10], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &m[14], &m[8], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &m[7], &m[2], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &m[5], &m[3], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &m[0], &m[1], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &m[6], &m[4], r16, r12, r8, r7)
	// round 5
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &m[9], &m[14], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &m[11], &m[5], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &m[8], &m[12], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &m[15], &m[1], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &m[13], &m[3], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &m[0], &m[10], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &m[2], &m[6], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &m[4], &m[7], r16, r12, r8, r7)
	// round 6
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &m[11], &m[15], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &m[5], &m[0], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &m[1], &m[9], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &m[8], &m[6], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &m[14], &m[10], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &m[2], &m[12], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &m[3], &m[4], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &m[7], &m[13], r16, r12, r8, r7)

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
	// Lane-major message buffer: m[i] lane j = word i of parent j. Passed as
	// pointers to gVec512 so the messages are read from memory per round.
	var scratch [16][simdLanesAvx512]uint32
	for j := 0; j < simdLanesAvx512; j++ {
		for w := 0; w < 8; w++ {
			scratch[w][j] = left[j][w]
			scratch[w+8][j] = right[j][w]
		}
	}

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
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &scratch[0], &scratch[1], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &scratch[2], &scratch[3], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &scratch[4], &scratch[5], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &scratch[6], &scratch[7], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &scratch[8], &scratch[9], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &scratch[10], &scratch[11], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &scratch[12], &scratch[13], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &scratch[14], &scratch[15], r16, r12, r8, r7)
	// round 1
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &scratch[2], &scratch[6], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &scratch[3], &scratch[10], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &scratch[7], &scratch[0], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &scratch[4], &scratch[13], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &scratch[1], &scratch[11], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &scratch[12], &scratch[5], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &scratch[9], &scratch[14], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &scratch[15], &scratch[8], r16, r12, r8, r7)
	// round 2
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &scratch[3], &scratch[4], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &scratch[10], &scratch[12], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &scratch[13], &scratch[2], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &scratch[7], &scratch[14], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &scratch[6], &scratch[5], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &scratch[9], &scratch[0], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &scratch[11], &scratch[15], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &scratch[8], &scratch[1], r16, r12, r8, r7)
	// round 3
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &scratch[10], &scratch[7], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &scratch[12], &scratch[9], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &scratch[14], &scratch[3], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &scratch[13], &scratch[15], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &scratch[4], &scratch[0], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &scratch[11], &scratch[2], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &scratch[5], &scratch[8], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &scratch[1], &scratch[6], r16, r12, r8, r7)
	// round 4
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &scratch[12], &scratch[13], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &scratch[9], &scratch[11], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &scratch[15], &scratch[10], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &scratch[14], &scratch[8], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &scratch[7], &scratch[2], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &scratch[5], &scratch[3], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &scratch[0], &scratch[1], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &scratch[6], &scratch[4], r16, r12, r8, r7)
	// round 5
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &scratch[9], &scratch[14], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &scratch[11], &scratch[5], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &scratch[8], &scratch[12], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &scratch[15], &scratch[1], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &scratch[13], &scratch[3], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &scratch[0], &scratch[10], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &scratch[2], &scratch[6], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &scratch[4], &scratch[7], r16, r12, r8, r7)
	// round 6
	v0, v4, v8, v12 = gVec512(v0, v4, v8, v12, &scratch[11], &scratch[15], r16, r12, r8, r7)
	v1, v5, v9, v13 = gVec512(v1, v5, v9, v13, &scratch[5], &scratch[0], r16, r12, r8, r7)
	v2, v6, v10, v14 = gVec512(v2, v6, v10, v14, &scratch[1], &scratch[9], r16, r12, r8, r7)
	v3, v7, v11, v15 = gVec512(v3, v7, v11, v15, &scratch[8], &scratch[6], r16, r12, r8, r7)
	v0, v5, v10, v15 = gVec512(v0, v5, v10, v15, &scratch[14], &scratch[10], r16, r12, r8, r7)
	v1, v6, v11, v12 = gVec512(v1, v6, v11, v12, &scratch[2], &scratch[12], r16, r12, r8, r7)
	v2, v7, v8, v13 = gVec512(v2, v7, v8, v13, &scratch[3], &scratch[4], r16, r12, r8, r7)
	v3, v4, v9, v14 = gVec512(v3, v4, v9, v14, &scratch[7], &scratch[13], r16, r12, r8, r7)

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
