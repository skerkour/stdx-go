//go:build amd64 && goexperiment.simd

package blake3

import (
	"bytes"
	"math/rand"
	"testing"

	"simd/archsimd"
)

// TestAVX2PathCrossCheck verifies that the AVX2 lane kernels produce results
// identical to the scalar chunk kernel, and reports whether the runtime AVX2
// gate is active (so we know the SIMD path is really exercised).
func TestAVX2PathCrossCheck(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("host/emulated CPU lacks AVX2; SIMD kernel not exercised")
	}
	rng := rand.New(rand.NewSource(42))
	data := make([]byte, simdLanes*chunkLen)
	for i := range data {
		data[i] = byte(rng.Intn(256))
	}
	var key [8]uint32
	for i := range key {
		key[i] = rng.Uint32()
	}

	// Chunk kernel vs scalar.
	var simdCV, scalarCV [simdLanes][8]uint32
	compressChunksAvx2(data, 0, 0, simdCV[:], key, 0)
	for i := 0; i < simdLanes; i++ {
		scalarCV[i] = compressChunkCV(data, i, 0, key, 0)
	}
	for i := 0; i < simdLanes; i++ {
		if simdCV[i] != scalarCV[i] {
			t.Fatalf("chunk %d: SIMD %x scalar %x", i, simdCV[i], scalarCV[i])
		}
	}

	// XOF kernel vs scalar.
	var cv [8]uint32 = key
	var blk [16]uint32
	for i := range blk {
		blk[i] = rng.Uint32()
	}
	var simdOut, scalarOut [simdLanes * blockLen]byte
	compressOutputsAvx2(simdOut[:], &cv, &blk, blockLen, flagRoot, 7)
	compressOutputsScalar(scalarOut[:], &cv, &blk, blockLen, flagRoot, 7)
	if !bytes.Equal(simdOut[:], scalarOut[:]) {
		t.Fatal("compressOutputsLanes != scalar")
	}
}
