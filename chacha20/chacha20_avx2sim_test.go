//go:build goexperiment.simd

package chacha20

import (
	"encoding/binary"
	"testing"
)

// This file simulates the amd64/AVX2 8-block implementation
// (chacha20_amd64.go) in pure Go, so that the row layout, row shuffles and
// VPSHUFB rotation constants can be validated on any host, without AVX2
// hardware. It mirrors chacha8BlocksAVX2 operation for operation.

var (
	simRotl8Idx  = [32]int8{3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14, 3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14}
	simRotl16Idx = [32]int8{2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13}
)

type simReg [8]uint32

func simAdd(x, y simReg) simReg {
	for i := range x {
		x[i] += y[i]
	}
	return x
}

func simXor(x, y simReg) simReg {
	for i := range x {
		x[i] ^= y[i]
	}
	return x
}

func simOr(x, y simReg) simReg {
	for i := range x {
		x[i] |= y[i]
	}
	return x
}

func simShl(x simReg, n uint) simReg {
	for i := range x {
		x[i] <<= n
	}
	return x
}

func simShr(x simReg, n uint) simReg {
	for i := range x {
		x[i] >>= n
	}
	return x
}

// simPerm implements Uint32x8.PermuteScalarsGrouped: the 4 selected elements
// are applied independently to each 128-bit lane.
func simPerm(x simReg, a, b, c, d int) simReg {
	return simReg{x[a], x[b], x[c], x[d], x[a+4], x[b+4], x[c+4], x[d+4]}
}

// simPshufbRot implements the VPSHUFB-based rotation of chacha20_amd64.go:
// reshape to bytes, byte-permute within each 128-bit lane, reshape back.
func simPshufbRot(x simReg, idx [32]int8) simReg {
	var buf [32]uint8
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(buf[i*4:], x[i])
	}
	var out [32]uint8
	for lane := 0; lane < 2; lane++ {
		for i := 0; i < 16; i++ {
			s := idx[lane*16+i]
			if s < 0 {
				out[lane*16+i] = 0
			} else {
				out[lane*16+i] = buf[lane*16+int(s)]
			}
		}
	}
	var r simReg
	for i := 0; i < 8; i++ {
		r[i] = binary.LittleEndian.Uint32(out[i*4:])
	}
	return r
}

func simQuarterRound(a, b, c, d simReg) (simReg, simReg, simReg, simReg) {
	a = simAdd(a, b)
	d = simXor(d, a)
	d = simPshufbRot(d, simRotl16Idx)

	c = simAdd(c, d)
	b = simXor(b, c)
	b = simOr(simShl(b, 12), simShr(b, 20))

	a = simAdd(a, b)
	d = simXor(d, a)
	d = simPshufbRot(d, simRotl8Idx)

	c = simAdd(c, d)
	b = simXor(b, c)
	b = simOr(simShl(b, 7), simShr(b, 25))

	return a, b, c, d
}

// simConcat128 implements Uint32x8.ConcatPermute128Scalars: the 256-bit
// vectors x and y are treated as four 128-bit elements [x.lo, x.hi, y.lo,
// y.hi]; the result is the concatenation [elem[lo], elem[hi]].
func simConcat128(x, y simReg, lo, hi uint8) simReg {
	elem := [4][4]uint32{
		{x[0], x[1], x[2], x[3]},
		{x[4], x[5], x[6], x[7]},
		{y[0], y[1], y[2], y[3]},
		{y[4], y[5], y[6], y[7]},
	}
	var r simReg
	copy(r[:4], elem[lo][:])
	copy(r[4:], elem[hi][:])
	return r
}

// simChacha8Blocks simulates chacha8BlocksAVX2 and returns the 8 64-byte
// blocks in block-major order.
func simChacha8Blocks(state [16]uint32, counter uint32) [8][16]uint32 {
	var lo, hi simReg
	for i := 0; i < 8; i++ {
		lo[i] = state[i]
		hi[i] = state[8+i]
	}
	row0 := simConcat128(lo, lo, 0, 0)
	row1 := simConcat128(lo, lo, 1, 1)
	row2 := simConcat128(hi, hi, 0, 0)
	row3 := func(g uint32) simReg {
		return simReg{counter + 2*g, hi[5], hi[6], hi[7], counter + 2*g + 1, hi[5], hi[6], hi[7]}
	}

	var g [4][4]simReg
	g[0] = [4]simReg{row0, row1, row2, row3(0)}
	g[1] = [4]simReg{row0, row1, row2, row3(1)}
	g[2] = [4]simReg{row0, row1, row2, row3(2)}
	g[3] = [4]simReg{row0, row1, row2, row3(3)}

	for r := 0; r < 10; r++ {
		for i := 0; i < 4; i++ {
			g[i][0], g[i][1], g[i][2], g[i][3] = simQuarterRound(g[i][0], g[i][1], g[i][2], g[i][3])
		}
		for i := 0; i < 4; i++ {
			g[i][1] = simPerm(g[i][1], 1, 2, 3, 0)
			g[i][2] = simPerm(g[i][2], 2, 3, 0, 1)
			g[i][3] = simPerm(g[i][3], 3, 0, 1, 2)
		}
		for i := 0; i < 4; i++ {
			g[i][0], g[i][1], g[i][2], g[i][3] = simQuarterRound(g[i][0], g[i][1], g[i][2], g[i][3])
		}
		for i := 0; i < 4; i++ {
			g[i][1] = simPerm(g[i][1], 3, 0, 1, 2)
			g[i][2] = simPerm(g[i][2], 2, 3, 0, 1)
			g[i][3] = simPerm(g[i][3], 1, 2, 3, 0)
		}
	}

	// add-back
	for i := 0; i < 4; i++ {
		g[i][0] = simAdd(g[i][0], row0)
		g[i][1] = simAdd(g[i][1], row1)
		g[i][2] = simAdd(g[i][2], row2)
		g[i][3] = simAdd(g[i][3], row3(uint32(i)))
	}

	// de-interleave each group into its two blocks, using the same
	// ConcatPermute128Scalars arguments as store8/xorStore8.
	var blocks [8][16]uint32
	for gid := 0; gid < 4; gid++ {
		v0, v1, v2, v3 := g[gid][0], g[gid][1], g[gid][2], g[gid][3]
		c0 := simConcat128(v0, v1, 0, 2) // block 2g rows 0-1 (words 0-7)
		c1 := simConcat128(v2, v3, 0, 2) // block 2g rows 2-3 (words 8-15)
		c2 := simConcat128(v0, v1, 1, 3) // block 2g+1 rows 0-1 (words 0-7)
		c3 := simConcat128(v2, v3, 1, 3) // block 2g+1 rows 2-3 (words 8-15)
		copy(blocks[2*gid][0:8], c0[:])
		copy(blocks[2*gid][8:16], c1[:])
		copy(blocks[2*gid+1][0:8], c2[:])
		copy(blocks[2*gid+1][8:16], c3[:])
	}
	return blocks
}

func TestAVX2Layout(t *testing.T) {
	key, nonce := testKeyNonce()
	var state [16]uint32
	copy(state[:4], constant[:])
	for i := 0; i < 8; i++ {
		state[4+i] = binary.LittleEndian.Uint32(key[i*4 : i*4+4])
	}
	for i := 0; i < 3; i++ {
		state[13+i] = binary.LittleEndian.Uint32(nonce[i*4 : i*4+4])
	}

	for _, counter := range []uint32{0, 1, 0xfffffffc, 0xffffffff - 7} {
		state[12] = counter
		blocks := simChacha8Blocks(state, counter)
		for b := 0; b < 8; b++ {
			var out [64]byte
			for i := 0; i < 16; i++ {
				binary.LittleEndian.PutUint32(out[i*4:], blocks[b][i])
			}
			if out != chacha20Block(state) {
				t.Fatalf("AVX2 layout mismatch: counter=%d block=%d", counter, b)
			}
			state[12]++
		}
	}
}
