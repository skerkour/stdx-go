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

// simChacha8Blocks simulates chacha8BlocksAVX2 and returns the 8 64-byte
// blocks in block-major order.
func simChacha8Blocks(state [16]uint32, counter uint32) [8][16]uint32 {
	var lo, hi simReg
	for i := 0; i < 8; i++ {
		lo[i] = state[i]
		hi[i] = state[8+i]
	}
	row0 := simReg{lo[0], lo[1], lo[2], lo[3], lo[0], lo[1], lo[2], lo[3]}
	row1 := simReg{lo[4], lo[5], lo[6], lo[7], lo[4], lo[5], lo[6], lo[7]}
	row2 := simReg{hi[0], hi[1], hi[2], hi[3], hi[0], hi[1], hi[2], hi[3]}
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

	var blocks [8][16]uint32
	for gid := 0; gid < 4; gid++ {
		// block 2*gid: low halves, block 2*gid+1: high halves
		for r := 0; r < 4; r++ {
			for w := 0; w < 4; w++ {
				blocks[2*gid][4*r+w] = g[gid][r][w]
				blocks[2*gid+1][4*r+w] = g[gid][r][4+w]
			}
		}
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
