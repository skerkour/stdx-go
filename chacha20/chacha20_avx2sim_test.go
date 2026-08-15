//go:build goexperiment.simd

package chacha20

import (
	"encoding/binary"
	"testing"
)

// This file simulates the amd64/AVX2 8-block implementation
// (chacha20_amd64.go) in pure Go, so that the word-major layout, the quarter
// rounds and the 4x4 transpose can be validated on any host, without AVX2
// hardware. It mirrors chacha8BlocksAVX2 and transpose4AVX2 operation for
// operation.

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

func simBroadcast(v uint32) simReg {
	var r simReg
	for i := range r {
		r[i] = v
	}
	return r
}

// simCounterVec mirrors counterVec: lane i holds counter+i.
func simCounterVec(counter uint32) simReg {
	return simReg{counter, counter + 1, counter + 2, counter + 3, counter + 4, counter + 5, counter + 6, counter + 7}
}

func simQuarterRound(a, b, c, d simReg) (simReg, simReg, simReg, simReg) {
	a = simAdd(a, b)
	d = simXor(d, a)
	d = simOr(simShl(d, 16), simShr(d, 16))

	c = simAdd(c, d)
	b = simXor(b, c)
	b = simOr(simShl(b, 12), simShr(b, 20))

	a = simAdd(a, b)
	d = simXor(d, a)
	d = simOr(simShl(d, 8), simShr(d, 24))

	c = simAdd(c, d)
	b = simXor(b, c)
	b = simOr(simShl(b, 7), simShr(b, 25))

	return a, b, c, d
}

// simInterleaveLo implements InterleaveLoGrouped: within each 128-bit lane,
// [a0 b0 a1 b1].
func simInterleaveLo(a, b simReg) simReg {
	var r simReg
	for lane := 0; lane < 2; lane++ {
		base := lane * 4
		r[base+0] = a[base+0]
		r[base+1] = b[base+0]
		r[base+2] = a[base+1]
		r[base+3] = b[base+1]
	}
	return r
}

// simInterleaveHi implements InterleaveHiGrouped: within each 128-bit lane,
// [a2 b2 a3 b3].
func simInterleaveHi(a, b simReg) simReg {
	var r simReg
	for lane := 0; lane < 2; lane++ {
		base := lane * 4
		r[base+0] = a[base+2]
		r[base+1] = b[base+2]
		r[base+2] = a[base+3]
		r[base+3] = b[base+3]
	}
	return r
}

// simConcatPermuteGrouped implements ConcatPermuteScalarsGrouped: per 128-bit
// lane, select 4 elements where selectors 0-3 index x and 4-7 index y.
func simConcatPermuteGrouped(x, y simReg, a, b, c, d int) simReg {
	sel := [4]int{a, b, c, d}
	var r simReg
	for lane := 0; lane < 2; lane++ {
		base := lane * 4
		for i := 0; i < 4; i++ {
			if sel[i] < 4 {
				r[base+i] = x[base+sel[i]]
			} else {
				r[base+i] = y[base+sel[i]-4]
			}
		}
	}
	return r
}

// simTranspose4 mirrors transpose4AVX2.
func simTranspose4(a, b, c, d simReg) (simReg, simReg, simReg, simReg) {
	t0 := simInterleaveLo(a, b)
	t1 := simInterleaveHi(a, b)
	t2 := simInterleaveLo(c, d)
	t3 := simInterleaveHi(c, d)
	return simConcatPermuteGrouped(t0, t2, 0, 1, 4, 5),
		simConcatPermuteGrouped(t0, t2, 2, 3, 6, 7),
		simConcatPermuteGrouped(t1, t3, 0, 1, 4, 5),
		simConcatPermuteGrouped(t1, t3, 2, 3, 6, 7)
}

// simChacha8Blocks simulates chacha8BlocksAVX2 + transpose4AVX2 and returns
// the 8 64-byte blocks in block-major order.
func simChacha8Blocks(state [16]uint32, counter uint32) [8][16]uint32 {
	var w [16]simReg
	for i := 0; i < 16; i++ {
		if i == 12 {
			w[i] = simCounterVec(counter)
		} else {
			w[i] = simBroadcast(state[i])
		}
	}

	for r := 0; r < 10; r++ {
		w[0], w[4], w[8], w[12] = simQuarterRound(w[0], w[4], w[8], w[12])
		w[1], w[5], w[9], w[13] = simQuarterRound(w[1], w[5], w[9], w[13])
		w[2], w[6], w[10], w[14] = simQuarterRound(w[2], w[6], w[10], w[14])
		w[3], w[7], w[11], w[15] = simQuarterRound(w[3], w[7], w[11], w[15])

		w[0], w[5], w[10], w[15] = simQuarterRound(w[0], w[5], w[10], w[15])
		w[1], w[6], w[11], w[12] = simQuarterRound(w[1], w[6], w[11], w[12])
		w[2], w[7], w[8], w[13] = simQuarterRound(w[2], w[7], w[8], w[13])
		w[3], w[4], w[9], w[14] = simQuarterRound(w[3], w[4], w[9], w[14])
	}

	// add-back (constant reload)
	for i := 0; i < 16; i++ {
		if i == 12 {
			w[i] = simAdd(w[i], simCounterVec(counter))
		} else {
			w[i] = simAdd(w[i], simBroadcast(state[i]))
		}
	}

	// transpose each word group and de-interleave into blocks
	var tr [4][4]simReg
	for g := 0; g < 4; g++ {
		tr[g][0], tr[g][1], tr[g][2], tr[g][3] = simTranspose4(w[4*g], w[4*g+1], w[4*g+2], w[4*g+3])
	}

	var blocks [8][16]uint32
	for b := 0; b < 4; b++ {
		for g := 0; g < 4; g++ {
			// tr[g][b]: block b (low half) and block b+4 (high half), words 4g..4g+3
			copy(blocks[b][4*g:4*g+4], tr[g][b][0:4])
			copy(blocks[b+4][4*g:4*g+4], tr[g][b][4:8])
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
