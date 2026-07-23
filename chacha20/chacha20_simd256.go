package chacha20

import (
	"crypto/subtle"
	"encoding/binary"
	"simd/archsimd"
)

func XORKeyStream4(key [32]byte, nonce [12]byte, counter uint32, dst, src []byte) {
	// load constant
	i0 := archsimd.BroadcastUint32x4(0x61707865)
	i1 := archsimd.BroadcastUint32x4(0x3320646e)
	i2 := archsimd.BroadcastUint32x4(0x79622d32)
	i3 := archsimd.BroadcastUint32x4(0x6b206574)

	// load key
	i4 := archsimd.BroadcastUint32x4(binary.LittleEndian.Uint32(key[0:4]))
	i5 := archsimd.BroadcastUint32x4(binary.LittleEndian.Uint32(key[4:8]))
	i6 := archsimd.BroadcastUint32x4(binary.LittleEndian.Uint32(key[8:12]))
	i7 := archsimd.BroadcastUint32x4(binary.LittleEndian.Uint32(key[12:16]))
	i8 := archsimd.BroadcastUint32x4(binary.LittleEndian.Uint32(key[16:20]))
	i9 := archsimd.BroadcastUint32x4(binary.LittleEndian.Uint32(key[20:24]))
	i10 := archsimd.BroadcastUint32x4(binary.LittleEndian.Uint32(key[24:28]))
	i11 := archsimd.BroadcastUint32x4(binary.LittleEndian.Uint32(key[28:32]))

	// load nonce
	i13 := archsimd.BroadcastUint32x4(binary.LittleEndian.Uint32(nonce[0:4]))
	i14 := archsimd.BroadcastUint32x4(binary.LittleEndian.Uint32(nonce[4:8]))
	i15 := archsimd.BroadcastUint32x4(binary.LittleEndian.Uint32(nonce[8:12]))

	for len(src) > 0 {
		i12 := archsimd.LoadUint32x4Array(&[4]uint32{counter, counter + 1, counter + 2, counter + 3})

		w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15 :=
			i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15

		for r := 0; r < 10; r++ {
			w0, w4, w8, w12 = quarterRound4(w0, w4, w8, w12)
			w1, w5, w9, w13 = quarterRound4(w1, w5, w9, w13)
			w2, w6, w10, w14 = quarterRound4(w2, w6, w10, w14)
			w3, w7, w11, w15 = quarterRound4(w3, w7, w11, w15)

			w0, w5, w10, w15 = quarterRound4(w0, w5, w10, w15)
			w1, w6, w11, w12 = quarterRound4(w1, w6, w11, w12)
			w2, w7, w8, w13 = quarterRound4(w2, w7, w8, w13)
			w3, w4, w9, w14 = quarterRound4(w3, w4, w9, w14)
		}

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

		var keystream [256]byte

		// Block 0
		binary.LittleEndian.PutUint32(keystream[0:], w0.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[4:], w1.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[8:], w2.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[12:], w3.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[16:], w4.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[20:], w5.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[24:], w6.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[28:], w7.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[32:], w8.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[36:], w9.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[40:], w10.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[44:], w11.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[48:], w12.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[52:], w13.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[56:], w14.GetElem(0))
		binary.LittleEndian.PutUint32(keystream[60:], w15.GetElem(0))

		// Block 1
		binary.LittleEndian.PutUint32(keystream[64:], w0.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[68:], w1.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[72:], w2.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[76:], w3.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[80:], w4.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[84:], w5.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[88:], w6.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[92:], w7.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[96:], w8.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[100:], w9.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[104:], w10.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[108:], w11.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[112:], w12.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[116:], w13.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[120:], w14.GetElem(1))
		binary.LittleEndian.PutUint32(keystream[124:], w15.GetElem(1))

		// Block 2
		binary.LittleEndian.PutUint32(keystream[128:], w0.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[132:], w1.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[136:], w2.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[140:], w3.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[144:], w4.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[148:], w5.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[152:], w6.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[156:], w7.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[160:], w8.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[164:], w9.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[168:], w10.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[172:], w11.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[176:], w12.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[180:], w13.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[184:], w14.GetElem(2))
		binary.LittleEndian.PutUint32(keystream[188:], w15.GetElem(2))

		// Block 3
		binary.LittleEndian.PutUint32(keystream[192:], w0.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[196:], w1.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[200:], w2.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[204:], w3.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[208:], w4.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[212:], w5.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[216:], w6.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[220:], w7.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[224:], w8.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[228:], w9.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[232:], w10.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[236:], w11.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[240:], w12.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[244:], w13.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[248:], w14.GetElem(3))
		binary.LittleEndian.PutUint32(keystream[252:], w15.GetElem(3))

		// n := 256
		// if len(src) < n {
		// 	n = len(src)
		// }
		// for i := 0; i < n; i++ {
		// 	dst[i] = src[i] ^ ks[i]
		// }
		n := subtle.XORBytes(dst, src, keystream[:])
		src = src[n:]
		dst = dst[n:]
		counter += 4
	}
}

func quarterRound4(a, b, c, d archsimd.Uint32x4) (archsimd.Uint32x4, archsimd.Uint32x4, archsimd.Uint32x4, archsimd.Uint32x4) {
	// a += b; d ^= a; d <<<= 16
	a = a.Add(b)
	d = d.Xor(a)
	d = d.RotateAllLeft(16)

	// c += d; b ^= c; b <<<= 12
	c = c.Add(d)
	b = b.Xor(c)
	b = b.RotateAllLeft(12)

	// a += b; d ^= a; d <<<= 8
	a = a.Add(b)
	d = d.Xor(a)
	d = d.RotateAllLeft(8)

	// c += d; b ^= c; b <<<= 7
	c = c.Add(d)
	b = b.Xor(c)
	b = b.RotateAllLeft(7)

	return a, b, c, d
}
