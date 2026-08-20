package chacha20

import (
	"crypto/rand"
	"fmt"
	"testing"

	chacha20std "golang.org/x/crypto/chacha20"
)

var benchSizes = []int{64, 512, 1024, 4096, 65536, 1048576}

func BenchmarkChaCha20(b *testing.B) {
	var key [32]byte
	var nonce [12]byte
	rand.Read(key[:])
	rand.Read(nonce[:])

	for _, size := range benchSizes {
		src := make([]byte, size)
		dst := make([]byte, size)
		rand.Read(src)

		b.Run(fmt.Sprintf("chacha20_stdlib/%d", size), func(b *testing.B) {
			b.SetBytes(int64(size))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				stdlibCipher, err := chacha20std.NewUnauthenticatedCipher(key[:], nonce[:])
				if err != nil {
					b.Fatal(err)
				}
				stdlibCipher.XORKeyStream(dst, src)
			}
		})

		b.Run(fmt.Sprintf("chacha20_simd/%d", size), func(b *testing.B) {
			b.SetBytes(int64(size))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				chacha20Cipher := NewIetf(key, nonce)
				chacha20Cipher.XORKeyStream(dst, src)
			}
		})
	}
}
