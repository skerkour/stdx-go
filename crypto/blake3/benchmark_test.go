package blake3_test

import (
	"strconv"
	"testing"

	"github.com/skerkour/stdx-go/crypto/blake3"
)

func BenchmarkSum256(b *testing.B) {
	data := make([]byte, 1<<20)
	for size := 64; size <= 1<<20; size *= 64 {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			b.SetBytes(int64(size))
			in := data[:size]
			b.ResetTimer()
			for b.Loop() {
				blake3.Sum256(in)
			}
		})
	}
}

func BenchmarkHasherStream(b *testing.B) {
	data := make([]byte, 1<<20)
	var out [32]byte
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for b.Loop() {
		h := blake3.New()
		h.Write(data)
		h.Sum(out[:0])
	}
}

func BenchmarkKeyedHash(b *testing.B) {
	var key [32]byte
	data := make([]byte, 1<<20)
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for b.Loop() {
		blake3.KeyedHash(key, data)
	}
}

func BenchmarkDeriveKey(b *testing.B) {
	data := make([]byte, 1<<20)
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for b.Loop() {
		blake3.DeriveKey("stdx-go blake3 benchmark", data)
	}
}

func BenchmarkXof(b *testing.B) {
	h := blake3.New()
	h.Write([]byte("xof benchmark"))
	out := make([]byte, 1<<20)
	b.SetBytes(int64(len(out)))
	b.ResetTimer()
	for b.Loop() {
		r := h.Xof()
		for p := out; len(p) > 0; {
			n := len(p)
			if n > 64 {
				n = 64
			}
			r.Read(p[:n])
			p = p[n:]
		}
	}
}
