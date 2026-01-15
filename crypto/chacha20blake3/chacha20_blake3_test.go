package chacha20blake3_test

import (
	"bytes"
	"encoding/hex"
	"testing"

	"github.com/skerkour/stdx-go/crypto/chacha20blake3"
)

func toHex(bits []byte) string {
	return hex.EncodeToString(bits)
}

func fromHex(bits string) []byte {
	b, err := hex.DecodeString(bits)
	if err != nil {
		panic(err)
	}
	return b
}

func TestVectorsChaCha20Blake3(t *testing.T) {
	for i, v := range chacha20Blake3Vectors {
		dst := make([]byte, len(v.plaintext)+chacha20blake3.TagSize)

		cipher, err := chacha20blake3.New(v.key)
		if err != nil {
			t.Errorf("plaintext: %s", toHex(v.plaintext))
			t.Errorf("nonce: %s", toHex(v.nonce))
			t.Fatal(err)
		}

		cipher.Seal(dst[:0], v.nonce, v.plaintext, v.additionalData)
		if !bytes.Equal(dst, v.ciphertext) {
			t.Errorf("Test %d: ciphertext mismatch:\ngot:  %s\nwant: %s", i, toHex(dst), toHex(v.ciphertext))
		}

		decryptedPlaintext, err := cipher.Open(nil, v.nonce, dst, v.additionalData)
		if err != nil {
			t.Errorf("Test %d: %v", i, err)
		}
		if !bytes.Equal(decryptedPlaintext, v.plaintext) {
			t.Errorf("Test %d: plaintext mismatch:\ngot:  %s\nwant: %s", i, toHex(decryptedPlaintext), toHex(v.plaintext))
		}
	}
}

// func TestBasicX(t *testing.T) {
// 	var key [chacha20blake3.KeySize]byte
// 	var nonce [chacha20blake3.NonceSizeX]byte

// 	originalPlaintext := []byte("Hello World")
// 	additionalData := []byte("!")

// 	rand.Read(key[:])
// 	rand.Read(nonce[:])

// 	cipher, _ := chacha20blake3.NewX(key[:])
// 	ciphertext := cipher.Seal(nil, nonce[:], originalPlaintext, additionalData)

// 	decryptedPlaintext, err := cipher.Open(nil, nonce[:], ciphertext, additionalData)
// 	if err != nil {
// 		t.Errorf("decrypting message: %s", err)
// 		return
// 	}

// 	if !bytes.Equal(decryptedPlaintext, originalPlaintext) {
// 		t.Errorf("original message (%s) != decrypted message (%s)", string(originalPlaintext), string(decryptedPlaintext))
// 		return
// 	}
// }

// func TestAdditionalDataX(t *testing.T) {
// 	var key [chacha20blake3.KeySize]byte
// 	var nonce [chacha20blake3.NonceSizeX]byte

// 	originalPlaintext := []byte("Hello World")
// 	additionalData := []byte("!")

// 	rand.Read(key[:])
// 	rand.Read(nonce[:])

// 	cipher, _ := chacha20blake3.NewX(key[:])
// 	ciphertext := cipher.Seal(nil, nonce[:], originalPlaintext, additionalData)

// 	_, err := cipher.Open(nil, nonce[:], ciphertext, []byte{})
// 	if !errors.Is(err, chacha20blake3.ErrOpen) {
// 		t.Errorf("expected error (%s) | got (%s)", chacha20blake3.ErrOpen, err)
// 		return
// 	}
// }

var chacha20Blake3Vectors = []struct {
	key            []byte
	nonce          []byte
	plaintext      []byte
	additionalData []byte
	ciphertext     []byte
}{
	{
		fromHex("0000000000000000000000000000000000000000000000000000000000000000"),
		fromHex("0000000000000000000000000000000000000000000000000000000000000000"),
		fromHex("48656c6c6f20576f726c6421"), //  Hello World!
		nil,
		fromHex("272659c2b44c64b01cb0fc383f9d2664e940dd410bd752390cf701dacd2d847980ee4e8a82bf87a1d57771fd"),
	},
	{
		fromHex("0100000000000000000000000000000000000000000000000000000000000010"),
		fromHex("0100000000000000000000000000000000000000000000000000000000000010"),
		fromHex("4368614368613230"), // ChaCha20
		fromHex("424c414b4533"),     // BLAKE3
		fromHex("f219d44d74d66e28cee192edc71ead02fa79897a195c36a21512fbd0625490a4cff26c0855d6d5bc"),
	},
	{
		fromHex("3eb02a239a2a66de159b9bb5486ccc10a6f63ddf5862ef076650513372353622"),
		fromHex("768e9bda14afb5686cc34de26210f9ff6fa1dfadc64ee3f0793e4979a30fc304"),
		fromHex("b8f60975cd7057a003ac84df00d514624fe40cb7855c50dd6594f59b3a2580e5"),
		fromHex("c8d69ca92da6c5fd22f1805179fcd36cb7a9d45848fa346ba7118c2f34d23a48"),
		fromHex("1e72c231100afaa0d4922faf0536a0d2ce139f21d4009fa3268595e5246f6c3c2fdf26cc4723d5bba49ed6c7f050b3db2eec2353419fa3d1ddfc908f0e120984"),
	},
}
