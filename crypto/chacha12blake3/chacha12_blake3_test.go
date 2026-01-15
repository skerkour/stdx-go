package chacha12blake3_test

import (
	"bytes"
	"encoding/hex"
	"testing"

	"github.com/skerkour/stdx-go/crypto/chacha12blake3"
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

func TestVectorsChaCha12Blake3(t *testing.T) {
	for i, v := range chacha12Blake3Vectors {
		dst := make([]byte, len(v.plaintext)+chacha12blake3.TagSize)

		cipher, err := chacha12blake3.New(v.key)
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

var chacha12Blake3Vectors = []struct {
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
		fromHex("cd0db778a52f260a7bfe1539767c8cc8e95d8bb0587632615dfab2b54b59439488fc47ef37c073f00d8e5ff5"),
	},
	{
		fromHex("0100000000000000000000000000000000000000000000000000000000000010"),
		fromHex("0100000000000000000000000000000000000000000000000000000000000010"),
		fromHex("4368614368613132"), // ChaCha12
		fromHex("424c414b4533"),     // BLAKE3
		fromHex("3d7c1b8dace1414dd433549f7b25489c4f074a523025effcf048f77ea14a8b4b990e06a35ab0ec35"),
	},
	{
		fromHex("3eb02a239a2a66de159b9bb5486ccc10a6f63ddf5862ef076650513372353622"),
		fromHex("768e9bda14afb5686cc34de26210f9ff6fa1dfadc64ee3f0793e4979a30fc304"),
		fromHex("b8f60975cd7057a003ac84df00d514624fe40cb7855c50dd6594f59b3a2580e5"),
		fromHex("c8d69ca92da6c5fd22f1805179fcd36cb7a9d45848fa346ba7118c2f34d23a48"),
		fromHex("5f10321d8b50842182bb37bcebcd6abd45b77d3b420cf9ff6040cb46c57ebda990e18e4a71f291976f34505a8a0bd7a2ffd76bd2a7170acaef2a5dd42e0dd8b4"),
	},
}
