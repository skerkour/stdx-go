package xwing_test

import (
	"fmt"
	"log"

	"github.com/skerkour/stdx-go/xwing"
)

func Example() {
	alicePrivateKey := xwing.GenerateDecapsulationKey()

	// send alicePublicKey to Bob with alciePublicKey.Bytes()
	alicePublicKey := alicePrivateKey.EncapsulationKey()
	// Bob can now compute a sharedSecret and a ciphertext
	bobSharedSecret, ciphertext, err := alicePublicKey.Encapsulate()
	if err != nil {
		log.Fatal(err)
	}

	// send the ciphertext to Alice
	// Alice can know compute the same shared secret as Bob
	aliceSharedSecret, err := alicePrivateKey.Decapsulate(ciphertext)
	if err != nil {
		log.Fatal(err)
	}

	if aliceSharedSecret == bobSharedSecret {
		fmt.Println("shared secrets match")
	}

	// Output:
	// shared secrets match
}
