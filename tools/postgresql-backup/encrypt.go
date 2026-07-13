package main

import (
	"crypto/ecdh"
	"crypto/rand"
	"crypto/sha512"
	"encoding/hex"
	"errors"
	"fmt"
	"io"

	"golang.org/x/crypto/chacha20poly1305"
	"golang.org/x/crypto/hkdf"
)

const (
	encryptionKeySize = 32
	nonceSize         = 24
	ephemeralKeySize  = 32
)

func encrypt(plaintext []byte, publicKeyHex string) ([]byte, error) {
	pubKeyBytes, err := hex.DecodeString(publicKeyHex)
	if err != nil {
		return nil, err
	}

	curve := ecdh.X25519()
	recipientPubKey, err := curve.NewPublicKey(pubKeyBytes)
	if err != nil {
		return nil, err
	}

	ephemeralPrivKey, err := curve.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	ephemeralPubKey := ephemeralPrivKey.PublicKey()

	sharedSecret, err := ephemeralPrivKey.ECDH(recipientPubKey)
	if err != nil {
		return nil, err
	}

	encKey, err := deriveEncryptionKey(sharedSecret)
	if err != nil {
		return nil, err
	}

	nonce := make([]byte, nonceSize)
	if _, err := rand.Read(nonce); err != nil {
		return nil, err
	}

	cipher, err := chacha20poly1305.NewX(encKey)
	if err != nil {
		return nil, err
	}

	ciphertext := cipher.Seal(nil, nonce, plaintext, nil)

	result := make([]byte, 0, ephemeralKeySize+nonceSize+len(ciphertext))
	result = append(result, ephemeralPubKey.Bytes()...)
	result = append(result, nonce...)
	result = append(result, ciphertext...)

	return result, nil
}

func decrypt(data []byte, privateKeyHex string) ([]byte, error) {
	privKeyBytes, err := hex.DecodeString(privateKeyHex)
	if err != nil {
		return nil, err
	}

	curve := ecdh.X25519()
	privKey, err := curve.NewPrivateKey(privKeyBytes)
	if err != nil {
		return nil, err
	}

	if len(data) < ephemeralKeySize+nonceSize {
		return nil, errors.New("invalid encrypted payload: too short")
	}

	ephemeralPubKeyBytes := data[:ephemeralKeySize]
	nonce := data[ephemeralKeySize : ephemeralKeySize+nonceSize]
	ciphertext := data[ephemeralKeySize+nonceSize:]

	ephemeralPubKey, err := curve.NewPublicKey(ephemeralPubKeyBytes)
	if err != nil {
		return nil, err
	}

	sharedSecret, err := privKey.ECDH(ephemeralPubKey)
	if err != nil {
		return nil, err
	}

	encKey, err := deriveEncryptionKey(sharedSecret)
	if err != nil {
		return nil, err
	}

	cipher, err := chacha20poly1305.NewX(encKey)
	if err != nil {
		return nil, err
	}

	plaintext, err := cipher.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, err
	}

	return plaintext, nil
}

func deriveEncryptionKey(sharedSecret []byte) ([]byte, error) {
	hkdfReader := hkdf.New(sha512.New, sharedSecret, nil, []byte("postgresql-backup"))
	key := make([]byte, encryptionKeySize)
	if _, err := io.ReadFull(hkdfReader, key); err != nil {
		return nil, fmt.Errorf("hkdf: %w", err)
	}
	return key, nil
}
