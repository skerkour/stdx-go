package main

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha3"
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/skerkour/stdx-go/crypto/chacha20blake3"
	"golang.org/x/crypto/argon2"
	"golang.org/x/term"
)

const (
	keyLength                 = 32
	aesNonceLength            = 12
	chacha20Blake3NonceLength = 24
	kdfInfoChaCha20Blake3Key  = "ChaCha20-BLAKE3 key"
	kdfInfoAesKey             = "AES-256-GCM key"

	argon2SaltLength = 32
	argon2Iterations = 4
	argon2MemoryKB   = 512 * 1024 // 512 MB
	argon2Threads    = 4
)

func main() {
	if len(os.Args) != 4 {
		printHelpAndExit(1)
	}

	action := os.Args[1]
	fileIn := os.Args[2]
	fileOut := os.Args[3]
	var err error

	switch action {
	case "encrypt":
		err = processFile(fileIn, fileOut, encrypt)
	case "decrypt":
		err = processFile(fileIn, fileOut, decrypt)
	default:
		printHelpAndExit(1)
	}

	if err != nil {
		log.Fatal(err)
	}
}

func processFile(fileIn, fileOut string, fn func(password, input []byte) (output []byte, err error)) error {
	if fileIn == fileOut {
		return errors.New("input file can't be the same as output file")
	}

	dataIn, err := os.ReadFile(fileIn)
	if err != nil {
		return fmt.Errorf("error reading [%s]: %w", fileIn, err)
	}

	// Prompt for password (no echo)
	os.Stderr.WriteString("Password: ")
	password, err := term.ReadPassword(int(os.Stdin.Fd()))
	os.Stderr.WriteString("\n")
	if err != nil {
		return fmt.Errorf("error reading password: %w", err)
	}
	if len(password) == 0 {
		return errors.New("password is empty")
	}

	dataOut, err := fn(password, dataIn)
	zerooize(password)

	if err != nil {
		return err
	}

	err = os.WriteFile(fileOut, dataOut, 0600)
	if err != nil {
		return fmt.Errorf("error writing to [%s]: %w", fileOut, err)
	}

	return nil
}

func encrypt(password, plaintext []byte) (ciphertext []byte, err error) {
	var aesNonce [aesNonceLength]byte
	var chacha20Blake3Nonce [chacha20Blake3NonceLength]byte
	var argon2Salt [argon2SaltLength]byte

	rand.Read(aesNonce[:])
	rand.Read(chacha20Blake3Nonce[:])
	rand.Read(argon2Salt[:])

	rootKey := argon2.IDKey(password, argon2Salt[:], argon2Iterations, argon2MemoryKB, uint8(argon2Threads), keyLength)

	aesKey := deriveKey(rootKey, kdfInfoAesKey, keyLength)
	chacha20Key := deriveKey(rootKey, kdfInfoChaCha20Blake3Key, keyLength)
	zerooize(rootKey)

	chacha20Blake3, err := chacha20blake3.New(chacha20Key[:])
	if err != nil {
		return nil, fmt.Errorf("error instantiating ChaCha20-BLAKE3 cipher: %w", err)
	}

	aesCipher, err := aes.NewCipher(aesKey[:])
	if err != nil {
		return nil, fmt.Errorf("error instantiating AES cipher: %w", err)
	}
	aes256Gcm, err := cipher.NewGCM(aesCipher)
	if err != nil {
		return nil, fmt.Errorf("error instantiating AES-256-GCM cipher: %w", err)
	}

	ciphertextAes := aes256Gcm.Seal(nil, aesNonce[:], plaintext, nil)
	zerooize(plaintext)
	zerooize(aesKey)

	ciphertext = chacha20Blake3.Seal(nil, chacha20Blake3Nonce[:], ciphertextAes, nil)
	zerooize(ciphertextAes)
	zerooize(chacha20Key[:])

	finalOutput := bytes.NewBuffer(make([]byte, 0, argon2SaltLength+chacha20Blake3NonceLength+aesNonceLength+len(ciphertext)))
	finalOutput.Write(argon2Salt[:])
	finalOutput.Write(chacha20Blake3Nonce[:])
	finalOutput.Write(aesNonce[:])
	finalOutput.Write(ciphertext)

	return finalOutput.Bytes(), nil
}

func decrypt(password, ciphertext []byte) (plaintext []byte, err error) {
	argon2Salt := ciphertext[:argon2SaltLength]
	chacha20Blake3Nonce := ciphertext[argon2SaltLength : argon2SaltLength+chacha20Blake3NonceLength]
	aesNonce := ciphertext[argon2SaltLength+chacha20Blake3NonceLength : argon2SaltLength+chacha20Blake3NonceLength+aesNonceLength]
	ciphertext = ciphertext[argon2SaltLength+chacha20Blake3NonceLength+aesNonceLength:]

	rootKey := argon2.IDKey(password, argon2Salt[:], argon2Iterations, argon2MemoryKB, uint8(argon2Threads), keyLength)
	zerooize(argon2Salt)

	aesKey := deriveKey(rootKey, kdfInfoAesKey, keyLength)
	chacha20Key := deriveKey(rootKey, kdfInfoChaCha20Blake3Key, keyLength)
	zerooize(rootKey)

	chacha20Blake3, err := chacha20blake3.New(chacha20Key[:])
	if err != nil {
		return nil, fmt.Errorf("error instantiating ChaCha20-BLAKE3 cipher: %w", err)
	}

	aesCipher, err := aes.NewCipher(aesKey[:])
	if err != nil {
		return nil, fmt.Errorf("error instantiating AES cipher: %w", err)
	}
	aes256Gcm, err := cipher.NewGCM(aesCipher)
	if err != nil {
		return nil, fmt.Errorf("error instantiating AES-256-GCM cipher: %w", err)
	}

	ciphertextAes, err := chacha20Blake3.Open(nil, chacha20Blake3Nonce[:], ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("error decrypting data with ChaCha20-BLAKE3: %w", err)
	}
	zerooize(ciphertext)
	zerooize(chacha20Key[:])

	plaintext, err = aes256Gcm.Open(nil, aesNonce[:], ciphertextAes, nil)
	if err != nil {
		return nil, fmt.Errorf("error decrypting data with AES-256-GCM: %w", err)

	}
	zerooize(ciphertextAes)
	zerooize(aesKey[:])

	return plaintext, nil
}

func deriveKey(rootKey []byte, info string, length int) []byte {
	out := make([]byte, length)

	hasher := sha3.NewSHAKE256()
	hasher.Write([]byte(info))
	binary.Write(hasher, binary.LittleEndian, len(info))
	hasher.Write([]byte(rootKey))
	binary.Write(hasher, binary.LittleEndian, len(rootKey))

	hasher.Read(out)
	return out
}

func printHelpAndExit(exitCode int) {
	fmt.Println("usage: crypt <encrypt|decrypt> <in> <out>")
	os.Exit(exitCode)
}

func zerooize(b []byte) {
	for i := range b {
		b[i] = 0
	}
}
