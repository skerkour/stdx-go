package jst

import (
	"bytes"
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/skerkour/stdx-go/crypto/blake3"
	"github.com/skerkour/stdx-go/crypto/chacha20"
)

// jst.v1.local.[header].[payload].[signature]

const (
	V1KeySize = 32
	// nonceSize is the size of the nonce to encrypt tokens, in bytes.
	v1nonceSize = chacha20.NonceSizeX

	v1EncryptionKeyContext     = "jst-v1 2023-12-31 23:59:59.999 encryption-key"
	v1AuthenticationKeyContext = "jst-v1 2024-01-01 00:00:00.000 authentication-key"
)

var (
	ErrTokenIsNotValid     = errors.New("jst: token is not valid")
	ErrSignatureIsNotValid = errors.New("jst: signature is not valid")
	ErrTokenHasExpired     = errors.New("jst: token has expired")
)

type Provider struct {
	defaultKey  string
	keyProvider KeyProvider
}

func NewProvider(keyProvider KeyProvider, defaultKey string) (provider *Provider, err error) {
	provider = &Provider{
		defaultKey:  defaultKey,
		keyProvider: keyProvider,
	}
	return
}

type TokenOptions struct {
	NotBefore   *time.Time
	IssuedAt    *time.Time
	KeyID       string
	Compression string
}

type HeaderV1 struct {
	ExpiresAt   *time.Time `json:"expires_at,omitempty"`
	NotBefore   *time.Time `json:"not_before,omitempty"`
	IssuedAt    *time.Time `json:"issued_at,omitempty"`
	KeyID       string     `json:"key_id,omitempty"`
	Compression string     `json:"compression,omitempty"`
	Nonce       []byte     `json:"nonce"`
}

func (provider *Provider) IssueToken(payload any, expiresAt *time.Time, options *TokenOptions) (token string, err error) {
	tokenBuffer := bytes.NewBuffer(make([]byte, 0, 120))

	if options == nil {
		options = &TokenOptions{}
	}

	keyId := options.KeyID
	if keyId == "" {
		keyId = provider.defaultKey
	}

	maskterKey, err := provider.keyProvider.GetKey(keyId)
	if err != nil {
		return
	}
	if len(maskterKey) != V1KeySize {
		err = fmt.Errorf("jst: key %s is invalid. Expected size: %d bytes", keyId, V1KeySize)
		return
	}

	nonce := make([]byte, v1nonceSize)
	_, err = rand.Read(nonce)
	if err != nil {
		err = fmt.Errorf("jst: error generating random nonce: %w", err)
		return
	}

	// derive keys
	encryptionKey := deriveKey(maskterKey, v1EncryptionKeyContext, nonce)
	authenticationKey := deriveKey(maskterKey, v1AuthenticationKeyContext, nonce)

	// prefix
	_, err = tokenBuffer.WriteString("jst.v1.")
	if err != nil {
		err = fmt.Errorf("jst: generating token: %w", err)
		return
	}

	// header
	header := HeaderV1{
		ExpiresAt: expiresAt,
		NotBefore: options.NotBefore,
		IssuedAt:  options.IssuedAt,
		KeyID:     keyId,
		Nonce:     nonce,
	}
	headerJson, err := json.Marshal(header)
	if err != nil {
		err = fmt.Errorf("jst: error encoding header to JSON: %w", err)
		return
	}
	headerBase64 := base64.RawURLEncoding.EncodeToString(headerJson)

	// we can ignore some errors as Buffer.Write* methods never return an error
	_, _ = tokenBuffer.WriteString(headerBase64)

	// payload
	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		err = fmt.Errorf("jst: error encoding payload to JSON: %w", err)
		return
	}

	// we can ignore error as we already checked that the key and nonce are of the correct size
	cipher, _ := chacha20.NewX(encryptionKey, nonce)
	cipherTextBuffer := make([]byte, len(payloadJSON))
	cipher.XORKeyStream(cipherTextBuffer, payloadJSON)
	payloadBase64 := base64.RawURLEncoding.EncodeToString(cipherTextBuffer)

	_ = tokenBuffer.WriteByte('.')
	_, _ = tokenBuffer.WriteString(payloadBase64)

	// we can ignore error as we are sure that the key is of the good size
	macHasher := blake3.New(32, authenticationKey)
	macHasher.Write(tokenBuffer.Bytes())
	signature := macHasher.Sum(nil)
	signatureBase64 := base64.RawURLEncoding.EncodeToString(signature)

	_ = tokenBuffer.WriteByte('.')
	_, _ = tokenBuffer.WriteString(signatureBase64)

	token = tokenBuffer.String()

	return
}

func (provider *Provider) VerifyToken(token string, data any) (header HeaderV1, err error) {
	if strings.Count(token, ".") != 4 {
		err = ErrTokenIsNotValid
		return
	}

	if !strings.HasPrefix(token, "jst.v1.") {
		err = ErrTokenIsNotValid
		return
	}

	// Header
	headerEnd := strings.IndexByte(token[7:], '.') + 7
	encodedHeader := token[7:headerEnd]
	headerJson, err := base64.RawURLEncoding.DecodeString(encodedHeader)
	if err != nil {
		err = ErrTokenIsNotValid
		return
	}
	err = json.Unmarshal(headerJson, &header)
	if err != nil {
		err = ErrTokenIsNotValid
		return
	}

	if len(header.Nonce) != v1nonceSize {
		err = ErrTokenIsNotValid
		return
	}

	maskterKey, err := provider.keyProvider.GetKey(header.KeyID)
	if err != nil {
		return
	}

	// derive keys
	encryptionKey := deriveKey(maskterKey, v1EncryptionKeyContext, header.Nonce)
	authenticationKey := deriveKey(maskterKey, v1AuthenticationKeyContext, header.Nonce)

	signatureStart := strings.LastIndexByte(token, '.')
	encodedSignature := token[signatureStart+1:]
	tokenSignature, err := base64.RawURLEncoding.DecodeString(encodedSignature)
	if err != nil {
		err = ErrTokenIsNotValid
		return
	}

	encodedHeaderAndPayload := token[:signatureStart]

	// we can ignore error as we are sure that the key is of the good size
	macHasher, _ := blake3.NewKeyed(authenticationKey)
	macHasher.Write([]byte(encodedHeaderAndPayload))
	signature := macHasher.Sum(nil)

	if subtle.ConstantTimeCompare(tokenSignature, signature) != 1 {
		err = ErrSignatureIsNotValid
		return
	}

	// Payload
	encodedPayload := token[headerEnd+1 : signatureStart]
	encryptedPayload, err := base64.RawURLEncoding.DecodeString(encodedPayload)
	if err != nil {
		err = ErrTokenIsNotValid
		return
	}
	// we can ignore error as we already checked that the key and nonce are of the correct size
	cipher, _ := chacha20.NewX(encryptionKey, header.Nonce)
	payloadJson := make([]byte, len(encryptedPayload))
	cipher.XORKeyStream(payloadJson, encryptedPayload)

	err = json.Unmarshal(payloadJson, data)
	if err != nil {
		err = ErrTokenIsNotValid
		return
	}

	return
}

func deriveKey(parentKey []byte, context string, nonce []byte) []byte {
	hasher := blake3.NewDeriveKey(context)
	hasher.Write(nonce)
	hasher.Write(parentKey)
	// hasher.Write(binary.LittleEndian.AppendUint64([]byte{}, uint64(len(nonce))))
	// hasher.Write(binary.LittleEndian.AppendUint64([]byte{}, uint64(len(parentKey))))
	return hasher.Sum(nil)
}
