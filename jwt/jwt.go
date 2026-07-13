// Package jwt provides functions for creating, parsing, and verifying JSON Web Tokens
// (JWT) as defined in RFC 7519, with support for JSON Web Keys (JWK) per RFC 7517.
//
// It supports EdDSA (Ed25519), ECDSA (P-256/P-384/P-521), RSA (PKCS1v15 and PSS),
// and HMAC (SHA-256/384/512) signing and verification.
//
// # Sign, ParseHeader, ParseAndVerify
//
// The following example demonstrates the complete JWT lifecycle: signing a token,
// inspecting its header, and verifying signature and claims.
//
//	import "github.com/skerkour/stdx-go/jwt"
//
//	 // Generate or load a key pair (Ed25519 shown here).
//	 _, priv, _ := ed25519.GenerateKey(rand.Reader)
//	 pub := priv.Public().(ed25519.PublicKey)
//
//	 // Sign a token.
//	 header := jwt.Header{Typ: jwt.JWT, Alg: jwt.EdDSA, KID: "my-key"}
//	 claims := map[string]any{"sub": "user123", "exp": time.Now().Add(time.Hour).Unix()}
//	 token, err := jwt.Sign(priv, &header, claims)
//	 if err != nil { log.Fatal(err) }
//
//	 // ParseHeader extracts the header without verifying (useful for kid lookup).
//	 parsedHeader, err := jwt.ParseHeader(token)
//	 if err != nil { log.Fatal(err) }
//	 fmt.Println("Key ID:", parsedHeader.KID)
//
//	 // ParseAndVerify verifies the signature, validates claims, and decodes claims.
//	 opts := jwt.VerifyOptions{Exp: true, AllowedTimeDrift: time.Minute}
//	 result, err := jwt.ParseAndVerify[map[string]any](pub, &parsedHeader, token, &opts)
//	 if err != nil { log.Fatal(err) }
//	 fmt.Println("Subject:", result["sub"])
//
// # JWK
//
// Example of encoding a Go crypto key to JWK format and parsing it back:
//
//	import "github.com/skerkour/stdx-go/jwt"
//
//	 // Encode a key to RFC 7517 JWK JSON.
//	 _, priv, _ := ed25519.GenerateKey(rand.Reader)
//	 jwkJSON, err := jwt.EncodeToJWK(priv, jwt.EdDSA, "my-key-id")
//	 if err != nil {
//	 	log.Fatal(err) }
//	 }
//	 fmt.Println(string(jwkJSON))
//
//	 // Parse JWK JSON back into a Go crypto key.
//	 jwk, err := jwt.ParseJWK(jwkJSON)
//	 if err != nil { log.Fatal(err) }
//	 fmt.Printf("Algorithm: %s, Key ID: %s\n", jwk.Alg, jwk.KID)
//	 _ = jwk.Key.(ed25519.PrivateKey)
package jwt

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/hmac"
	cryptoRand "crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/base64"
	"encoding/json"
	"errors"
	"hash"
	"slices"
	"strings"
	"time"
)

var (
	ErrUnsupportedKeyType = errors.New("jwt: unsupported key type")
	ErrUnsupportedCurve   = errors.New("jwt: unsupported elliptic curve")
	ErrInvalidAlgorithm   = errors.New("jwt: invalid algorithm for key type")
	ErrInvalidJWK         = errors.New("jwt: invalid JWK")
	ErrMissingField       = errors.New("jwt: missing required JWK field")
	ErrInvalidSignature   = errors.New("jwt: invalid signature")
	ErrInvalidToken       = errors.New("jwt: invalid token format")
	ErrTokenExpired       = errors.New("jwt: token is expired")
	ErrTokenNotYetValid   = errors.New("jwt: token is not yet valid")
	ErrInvalidAudience    = errors.New("jwt: invalid audience")
	ErrInvalidIssuer      = errors.New("jwt: invalid issuer")
)

// Algorithm identifies the signing/verification algorithm for a JWT.
type Algorithm string

const (
	// HMAC using SHA-256.
	HS256 Algorithm = "HS256"

	// HMAC using SHA-384.
	HS384 Algorithm = "HS384"

	// HMAC using SHA-512.
	HS512 Algorithm = "HS512"

	// Edwards-curve Digital Signature Algorithm (Ed25519).
	EdDSA Algorithm = "EdDSA"

	// ECDSA using P-256 and SHA-256.
	ES256 Algorithm = "ES256"

	// ECDSA using P-384 and SHA-384.
	ES384 Algorithm = "ES384"

	// ECDSA using P-521 and SHA-512.
	ES512 Algorithm = "ES512"

	// RSASSA-PKCS1-v1.5 with SHA-256.
	RS256 Algorithm = "RS256"

	// RSASSA-PKCS1-v1.5 with SHA-384.
	RS384 Algorithm = "RS384"

	// RSASSA-PKCS1-v1.5 with SHA-512.
	RS512 Algorithm = "RS512"

	// RSASSA-PSS with SHA-256.
	PS256 Algorithm = "PS256"

	// RSASSA-PSS with SHA-384.
	PS384 Algorithm = "PS384"

	// RSASSA-PSS with SHA-512.
	PS512 Algorithm = "PS512"
)

func (a Algorithm) isHMAC() bool {
	switch a {
	case HS256, HS384, HS512:
		return true
	}
	return false
}

func (a Algorithm) isRSA() bool {
	switch a {
	case RS256, RS384, RS512, PS256, PS384, PS512:
		return true
	}
	return false
}

func (a Algorithm) isECDSA() bool {
	switch a {
	case ES256, ES384, ES512:
		return true
	}
	return false
}

// TokenType represents the type of a JWT token.
type TokenType string

const JWT TokenType = "JWT"

// Header represents a JWT header.
type Header struct {
	Typ     TokenType `json:"typ"`
	Alg     Algorithm `json:"alg"`
	CTY     string    `json:"cty,omitempty"`
	JKU     string    `json:"jku,omitempty"`
	KID     string    `json:"kid,omitempty"`
	X5U     string    `json:"x5u,omitempty"`
	X5C     []string  `json:"x5c,omitempty"`
	X5T     string    `json:"x5t,omitempty"`
	X5TS256 string    `json:"x5t#S256,omitempty"`
}

// RegisteredClaims holds the standard JWT registered claim names.
// https://www.rfc-editor.org/rfc/rfc7519#section-4.1
type RegisteredClaims struct {
	ISS string `json:"iss,omitempty"`
	SUB string `json:"sub,omitempty"`
	AUD string `json:"aud,omitempty"`
	EXP int64  `json:"exp,omitempty"`
	NBF int64  `json:"nbf,omitempty"`
	IAT int64  `json:"iat,omitempty"`
	JTI string `json:"jti,omitempty"`
}

// VerifyOptions configures claim verification for ParseAndVerify.
type VerifyOptions struct {
	AllowedTimeDrift time.Duration
	Exp              bool
	Nbf              bool
	Aud              []string
	Iss              []string
}

// Sign creates a signed JWT token string.
//
// Supported key types for signing:
//
//	ed25519.PrivateKey  -> EdDSA
//	*ecdsa.PrivateKey   -> ES256/ES384/ES512 (derived from curve)
//	[]byte              -> HS256/HS384/HS512 (must match header alg)
func Sign(key any, header *Header, claims any) (string, error) {
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return "", err
	}
	claimsJSON, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}

	base64HeaderLength := base64.RawURLEncoding.EncodedLen(len(headerJSON))
	base64ClaimsLength := base64.RawURLEncoding.EncodedLen(len(claimsJSON))

	token := strings.Builder{}
	token.Grow(base64HeaderLength + base64ClaimsLength + 2 + base64.RawURLEncoding.EncodedLen(64))

	token.WriteString(base64.RawURLEncoding.EncodeToString(headerJSON))
	token.WriteByte('.')
	token.WriteString(base64.RawURLEncoding.EncodeToString(claimsJSON))

	var signature []byte
	switch k := key.(type) {
	case ed25519.PrivateKey:
		if header.Alg != EdDSA {
			return "", ErrInvalidAlgorithm
		}
		signature = ed25519.Sign(k, []byte(token.String()))

	case *ecdsa.PrivateKey:
		_, expectedAlg, _, err := curveInfo(k.Curve)
		if err != nil {
			return "", err
		}
		if header.Alg != expectedAlg {
			return "", ErrInvalidAlgorithm
		}
		signature, err = ecdsaSign(k, []byte(token.String()), header.Alg)
		if err != nil {
			return "", err
		}

	case *rsa.PrivateKey:
		if !header.Alg.isRSA() {
			return "", ErrInvalidAlgorithm
		}
		signature, err = rsaSign(k, []byte(token.String()), header.Alg)
		if err != nil {
			return "", err
		}

	case []byte:
		if !header.Alg.isHMAC() {
			return "", ErrInvalidAlgorithm
		}
		signature = hmacSign([]byte(token.String()), k, header.Alg)

	default:
		return "", ErrUnsupportedKeyType
	}

	token.WriteByte('.')
	token.WriteString(base64.RawURLEncoding.EncodeToString(signature))
	return token.String(), nil
}

// ParseHeader extracts and decodes the header from a JWT token without verifying.
// Use this to inspect the kid field before looking up the verification key.
func ParseHeader(token string) (Header, error) {
	dotsCount := strings.Count(token, ".")
	if dotsCount != 2 {
		return Header{}, ErrInvalidToken
	}

	firstDotIndex := strings.IndexByte(token, '.')
	if firstDotIndex <= 0 {
		return Header{}, ErrInvalidToken
	}

	headerBase64 := token[:firstDotIndex]
	headerJSON, err := base64.RawURLEncoding.DecodeString(headerBase64)
	if err != nil {
		return Header{}, ErrInvalidToken
	}

	var header Header
	if err := json.Unmarshal(headerJSON, &header); err != nil {
		return Header{}, ErrInvalidToken
	}

	if header.Alg == "" || header.Typ != JWT {
		return Header{}, ErrInvalidToken
	}

	return header, nil
}

// ParseAndVerify verifies the JWT signature, validates claims per opts,
// and unmarshals the claims into dst (which must be a pointer).
//
// Supported key types for verification:
//
//	ed25519.PrivateKey / ed25519.PublicKey  -> EdDSA
//	*ecdsa.PrivateKey / *ecdsa.PublicKey    -> ES256/ES384/ES512
//	*rsa.PublicKey                          -> RS*/PS*
//	[]byte                                  -> HS*
func ParseAndVerify[C any](key any, header *Header, token string, opts *VerifyOptions) (claims C, err error) {
	dotsCount := strings.Count(token, ".")
	if dotsCount != 2 {
		return claims, ErrInvalidToken
	}

	firstDotIndex := strings.IndexByte(token, '.')
	if firstDotIndex <= 0 {
		return claims, ErrInvalidToken
	}

	secondDotIndex := strings.LastIndexByte(token, '.')
	if secondDotIndex <= 0 {
		return claims, ErrInvalidToken
	}

	claimsBase64 := token[firstDotIndex+1 : secondDotIndex]
	signatureBase64 := token[secondDotIndex+1:]

	signedMessage := token[:secondDotIndex]

	rawSignature, err := base64.RawURLEncoding.DecodeString(signatureBase64)
	if err != nil {
		return claims, ErrInvalidToken
	}

	if err = verify(key, []byte(signedMessage), rawSignature, header.Alg); err != nil {
		return claims, err
	}

	claimsJSON, err := base64.RawURLEncoding.DecodeString(claimsBase64)
	if err != nil {
		return claims, ErrInvalidToken
	}

	if err = validateClaims(claimsJSON, opts); err != nil {
		return claims, err
	}

	if err = json.Unmarshal(claimsJSON, &claims); err != nil {
		return claims, err
	}

	return claims, nil
}

// ── signing / verification helpers ──────────────────────────

func hmacSign(message, key []byte, alg Algorithm) []byte {
	mac := hmac.New(hashFunc(alg), key)
	mac.Write(message)
	return mac.Sum(nil)
}

func hmacVerify(message, key, sig []byte, alg Algorithm) bool {
	expected := hmacSign(message, key, alg)
	return hmac.Equal(sig, expected)
}

func ecdsaSign(key *ecdsa.PrivateKey, message []byte, alg Algorithm) ([]byte, error) {
	h := hashFunc(alg)()
	h.Write(message)
	digest := h.Sum(nil)
	return ecdsa.SignASN1(cryptoRand.Reader, key, digest)
}

func rsaSign(key *rsa.PrivateKey, message []byte, alg Algorithm) ([]byte, error) {
	h := hashFunc(alg)()
	h.Write(message)
	digest := h.Sum(nil)

	switch alg {
	case RS256, RS384, RS512:
		return rsa.SignPKCS1v15(nil, key, hashFuncCrypto(alg), digest)
	case PS256, PS384, PS512:
		return rsa.SignPSS(cryptoRand.Reader, key, hashFuncCrypto(alg), digest, nil)
	}
	return nil, ErrInvalidAlgorithm
}

func ecdsaVerify(pub *ecdsa.PublicKey, message, sig []byte, alg Algorithm) bool {
	h := hashFunc(alg)()
	h.Write(message)
	digest := h.Sum(nil)
	return ecdsa.VerifyASN1(pub, digest, sig)
}

func rsaVerify(pub *rsa.PublicKey, message, sig []byte, alg Algorithm) error {
	h := hashFunc(alg)()
	h.Write(message)
	digest := h.Sum(nil)

	switch alg {
	case RS256, RS384, RS512:
		return rsa.VerifyPKCS1v15(pub, hashFuncCrypto(alg), digest, sig)
	case PS256, PS384, PS512:
		return rsa.VerifyPSS(pub, hashFuncCrypto(alg), digest, sig, nil)
	}
	return ErrInvalidAlgorithm
}

func hashFunc(alg Algorithm) func() hash.Hash {
	switch alg {
	case HS256, ES256, RS256, PS256:
		return sha256.New
	case HS384, ES384, RS384, PS384:
		return sha512.New384
	case HS512, ES512, RS512, PS512:
		return sha512.New
	}
	return sha256.New
}

func hashFuncCrypto(alg Algorithm) crypto.Hash {
	switch alg {
	case RS256, PS256:
		return crypto.SHA256
	case RS384, PS384:
		return crypto.SHA384
	case RS512, PS512:
		return crypto.SHA512
	}
	return crypto.SHA256
}

func verify(key any, message, sig []byte, alg Algorithm) error {
	switch k := key.(type) {
	case ed25519.PrivateKey:
		if alg != EdDSA {
			return ErrInvalidAlgorithm
		}
		if !ed25519.Verify(k.Public().(ed25519.PublicKey), message, sig) {
			return ErrInvalidSignature
		}
		return nil

	case ed25519.PublicKey:
		if alg != EdDSA {
			return ErrInvalidAlgorithm
		}
		if !ed25519.Verify(k, message, sig) {
			return ErrInvalidSignature
		}
		return nil

	case *ecdsa.PrivateKey:
		_, expectedAlg, _, err := curveInfo(k.Curve)
		if err != nil {
			return err
		}
		if alg != expectedAlg {
			return ErrInvalidAlgorithm
		}
		if !ecdsaVerify(&k.PublicKey, message, sig, alg) {
			return ErrInvalidSignature
		}
		return nil

	case *ecdsa.PublicKey:
		_, expectedAlg, _, err := curveInfo(k.Curve)
		if err != nil {
			return err
		}
		if alg != expectedAlg {
			return ErrInvalidAlgorithm
		}
		if !ecdsaVerify(k, message, sig, alg) {
			return ErrInvalidSignature
		}
		return nil

	case *rsa.PublicKey:
		if !alg.isRSA() {
			return ErrInvalidAlgorithm
		}
		return rsaVerify(k, message, sig, alg)

	case *rsa.PrivateKey:
		if !alg.isRSA() {
			return ErrInvalidAlgorithm
		}
		return rsaVerify(&k.PublicKey, message, sig, alg)

	case []byte:
		if !alg.isHMAC() {
			return ErrInvalidAlgorithm
		}
		if !hmacVerify(message, k, sig, alg) {
			return ErrInvalidSignature
		}
		return nil

	default:
		return ErrUnsupportedKeyType
	}
}

func validateClaims(claimsJSON []byte, opts *VerifyOptions) error {
	if opts == nil {
		return nil
	}

	hasValidation := opts.Exp || opts.Nbf || len(opts.Aud) > 0 || len(opts.Iss) > 0
	if !hasValidation {
		return nil
	}

	raw := make(map[string]any, 4)
	if err := json.Unmarshal(claimsJSON, &raw); err != nil {
		return nil
	}

	now := time.Now().Unix()

	if opts.Exp {
		if expRaw, ok := raw["exp"]; ok {
			exp := numericDate(expRaw)
			if exp == 0 {
				return ErrMissingField
			}
			if exp < now-int64(opts.AllowedTimeDrift.Seconds()) {
				return ErrTokenExpired
			}
		} else {
			return ErrMissingField
		}
	}

	if opts.Nbf {
		if nbfRaw, ok := raw["nbf"]; ok {
			nbf := numericDate(nbfRaw)
			if nbf == 0 {
				return ErrMissingField
			}
			if nbf > now+int64(opts.AllowedTimeDrift.Seconds()) {
				return ErrTokenNotYetValid
			}
		} else {
			return ErrMissingField
		}
	}

	if len(opts.Aud) > 0 {
		audRaw, ok := raw["aud"]
		if !ok {
			return ErrMissingField
		}
		var tokenAud string
		switch v := audRaw.(type) {
		case string:
			tokenAud = v
		default:
			return ErrInvalidAudience
		}

		if !slices.Contains(opts.Aud, tokenAud) {
			return ErrInvalidAudience
		}
	}

	if len(opts.Iss) > 0 {
		issRaw, ok := raw["iss"]
		if !ok {
			return ErrMissingField
		}
		tokenIss, ok := issRaw.(string)
		if !ok {
			return ErrMissingField
		}

		if !slices.Contains(opts.Iss, tokenIss) {
			return ErrInvalidIssuer
		}
	}

	return nil
}

func numericDate(v any) int64 {
	switch n := v.(type) {
	case float64:
		return int64(n)
	case int64:
		return n
	case uint64:
		return int64(n)
	case int:
		return int64(n)
	}
	return 0
}
