package jwt

import (
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"testing"
	"time"
)

func TestSignAndVerify_Ed25519(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	pub := priv.Public().(ed25519.PublicKey)

	header := Header{Typ: JWT, Alg: EdDSA, KID: "ed25519-key"}
	claims := map[string]any{"sub": "user123", "exp": time.Now().Add(time.Hour).Unix()}

	token, err := Sign(priv, &header, claims)
	if err != nil {
		t.Fatal(err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}
	if parsedHeader.Alg != EdDSA {
		t.Fatalf("expected EdDSA, got %s", parsedHeader.Alg)
	}
	if parsedHeader.KID != "ed25519-key" {
		t.Fatalf("expected ed25519-key, got %s", parsedHeader.KID)
	}

	opts := VerifyOptions{Exp: true, Nbf: false, AllowedTimeDrift: time.Minute}
	result, err := ParseAndVerify[map[string]any](pub, parsedHeader, token, &opts)
	if err != nil {
		t.Fatal(err)
	}
	if result["sub"] != "user123" {
		t.Fatalf("expected user123, got %v", result["sub"])
	}
}

func TestSignAndVerify_Ed25519PrivateKey(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	header := Header{Typ: JWT, Alg: EdDSA}
	token, err := Sign(priv, &header, map[string]any{"sub": "test"})
	if err != nil {
		t.Fatal(err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}

	_, err = ParseAndVerify[map[string]any](priv, parsedHeader, token, nil)
	if err != nil {
		t.Fatal(err)
	}
}

func TestSignAndVerify_ECDSAP256(t *testing.T) {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	pub := &priv.PublicKey

	header := Header{Typ: JWT, Alg: ES256}
	claims := map[string]any{"sub": "alice"}

	token, err := Sign(priv, &header, claims)
	if err != nil {
		t.Fatal(err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}
	if parsedHeader.Alg != ES256 {
		t.Fatalf("expected ES256, got %s", parsedHeader.Alg)
	}

	result, err := ParseAndVerify[map[string]any](pub, parsedHeader, token, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result["sub"] != "alice" {
		t.Fatalf("expected alice, got %v", result["sub"])
	}
}

func TestSignAndVerify_ECDSAP384(t *testing.T) {
	priv, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	pub := &priv.PublicKey

	header := Header{Typ: JWT, Alg: ES384}
	token, err := Sign(priv, &header, map[string]any{"sub": "bob"})
	if err != nil {
		t.Fatal(err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}
	if parsedHeader.Alg != ES384 {
		t.Fatalf("expected ES384, got %s", parsedHeader.Alg)
	}

	_, err = ParseAndVerify[map[string]any](pub, parsedHeader, token, nil)
	if err != nil {
		t.Fatal(err)
	}
}

func TestSignAndVerify_ECDSAP521(t *testing.T) {
	priv, err := ecdsa.GenerateKey(elliptic.P521(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	pub := &priv.PublicKey

	header := Header{Typ: JWT, Alg: ES512}
	token, err := Sign(priv, &header, map[string]any{"sub": "carol"})
	if err != nil {
		t.Fatal(err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}
	if parsedHeader.Alg != ES512 {
		t.Fatalf("expected ES512, got %s", parsedHeader.Alg)
	}

	_, err = ParseAndVerify[map[string]any](pub, parsedHeader, token, nil)
	if err != nil {
		t.Fatal(err)
	}
}

func TestSignAndVerify_HMAC(t *testing.T) {
	tests := []struct {
		alg Algorithm
		key []byte
	}{
		{HS256, make([]byte, 32)},
		{HS384, make([]byte, 48)},
		{HS512, make([]byte, 64)},
	}

	for _, tt := range tests {
		if _, err := rand.Read(tt.key); err != nil {
			t.Fatal(err)
		}

		header := Header{Typ: JWT, Alg: tt.alg}
		token, err := Sign([]byte(tt.key), &header, map[string]any{"sub": "test"})
		if err != nil {
			t.Fatalf("Sign(%s): %v", tt.alg, err)
		}

		parsedHeader, err := ParseHeader(token)
		if err != nil {
			t.Fatalf("ParseHeader(%s): %v", tt.alg, err)
		}

		_, err = ParseAndVerify[map[string]any]([]byte(tt.key), parsedHeader, token, nil)
		if err != nil {
			t.Fatalf("ParseAndVerify(%s): %v", tt.alg, err)
		}
	}
}

func TestSignAndVerify_RSAPublicKey(t *testing.T) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	pub := &priv.PublicKey

	header := Header{Typ: JWT, Alg: RS256}
	token, err := Sign(priv, &header, map[string]any{"sub": "rsa-user"})
	if err != nil {
		t.Fatalf("Sign: %v", err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}

	result, err := ParseAndVerify[map[string]any](pub, parsedHeader, token, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result["sub"] != "rsa-user" {
		t.Fatalf("expected rsa-user, got %v", result["sub"])
	}
}

func TestSignAndVerify_WithRegisteredClaims(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	claims := RegisteredClaims{
		SUB: "user1",
		ISS: "my-app",
		AUD: "my-api",
		EXP: time.Now().Add(time.Hour).Unix(),
		NBF: time.Now().Add(-time.Minute).Unix(),
		IAT: time.Now().Unix(),
		JTI: "unique-id-123",
	}

	header := Header{Typ: JWT, Alg: EdDSA}
	token, err := Sign(priv, &header, claims)
	if err != nil {
		t.Fatal(err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}

	opts := VerifyOptions{
		Exp:              true,
		Nbf:              true,
		Aud:              []string{"my-api"},
		Iss:              []string{"my-app"},
		AllowedTimeDrift: time.Minute,
	}

	result, err := ParseAndVerify[RegisteredClaims](priv, parsedHeader, token, &opts)
	if err != nil {
		t.Fatal(err)
	}
	if result.SUB != "user1" {
		t.Fatalf("expected user1, got %s", result.SUB)
	}
}

func TestSignAndVerify_ExpiredToken(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	claims := map[string]any{"exp": time.Now().Add(-time.Hour).Unix()}

	header := Header{Typ: JWT, Alg: EdDSA}
	token, err := Sign(priv, &header, claims)
	if err != nil {
		t.Fatal(err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}

	opts := VerifyOptions{Exp: true}
	_, err = ParseAndVerify[map[string]any](priv, parsedHeader, token, &opts)
	if err != ErrTokenExpired {
		t.Fatalf("expected ErrTokenExpired, got %v", err)
	}
}

func TestSignAndVerify_NotYetValid(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	claims := map[string]any{"nbf": time.Now().Add(time.Hour).Unix()}

	header := Header{Typ: JWT, Alg: EdDSA}
	token, err := Sign(priv, &header, claims)
	if err != nil {
		t.Fatal(err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}

	opts := VerifyOptions{Nbf: true}
	_, err = ParseAndVerify[map[string]any](priv, parsedHeader, token, &opts)
	if err != ErrTokenNotYetValid {
		t.Fatalf("expected ErrTokenNotYetValid, got %v", err)
	}
}

func TestSignAndVerify_BadAudience(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	claims := map[string]any{"aud": "wrong-aud"}

	header := Header{Typ: JWT, Alg: EdDSA}
	token, err := Sign(priv, &header, claims)
	if err != nil {
		t.Fatal(err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}

	opts := VerifyOptions{Aud: []string{"expected-aud"}}
	_, err = ParseAndVerify[map[string]any](priv, parsedHeader, token, &opts)
	if err != ErrInvalidAudience {
		t.Fatalf("expected ErrInvalidAudience, got %v", err)
	}
}

func TestSignAndVerify_BadIssuer(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	claims := map[string]any{"iss": "wrong-iss"}

	header := Header{Typ: JWT, Alg: EdDSA}
	token, err := Sign(priv, &header, claims)
	if err != nil {
		t.Fatal(err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}

	opts := VerifyOptions{Iss: []string{"expected-iss"}}
	_, err = ParseAndVerify[map[string]any](priv, parsedHeader, token, &opts)
	if err != ErrInvalidIssuer {
		t.Fatalf("expected ErrInvalidIssuer, got %v", err)
	}
}

func TestParseAndVerify_WrongKey(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	_, wrongPriv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	header := Header{Typ: JWT, Alg: EdDSA}
	token, err := Sign(priv, &header, map[string]any{"sub": "test"})
	if err != nil {
		t.Fatal(err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}

	_, err = ParseAndVerify[map[string]any](wrongPriv, parsedHeader, token, nil)
	if err != ErrInvalidSignature {
		t.Fatalf("expected ErrInvalidSignature, got %v", err)
	}
}

func TestParseHeader_InvalidToken(t *testing.T) {
	_, err := ParseHeader("not-a-jwt")
	if err != ErrInvalidToken {
		t.Fatalf("expected ErrInvalidToken, got %v", err)
	}

	_, err = ParseHeader("only.two.parts")
	if err == nil {
		t.Fatal("expected error for two-part token")
	}
}

func TestSign_UnsupportedKeyType(t *testing.T) {
	header := Header{Typ: JWT, Alg: EdDSA}
	_, err := Sign("not-a-key", &header, map[string]any{})
	if err != ErrUnsupportedKeyType {
		t.Fatalf("expected ErrUnsupportedKeyType, got %v", err)
	}
}

func TestSign_WrongAlgorithm(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	header := Header{Typ: JWT, Alg: HS256}
	_, err = Sign(priv, &header, map[string]any{})
	if err != ErrInvalidAlgorithm {
		t.Fatalf("expected ErrInvalidAlgorithm, got %v", err)
	}
}

func TestVerify_UnsupportedKeyType(t *testing.T) {
	header := Header{Typ: JWT, Alg: EdDSA}
	_, err := ParseAndVerify[any]("not-a-key", header, "a.b.cA", nil)
	if err != ErrUnsupportedKeyType {
		t.Fatalf("expected ErrUnsupportedKeyType, got %v", err)
	}
}

func TestParseAndVerify_RSAPSS(t *testing.T) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}

	tests := []Algorithm{PS256, PS384, PS512}
	for _, alg := range tests {
		header := Header{Typ: JWT, Alg: alg}
		token, err := Sign(priv, &header, map[string]any{"sub": "pss"})
		if err != nil {
			t.Fatalf("Sign(%s): %v", alg, err)
		}

		parsedHeader, err := ParseHeader(token)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ParseAndVerify[map[string]any](&priv.PublicKey, parsedHeader, token, nil)
		if err != nil {
			t.Fatalf("Verify(%s): %v", alg, err)
		}
	}
}

func TestSignAndVerify_TimeDrift(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	opts := VerifyOptions{
		Exp:              true,
		AllowedTimeDrift: 10 * time.Second,
	}

	withinDrift := time.Now().Add(-5 * time.Second).Unix()

	claims := map[string]any{"exp": withinDrift}
	header := Header{Typ: JWT, Alg: EdDSA}
	token, err := Sign(priv, &header, claims)
	if err != nil {
		t.Fatal(err)
	}

	parsedHeader, err := ParseHeader(token)
	if err != nil {
		t.Fatal(err)
	}

	_, err = ParseAndVerify[map[string]any](priv, parsedHeader, token, &opts)
	if err != nil {
		t.Fatalf("expected token within drift to be valid, got %v", err)
	}
}
