package jwt

import (
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"encoding/base64"
	"encoding/json"
	"testing"
)

func TestJWKRoundTrip_Ed25519PrivateKey(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	kid := "test-ed25519-priv"
	jsonData, err := EncodeToJWK(priv, EdDSA, kid)
	if err != nil {
		t.Fatal(err)
	}

	result, err := ParseJWK(jsonData)
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != EdDSA {
		t.Fatalf("expected Alg EdDSA, got %s", result.Alg)
	}
	if result.KID != kid {
		t.Fatalf("expected KID %s, got %s", kid, result.KID)
	}

	parsed, ok := result.Key.(ed25519.PrivateKey)
	if !ok {
		t.Fatalf("expected ed25519.PrivateKey, got %T", result.Key)
	}
	if !parsed.Equal(priv) {
		t.Fatal("keys not equal")
	}
}

func TestJWKRoundTrip_Ed25519PublicKey(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	jsonData, err := EncodeToJWK(pub, EdDSA, "")
	if err != nil {
		t.Fatal(err)
	}

	result, err := ParseJWK(jsonData)
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != EdDSA {
		t.Fatalf("expected Alg EdDSA, got %s", result.Alg)
	}

	parsed, ok := result.Key.(ed25519.PublicKey)
	if !ok {
		t.Fatalf("expected ed25519.PublicKey, got %T", result.Key)
	}
	if len(parsed) == 0 {
		t.Fatal("empty public key")
	}
}

func TestJWKRoundTrip_ECDSAP256PrivateKey(t *testing.T) {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	jsonData, err := EncodeToJWK(priv, ES256, "p256")
	if err != nil {
		t.Fatal(err)
	}

	result, err := ParseJWK(jsonData)
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != ES256 {
		t.Fatalf("expected Alg ES256, got %s", result.Alg)
	}
	if result.KID != "p256" {
		t.Fatalf("expected KID p256, got %s", result.KID)
	}

	parsed, ok := result.Key.(*ecdsa.PrivateKey)
	if !ok {
		t.Fatalf("expected *ecdsa.PrivateKey, got %T", result.Key)
	}
	if priv.D.Cmp(parsed.D) != 0 {
		t.Fatal("private keys not equal")
	}
}

func TestJWKRoundTrip_ECDSAP256PublicKey(t *testing.T) {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	pub := &priv.PublicKey

	jsonData, err := EncodeToJWK(pub, ES256, "")
	if err != nil {
		t.Fatal(err)
	}

	result, err := ParseJWK(jsonData)
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != ES256 {
		t.Fatalf("expected Alg ES256, got %s", result.Alg)
	}

	parsed, ok := result.Key.(*ecdsa.PublicKey)
	if !ok {
		t.Fatalf("expected *ecdsa.PublicKey, got %T", result.Key)
	}
	if parsed.X == nil || parsed.Y == nil {
		t.Fatal("empty public key coordinates")
	}
	_ = parsed
}

func TestJWKRoundTrip_ECDSAP384PrivateKey(t *testing.T) {
	priv, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	jsonData, err := EncodeToJWK(priv, ES384, "")
	if err != nil {
		t.Fatal(err)
	}

	result, err := ParseJWK(jsonData)
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != ES384 {
		t.Fatalf("expected Alg ES384, got %s", result.Alg)
	}

	parsed, ok := result.Key.(*ecdsa.PrivateKey)
	if !ok {
		t.Fatalf("expected *ecdsa.PrivateKey, got %T", result.Key)
	}
	if priv.D.Cmp(parsed.D) != 0 {
		t.Fatal("keys not equal")
	}
}

func TestJWKRoundTrip_ECDSAP521PrivateKey(t *testing.T) {
	priv, err := ecdsa.GenerateKey(elliptic.P521(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	jsonData, err := EncodeToJWK(priv, ES512, "")
	if err != nil {
		t.Fatal(err)
	}

	result, err := ParseJWK(jsonData)
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != ES512 {
		t.Fatalf("expected Alg ES512, got %s", result.Alg)
	}

	parsed, ok := result.Key.(*ecdsa.PrivateKey)
	if !ok {
		t.Fatalf("expected *ecdsa.PrivateKey, got %T", result.Key)
	}
	if priv.D.Cmp(parsed.D) != 0 {
		t.Fatal("keys not equal")
	}
}

func TestJWKRoundTrip_RSAPrivateKey(t *testing.T) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}

	jsonData, err := EncodeToJWK(priv, RS256, "rsa-2048")
	if err != nil {
		t.Fatal(err)
	}

	result, err := ParseJWK(jsonData)
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != RS256 {
		t.Fatalf("expected Alg RS256, got %s", result.Alg)
	}
	if result.KID != "rsa-2048" {
		t.Fatalf("expected KID rsa-2048, got %s", result.KID)
	}

	parsed, ok := result.Key.(*rsa.PrivateKey)
	if !ok {
		t.Fatalf("expected *rsa.PrivateKey, got %T", result.Key)
	}
	if priv.D.Cmp(parsed.D) != 0 {
		t.Fatal("keys not equal")
	}
	if len(parsed.Primes) == 0 {
		t.Fatal("missing prime factors")
	}
}

func TestJWKRoundTrip_RSAPublicKey(t *testing.T) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	pub := &priv.PublicKey

	jsonData, err := EncodeToJWK(pub, PS384, "")
	if err != nil {
		t.Fatal(err)
	}

	result, err := ParseJWK(jsonData)
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != PS384 {
		t.Fatalf("expected Alg PS384, got %s", result.Alg)
	}

	parsed, ok := result.Key.(*rsa.PublicKey)
	if !ok {
		t.Fatalf("expected *rsa.PublicKey, got %T", result.Key)
	}
	if parsed.N.Cmp(pub.N) != 0 || parsed.E != pub.E {
		t.Fatal("keys not equal")
	}
}

func TestJWKRoundTrip_HMAC(t *testing.T) {
	tests := []struct {
		alg Algorithm
		len int
	}{
		{HS256, 32},
		{HS384, 48},
		{HS512, 64},
	}

	for _, tt := range tests {
		key := make([]byte, tt.len)
		if _, err := rand.Read(key); err != nil {
			t.Fatal(err)
		}

		jsonData, err := EncodeToJWK([]byte(key), tt.alg, "")
		if err != nil {
			t.Fatalf("EncodeToJWK(%s): %v", tt.alg, err)
		}

		result, err := ParseJWK(jsonData)
		if err != nil {
			t.Fatalf("ParseJWK(%s): %v", tt.alg, err)
		}

		if result.Alg != tt.alg {
			t.Fatalf("expected Alg %s, got %s", tt.alg, result.Alg)
		}

		parsed, ok := result.Key.([]byte)
		if !ok {
			t.Fatalf("expected []byte, got %T", result.Key)
		}
		if len(parsed) != tt.len {
			t.Fatalf("expected key length %d, got %d", tt.len, len(parsed))
		}
	}
}

func TestJWKRoundTrip_ECDSAalgIgnored(t *testing.T) {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	jsonData, err := EncodeToJWK(priv, "WRONG_ALG", "")
	if err != nil {
		t.Fatal(err)
	}

	result, err := ParseJWK(jsonData)
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != ES256 {
		t.Fatalf("expected Alg ES256 (derived from curve), got %s", result.Alg)
	}
}

func TestJWKRoundTrip_EdDSAalgIgnored(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	jsonData, err := EncodeToJWK(priv, "WRONG", "")
	if err != nil {
		t.Fatal(err)
	}

	result, err := ParseJWK(jsonData)
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != EdDSA {
		t.Fatalf("expected Alg EdDSA, got %s", result.Alg)
	}
}

// ── Error cases ─────────────────────────────────────────────

func TestJWKError_UnsupportedKeyType(t *testing.T) {
	_, err := EncodeToJWK("not-a-key", HS256, "")
	if err != ErrUnsupportedKeyType {
		t.Fatalf("expected ErrUnsupportedKeyType, got %v", err)
	}
}

func TestJWKError_InvalidEd25519Size(t *testing.T) {
	_, err := EncodeToJWK(ed25519.PrivateKey{1, 2, 3}, EdDSA, "")
	if err != ErrInvalidJWK {
		t.Fatalf("expected ErrInvalidJWK, got %v", err)
	}
}

func TestJWKError_ECDSAwithP224(t *testing.T) {
	priv, err := ecdsa.GenerateKey(elliptic.P224(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	_, err = EncodeToJWK(priv, ES256, "")
	if err != ErrUnsupportedCurve {
		t.Fatalf("expected ErrUnsupportedCurve, got %v", err)
	}
}

func TestJWKError_RSABadAlgorithm(t *testing.T) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}

	_, err = EncodeToJWK(priv, HS256, "")
	if err != ErrInvalidAlgorithm {
		t.Fatalf("expected ErrInvalidAlgorithm, got %v", err)
	}

	_, err = EncodeToJWK(&priv.PublicKey, HS256, "")
	if err != ErrInvalidAlgorithm {
		t.Fatalf("expected ErrInvalidAlgorithm, got %v", err)
	}
}

func TestJWKError_HMACBadAlgorithm(t *testing.T) {
	key := make([]byte, 32)

	_, err := EncodeToJWK(key, RS256, "")
	if err != ErrInvalidAlgorithm {
		t.Fatalf("expected ErrInvalidAlgorithm, got %v", err)
	}
}

func TestJWKError_ParseBadJSON(t *testing.T) {
	_, err := ParseJWK([]byte("{bad json"))
	if err != ErrInvalidJWK {
		t.Fatalf("expected ErrInvalidJWK, got %v", err)
	}
}

func TestJWKError_ParseUnknownKTY(t *testing.T) {
	_, err := ParseJWK([]byte(`{"kty":"UNKNOWN"}`))
	if err != ErrUnsupportedKeyType {
		t.Fatalf("expected ErrUnsupportedKeyType, got %v", err)
	}
}

func TestJWKError_ParseUnsupportedCurve(t *testing.T) {
	_, err := ParseJWK([]byte(`{"kty":"EC","crv":"P-224","x":"","y":""}`))
	if err != ErrUnsupportedCurve {
		t.Fatalf("expected ErrUnsupportedCurve, got %v", err)
	}
}

func TestJWKError_ParseBadBase64(t *testing.T) {
	_, err := ParseJWK([]byte(`{"kty":"oct","k":"!!!not-base64!!!"}`))
	if err != ErrInvalidJWK {
		t.Fatalf("expected ErrInvalidJWK, got %v", err)
	}
}

func TestJWKError_ParseMissingKTY(t *testing.T) {
	_, err := ParseJWK([]byte(`{"alg":"RS256"}`))
	if err != ErrUnsupportedKeyType {
		t.Fatalf("expected ErrUnsupportedKeyType, got %v", err)
	}
}

func TestJWKError_ParseNonRSAalg(t *testing.T) {
	data := []byte(`{"kty":"RSA","n":"AQAB","e":"AQAB","alg":"HS256"}`)
	_, err := ParseJWK(data)
	if err != ErrInvalidAlgorithm {
		t.Fatalf("expected ErrInvalidAlgorithm, got %v", err)
	}
}

// ── Known test vectors ──────────────────────────────────────

func TestJWKVector_Ed25519PublicKey(t *testing.T) {
	knownJWK := `{
		"kty":"OKP",
		"crv":"Ed25519",
		"x":"11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaNcAlUM",
		"alg":"EdDSA"
	}`

	result, err := ParseJWK([]byte(knownJWK))
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != EdDSA {
		t.Fatalf("expected Alg EdDSA, got %s", result.Alg)
	}

	pub, ok := result.Key.(ed25519.PublicKey)
	if !ok {
		t.Fatalf("expected ed25519.PublicKey, got %T", result.Key)
	}

	expectedX, _ := base64.RawURLEncoding.DecodeString("11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaNcAlUM")
	if len(pub) != len(expectedX) {
		t.Fatalf("expected %d bytes, got %d", len(expectedX), len(pub))
	}
	_ = pub
}

func TestJWKVector_Ed25519PrivateKey(t *testing.T) {
	knownJWK := `{
		"kty":"OKP",
		"crv":"Ed25519",
		"d":"nWGxne_9WmC6hEr0kuwsxERJxWl7M0ZQWPN7i_J2G6E",
		"x":"11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaNcAlUM",
		"alg":"EdDSA"
	}`

	result, err := ParseJWK([]byte(knownJWK))
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != EdDSA {
		t.Fatalf("expected Alg EdDSA, got %s", result.Alg)
	}

	priv, ok := result.Key.(ed25519.PrivateKey)
	if !ok {
		t.Fatalf("expected ed25519.PrivateKey, got %T", result.Key)
	}

	if len(priv) != ed25519.PrivateKeySize {
		t.Fatalf("expected %d bytes, got %d", ed25519.PrivateKeySize, len(priv))
	}
}

func TestJWKVector_ECDSAP256Public(t *testing.T) {
	knownJWK := `{
		"kty":"EC",
		"crv":"P-256",
		"x":"MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
		"y":"4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM",
		"alg":"ES256"
	}`

	result, err := ParseJWK([]byte(knownJWK))
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != ES256 {
		t.Fatalf("expected Alg ES256, got %s", result.Alg)
	}

	pub, ok := result.Key.(*ecdsa.PublicKey)
	if !ok {
		t.Fatalf("expected *ecdsa.PublicKey, got %T", result.Key)
	}

	if pub.X == nil || pub.Y == nil {
		t.Fatal("empty coordinates")
	}
	if pub.Curve != elliptic.P256() {
		t.Fatal("wrong curve")
	}
}

func TestJWKVector_ECDSAP256Private(t *testing.T) {
	knownJWK := `{
		"kty":"EC",
		"crv":"P-256",
		"x":"MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
		"y":"4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM",
		"d":"870MB6gfuTJ4HtUnUvYMyJpr5eUZNP4Bk43bVdj3eAE",
		"alg":"ES256"
	}`

	result, err := ParseJWK([]byte(knownJWK))
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != ES256 {
		t.Fatalf("expected Alg ES256, got %s", result.Alg)
	}

	priv, ok := result.Key.(*ecdsa.PrivateKey)
	if !ok {
		t.Fatalf("expected *ecdsa.PrivateKey, got %T", result.Key)
	}

	if priv.D == nil {
		t.Fatal("missing private scalar")
	}
}

func TestJWKVector_RSA(t *testing.T) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}

	jsonData, err := EncodeToJWK(priv, RS256, "test-rsa-kid")
	if err != nil {
		t.Fatal(err)
	}
	result, err := ParseJWK(jsonData)
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != RS256 {
		t.Fatalf("expected Alg RS256, got %s", result.Alg)
	}
	if result.KID != "test-rsa-kid" {
		t.Fatalf("expected KID test-rsa-kid, got %s", result.KID)
	}

	pub, ok := result.Key.(*rsa.PrivateKey)
	if !ok {
		t.Fatalf("expected *rsa.PrivateKey, got %T", result.Key)
	}
	if pub.N == nil || pub.D == nil {
		t.Fatal("empty RSA key")
	}
}

// ── EncodeToJWK with kid ────────────────────────────────────

func TestJWKEncode_WithKID(t *testing.T) {
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	jsonData, err := EncodeToJWK(priv, EdDSA, "my-key-1")
	if err != nil {
		t.Fatal(err)
	}

	var raw struct {
		KID string `json:"kid"`
	}
	if err := json.Unmarshal(jsonData, &raw); err != nil {
		t.Fatal(err)
	}
	if raw.KID != "my-key-1" {
		t.Fatalf("expected kid my-key-1, got %s", raw.KID)
	}
}

func TestJWKEncode_WithoutKID(t *testing.T) {
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	jsonData, err := EncodeToJWK(priv, EdDSA, "")
	if err != nil {
		t.Fatal(err)
	}

	var raw struct {
		KID string `json:"kid,omitempty"`
	}
	if err := json.Unmarshal(jsonData, &raw); err != nil {
		t.Fatal(err)
	}
	if raw.KID != "" {
		t.Fatalf("expected empty kid, got %s", raw.KID)
	}
}

// ── Edge cases ──────────────────────────────────────────────

func TestJWKRoundTrip_Ed25519EmptyKID(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	jsonData, err := EncodeToJWK(priv, EdDSA, "")
	if err != nil {
		t.Fatal(err)
	}

	result, err := ParseJWK(jsonData)
	if err != nil {
		t.Fatal(err)
	}
	if result.KID != "" {
		t.Fatalf("expected empty KID, got %s", result.KID)
	}
}

func TestJWKParse_OctWithAlg(t *testing.T) {
	data := []byte(`{"kty":"oct","k":"Zm9vYmFy","alg":"HS256","kid":"sym-key"}`)
	result, err := ParseJWK(data)
	if err != nil {
		t.Fatal(err)
	}

	if result.Alg != HS256 {
		t.Fatalf("expected HS256, got %s", result.Alg)
	}
	if result.KID != "sym-key" {
		t.Fatalf("expected sym-key, got %s", result.KID)
	}
	key, ok := result.Key.([]byte)
	if !ok {
		t.Fatalf("expected []byte, got %T", result.Key)
	}
	if string(key) != "foobar" {
		t.Fatalf("expected foobar, got %s", key)
	}
}
