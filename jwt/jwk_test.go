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
	if result.ID != kid {
		t.Fatalf("expected KID %s, got %s", kid, result.ID)
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
	if result.ID != "p256" {
		t.Fatalf("expected KID p256, got %s", result.ID)
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
	if result.ID != "rsa-2048" {
		t.Fatalf("expected KID rsa-2048, got %s", result.ID)
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
	if result.ID != "test-rsa-kid" {
		t.Fatalf("expected KID test-rsa-kid, got %s", result.ID)
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
	if result.ID != "" {
		t.Fatalf("expected empty KID, got %s", result.ID)
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
	if result.ID != "sym-key" {
		t.Fatalf("expected sym-key, got %s", result.ID)
	}
	key, ok := result.Key.([]byte)
	if !ok {
		t.Fatalf("expected []byte, got %T", result.Key)
	}
	if string(key) != "foobar" {
		t.Fatalf("expected foobar, got %s", key)
	}
}

// ── JWKS (JSON Web Key Set) ─────────────────────────────────

func TestJWKSRoundTrip_MixedKeys(t *testing.T) {
	_, edPriv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	ecPriv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	rsaPriv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	hmacKey := make([]byte, 32)
	rand.Read(hmacKey)

	edJWK, err := EncodeToJWK(edPriv, EdDSA, "ed25519-key")
	if err != nil {
		t.Fatal(err)
	}
	ecJWK, err := EncodeToJWK(ecPriv, ES256, "ecdsa-key")
	if err != nil {
		t.Fatal(err)
	}
	rsaJWK, err := EncodeToJWK(rsaPriv, RS256, "rsa-key")
	if err != nil {
		t.Fatal(err)
	}
	hmacJWK, err := EncodeToJWK([]byte(hmacKey), HS256, "hmac-key")
	if err != nil {
		t.Fatal(err)
	}

	jwksJSON := `{"keys":[` +
		string(edJWK) + `,` +
		string(ecJWK) + `,` +
		string(rsaJWK) + `,` +
		string(hmacJWK) +
		`]}`

	keys, err := ParseJWKS([]byte(jwksJSON))
	if err != nil {
		t.Fatal(err)
	}

	if len(keys) != 4 {
		t.Fatalf("expected 4 keys, got %d", len(keys))
	}

	if keys[0].ID != "ed25519-key" || keys[0].Alg != EdDSA {
		t.Fatalf("unexpected key 0: Alg=%s ID=%s", keys[0].Alg, keys[0].ID)
	}
	if _, ok := keys[0].Key.(ed25519.PrivateKey); !ok {
		t.Fatalf("expected ed25519.PrivateKey, got %T", keys[0].Key)
	}

	if keys[1].ID != "ecdsa-key" || keys[1].Alg != ES256 {
		t.Fatalf("unexpected key 1: Alg=%s ID=%s", keys[1].Alg, keys[1].ID)
	}
	if _, ok := keys[1].Key.(*ecdsa.PrivateKey); !ok {
		t.Fatalf("expected *ecdsa.PrivateKey, got %T", keys[1].Key)
	}

	if keys[2].ID != "rsa-key" || keys[2].Alg != RS256 {
		t.Fatalf("unexpected key 2: Alg=%s ID=%s", keys[2].Alg, keys[2].ID)
	}
	if _, ok := keys[2].Key.(*rsa.PrivateKey); !ok {
		t.Fatalf("expected *rsa.PrivateKey, got %T", keys[2].Key)
	}

	if keys[3].ID != "hmac-key" || keys[3].Alg != HS256 {
		t.Fatalf("unexpected key 3: Alg=%s ID=%s", keys[3].Alg, keys[3].ID)
	}
	if _, ok := keys[3].Key.([]byte); !ok {
		t.Fatalf("expected []byte, got %T", keys[3].Key)
	}
}

func TestJWKS_EmptySet(t *testing.T) {
	keys, err := ParseJWKS([]byte(`{"keys":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) != 0 {
		t.Fatalf("expected 0 keys, got %d", len(keys))
	}
}

func TestJWKS_SingleKey(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	jwkJSON, err := EncodeToJWK(priv, EdDSA, "single")
	if err != nil {
		t.Fatal(err)
	}
	jwksJSON := `{"keys":[` + string(jwkJSON) + `]}`

	keys, err := ParseJWKS([]byte(jwksJSON))
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) != 1 {
		t.Fatalf("expected 1 key, got %d", len(keys))
	}
	if keys[0].ID != "single" {
		t.Fatalf("expected ID single, got %s", keys[0].ID)
	}
}

func TestJWKS_WithPublicKeys(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	ecPriv, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	edJWK, err := EncodeToJWK(pub, EdDSA, "")
	if err != nil {
		t.Fatal(err)
	}
	ecJWK, err := EncodeToJWK(&ecPriv.PublicKey, ES384, "")
	if err != nil {
		t.Fatal(err)
	}

	jwksJSON := `{"keys":[` + string(edJWK) + `,` + string(ecJWK) + `]}`
	keys, err := ParseJWKS([]byte(jwksJSON))
	if err != nil {
		t.Fatal(err)
	}

	if len(keys) != 2 {
		t.Fatalf("expected 2 keys, got %d", len(keys))
	}
	if keys[0].Alg != EdDSA {
		t.Fatalf("expected EdDSA, got %s", keys[0].Alg)
	}
	if keys[1].Alg != ES384 {
		t.Fatalf("expected ES384, got %s", keys[1].Alg)
	}
}

// ── JWKS Error cases ────────────────────────────────────────

func TestJWKSError_BadJSON(t *testing.T) {
	_, err := ParseJWKS([]byte(`{bad json`))
	if err != ErrInvalidJWK {
		t.Fatalf("expected ErrInvalidJWK, got %v", err)
	}
}

func TestJWKSError_NoKeysField(t *testing.T) {
	_, err := ParseJWKS([]byte(`{"kty":"RSA"}`))
	if err != ErrInvalidJWK {
		t.Fatalf("expected ErrInvalidJWK, got %v", err)
	}
}

func TestJWKSError_KeysNotArray(t *testing.T) {
	_, err := ParseJWKS([]byte(`{"keys":"not-an-array"}`))
	if err != ErrInvalidJWK {
		t.Fatalf("expected ErrInvalidJWK, got %v", err)
	}
}

func TestJWKSError_InvalidKeyInSet(t *testing.T) {
	jwksJSON := `{"keys":[
		{"kty":"OKP","crv":"Ed25519","x":"11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaNcAlUM"},
		{"kty":"UNKNOWN"}
	]}`

	_, err := ParseJWKS([]byte(jwksJSON))
	if err != ErrUnsupportedKeyType {
		t.Fatalf("expected ErrUnsupportedKeyType, got %v", err)
	}
}

func TestJWKSError_BadBase64InSet(t *testing.T) {
	jwksJSON := `{"keys":[
		{"kty":"oct","k":"!!!not-base64!!!"}
	]}`

	_, err := ParseJWKS([]byte(jwksJSON))
	if err != ErrInvalidJWK {
		t.Fatalf("expected ErrInvalidJWK, got %v", err)
	}
}
