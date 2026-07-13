package jwt

import (
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rsa"
	"encoding/base64"
	"encoding/json"
	"math/big"
)

// JWK holds the decoded key and its JWK metadata.
// Returned by value to minimize allocations.
type JWK struct {
	// Key ID
	ID  string
	Alg Algorithm
	Key any
}

type jwkFields struct {
	KTY string `json:"kty"`
	USE string `json:"use,omitempty"`
	CRV string `json:"crv,omitempty"`
	ALG string `json:"alg,omitempty"`
	KID string `json:"kid,omitempty"`
	X   string `json:"x,omitempty"`
	Y   string `json:"y,omitempty"`
	D   string `json:"d,omitempty"`
	N   string `json:"n,omitempty"`
	E   string `json:"e,omitempty"`
	P   string `json:"p,omitempty"`
	Q   string `json:"q,omitempty"`
	DP  string `json:"dp,omitempty"`
	DQ  string `json:"dq,omitempty"`
	QI  string `json:"qi,omitempty"`
	K   string `json:"k,omitempty"`
}

// EncodeToJWK serializes a Go crypto key to RFC 7517 JSON.
//
// Supported key types:
//
//	ed25519.PrivateKey / ed25519.PublicKey  -> OKP Ed25519 (alg=EdDSA, alg parameter ignored)
//	*ecdsa.PrivateKey / *ecdsa.PublicKey    -> EC P-256/P-384/P-521 (alg derived from curve)
//	*rsa.PrivateKey / *rsa.PublicKey        -> RSA (alg parameter required: RS*/PS*)
//	[]byte                                  -> oct symmetric key (alg required: HS*)
func EncodeToJWK(keyAny any, alg Algorithm, kid string) ([]byte, error) {
	w := newJwkWriter(128)
	w.writeByte('{')

	switch key := keyAny.(type) {
	case ed25519.PrivateKey:
		if len(key) != ed25519.PrivateKeySize {
			return nil, ErrInvalidJWK
		}
		w.writeString("kty", "OKP")
		w.writeString("crv", "Ed25519")
		w.writeString("alg", string(EdDSA))
		w.writeBase64("x", key[ed25519.SeedSize:])
		w.writeBase64("d", key[:ed25519.SeedSize])

	case ed25519.PublicKey:
		if len(key) != ed25519.PublicKeySize {
			return nil, ErrInvalidJWK
		}
		w.writeString("kty", "OKP")
		w.writeString("crv", "Ed25519")
		w.writeString("alg", string(EdDSA))
		w.writeBase64("x", key)

	case *ecdsa.PrivateKey:
		crv, ecAlg, coordLen, err := curveInfo(key.Curve)
		if err != nil {
			return nil, err
		}
		w.writeString("kty", "EC")
		w.writeString("crv", crv)
		w.writeString("alg", string(ecAlg))
		w.writeBase64("x", bigIntFixed(key.X, coordLen))
		w.writeBase64("y", bigIntFixed(key.Y, coordLen))
		w.writeBase64("d", bigIntFixed(key.D, coordLen))

	case *ecdsa.PublicKey:
		crv, ecAlg, coordLen, err := curveInfo(key.Curve)
		if err != nil {
			return nil, err
		}
		w.writeString("kty", "EC")
		w.writeString("crv", crv)
		w.writeString("alg", string(ecAlg))
		w.writeBase64("x", bigIntFixed(key.X, coordLen))
		w.writeBase64("y", bigIntFixed(key.Y, coordLen))

	case *rsa.PrivateKey:
		if !alg.isRSA() {
			return nil, ErrInvalidAlgorithm
		}
		key.Precompute()
		w.writeString("kty", "RSA")
		w.writeString("alg", string(alg))
		w.writeBase64("n", key.N.Bytes())
		w.writeBase64("e", intToBytes(uint32(key.E)))
		w.writeBase64("d", key.D.Bytes())
		if len(key.Primes) >= 2 {
			w.writeBase64("p", key.Primes[0].Bytes())
			w.writeBase64("q", key.Primes[1].Bytes())
		}
		if key.Precomputed.Dp != nil {
			w.writeBase64("dp", key.Precomputed.Dp.Bytes())
		}
		if key.Precomputed.Dq != nil {
			w.writeBase64("dq", key.Precomputed.Dq.Bytes())
		}
		if key.Precomputed.Qinv != nil {
			w.writeBase64("qi", key.Precomputed.Qinv.Bytes())
		}

	case *rsa.PublicKey:
		if !alg.isRSA() {
			return nil, ErrInvalidAlgorithm
		}
		w.writeString("kty", "RSA")
		w.writeString("alg", string(alg))
		w.writeBase64("n", key.N.Bytes())
		w.writeBase64("e", intToBytes(uint32(key.E)))

	case []byte:
		if !alg.isHMAC() {
			return nil, ErrInvalidAlgorithm
		}
		w.writeString("kty", "oct")
		w.writeString("alg", string(alg))
		w.writeBase64("k", key)

	default:
		return nil, ErrUnsupportedKeyType
	}

	if kid != "" {
		w.writeString("kid", kid)
	}

	w.writeByte('}')
	return w.buf, nil
}

// ParseJWK decodes a JSON Web Key and returns the Go crypto key.
// The returned JWK.Key is one of:
//
//	ed25519.PrivateKey, ed25519.PublicKey,
//	*ecdsa.PrivateKey, *ecdsa.PublicKey,
//	*rsa.PrivateKey, *rsa.PublicKey,
//	[]byte (symmetric).
func ParseJWK(data []byte) (JWK, error) {
	var raw jwkFields
	if err := json.Unmarshal(data, &raw); err != nil {
		return JWK{}, ErrInvalidJWK
	}
	return parseJWK(&raw)
}

// ParseJWKS decodes a JSON Web Key Set (JWKS) as defined in RFC 7517 Section 5.
// The returned slice contains one JWK per entry in the "keys" array.
func ParseJWKS(data []byte) ([]JWK, error) {
	var raw struct {
		Keys []jwkFields `json:"keys"`
	}
	var err error

	if err = json.Unmarshal(data, &raw); err != nil || raw.Keys == nil {
		return nil, ErrInvalidJWK
	}

	keys := make([]JWK, len(raw.Keys))
	for i := range raw.Keys {
		keys[i], err = parseJWK(&raw.Keys[i])
		if err != nil {
			return nil, err
		}
	}

	return keys, nil
}

func parseJWK(raw *jwkFields) (JWK, error) {
	switch raw.KTY {
	case "OKP":
		x, err := base64.RawURLEncoding.DecodeString(raw.X)
		if err != nil {
			return JWK{}, ErrInvalidJWK
		}

		switch raw.CRV {
		case "Ed25519":
			jwk := JWK{
				Alg: EdDSA,
				ID:  raw.KID,
			}
			if raw.D != "" {
				d, err := base64.RawURLEncoding.DecodeString(raw.D)
				if err != nil || len(d) != ed25519.SeedSize || len(x) != ed25519.PublicKeySize {
					return JWK{}, ErrInvalidJWK
				}
				priv := make(ed25519.PrivateKey, ed25519.PrivateKeySize)
				copy(priv[:ed25519.SeedSize], d)
				copy(priv[ed25519.SeedSize:], x)
				jwk.Key = priv
			} else {
				if len(x) != ed25519.PublicKeySize {
					return JWK{}, ErrInvalidJWK
				}
				jwk.Key = ed25519.PublicKey(x)
			}
			return jwk, nil

		default:
			return JWK{}, ErrUnsupportedCurve
		}

	case "EC":
		x, err := base64.RawURLEncoding.DecodeString(raw.X)
		if err != nil {
			return JWK{}, ErrInvalidJWK
		}
		y, err := base64.RawURLEncoding.DecodeString(raw.Y)
		if err != nil {
			return JWK{}, ErrInvalidJWK
		}

		var curve elliptic.Curve
		var alg Algorithm
		switch raw.CRV {
		case "P-256":
			curve = elliptic.P256()
			alg = ES256
		case "P-384":
			curve = elliptic.P384()
			alg = ES384
		case "P-521":
			curve = elliptic.P521()
			alg = ES512
		default:
			return JWK{}, ErrUnsupportedCurve
		}

		result := JWK{
			Alg: alg,
			ID:  raw.KID,
		}
		xBig := new(big.Int).SetBytes(x)
		yBig := new(big.Int).SetBytes(y)

		if raw.D != "" {
			d, err := base64.RawURLEncoding.DecodeString(raw.D)
			if err != nil {
				return JWK{}, ErrInvalidJWK
			}
			dBig := new(big.Int).SetBytes(d)
			result.Key = &ecdsa.PrivateKey{
				PublicKey: ecdsa.PublicKey{
					Curve: curve,
					X:     xBig,
					Y:     yBig,
				},
				D: dBig,
			}
		} else {
			result.Key = &ecdsa.PublicKey{
				Curve: curve,
				X:     xBig,
				Y:     yBig,
			}
		}
		return result, nil

	case "RSA":
		n, err := base64.RawURLEncoding.DecodeString(raw.N)
		if err != nil {
			return JWK{}, ErrInvalidJWK
		}
		e, err := base64.RawURLEncoding.DecodeString(raw.E)
		if err != nil {
			return JWK{}, ErrInvalidJWK
		}

		alg := Algorithm(raw.ALG)
		if !alg.isRSA() {
			return JWK{}, ErrInvalidAlgorithm
		}

		result := JWK{
			Alg: alg,
			ID:  raw.KID,
		}
		nBig := new(big.Int).SetBytes(n)
		eBig := new(big.Int).SetBytes(e)

		if raw.D != "" {
			d, err := base64.RawURLEncoding.DecodeString(raw.D)
			if err != nil {
				return JWK{}, ErrInvalidJWK
			}
			dBig := new(big.Int).SetBytes(d)

			priv := &rsa.PrivateKey{
				PublicKey: rsa.PublicKey{
					N: nBig,
					E: int(eBig.Int64()),
				},
				D: dBig,
			}

			if raw.P != "" && raw.Q != "" {
				p, err := base64.RawURLEncoding.DecodeString(raw.P)
				if err != nil {
					return JWK{}, ErrInvalidJWK
				}
				q, err := base64.RawURLEncoding.DecodeString(raw.Q)
				if err != nil {
					return JWK{}, ErrInvalidJWK
				}
				priv.Primes = []*big.Int{
					new(big.Int).SetBytes(p),
					new(big.Int).SetBytes(q),
				}

				if raw.DP != "" {
					dp, _ := base64.RawURLEncoding.DecodeString(raw.DP)
					if dp != nil {
						priv.Precomputed.Dp = new(big.Int).SetBytes(dp)
					}
				}
				if raw.DQ != "" {
					dq, _ := base64.RawURLEncoding.DecodeString(raw.DQ)
					if dq != nil {
						priv.Precomputed.Dq = new(big.Int).SetBytes(dq)
					}
				}
				if raw.QI != "" {
					qi, _ := base64.RawURLEncoding.DecodeString(raw.QI)
					if qi != nil {
						priv.Precomputed.Qinv = new(big.Int).SetBytes(qi)
					}
				}
			}

			result.Key = priv
		} else {
			result.Key = &rsa.PublicKey{
				N: nBig,
				E: int(eBig.Int64()),
			}
		}
		return result, nil

	case "oct":
		k, err := base64.RawURLEncoding.DecodeString(raw.K)
		if err != nil {
			return JWK{}, ErrInvalidJWK
		}
		return JWK{
			Key: k,
			Alg: Algorithm(raw.ALG),
			ID:  raw.KID,
		}, nil

	default:
		return JWK{}, ErrUnsupportedKeyType
	}
}

// ── internal helpers ────────────────────────────────────────

type jwkWriter struct {
	buf    []byte
	hasKey bool
}

func newJwkWriter(capacity int) jwkWriter {
	return jwkWriter{
		buf:    make([]byte, 0, capacity),
		hasKey: false,
	}
}

func (w *jwkWriter) writeString(key, value string) {
	if w.hasKey {
		w.buf = append(w.buf, ',')
	}
	w.hasKey = true
	w.buf = append(w.buf, '"')
	w.buf = append(w.buf, key...)
	w.buf = append(w.buf, '"', ':')
	w.buf = append(w.buf, '"')
	w.buf = append(w.buf, value...)
	w.buf = append(w.buf, '"')
}

func (w *jwkWriter) writeBase64(key string, data []byte) {
	if w.hasKey {
		w.buf = append(w.buf, ',')
	}
	w.hasKey = true
	w.buf = append(w.buf, '"')
	w.buf = append(w.buf, key...)
	w.buf = append(w.buf, '"', ':')
	w.buf = append(w.buf, '"')
	w.buf = appendBase64(w.buf, data)
	w.buf = append(w.buf, '"')
}

func (w *jwkWriter) writeByte(b byte) {
	w.buf = append(w.buf, b)
}

func intToBytes(v uint32) []byte {
	var tmp [4]byte
	tmp[0] = byte(v >> 24)
	tmp[1] = byte(v >> 16)
	tmp[2] = byte(v >> 8)
	tmp[3] = byte(v)
	i := 0
	for i < 3 && tmp[i] == 0 {
		i++
	}
	out := make([]byte, 4-i)
	copy(out, tmp[i:])
	return out
}

func bigIntFixed(x *big.Int, minLen int) []byte {
	b := x.Bytes()
	if len(b) >= minLen {
		return b
	}
	padded := make([]byte, minLen)
	copy(padded[minLen-len(b):], b)
	return padded
}

func curveInfo(curve elliptic.Curve) (crv string, alg Algorithm, coordLen int, err error) {
	switch curve {
	case elliptic.P256():
		return "P-256", ES256, 32, nil
	case elliptic.P384():
		return "P-384", ES384, 48, nil
	case elliptic.P521():
		return "P-521", ES512, 66, nil
	default:
		return "", "", 0, ErrUnsupportedCurve
	}
}
