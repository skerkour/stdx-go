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
	Key any
	Alg Algorithm
	KID string
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
	var fields jwkFields

	switch key := keyAny.(type) {
	case ed25519.PrivateKey:
		if len(key) != ed25519.PrivateKeySize {
			return nil, ErrInvalidJWK
		}
		fields = jwkFields{
			KTY: "OKP",
			CRV: "Ed25519",
			ALG: string(EdDSA),
			X:   base64.RawURLEncoding.EncodeToString(key[ed25519.SeedSize:]),
			D:   base64.RawURLEncoding.EncodeToString(key[:ed25519.SeedSize]),
		}

	case ed25519.PublicKey:
		if len(key) != ed25519.PublicKeySize {
			return nil, ErrInvalidJWK
		}
		fields = jwkFields{
			KTY: "OKP",
			CRV: "Ed25519",
			ALG: string(EdDSA),
			X:   base64.RawURLEncoding.EncodeToString(key),
		}

	case *ecdsa.PrivateKey:
		crv, ecAlg, coordLen, err := curveInfo(key.Curve)
		if err != nil {
			return nil, err
		}
		fields = jwkFields{
			KTY: "EC",
			CRV: crv,
			ALG: string(ecAlg),
			X:   base64.RawURLEncoding.EncodeToString(bigIntFixed(key.X, coordLen)),
			Y:   base64.RawURLEncoding.EncodeToString(bigIntFixed(key.Y, coordLen)),
			D:   base64.RawURLEncoding.EncodeToString(bigIntFixed(key.D, coordLen)),
		}

	case *ecdsa.PublicKey:
		crv, ecAlg, coordLen, err := curveInfo(key.Curve)
		if err != nil {
			return nil, err
		}
		fields = jwkFields{
			KTY: "EC",
			CRV: crv,
			ALG: string(ecAlg),
			X:   base64.RawURLEncoding.EncodeToString(bigIntFixed(key.X, coordLen)),
			Y:   base64.RawURLEncoding.EncodeToString(bigIntFixed(key.Y, coordLen)),
		}

	case *rsa.PrivateKey:
		if !alg.isRSA() {
			return nil, ErrInvalidAlgorithm
		}
		key.Precompute()
		fields = jwkFields{
			KTY: "RSA",
			ALG: string(alg),
			N:   base64.RawURLEncoding.EncodeToString(key.N.Bytes()),
			E:   base64.RawURLEncoding.EncodeToString(big.NewInt(int64(key.E)).Bytes()),
			D:   base64.RawURLEncoding.EncodeToString(key.D.Bytes()),
		}
		if len(key.Primes) >= 2 {
			fields.P = base64.RawURLEncoding.EncodeToString(key.Primes[0].Bytes())
			fields.Q = base64.RawURLEncoding.EncodeToString(key.Primes[1].Bytes())
		}
		if key.Precomputed.Dp != nil {
			fields.DP = base64.RawURLEncoding.EncodeToString(key.Precomputed.Dp.Bytes())
		}
		if key.Precomputed.Dq != nil {
			fields.DQ = base64.RawURLEncoding.EncodeToString(key.Precomputed.Dq.Bytes())
		}
		if key.Precomputed.Qinv != nil {
			fields.QI = base64.RawURLEncoding.EncodeToString(key.Precomputed.Qinv.Bytes())
		}

	case *rsa.PublicKey:
		if !alg.isRSA() {
			return nil, ErrInvalidAlgorithm
		}
		fields = jwkFields{
			KTY: "RSA",
			ALG: string(alg),
			N:   base64.RawURLEncoding.EncodeToString(key.N.Bytes()),
			E:   base64.RawURLEncoding.EncodeToString(big.NewInt(int64(key.E)).Bytes()),
		}

	case []byte:
		if !alg.isHMAC() {
			return nil, ErrInvalidAlgorithm
		}
		fields = jwkFields{
			KTY: "oct",
			ALG: string(alg),
			K:   base64.RawURLEncoding.EncodeToString(key),
		}

	default:
		return nil, ErrUnsupportedKeyType
	}

	if kid != "" {
		fields.KID = kid
	}

	return json.Marshal(fields)
}

// ParseJWK decodes a JSON Web Key and returns the Go crypto key.
// The returned JWKResult.Key is one of:
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

	switch raw.KTY {
	case "OKP":
		x, err := base64.RawURLEncoding.DecodeString(raw.X)
		if err != nil {
			return JWK{}, ErrInvalidJWK
		}

		switch raw.CRV {
		case "Ed25519":
			result := JWK{
				Alg: EdDSA,
				KID: raw.KID,
			}
			if raw.D != "" {
				d, err := base64.RawURLEncoding.DecodeString(raw.D)
				if err != nil || len(d) != ed25519.SeedSize || len(x) != ed25519.PublicKeySize {
					return JWK{}, ErrInvalidJWK
				}
				priv := make(ed25519.PrivateKey, ed25519.PrivateKeySize)
				copy(priv[:ed25519.SeedSize], d)
				copy(priv[ed25519.SeedSize:], x)
				result.Key = priv
			} else {
				if len(x) != ed25519.PublicKeySize {
					return JWK{}, ErrInvalidJWK
				}
				result.Key = ed25519.PublicKey(x)
			}
			return result, nil

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
			KID: raw.KID,
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
			KID: raw.KID,
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
			KID: raw.KID,
		}, nil

	default:
		return JWK{}, ErrUnsupportedKeyType
	}
}

// ── internal helpers ────────────────────────────────────────

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
