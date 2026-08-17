package iron

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"errors"
	"math/big"
	"time"

	"github.com/skerkour/stdx-go/iron/base"
)

// nodeCertificate builds a self-signed X.509 certificate whose key IS the
// node's Ed25519 key. Raw public keys are not supported in Go, so
// authenticating a peer reduces to checking that the presented leaf cert
// carries the expected public key.
func nodeCertificate(secret *base.NodeSecret) (tls.Certificate, error) {
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: secret.Public().String()},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, secret.PublicKey(), secret.PrivateKey())
	if err != nil {
		return tls.Certificate{}, err
	}
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: secret.PrivateKey()}, nil
}

// serverTLSConfig configures the server side: it presents the node's cert and
// demands a client certificate. Any client proving ownership of an Ed25519 key
// is accepted; that key becomes its identity.
func serverTLSConfig(secret *base.NodeSecret, alpn string) (*tls.Config, error) {
	cert, err := nodeCertificate(secret)
	if err != nil {
		return nil, err
	}
	return &tls.Config{
		MinVersion:   tls.VersionTLS13,
		NextProtos:   []string{alpn},
		Certificates: []tls.Certificate{cert},
		ClientAuth:   tls.RequireAnyClientCert,
		VerifyPeerCertificate: func(raw [][]byte, _ [][]*x509.Certificate) error {
			if len(raw) == 0 {
				return errors.New("peer did not present a certificate")
			}
			leaf, err := x509.ParseCertificate(raw[0])
			if err != nil {
				return err
			}
			if _, ok := leaf.PublicKey.(ed25519.PublicKey); !ok {
				return errors.New("peer certificate is not an ed25519 key")
			}
			return nil
		},
	}, nil
}

// clientTLSConfig configures the dial side: it presents the node's cert, sets
// the SNI to the peer's identity-based name, and — instead of hostname
// verification — authenticates the peer by checking that the presented cert
// carries exactly the dialed NodeID.
func clientTLSConfig(secret *base.NodeSecret, peer base.NodeID, alpn string) (*tls.Config, error) {
	cert, err := nodeCertificate(secret)
	if err != nil {
		return nil, err
	}
	return &tls.Config{
		MinVersion:         tls.VersionTLS13,
		NextProtos:         []string{alpn},
		Certificates:       []tls.Certificate{cert},
		ServerName:         peer.Name(),
		InsecureSkipVerify: true, // authentication happens in VerifyConnection below
		VerifyConnection: func(cs tls.ConnectionState) error {
			got, err := peerIDFromTLS(cs)
			if err != nil {
				return err
			}
			if got != peer {
				return errors.New("connected peer does not match the dialed node id")
			}
			return nil
		},
	}, nil
}

// peerIDFromTLS extracts the peer's NodeID from the presented certificate.
func peerIDFromTLS(cs tls.ConnectionState) (base.NodeID, error) {
	if len(cs.PeerCertificates) == 0 {
		return base.NodeID{}, errors.New("peer did not present a certificate")
	}
	pub, ok := cs.PeerCertificates[0].PublicKey.(ed25519.PublicKey)
	if !ok {
		return base.NodeID{}, errors.New("peer certificate is not an ed25519 key")
	}
	return base.NodeIDFromBytes(pub)
}
