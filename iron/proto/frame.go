package proto

import (
	"errors"
	"net"

	"github.com/skerkour/stdx-go/iron/base"
)

// Relay protocol versions negotiated via the WebSocket subprotocol.
const (
	RelayProtocolV1 = "iron-relay-v1"
	RelayProtocolV2 = "iron-relay-v2"
)

// HandshakeDomainSep is the domain-separation string for the relay
// challenge signature.
const HandshakeDomainSep = "iron-relay handshake v1 challenge signature"

// Frame tags
const (
	ServerChallenge    = 0  // [16B challenge]
	ClientAuth         = 1  // [32B pubkey][64B signature]
	ServerConfirmsAuth = 2  // (empty payload)
	ServerDeniesAuth   = 3  // [postcard String]
	ClientToRelay      = 4  // [32B dst id][1B ecn][packet]
	ClientToRelayBatch = 5  // [32B dst id][1B ecn][u16 segment][packets]
	RelayToClient      = 6  // [32B src id][1B ecn][packet]
	RelayToClientBatch = 7  // [32B src id][1B ecn][u16 segment][packets]
	EndpointGone       = 8  // [32B id]
	Ping               = 9  // [8B]
	Pong               = 10 // [8B]
	Status             = 13 // [1B]

	// Extension frames for direct-address discovery
	SetEndpoints = 20 // client->relay [nn[{1B fam}{4/16B ip}{2B port BE}]]
	GetEndpoints = 21 // client->relay [32B nodeID]
	EndpointList = 22 // relay->client [32B nodeID][nn[...]]
	ObservedAddr = 23 // relay->client [1B family][4/16B ip]
)

const (
	// MaxPacketSize mirrors MAX_PACKET_SIZE: 64 KiB per datagram.
	MaxPacketSize = 64 * 1024
	// MaxFrameSize mirrors MAX_FRAME_SIZE: 1 MiB per websocket message.
	MaxFrameSize = 1024 * 1024
)

// Parse reads the frame tag from a received websocket message and returns
// the tag and the remaining payload.
func Parse(msg []byte) (uint64, []byte, error) {
	tag, n := readVarint(msg)
	if n == 0 {
		return 0, nil, errors.New("invalid frame: missing tag")
	}
	payload := msg[n:]
	if len(payload) > MaxFrameSize {
		return 0, nil, errors.New("invalid frame: too large")
	}
	return tag, payload, nil
}

// Datagram is a single QUIC packet forwarded through the relay.
type Datagram struct {
	// Remote is the sender for RelayToClient frames, or the destination
	// for ClientToRelay frames.
	Remote base.NodeID
	Ecn    byte
	Pkt    []byte
}

// EncodeDatagram wraps a single QUIC packet into a one-datagram relay frame
// with the given tag (ClientToRelay or RelayToClient).
func EncodeDatagram(dst []byte, tag uint64, id base.NodeID, ecn byte, pkt []byte) []byte {
	dst = appendVarint(dst, tag)
	dst = append(dst, id[:]...)
	dst = append(dst, ecn)
	dst = append(dst, pkt...)
	return dst
}

// ParseDatagram unwraps the payload of a one-datagram frame (tag 4 or 6).
func ParseDatagram(payload []byte) (*Datagram, error) {
	if len(payload) < base.NodeIDLen+1 {
		return nil, errors.New("invalid datagram frame: too short")
	}
	var id base.NodeID
	copy(id[:], payload[:base.NodeIDLen])
	return &Datagram{Remote: id, Ecn: payload[base.NodeIDLen], Pkt: payload[base.NodeIDLen+1:]}, nil
}

// DatagramBatch is a set of same-sized QUIC packets forwarded through the
// relay in one frame (tag 5 or 7). The contents are an integer number of
// packets, each SegmentSize bytes long.
type DatagramBatch struct {
	// Remote is the sender for RelayToClient frames, or the destination
	// for ClientToRelay frames.
	Remote      base.NodeID
	Ecn         byte
	SegmentSize uint16
	Contents    []byte
}

// EncodeDatagramBatch wraps a batch of same-sized packets into a relay frame
// with the given tag (ClientToRelayBatch or RelayToClientBatch).
func EncodeDatagramBatch(dst []byte, tag uint64, id base.NodeID, ecn byte, segmentSize uint16, contents []byte) []byte {
	dst = appendVarint(dst, tag)
	dst = append(dst, id[:]...)
	dst = append(dst, ecn)
	dst = append(dst, byte(segmentSize>>8), byte(segmentSize))
	dst = append(dst, contents...)
	return dst
}

// ParseDatagramBatch unwraps the payload of a batch frame (tag 5 or 7).
func ParseDatagramBatch(payload []byte) (*DatagramBatch, error) {
	if len(payload) < base.NodeIDLen+3 {
		return nil, errors.New("invalid datagram batch: too short")
	}
	var id base.NodeID
	copy(id[:], payload[:base.NodeIDLen])
	ecn := payload[base.NodeIDLen]
	segmentSize := uint16(payload[base.NodeIDLen+1])<<8 | uint16(payload[base.NodeIDLen+2])
	contents := payload[base.NodeIDLen+3:]
	if segmentSize == 0 || len(contents)%int(segmentSize) != 0 {
		return nil, errors.New("invalid datagram batch: contents not a multiple of segment size")
	}
	return &DatagramBatch{
		Remote:      id,
		Ecn:         ecn,
		SegmentSize: segmentSize,
		Contents:    contents,
	}, nil
}

// Packets splits a DatagramBatch into its individual packets.
func (b *DatagramBatch) Packets() [][]byte {
	n := len(b.Contents) / int(b.SegmentSize)
	out := make([][]byte, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, b.Contents[i*int(b.SegmentSize):(i+1)*int(b.SegmentSize)])
	}
	return out
}

// EncodeFrameTag frames a payload with the given tag.
func EncodeFrameTag(tag uint64, payload []byte) []byte {
	return append(appendVarint(nil, tag), payload...)
}

// EncodeServerChallenge encodes the challenge for the auth handshake.
func EncodeServerChallenge(challenge [16]byte) []byte {
	return EncodeFrameTag(ServerChallenge, challenge[:])
}

// EncodeClientAuth encodes the client's public identity and the signature of
// the blake3-derived challenge key.
func EncodeClientAuth(id base.NodeID, sig [64]byte) []byte {
	out := appendVarint(nil, ClientAuth)
	out = append(out, id[:]...)
	return append(out, sig[:]...)
}

// EncodeServerConfirmsAuth encodes the successful auth confirmation.
func EncodeServerConfirmsAuth() []byte { return EncodeFrameTag(ServerConfirmsAuth, nil) }

// EncodeServerDeniesAuth encodes a denial. postcard encodes an empty String
// as a single 0x00 length byte.
func EncodeServerDeniesAuth() []byte { return EncodeFrameTag(ServerDeniesAuth, []byte{0}) }

// EncodePing encodes a relay ping with the given 8-byte payload.
func EncodePing(p [8]byte) []byte { return EncodeFrameTag(Ping, p[:]) }

// EncodePong encodes a relay pong echoing an 8-byte ping payload.
func EncodePong(p [8]byte) []byte { return EncodeFrameTag(Pong, p[:]) }

// MaxEndpoints bounds the number of direct addresses a client may announce.
const MaxEndpoints = 32

// AppendAddr appends a single address entry: [1B family][4 or 16B ip][2B port BE].
func AppendAddr(dst []byte, a *net.UDPAddr) []byte {
	if ip4 := a.IP.To4(); ip4 != nil {
		dst = append(dst, 4)
		dst = append(dst, ip4...)
	} else {
		ip6 := a.IP.To16()
		if ip6 == nil {
			ip6 = make(net.IP, 16)
		}
		dst = append(dst, 16)
		dst = append(dst, ip6...)
	}
	return append(dst, byte(a.Port>>8), byte(a.Port))
}

// ParseAddr parses a single address entry, returning the address and the
// number of bytes consumed.
func ParseAddr(payload []byte) (*net.UDPAddr, int, error) {
	if len(payload) < 3 {
		return nil, 0, errors.New("short address entry")
	}
	fam := payload[0]
	n := 0
	var ip net.IP
	switch fam {
	case 4:
		if len(payload) < 1+4+2 {
			return nil, 0, errors.New("short ipv4 address")
		}
		ip = net.IP(append([]byte(nil), payload[1:5]...))
		n = 1 + 4
	case 16:
		if len(payload) < 1+16+2 {
			return nil, 0, errors.New("short ipv6 address")
		}
		ip = net.IP(append([]byte(nil), payload[1:17]...))
		n = 1 + 16
	default:
		return nil, 0, errors.New("unknown address family")
	}
	port := int(payload[n])<<8 | int(payload[n+1])
	return &net.UDPAddr{IP: ip, Port: port}, n + 2, nil
}

// EncodeAddrs encodes an address list: [varint count][entries...].
func EncodeAddrs(dst []byte, addrs []*net.UDPAddr) []byte {
	dst = appendVarint(dst, uint64(len(addrs)))
	for _, a := range addrs {
		dst = AppendAddr(dst, a)
	}
	return dst
}

// ParseAddrs decodes the address list produced by EncodeAddrs.
func ParseAddrs(payload []byte) ([]*net.UDPAddr, error) {
	count, n := readVarint(payload)
	if n == 0 {
		return nil, errors.New("invalid address list")
	}
	payload = payload[n:]
	if count > MaxEndpoints {
		return nil, errors.New("too many addresses")
	}
	addrs := make([]*net.UDPAddr, 0, count)
	for i := uint64(0); i < count; i++ {
		a, used, err := ParseAddr(payload)
		if err != nil {
			return nil, err
		}
		payload = payload[used:]
		addrs = append(addrs, a)
	}
	return addrs, nil
}

// EncodeSetEndpoints frames the client's announced direct addresses.
func EncodeSetEndpoints(addrs []*net.UDPAddr) []byte {
	return EncodeFrameTag(SetEndpoints, EncodeAddrs(nil, addrs))
}

// EncodeGetEndpoints frames a lookup request for a peer's direct addresses.
func EncodeGetEndpoints(id base.NodeID) []byte {
	out := appendVarint(nil, GetEndpoints)
	return append(out, id[:]...)
}

// EncodeEndpointList frames the relay's answer to GetEndpoints.
func EncodeEndpointList(id base.NodeID, addrs []*net.UDPAddr) []byte {
	out := appendVarint(nil, EndpointList)
	out = append(out, id[:]...)
	return EncodeAddrs(out, addrs)
}

// ParseEndpointList decodes an EndpointList frame payload.
func ParseEndpointList(payload []byte) (base.NodeID, []*net.UDPAddr, error) {
	if len(payload) < base.NodeIDLen {
		return base.NodeID{}, nil, errors.New("short endpoint list")
	}
	var id base.NodeID
	copy(id[:], payload[:base.NodeIDLen])
	addrs, err := ParseAddrs(payload[base.NodeIDLen:])
	return id, addrs, err
}

// EncodeObservedAddr frames the relay's view of the client's source address
// (its public IP, or its LAN IP when the relay is local). The client uses it
// to skip dialing its own NAT's public address, which can never reach a peer.
func EncodeObservedAddr(ip net.IP) []byte {
	out := appendVarint(nil, ObservedAddr)
	if ip4 := ip.To4(); ip4 != nil {
		out = append(out, 4)
		out = append(out, ip4...)
	} else if ip6 := ip.To16(); ip6 != nil {
		out = append(out, 16)
		out = append(out, ip6...)
	}
	return out
}

// ParseObservedAddr decodes an ObservedAddr frame payload.
func ParseObservedAddr(payload []byte) (net.IP, error) {
	if len(payload) == 0 {
		return nil, errors.New("short observed address")
	}
	switch payload[0] {
	case 4:
		if len(payload) != 1+4 {
			return nil, errors.New("short observed address")
		}
		return net.IP(append([]byte(nil), payload[1:]...)), nil
	case 16:
		if len(payload) != 1+16 {
			return nil, errors.New("short observed address")
		}
		return net.IP(append([]byte(nil), payload[1:]...)), nil
	default:
		return nil, errors.New("unknown address family")
	}
}
