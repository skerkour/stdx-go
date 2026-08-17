// Package relayserver implements a minimal relay server: it
// authenticates clients by challenge/signature and forwards opaque QUIC
// datagrams between them.
package relayserver

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/zeebo/blake3"

	"github.com/skerkour/stdx-go/iron/base"
	"github.com/skerkour/stdx-go/iron/proto"
)

const relayPath = "/relay"

// Keepalive tuning, mirroring the client: clients are pinged every
// pingInterval and disconnected when silent for pingTimeout. Package vars so
// tests can shrink them.
var (
	pingInterval = 15 * time.Second
	pingTimeout  = 45 * time.Second
)

// parseRemoteAddr converts a "host:port" string (e.g. r.RemoteAddr) into a
// UDP-style address holding the observed source IP.
func parseRemoteAddr(s string) *net.UDPAddr {
	host, portStr, err := net.SplitHostPort(s)
	if err != nil {
		return nil
	}
	port, _ := strconv.Atoi(portStr)
	return &net.UDPAddr{IP: net.ParseIP(host), Port: port}
}

// Server is a relay. Clients connect to /relay, authenticate, and exchange
// datagrams keyed by NodeID.
type Server struct {
	Log       *slog.Logger
	mu        sync.Mutex
	clients   map[base.NodeID]*client
	endpoints map[base.NodeID][]*net.UDPAddr // direct addresses announced by clients
}

// client is an authenticated relay connection.
type client struct {
	id       base.NodeID
	ws       *websocket.Conn
	ctx      context.Context
	observed net.Addr // source address the relay sees on this connection

	mu sync.Mutex // serializes outbound writes

	lastRecvNano atomic.Int64 // unixnano of the last received frame (0 = none)
}

// NewServer creates an empty relay server.
func NewServer() *Server {
	return &Server{
		clients:   make(map[base.NodeID]*client),
		endpoints: make(map[base.NodeID][]*net.UDPAddr),
	}
}

// Handler returns the HTTP handler serving the relay.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc(relayPath, s.handleRelay)
	return mux
}

// ListenAndServe starts the relay on addr (e.g. ":3333").
func (s *Server) ListenAndServe(addr string) error {
	return http.ListenAndServe(addr, s.Handler())
}

// Serve starts the relay on the given listener.
func (s *Server) Serve(l net.Listener) error {
	return http.Serve(l, s.Handler())
}

// Close force-closes all connected clients. Active websocket connections are
// hijacked (not tracked by the http.Server), so they must be torn down
// explicitly for the clients to observe the outage.
func (s *Server) Close() error {
	s.mu.Lock()
	clients := make([]*client, 0, len(s.clients))
	for _, cl := range s.clients {
		clients = append(clients, cl)
	}
	s.mu.Unlock()
	for _, cl := range clients {
		cl.mu.Lock()
		cl.ws.CloseNow()
		cl.mu.Unlock()
	}
	return nil
}

func (s *Server) handleRelay(w http.ResponseWriter, r *http.Request) {
	ws, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		Subprotocols:       []string{proto.RelayProtocolV2},
		InsecureSkipVerify: true,
	})
	if err != nil {
		return
	}
	if ws.Subprotocol() != proto.RelayProtocolV2 {
		ws.Close(websocket.StatusPolicyViolation, "unsupported relay protocol")
		return
	}

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	id, err := s.serverHandshake(ctx, ws)
	if err != nil {
		if s.Log != nil {
			s.Log.Info("relay handshake failed", "err", err)
		}
		ws.Close(websocket.StatusPolicyViolation, "authentication failed")
		return
	}

	cl := &client{id: id, ws: ws, ctx: ctx, observed: parseRemoteAddr(r.RemoteAddr)}
	cl.lastRecvNano.Store(time.Now().UnixNano()) // base for the keepalive watchdog
	// Tell the client the address we observe for it, so it can recognize and
	// skip candidates that point at its own NAT (see Endpoint.filterCandidates).
	if ua, ok := cl.observed.(*net.UDPAddr); ok && ua.IP != nil {
		_ = cl.write(proto.EncodeObservedAddr(ua.IP))
	}
	if old, replaced := s.register(cl); replaced {
		// Kick the previous connection that used the same identity.
		old.mu.Lock()
		old.ws.CloseNow()
		old.mu.Unlock()
	}

	s.logInfo("client connected", id)
	defer func() {
		s.unregister(id, cl)
		s.logInfo("client disconnected", id)
	}()

	go cl.watchdog()
	s.readLoop(cl)
}

// watchdog pings the client periodically and force-closes the connection if
// the client stays silent for pingTimeout, freeing the relay's resources.
func (cl *client) watchdog() {
	tick := time.NewTicker(pingInterval)
	defer tick.Stop()
	for {
		select {
		case <-cl.ctx.Done():
			return
		case <-tick.C:
			var p [8]byte
			binary.LittleEndian.PutUint64(p[:], uint64(time.Now().UnixNano()))
			_ = cl.write(proto.EncodePing(p))
			if time.Since(time.Unix(0, cl.lastRecvNano.Load())) > pingTimeout {
				cl.mu.Lock()
				cl.ws.CloseNow()
				cl.mu.Unlock()
				return
			}
		}
	}
}

// serverHandshake runs the challenge/response auth protocol and returns the
// authenticated client's NodeID.
func (s *Server) serverHandshake(ctx context.Context, ws *websocket.Conn) (base.NodeID, error) {
	var challenge [16]byte
	if _, err := rand.Read(challenge[:]); err != nil {
		return base.NodeID{}, err
	}
	if err := ws.Write(ctx, websocket.MessageBinary, proto.EncodeServerChallenge(challenge)); err != nil {
		return base.NodeID{}, err
	}

	typ, msg, err := ws.Read(ctx)
	if err != nil {
		return base.NodeID{}, err
	}
	if typ != websocket.MessageBinary {
		return base.NodeID{}, errors.New("expected binary auth frame")
	}
	tag, payload, err := proto.Parse(msg)
	if err != nil {
		return base.NodeID{}, err
	}
	if tag != proto.ClientAuth || len(payload) != base.NodeIDLen+64 {
		_ = ws.Write(ctx, websocket.MessageBinary, proto.EncodeServerDeniesAuth())
		return base.NodeID{}, errors.New("invalid client auth frame")
	}

	id, err := base.NodeIDFromBytes(payload[:base.NodeIDLen])
	if err != nil {
		_ = ws.Write(ctx, websocket.MessageBinary, proto.EncodeServerDeniesAuth())
		return base.NodeID{}, err
	}
	var key [32]byte
	blake3.DeriveKey(proto.HandshakeDomainSep, challenge[:], key[:])
	if !base.Verify(id, key[:], payload[base.NodeIDLen:]) {
		_ = ws.Write(ctx, websocket.MessageBinary, proto.EncodeServerDeniesAuth())
		return base.NodeID{}, errors.New("invalid client signature")
	}

	if err := ws.Write(ctx, websocket.MessageBinary, proto.EncodeServerConfirmsAuth()); err != nil {
		return base.NodeID{}, err
	}
	return id, nil
}

// readLoop reads datagrams from a client and forwards them to their
// destinations.
func (s *Server) readLoop(cl *client) {
	for {
		typ, msg, err := cl.ws.Read(cl.ctx)
		if err != nil {
			return
		}
		cl.lastRecvNano.Store(time.Now().UnixNano())
		if typ != websocket.MessageBinary {
			continue
		}
		tag, payload, err := proto.Parse(msg)
		if err != nil {
			continue
		}
		switch tag {
		case proto.ClientToRelay:
			d, err := proto.ParseDatagram(payload)
			if err != nil {
				continue
			}
			s.forward(cl, d.Remote, d.Ecn, d.Pkt)
		case proto.ClientToRelayBatch:
			b, err := proto.ParseDatagramBatch(payload)
			if err != nil {
				continue
			}
			s.forwardBatch(cl, b.Remote, b.Ecn, b.SegmentSize, b.Contents)
		case proto.Ping:
			if len(payload) == 8 {
				var p [8]byte
				copy(p[:], payload)
				_ = cl.write(proto.EncodePong(p))
			}
		case proto.SetEndpoints:
			addrs, err := proto.ParseAddrs(payload)
			if err != nil {
				continue
			}
			s.mu.Lock()
			s.endpoints[cl.id] = addrs
			s.mu.Unlock()
		case proto.GetEndpoints:
			if len(payload) != base.NodeIDLen {
				continue
			}
			var target base.NodeID
			copy(target[:], payload)
			// Always reply with the peer's addresses (possibly empty) so the
			// client's lookup is not left dangling.
			addrs, _ := s.answerFor(target)
			_ = cl.write(proto.EncodeEndpointList(target, addrs))
		}
	}
}

// answerFor composes the direct addresses for a peer: its announced list plus
// a best-effort candidate using the address the relay observed (its public IP
// and, under port-preserving NAT, our UDP port guess). The bool reports
// whether the peer had announced any addresses.
func (s *Server) answerFor(target base.NodeID) ([]*net.UDPAddr, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	addrs, ok := s.endpoints[target]
	if !ok || len(addrs) == 0 {
		return nil, false
	}
	res := append([]*net.UDPAddr(nil), addrs...)

	if cl, ok2 := s.clients[target]; ok2 && cl.observed != nil {
		if observedIP := cl.observed.(*net.UDPAddr).IP; observedIP != nil {
			res = append(res, &net.UDPAddr{IP: observedIP, Port: addrs[0].Port})
		}
	}
	return res, true
}

// forward ships a datagram received from `from` to the client with the given
// destination id, tagging it with the sender's id.
func (s *Server) forward(from *client, dstID base.NodeID, ecn byte, pkt []byte) {
	s.mu.Lock()
	dst, ok := s.clients[dstID]
	s.mu.Unlock()
	if !ok {
		return // destination not connected: drop silently
	}
	out := proto.EncodeDatagram(nil, proto.RelayToClient, from.id, ecn, pkt)
	_ = dst.write(out)
}

// forwardBatch ships a batch of same-sized datagrams to the destination,
// tagged with the sender's id. The batch is forwarded verbatim (the relay
// never re-segments it).
func (s *Server) forwardBatch(from *client, dstID base.NodeID, ecn byte, segmentSize uint16, contents []byte) {
	s.mu.Lock()
	dst, ok := s.clients[dstID]
	s.mu.Unlock()
	if !ok {
		return // destination not connected: drop silently
	}
	out := proto.EncodeDatagramBatch(nil, proto.RelayToClientBatch, from.id, ecn, segmentSize, contents)
	_ = dst.write(out)
}

func (c *client) write(frame []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ws.Write(c.ctx, websocket.MessageBinary, frame)
}

func (s *Server) register(cl *client) (*client, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	old, ok := s.clients[cl.id]
	s.clients[cl.id] = cl
	return old, ok
}

func (s *Server) unregister(id base.NodeID, cl *client) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.clients[id] == cl {
		delete(s.clients, id)
		delete(s.endpoints, id)
	}
}

func (s *Server) logInfo(msg string, id base.NodeID) {
	if s.Log != nil {
		s.Log.Info(msg, "id", id.String())
	}
}
