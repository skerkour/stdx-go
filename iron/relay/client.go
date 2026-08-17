// Package relay implements the client side of the relay protocol.
//
// The relay data plane carries opaque QUIC packets: each packet is wrapped in
// a relay datagram framed with the destination (or source) NodeID. RelayConn
// presents that tunnel as a net.PacketConn, so a quic.Transport can treat the
// relay exactly like a UDP socket while quic-go keeps routing packets to the
// right connection via the QUIC connection IDs.
package relay

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"net"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/zeebo/blake3"

	"github.com/skerkour/stdx-go/iron/base"
	"github.com/skerkour/stdx-go/iron/proto"
)

const relayPath = "/relay"

// Keepalive tuning. A connection is considered dead when no frame at all has
// been received within pingTimeout; the peer is pinged every pingInterval to
// detect that. Package vars so tests can shrink them.
var (
	pingInterval = 15 * time.Second
	pingTimeout  = 45 * time.Second
)

// RelayConn is the relay transport: a net.PacketConn whose "remote address"
// is a base.PeerAddr (a NodeID). It is safe for concurrent use.
type RelayConn struct {
	ws        *websocket.Conn
	ctx       context.Context
	cancel    context.CancelFunc
	localAddr net.Addr

	writeMu sync.Mutex // serializes outbound websocket writes

	closed chan struct{} // closed when the read loop exits

	q              chan datagram // incoming datagrams for ReadFrom
	readDeadlineMu sync.Mutex
	readDeadline   time.Time
	deadlineCh     chan struct{} // closed+replaced whenever the deadline changes

	lookupsMu sync.Mutex
	lookups   map[base.NodeID]chan []*net.UDPAddr // outstanding GetEndpoints requests

	observed net.IP // our address as seen by the relay (from ObservedAddr)

	batchSize  int // max bytes per batch frame (<=0 disables batching)
	batchCount int // max packets per batch frame
	drainDelay time.Duration

	batchMu sync.Mutex
	pending map[batchKey]*batch // buffered outbound packets

	pingInterval time.Duration // keepalive: ping period
	pingTimeout  time.Duration // keepalive: silence threshold

	lastRecvNano atomic.Int64 // unixnano of the last received frame (0 = none)
}

var deadlineFired = closedDeadline()

func closedDeadline() <-chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}

// timeout schedules firing of a struct channel after d, safe against close
// races.
type timeout struct {
	once sync.Once
	ch   chan struct{}
}

func newTimeout(d time.Duration) *timeout {
	t := &timeout{ch: make(chan struct{})}
	time.AfterFunc(d, t.fire)
	return t
}

func (t *timeout) fire() { t.once.Do(func() { close(t.ch) }) }

// datagram is a QUIC packet received from another peer through the relay.
type datagram struct {
	src  base.NodeID
	ecn  byte
	data []byte
}

// Option configures a relay connection.
type Option func(*options)

type options struct {
	batchSize         int
	batchCount        int
	drainDelay        time.Duration
	keepaliveInterval time.Duration
	keepaliveTimeout  time.Duration
}

// WithBatch configures outbound relay batching. QUIC packets destined for
// the same peer with the same size are coalesced into a single batch frame
// (tag 5). batchSize is the max payload bytes per frame, batchCount the max
// packets per frame, and drainDelay bounds how long a partial batch is kept
// before being flushed. A batchSize <= 0 disables batching entirely.
func WithBatch(batchSize, batchCount int, drainDelay time.Duration) Option {
	return func(o *options) {
		o.batchSize = batchSize
		o.batchCount = batchCount
		o.drainDelay = drainDelay
	}
}

// WithKeepalive overrides the keepalive tuning: the relay is pinged every
// interval and the connection is declared dead when nothing has been received
// for timeout. Defaults: 15s / 45s.
func WithKeepalive(interval, timeout time.Duration) Option {
	return func(o *options) {
		o.keepaliveInterval = interval
		o.keepaliveTimeout = timeout
	}
}

// Default batching parameters.
const (
	defaultBatchSize  = 64 * 1024
	defaultBatchCount = 16
	defaultDrainDelay = 500 * time.Microsecond
)

// batchKey groups buffered packets by destination and packet size (a batch
// frame can only contain packets of equal length).
type batchKey struct {
	dst  base.NodeID
	size int
}

type batch struct {
	contents []byte
	count    int
}

// Dial connects to the relay at ws(s)://host/relay, authenticates with the
// challenge/signature handshake and returns a ready-to-use transport.
func Dial(ctx context.Context, url string, secret *base.NodeSecret, opts ...Option) (*RelayConn, error) {
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	if o.batchSize == 0 {
		o.batchSize = defaultBatchSize
	}
	if o.batchCount == 0 {
		o.batchCount = defaultBatchCount
	}
	if o.drainDelay == 0 {
		o.drainDelay = defaultDrainDelay
	}
	if o.keepaliveInterval == 0 {
		o.keepaliveInterval = pingInterval
	}
	if o.keepaliveTimeout == 0 {
		o.keepaliveTimeout = pingTimeout
	}

	ws, _, err := websocket.Dial(ctx, url+relayPath, &websocket.DialOptions{
		Subprotocols: []string{proto.RelayProtocolV2},
	})
	if err != nil {
		return nil, err
	}
	if ws.Subprotocol() != proto.RelayProtocolV2 {
		ws.Close(websocket.StatusPolicyViolation, "unsupported relay protocol")
		return nil, errors.New("relay: unsupported subprotocol")
	}
	observedIP, err := clientHandshake(ctx, ws, secret)
	if err != nil {
		ws.Close(websocket.StatusPolicyViolation, err.Error())
		return nil, err
	}

	cctx, cancel := context.WithCancel(context.Background())
	c := &RelayConn{
		ws:         ws,
		ctx:        cctx,
		cancel:     cancel,
		localAddr:  &net.UDPAddr{IP: net.IPv6loopback},
		closed:     make(chan struct{}),
		q:          make(chan datagram, 256),
		deadlineCh: make(chan struct{}),
		lookups:    make(map[base.NodeID]chan []*net.UDPAddr),
		observed:   observedIP,
		batchSize:  o.batchSize,
		batchCount: o.batchCount,
		drainDelay: o.drainDelay,
		pending:    make(map[batchKey]*batch),

		pingInterval: o.keepaliveInterval,
		pingTimeout:  o.keepaliveTimeout,
	}
	c.lastRecvNano.Store(time.Now().UnixNano()) // base for the keepalive watchdog
	go c.readLoop()
	go c.flusher()
	go c.watchdog()
	return c, nil
}

// ObservedIP returns this client's address as seen by the relay (its public
// IP, or its LAN IP when the relay is local). It is nil until the relay has
// reported it, and is used to recognize candidates that point at our own NAT.
func (c *RelayConn) ObservedIP() net.IP {
	return c.observed
}

// clientHandshake runs the relay auth protocol: receive the challenge,
// sign a blake3-derived key of it, await confirmation. It also reads the
// ObservedAddr frame the relay sends right after the confirmation and returns
// the reported IP.
func clientHandshake(ctx context.Context, ws *websocket.Conn, secret *base.NodeSecret) (net.IP, error) {
	typ, msg, err := ws.Read(ctx)
	if err != nil {
		return nil, err
	}
	if typ != websocket.MessageBinary {
		return nil, errors.New("relay: expected binary challenge")
	}
	tag, payload, err := proto.Parse(msg)
	if err != nil {
		return nil, err
	}
	if tag != proto.ServerChallenge || len(payload) != 16 {
		return nil, errors.New("relay: expected server challenge")
	}

	var key [32]byte
	blake3.DeriveKey(proto.HandshakeDomainSep, payload, key[:])
	sig := secret.Sign(key[:])
	if err := ws.Write(ctx, websocket.MessageBinary, proto.EncodeClientAuth(secret.Public(), sig)); err != nil {
		return nil, err
	}

	typ, msg, err = ws.Read(ctx)
	if err != nil {
		return nil, err
	}
	tag, _, err = proto.Parse(msg)
	if err != nil {
		return nil, err
	}
	switch tag {
	case proto.ServerConfirmsAuth:
	case proto.ServerDeniesAuth:
		return nil, errors.New("relay: authentication denied")
	default:
		return nil, errors.New("relay: unexpected frame during handshake")
	}

	// The relay immediately follows the confirmation with our observed
	// address, so it is available before any Connect happens.
	typ, msg, err = ws.Read(ctx)
	if err != nil {
		return nil, err
	}
	if typ != websocket.MessageBinary {
		return nil, errors.New("relay: expected observed address")
	}
	tag, payload, err = proto.Parse(msg)
	if err != nil {
		return nil, err
	}
	if tag != proto.ObservedAddr {
		return nil, errors.New("relay: expected observed address")
	}
	return proto.ParseObservedAddr(payload)
}

// readLoop decodes incoming frames and dispatches datagrams to ReadFrom. It
// also answers relay keep-alive pings, as the protocol requires.
func (c *RelayConn) readLoop() {
	defer close(c.closed)
	for {
		typ, msg, err := c.ws.Read(c.ctx)
		if err != nil {
			return
		}
		c.lastRecvNano.Store(time.Now().UnixNano())
		if typ != websocket.MessageBinary {
			continue
		}
		tag, payload, err := proto.Parse(msg)
		if err != nil {
			continue
		}
		switch tag {
		case proto.RelayToClient:
			d, err := proto.ParseDatagram(payload)
			if err != nil {
				continue
			}
			select {
			case c.q <- datagram{src: d.Remote, ecn: d.Ecn, data: d.Pkt}:
			default: // queue full: drop, like a UDP socket would
			}
		case proto.RelayToClientBatch:
			b, err := proto.ParseDatagramBatch(payload)
			if err != nil {
				continue
			}
			for _, pkt := range b.Packets() {
				select {
				case c.q <- datagram{src: b.Remote, ecn: b.Ecn, data: pkt}:
				default: // queue full: drop, like a UDP socket would
				}
			}
		case proto.Ping:
			if len(payload) == 8 {
				var p [8]byte
				copy(p[:], payload)
				_ = c.writeFrame(proto.EncodePong(p))
			}
		case proto.EndpointList:
			id, addrs, err := proto.ParseEndpointList(payload)
			if err != nil {
				continue
			}
			c.lookupsMu.Lock()
			ch, ok := c.lookups[id]
			delete(c.lookups, id)
			c.lookupsMu.Unlock()
			if ok {
				ch <- addrs
			}
		case proto.EndpointGone, proto.Status:
			// no-op in this MVP (surfaced later for path management).
		}
	}
}

func (c *RelayConn) writeFrame(frame []byte) error {
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	return c.ws.Write(c.ctx, websocket.MessageBinary, frame)
}

// SetEndpoints announces this client's reachable direct addresses to the
// relay, which stores them for other nodes to discover.
func (c *RelayConn) SetEndpoints(addrs []*net.UDPAddr) error {
	if len(addrs) > proto.MaxEndpoints {
		addrs = addrs[:proto.MaxEndpoints]
	}
	return c.writeFrame(proto.EncodeSetEndpoints(addrs))
}

// Lookup asks the relay for another node's direct addresses.
func (c *RelayConn) Lookup(ctx context.Context, peer base.NodeID) ([]*net.UDPAddr, error) {
	ch := make(chan []*net.UDPAddr, 1)
	c.lookupsMu.Lock()
	if prev, ok := c.lookups[peer]; ok {
		close(prev) // only one outstanding lookup per peer
	}
	c.lookups[peer] = ch
	c.lookupsMu.Unlock()

	if err := c.writeFrame(proto.EncodeGetEndpoints(peer)); err != nil {
		c.lookupsMu.Lock()
		delete(c.lookups, peer)
		c.lookupsMu.Unlock()
		return nil, err
	}

	select {
	case addrs, ok := <-ch:
		if !ok {
			return nil, errors.New("relay: lookup cancelled")
		}
		return addrs, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-c.closed:
		return nil, io.EOF
	}
}

// WriteTo tunnels a QUIC packet destined for the peer in addr. Unless
// batching is disabled, packets are buffered briefly and sent as a batch
// frame to cut per-packet overhead.
func (c *RelayConn) WriteTo(p []byte, addr net.Addr) (int, error) {
	pa, ok := addr.(base.PeerAddr)
	if !ok {
		return 0, errors.New("relay: write to a non-peer address")
	}
	if c.batchSize <= 0 {
		frame := proto.EncodeDatagram(nil, proto.ClientToRelay, pa.ID, 0, p)
		if err := c.writeFrame(frame); err != nil {
			return 0, err
		}
		return len(p), nil
	}
	c.enqueue(pa.ID, p)
	return len(p), nil
}

// enqueue buffers a packet into its (dst, size) bucket, flushing immediately
// once a bucket reaches the configured size or count limits.
func (c *RelayConn) enqueue(dst base.NodeID, p []byte) {
	key := batchKey{dst: dst, size: len(p)}
	c.batchMu.Lock()
	b := c.pending[key]
	if b == nil {
		b = &batch{}
		c.pending[key] = b
	}
	b.contents = append(b.contents, p...)
	b.count++
	full := b.count >= c.batchCount || len(b.contents) >= c.batchSize
	c.batchMu.Unlock()
	if full {
		c.flushPending()
	}
}

// flusher periodically drains buffered packets into batch frames.
func (c *RelayConn) flusher() {
	tick := time.NewTicker(c.drainDelay)
	defer tick.Stop()
	for {
		select {
		case <-c.ctx.Done():
			return // Close() flushes synchronously before cancelling
		case <-tick.C:
			c.flushPending()
		}
	}
}

// watchdog pings the relay periodically and force-closes the connection if no
// frame at all has been received for pingTimeout, so a silently-dead relay is
// detected and surfaced via Closed().
func (c *RelayConn) watchdog() {
	tick := time.NewTicker(c.pingInterval)
	defer tick.Stop()
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-tick.C:
			var p [8]byte
			binary.LittleEndian.PutUint64(p[:], uint64(time.Now().UnixNano()))
			_ = c.writeFrame(proto.EncodePing(p))
			if time.Since(time.Unix(0, c.lastRecvNano.Load())) > c.pingTimeout {
				c.ws.CloseNow()
				return
			}
		}
	}
}

// flushPending sends all buffered packets as batch frames.
func (c *RelayConn) flushPending() {
	c.batchMu.Lock()
	if len(c.pending) == 0 {
		c.batchMu.Unlock()
		return
	}
	pending := c.pending
	c.pending = make(map[batchKey]*batch)
	c.batchMu.Unlock()

	for k, b := range pending {
		frame := proto.EncodeDatagramBatch(nil, proto.ClientToRelayBatch, k.dst, 0, uint16(k.size), b.contents)
		_ = c.writeFrame(frame)
	}
}

// ReadFrom returns the next QUIC packet received from a peer. The returned
// addr is a base.PeerAddr carrying the sender's NodeID.
func (c *RelayConn) ReadFrom(p []byte) (int, net.Addr, error) {
	for {
		c.readDeadlineMu.Lock()
		deadlineCh := c.deadlineCh
		var fired <-chan struct{}
		if !c.readDeadline.IsZero() {
			switch wait := time.Until(c.readDeadline); {
			case wait <= 0:
				fired = deadlineFired
			default:
				fired = newTimeout(wait).ch
			}
		}
		c.readDeadlineMu.Unlock()

		select {
		case <-c.closed:
			return 0, nil, io.EOF
		case d := <-c.q:
			n := copy(p, d.data)
			return n, base.PeerAddr{ID: d.src}, nil
		case <-deadlineCh: // deadline changed: re-evaluate
		case <-fired:
			return 0, nil, os.ErrDeadlineExceeded
		}
	}
}

// SetDeadline sets the read deadline (write deadlines are a no-op).
func (c *RelayConn) SetDeadline(t time.Time) error { return c.SetReadDeadline(t) }

// SetReadDeadline sets the read deadline. A change immediately wakes any
// blocked ReadFrom so quic-go can shut the listener down.
func (c *RelayConn) SetReadDeadline(t time.Time) error {
	c.readDeadlineMu.Lock()
	c.readDeadline = t
	close(c.deadlineCh)
	c.deadlineCh = make(chan struct{})
	c.readDeadlineMu.Unlock()
	return nil
}

// SetWriteDeadline is a no-op.
func (c *RelayConn) SetWriteDeadline(time.Time) error { return nil }

// SetReadBuffer and SetWriteBuffer satisfy quic-go's optional buffer-resizing
// hook; the relay's backpressure is bounded by its own queue instead.
func (c *RelayConn) SetReadBuffer(int) error  { return nil }
func (c *RelayConn) SetWriteBuffer(int) error { return nil }

// Close flushes any buffered packets, then tears down the relay connection.
func (c *RelayConn) Close() error {
	c.flushPending()
	c.cancel()
	return c.ws.CloseNow()
}

// Closed returns a channel that is closed when the relay connection dies
// (read loop exits), e.g. on a network error or an explicit Close.
func (c *RelayConn) Closed() <-chan struct{} {
	return c.closed
}

// LocalAddr returns the local address of the relay connection.
func (c *RelayConn) LocalAddr() net.Addr { return c.localAddr }

var _ net.PacketConn = (*RelayConn)(nil)
