// Package iron implements "dial by public key" networking on top of QUIC.
//
// Every node keeps a persistent connection to a relay and announces the
// direct addresses it is reachable at. To reach another node you dial its
// NodeID: the node tries the peer's direct addresses first and falls back to
// connecting through the relay. Connections are authenticated by the node's
// Ed25519 identity carried in self-signed X.509 certificates.
package iron

import (
	"context"
	"crypto/tls"
	"errors"
	"log/slog"
	"net"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/quic-go/quic-go"

	"github.com/skerkour/stdx-go/iron/base"
	"github.com/skerkour/stdx-go/iron/relay"
)

// DefaultALPN is the application protocol advertised by endpoints by default.
const DefaultALPN = "iron-example/echo/0"

// directAttemptTimeout caps how long one direct-dial attempt may take before
// trying the next candidate or falling back to the relay.
const directAttemptTimeout = 2 * time.Second

// Endpoint is a node. It maintains a relay connection, listens for direct UDP
// connections, and can accept inbound QUIC connections and dial outbound
// ones, both identified by NodeID.
type Endpoint struct {
	secret *base.NodeSecret
	log    *slog.Logger
	alpn   string

	relayURL   string
	relayMu    sync.RWMutex // guards relayConn/relayTr (replaced on reconnect)
	relayConn  *relay.RelayConn
	relayTr    *quic.Transport
	directTr   *quic.Transport
	directAddr *net.UDPAddr // our UDP socket, published as a candidate

	relayWaitTimeout time.Duration // how long Connect waits for the relay to come back

	relayBatchSize  int // batching config, preserved across reconnects
	relayBatchCount int
	relayDrainDelay time.Duration

	announceMu sync.Mutex
	announced  []*net.UDPAddr // override set via SetAnnouncedAddrs, else nil

	peerAddrs sync.Map // NodeID -> *candidateEntry (cached direct addresses)

	conns    chan *quic.Conn
	closeCh  chan struct{}
	closeOne sync.Once
}

// NewEndpoint binds the node to a relay and to a UDP socket for direct
// connections. relayURL is a ws:// (or wss://) address, e.g.
// "ws://127.0.0.1:3333". ALPN "" selects the default.
func NewEndpoint(ctx context.Context, relayURL string, secret *base.NodeSecret, alpn string, opts ...EndpointOption) (*Endpoint, error) {
	var o endpointOptions
	for _, opt := range opts {
		opt(&o)
	}
	if alpn == "" {
		alpn = DefaultALPN
	}
	rc, err := relay.Dial(ctx, relayURL, secret, relay.WithBatch(o.batchSize, o.batchCount, o.drainDelay))
	if err != nil {
		return nil, err
	}

	// Bind a dual-stack UDP socket (IPv6 with V6ONLY=0), so both IPv4 and IPv6
	// direct connections work from a single socket.
	udpConn, err := net.ListenUDP("udp", &net.UDPAddr{IP: net.IPv6unspecified})
	if err != nil {
		rc.Close()
		return nil, err
	}

	e := &Endpoint{
		secret:           secret,
		alpn:             alpn,
		relayURL:         relayURL,
		relayConn:        rc,
		relayTr:          &quic.Transport{Conn: rc},
		directTr:         &quic.Transport{Conn: udpConn},
		directAddr:       udpConn.LocalAddr().(*net.UDPAddr),
		conns:            make(chan *quic.Conn, 16),
		closeCh:          make(chan struct{}),
		relayWaitTimeout: o.relayWaitTimeout,
		relayBatchSize:   o.batchSize,
		relayBatchCount:  o.batchCount,
		relayDrainDelay:  o.drainDelay,
	}
	if e.relayWaitTimeout == 0 {
		e.relayWaitTimeout = defaultRelayWaitTimeout
	}

	serverCfg, err := serverTLSConfig(secret, alpn)
	if err != nil {
		e.Close()
		return nil, err
	}
	relayL, err := e.relayTr.Listen(serverCfg, defaultQUICConfig())
	if err != nil {
		e.Close()
		return nil, err
	}
	directL, err := e.directTr.Listen(serverCfg, defaultQUICConfig())
	if err != nil {
		e.Close()
		return nil, err
	}
	go e.acceptLoop(relayL, false)
	go e.acceptLoop(directL, true)

	// Watch the relay connection and reconnect with backoff if it drops.
	go e.watchRelay()

	// Announce our direct addresses so peers can discover them, unless the
	// caller opted to publish a specific set via SetAnnouncedAddrs.
	if !o.skipAnnounce {
		if err := rc.SetEndpoints(e.localCandidates()); err != nil {
			e.Close()
			return nil, err
		}
	}
	return e, nil
}

// EndpointOption configures Endpoint construction.
type EndpointOption func(*endpointOptions)

type endpointOptions struct {
	skipAnnounce     bool
	batchSize        int
	batchCount       int
	drainDelay       time.Duration
	relayWaitTimeout time.Duration
}

// defaultRelayWaitTimeout is how long Connect waits for the relay to come
// back before failing, when the relay is down.
const defaultRelayWaitTimeout = 5 * time.Second

// WithSkipAnnounce disables announcing this endpoint's direct addresses at
// startup. Use it when you want to publish a precise address set with
// SetAnnouncedAddrs instead of the auto-detected interface addresses.
func WithSkipAnnounce() EndpointOption {
	return func(o *endpointOptions) { o.skipAnnounce = true }
}

// WithRelayBatching configures outbound relay batching: max bytes per batch
// frame, max packets per batch frame, and how long a partial batch is held
// before flushing. A batchSize <= 0 disables batching. Defaults: 64 KiB,
// 16 packets, 500 microseconds.
func WithRelayBatching(batchSize, batchCount int, drainDelay time.Duration) EndpointOption {
	return func(o *endpointOptions) {
		o.batchSize = batchSize
		o.batchCount = batchCount
		o.drainDelay = drainDelay
	}
}

// WithRelayWaitTimeout sets how long Connect waits for the relay to come back
// after an outage before failing (default 5 seconds).
func WithRelayWaitTimeout(d time.Duration) EndpointOption {
	return func(o *endpointOptions) { o.relayWaitTimeout = d }
}

// WithLogger attaches a logger for diagnostics.
func (e *Endpoint) WithLogger(l *slog.Logger) *Endpoint {
	e.log = l
	return e
}

// NodeID returns this endpoint's public identity.
func (e *Endpoint) NodeID() base.NodeID { return e.secret.Public() }

// SetAnnouncedAddrs overrides the direct addresses this endpoint announces to
// the relay. Mostly useful for tests and unusual deployments.
func (e *Endpoint) SetAnnouncedAddrs(addrs []*net.UDPAddr) error {
	e.announceMu.Lock()
	e.announced = append([]*net.UDPAddr(nil), addrs...)
	e.announceMu.Unlock()
	return e.currentRelayConn().SetEndpoints(addrs)
}

// Connect dials another node by its NodeID: it tries the peer's announced
// direct addresses first and falls back to the relay. The returned
// Connection redials over the relay transparently if a direct connection
// drops.
func (e *Endpoint) Connect(ctx context.Context, peer base.NodeID) (*Connection, error) {
	if addrs, err := e.lookupCandidates(ctx, peer); err == nil {
		for _, addr := range e.filterCandidates(addrs) {
			dctx, cancel := context.WithTimeout(ctx, directAttemptTimeout)
			conn, derr := e.directTr.Dial(dctx, addr, e.clientTLSConfig(peer), defaultQUICConfig())
			cancel()
			if derr == nil {
				if e.log != nil {
					e.log.Info("connected direct",
						"peer", peer.String(), "addr", addr.String())
				}
				return newConnection(e, peer, conn, PathDirect, true), nil
			}
			if e.log != nil {
				e.log.Info("direct dial failed", "peer", peer.String(), "addr", addr.String(), "err", derr)
			}
		}
		// Every candidate failed, so the peer's addresses are likely stale:
		// evict them so the next Connect re-discovers.
		e.forgetCandidates(peer)
	}
	if err := e.waitForRelay(ctx); err != nil {
		return nil, err
	}
	conn, err := e.currentRelayTr().Dial(ctx, base.PeerAddr{ID: peer}, e.clientTLSConfig(peer), defaultQUICConfig())
	if err != nil {
		return nil, err
	}
	if e.log != nil {
		e.log.Info("connected via relay", "peer", peer.String())
	}
	return newConnection(e, peer, conn, PathRelay, true), nil
}

// Accept returns the next inbound QUIC connection (direct or through the
// relay).
func (e *Endpoint) Accept(ctx context.Context) (*Connection, error) {
	select {
	case conn := <-e.conns:
		path := PathRelay
		if _, ok := conn.RemoteAddr().(*net.UDPAddr); ok {
			path = PathDirect
		}
		peer, err := peerIDFromTLS(conn.ConnectionState().TLS)
		if err != nil {
			peer = base.NodeID{}
		}
		return newConnection(e, peer, conn, path, false), nil
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-e.closeCh:
		return nil, errors.New("endpoint closed")
	}
}

// PeerID returns the NodeID of the peer on an established connection.
func (e *Endpoint) PeerID(conn *Connection) (base.NodeID, error) {
	return conn.PeerID()
}

// Close shuts down the endpoint, its UDP socket and its relay connection.
func (e *Endpoint) Close() error {
	e.closeOne.Do(func() { close(e.closeCh) })
	err1 := e.currentRelayTr().Close()
	err2 := e.directTr.Close()
	_ = e.currentRelayConn().Close()
	if err1 != nil {
		return err1
	}
	return err2
}

// localCandidates returns this endpoint's direct addresses (or the override
// set via SetAnnouncedAddrs): every interface IP at our UDP port. Loopback
// and IPv6 link-local addresses are skipped: loopback is present on every
// host (and we never dial it), while fe80::/10 addresses are only dialable
// with an interface zone a remote peer cannot know, so announcing them is
// pure noise.
func (e *Endpoint) localCandidates() []*net.UDPAddr {
	e.announceMu.Lock()
	override := e.announced
	e.announceMu.Unlock()
	if override != nil {
		return override
	}

	port := e.directAddr.Port
	var addrs []*net.UDPAddr
	ifaces, err := net.Interfaces()
	if err != nil {
		return addrs
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 {
			continue
		}
		ifAddrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, a := range ifAddrs {
			var ip net.IP
			switch v := a.(type) {
			case *net.IPNet:
				ip = v.IP
			case *net.IPAddr:
				ip = v.IP
			}
			if ip == nil || ip.IsLoopback() || isV6LinkLocal(ip) {
				continue
			}
			addrs = append(addrs, &net.UDPAddr{IP: ip, Port: port})
		}
	}
	return sortCandidates(addrs)
}

// candidateTTL is how long a peer's direct addresses are cached before they
// are re-fetched from the relay.
var candidateTTL = 60 * time.Second

// candidateEntry is a cached set of a peer's direct addresses.
type candidateEntry struct {
	addrs []*net.UDPAddr
	at    time.Time
}

// lookupCandidates returns the peer's direct addresses, cached locally for
// candidateTTL. The relay may not have processed the peer's announcement yet,
// so an empty result is retried a couple of times and is never cached.
func (e *Endpoint) lookupCandidates(ctx context.Context, peer base.NodeID) ([]*net.UDPAddr, error) {
	if v, ok := e.peerAddrs.Load(peer); ok {
		ent := v.(*candidateEntry)
		if len(ent.addrs) > 0 && time.Since(ent.at) < candidateTTL {
			return ent.addrs, nil
		}
	}
	for attempt := 0; attempt < 3; attempt++ {
		addrs, err := e.currentRelayConn().Lookup(ctx, peer)
		if err != nil {
			return nil, err
		}
		if len(addrs) > 0 {
			e.peerAddrs.Store(peer, &candidateEntry{addrs: addrs, at: time.Now()})
			return addrs, nil
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
	return nil, nil
}

// forgetCandidates drops the cached direct addresses for a peer, forcing the
// next Connect to re-discover them.
func (e *Endpoint) forgetCandidates(peer base.NodeID) {
	e.peerAddrs.Delete(peer)
}

// dialRelay connects to the peer solely through the relay (used for the
// transparent fallback). It does not wait for the relay to come back.
func (e *Endpoint) dialRelay(ctx context.Context, peer base.NodeID) (*quic.Conn, error) {
	return e.currentRelayTr().Dial(ctx, base.PeerAddr{ID: peer}, e.clientTLSConfig(peer), defaultQUICConfig())
}

func (e *Endpoint) clientTLSConfig(peer base.NodeID) *tls.Config {
	cfg, err := clientTLSConfig(e.secret, peer, e.alpn)
	if err != nil {
		panic(err) // identity is static; cannot fail
	}
	return cfg
}

// acceptLoop feeds inbound connections into e.conns. If fatal is true (the
// direct listener), a listener error shuts the whole endpoint down; the relay
// listener is not fatal because watchRelay owns reconnection.
func (e *Endpoint) acceptLoop(l *quic.Listener, fatal bool) {
	for {
		conn, err := l.Accept(context.Background())
		if err != nil {
			if fatal {
				e.closeOne.Do(func() { close(e.closeCh) })
			}
			return
		}
		select {
		case e.conns <- conn:
		case <-e.closeCh:
			conn.CloseWithError(0, "")
			return
		}
	}
}

// sortCandidates orders addresses by likely reachability (same-host, then
// LAN, then public) and removes duplicates.
func sortCandidates(addrs []*net.UDPAddr) []*net.UDPAddr {
	seen := make(map[string]bool, len(addrs))
	out := make([]*net.UDPAddr, 0, len(addrs))
	for _, a := range addrs {
		key := ipKey(a.IP)
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, a)
	}
	sort.SliceStable(out, func(i, j int) bool {
		return rankIP(out[i].IP) < rankIP(out[j].IP)
	})
	return out
}

func rankIP(ip net.IP) int {
	switch {
	case ip.IsLoopback():
		return 0
	case ip.IsLinkLocalUnicast():
		return 2
	case ip.IsPrivate():
		return 1
	default:
		return 3
	}
}

func ipKey(ip net.IP) string {
	if ip4 := ip.To4(); ip4 != nil {
		return "4:" + ip4.String()
	}
	if ip6 := ip.To16(); ip6 != nil {
		return "6:" + ip6.String()
	}
	return ""
}

// localNet is a snapshot of this endpoint's network interfaces: the exact
// addresses (to detect same-host peers) and the prefixes (to detect LAN
// peers).
type localNet struct {
	addrs []net.IP
	nets  []net.IPNet
}

// localNetworks re-enumerates the up interfaces. It is called on every
// Connect so that interface changes (WiFi reconnects, etc.) are picked up.
func (e *Endpoint) localNetworks() localNet {
	var ln localNet
	ifaces, err := net.Interfaces()
	if err != nil {
		return ln
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 {
			continue
		}
		ifAddrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, a := range ifAddrs {
			var ip net.IP
			switch v := a.(type) {
			case *net.IPNet:
				ip = v.IP
				ln.nets = append(ln.nets, *v)
			case *net.IPAddr:
				ip = v.IP
			}
			if ip != nil {
				ln.addrs = append(ln.addrs, ip)
			}
		}
	}
	return ln
}

// filterCandidates drops direct-dial candidates that cannot be reachable from
// this host, then dedups and orders the remainder.
func (e *Endpoint) filterCandidates(addrs []*net.UDPAddr) []*net.UDPAddr {
	return filterCandidatesFor(addrs, e.localNetworks(), e.currentRelayConn().ObservedIP())
}

// filterCandidatesFor is the pure form of filterCandidates, parameterized by
// the interface snapshot and our observed address so it can be tested in
// isolation.
//
// When the peer is detected to be the same host (a candidate is strictly
// equal to one of our interface addresses), 127.0.0.1 is tried FIRST: both
// endpoints listen on 0.0.0.0, so loopback is the guaranteed-reachable,
// loopback-fast path (~microsecond RTT), while a same-host interface address
// might not actually be local to us (different network namespace, or a
// veth/docker peer) or could be dropped by strict reverse-path filtering —
// and each such failed attempt costs up to directAttemptTimeout. The real
// interface candidates are kept as follow-ups. Loopback is only ever added
// for genuinely same-host peers, so it does not reintroduce the cross-host
// loopback noise.
func filterCandidatesFor(addrs []*net.UDPAddr, ln localNet, observed net.IP) []*net.UDPAddr {
	sameHost := false
	kept := make([]*net.UDPAddr, 0, len(addrs))
	for _, a := range addrs {
		if reachableIP(a.IP, ln, observed) {
			kept = append(kept, a)
			if !sameHost && equalAnyIP(a.IP, ln.addrs) {
				sameHost = true
			}
		}
	}
	out := sortCandidates(kept)

	if sameHost {
		seen := make(map[string]bool, 1)
		var loopbacks []*net.UDPAddr
		for _, a := range kept {
			if !equalAnyIP(a.IP, ln.addrs) {
				continue
			}
			lb := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: a.Port}
			if key := "127.0.0.1:" + strconv.Itoa(a.Port); !seen[key] {
				seen[key] = true
				loopbacks = append(loopbacks, lb)
			}
		}
		out = append(loopbacks, out...)
	}
	return out
}

// reachableIP decides whether a candidate address is worth a direct-dial
// attempt.
//
// A candidate is tried only if it is plausibly reachable from this host:
//   - public and RFC 6598 CGNAT addresses are kept — they may be directly
//     reachable over the internet (CGNAT is not private per RFC 1918);
//   - RFC 1918 / ULA / link-local addresses are kept only when they fall on
//     one of our own subnets (the peer is on our LAN);
//   - a candidate strictly equal to one of our interface addresses is kept
//     (the peer is the same host);
//   - loopback candidates are always dropped: every endpoint announces
//     127.0.0.1, so keeping them would make us probe every peer's loopback.
//     Same-host peers are instead covered by the strict-equality rule on
//     their real interface addresses;
//   - a candidate equal to our own relay-observed address is dropped: that
//     address is our NAT's public IP. A peer behind the same home NAT shares
//     it, but dialing it would reach the router's WAN port, not the peer —
//     such a peer is reachable via its LAN candidates (same subnet) instead.
//     Two users behind carrier NAT (RFC 6598) may share a pool address; for
//     the same reason that shared address cannot route to a specific user, so
//     it is dropped there too, while users with *distinct* observed
//     addresses are unaffected and still get a direct attempt.
//
// Loopback never passes this predicate; when the peer is the same host,
// filterCandidatesFor appends 127.0.0.1 as a last-resort fallback instead.
func reachableIP(ip net.IP, ln localNet, observed net.IP) bool {
	switch {
	case ip == nil || ip.IsUnspecified() || ip.IsMulticast() || ip.IsLoopback():
		return false
	case isV6LinkLocal(ip):
		return false // not dialable without an interface zone
	case observed != nil && ip.Equal(observed):
		return false // our own NAT's public address (same-NAT / shared-CGNAT peer)
	case equalAnyIP(ip, ln.addrs):
		return true // same host
	case ip.IsLinkLocalUnicast() || isPrivateNet(ip):
		return containsIP(ln.nets, ip) // LAN peer
	default:
		return true // public / RFC 6598 CGNAT
	}
}

// isV6LinkLocal reports whether ip is an IPv6 link-local address (fe80::/10).
// Such addresses are only dialable with an interface zone, which a remote
// peer cannot supply, so they are never dialed.
func isV6LinkLocal(ip net.IP) bool {
	return ip.To4() == nil && len(ip) == net.IPv6len && ip[0] == 0xfe && ip[1]&0xc0 == 0x80
}

// isPrivateNet reports whether ip is in the RFC 1918 private ranges or the
// RFC 4193 ULA range. RFC 6598 CGNAT (100.64/10) is deliberately NOT
// included: CGNAT observed addresses can be directly reachable, so they are
// treated as public.
func isPrivateNet(ip net.IP) bool {
	if ip4 := ip.To4(); ip4 != nil {
		return ip4[0] == 10 ||
			(ip4[0] == 172 && ip4[1]&0xf0 == 16) ||
			(ip4[0] == 192 && ip4[1] == 168)
	}
	return len(ip) == net.IPv6len && ip[0]&0xfe == 0xfc
}

func equalAnyIP(ip net.IP, ours []net.IP) bool {
	for _, o := range ours {
		if o.Equal(ip) {
			return true
		}
	}
	return false
}

func containsIP(nets []net.IPNet, ip net.IP) bool {
	for _, n := range nets {
		if n.Contains(ip) {
			return true
		}
	}
	return false
}

// currentRelayConn returns the active relay connection (the one being used for
// dialing, lookups and announcements).
func (e *Endpoint) currentRelayConn() *relay.RelayConn {
	e.relayMu.RLock()
	defer e.relayMu.RUnlock()
	return e.relayConn
}

// currentRelayTr returns the active relay QUIC transport.
func (e *Endpoint) currentRelayTr() *quic.Transport {
	e.relayMu.RLock()
	defer e.relayMu.RUnlock()
	return e.relayTr
}

func (e *Endpoint) isClosing() bool {
	select {
	case <-e.closeCh:
		return true
	default:
		return false
	}
}

// Relay reconnection backoff bounds.
const (
	relayBackoffMin = 100 * time.Millisecond
	relayBackoffMax = 10 * time.Second
)

// installRelay makes nrc the endpoint's active relay connection: it builds a
// fresh QUIC transport, listens on it for inbound relayed connections,
// re-announces our direct addresses, and swaps it in atomically.
func (e *Endpoint) installRelay(nrc *relay.RelayConn) error {
	tr := &quic.Transport{Conn: nrc}
	serverCfg, err := serverTLSConfig(e.secret, e.alpn)
	if err != nil {
		return err
	}
	l, err := tr.Listen(serverCfg, defaultQUICConfig())
	if err != nil {
		return err
	}
	go e.acceptLoop(l, false)
	_ = nrc.SetEndpoints(e.localCandidates())

	e.relayMu.Lock()
	e.relayConn = nrc
	e.relayTr = tr
	e.relayMu.Unlock()
	return nil
}

// watchRelay reconnects to the relay with exponential backoff whenever the
// relay connection drops, keeping the endpoint usable across relay outages.
func (e *Endpoint) watchRelay() {
	backoff := relayBackoffMin
	for {
		rc := e.currentRelayConn()
		<-rc.Closed()
		if e.isClosing() {
			return
		}
		if e.log != nil {
			e.log.Warn("relay connection lost; reconnecting", "url", e.relayURL)
		}
		for {
			if e.isClosing() {
				return
			}
			nrc, err := relay.Dial(context.Background(), e.relayURL, e.secret,
				relay.WithBatch(e.relayBatchSize, e.relayBatchCount, e.relayDrainDelay))
			if err == nil {
				if ierr := e.installRelay(nrc); ierr == nil {
					if e.log != nil {
						e.log.Info("relay reconnected", "url", e.relayURL)
					}
					backoff = relayBackoffMin
					break
				}
				nrc.Close()
			}
			select {
			case <-e.closeCh:
				return
			case <-time.After(backoff):
			}
			backoff *= 2
			if backoff > relayBackoffMax {
				backoff = relayBackoffMax
			}
		}
	}
}

// waitForRelay blocks until the relay is usable or ctx/relayWaitTimeout
// expires. It is called before dialing over the relay so that a Connect
// during a relay outage briefly waits for reconnection instead of failing
// immediately.
func (e *Endpoint) waitForRelay(ctx context.Context) error {
	deadline := time.Now().Add(e.relayWaitTimeout)
	for {
		rc := e.currentRelayConn()
		select {
		case <-rc.Closed():
			if time.Now().After(deadline) {
				return errors.New("relay unavailable")
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-e.closeCh:
				return errors.New("endpoint closed")
			case <-time.After(50 * time.Millisecond):
			}
		default:
			return nil
		}
	}
}

func defaultQUICConfig() *quic.Config {
	return &quic.Config{
		EnableDatagrams:       true,
		MaxIncomingStreams:    1024,
		MaxIncomingUniStreams: 1024,
		KeepAlivePeriod:       5 * time.Second,
		MaxIdleTimeout:        30 * time.Second,
		HandshakeIdleTimeout:  30 * time.Second,
	}
}
