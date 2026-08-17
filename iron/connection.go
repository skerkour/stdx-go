package iron

import (
	"context"
	"errors"
	"sync"

	"github.com/quic-go/quic-go"

	"github.com/skerkour/stdx-go/iron/base"
)

// Path over which a Connection was established.
const (
	// PathDirect means the connection uses the peer's direct address.
	PathDirect = "direct"
	// PathRelay means the connection is tunnelled through the relay.
	PathRelay = "relay"
)

// Connection wraps a quic.Conn and remembers which path it came up on.
//
// For connections dialed by this endpoint (see Endpoint.Connect), it also
// provides transparent fallback: if the direct path dies, the endpoint
// automatically re-dials the peer over the relay and subsequent stream and
// datagram operations run on the replacement connection. Connections opened
// *by the peer* (from Endpoint.Accept) are not auto-redialed; that side sees
// the connection close and the peer re-dialing.
type Connection struct {
	ep   *Endpoint
	peer base.NodeID

	redialable bool

	mu   sync.RWMutex
	cur  *quic.Conn
	path string

	closed chan struct{}
	once   sync.Once

	redialOnce sync.Once
}

func newConnection(ep *Endpoint, peer base.NodeID, conn *quic.Conn, path string, redialable bool) *Connection {
	c := &Connection{
		ep:         ep,
		peer:       peer,
		cur:        conn,
		path:       path,
		redialable: redialable,
		closed:     make(chan struct{}),
	}
	go c.watch()
	return c
}

// Path reports whether the connection is direct or relayed.
func (c *Connection) Path() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.path
}

// PeerID returns the node id of the remote endpoint, taken from the
// authenticated TLS certificate.
func (c *Connection) PeerID() (base.NodeID, error) {
	return peerIDFromTLS(c.State().TLS)
}

// State returns the QUIC connection state of the current underlying
// connection.
func (c *Connection) State() quic.ConnectionState {
	return c.current().ConnectionState()
}

// Context returns the context of the current underlying connection. It is
// canceled when that connection closes.
func (c *Connection) Context() context.Context {
	return c.current().Context()
}

// Close closes the underlying connection.
func (c *Connection) Close() error {
	c.once.Do(func() { close(c.closed) })
	return c.current().CloseWithError(0, "connection closed")
}

func (c *Connection) current() *quic.Conn {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.cur
}

func (c *Connection) swap(conn *quic.Conn, path string) {
	c.mu.Lock()
	c.cur = conn
	c.path = path
	c.mu.Unlock()
}

// isDead reports whether the given underlying connection has closed.
func (c *Connection) isDead(conn *quic.Conn) bool {
	return conn.Context().Err() != nil
}

func (c *Connection) errIfClosed() error {
	select {
	case <-c.closed:
		return errors.New("connection closed")
	default:
		return nil
	}
}

// watch redials over the relay when a direct connection dies.
func (c *Connection) watch() {
	if !c.redialable {
		return
	}
	for {
		cur := c.current()
		<-cur.Context().Done()
		if c.isClosed() {
			return
		}
		c.mu.RLock()
		path := c.path
		c.mu.RUnlock()
		if path != PathDirect {
			return
		}
		c.redial()
		if nxt := c.current(); nxt == cur {
			return // redial failed; no further watching
		}
		return // on the relay now; if that dies, there is no fallback left
	}
}

func (c *Connection) isClosed() bool {
	select {
	case <-c.closed:
		return true
	default:
		return false
	}
}

// redial replaces a dead direct connection with a fresh one over the relay.
// It is guarded so it happens at most once.
func (c *Connection) redial() {
	c.redialOnce.Do(func() {
		conn, err := c.ep.dialRelay(context.Background(), c.peer)
		if err != nil {
			return
		}
		c.swap(conn, PathRelay)
	})
}

// ensureRedial attempts the direct->relay fallback, e.g. when a stream
// operation just observed a dead connection.
func (c *Connection) ensureRedial() { c.redial() }

// Forwards for stream operations. Each tries once on the underlying
// connection and, if that connection just died and was replaced, once more on
// the replacement.

// OpenStream opens a new bidirectional stream.
func (c *Connection) OpenStream() (*quic.Stream, error) {
	if err := c.errIfClosed(); err != nil {
		return nil, err
	}
	return c.openStream(func(conn *quic.Conn) (*quic.Stream, error) {
		return conn.OpenStream()
	})
}

// OpenStreamSync opens a new bidirectional stream, blocking until a stream
// can be opened.
func (c *Connection) OpenStreamSync(ctx context.Context) (*quic.Stream, error) {
	if err := c.errIfClosed(); err != nil {
		return nil, err
	}
	return c.openStream(func(conn *quic.Conn) (*quic.Stream, error) {
		return conn.OpenStreamSync(ctx)
	})
}

func (c *Connection) openStream(f func(*quic.Conn) (*quic.Stream, error)) (*quic.Stream, error) {
	for i := 0; i < 2; i++ {
		cur := c.current()
		st, err := f(cur)
		if err == nil {
			return st, nil
		}
		if i == 1 || !c.isDead(cur) {
			return nil, err
		}
		c.ensureRedial()
	}
	return nil, errors.New("open stream failed")
}

// OpenUniStream opens a new unidirectional stream.
func (c *Connection) OpenUniStream() (*quic.SendStream, error) {
	if err := c.errIfClosed(); err != nil {
		return nil, err
	}
	return c.openUniStream(func(conn *quic.Conn) (*quic.SendStream, error) {
		return conn.OpenUniStream()
	})
}

// OpenUniStreamSync opens a new unidirectional stream, blocking until a
// stream can be opened.
func (c *Connection) OpenUniStreamSync(ctx context.Context) (*quic.SendStream, error) {
	if err := c.errIfClosed(); err != nil {
		return nil, err
	}
	return c.openUniStream(func(conn *quic.Conn) (*quic.SendStream, error) {
		return conn.OpenUniStreamSync(ctx)
	})
}

func (c *Connection) openUniStream(f func(*quic.Conn) (*quic.SendStream, error)) (*quic.SendStream, error) {
	for i := 0; i < 2; i++ {
		cur := c.current()
		st, err := f(cur)
		if err == nil {
			return st, nil
		}
		if i == 1 || !c.isDead(cur) {
			return nil, err
		}
		c.ensureRedial()
	}
	return nil, errors.New("open stream failed")
}

// AcceptStream accepts the next bidirectional stream opened by the peer.
func (c *Connection) AcceptStream(ctx context.Context) (*quic.Stream, error) {
	if err := c.errIfClosed(); err != nil {
		return nil, err
	}
	return c.acceptStream(func(conn *quic.Conn) (*quic.Stream, error) {
		return conn.AcceptStream(ctx)
	})
}

func (c *Connection) acceptStream(f func(*quic.Conn) (*quic.Stream, error)) (*quic.Stream, error) {
	for i := 0; i < 2; i++ {
		cur := c.current()
		st, err := f(cur)
		if err == nil {
			return st, nil
		}
		if i == 1 || !c.isDead(cur) {
			return nil, err
		}
		c.ensureRedial()
	}
	return nil, errors.New("accept stream failed")
}

// AcceptUniStream accepts the next unidirectional stream opened by the peer.
func (c *Connection) AcceptUniStream(ctx context.Context) (*quic.ReceiveStream, error) {
	if err := c.errIfClosed(); err != nil {
		return nil, err
	}
	for i := 0; i < 2; i++ {
		cur := c.current()
		st, err := cur.AcceptUniStream(ctx)
		if err == nil {
			return st, nil
		}
		if i == 1 || !c.isDead(cur) {
			return nil, err
		}
		c.ensureRedial()
	}
	return nil, errors.New("accept stream failed")
}

// SendDatagram sends an unreliable datagram.
func (c *Connection) SendDatagram(p []byte) error {
	if err := c.errIfClosed(); err != nil {
		return err
	}
	for i := 0; i < 2; i++ {
		cur := c.current()
		err := cur.SendDatagram(p)
		if err == nil {
			return nil
		}
		if i == 1 || !c.isDead(cur) {
			return err
		}
		c.ensureRedial()
	}
	return errors.New("send datagram failed")
}

// ReceiveDatagram receives the next datagram sent by the peer.
func (c *Connection) ReceiveDatagram(ctx context.Context) ([]byte, error) {
	if err := c.errIfClosed(); err != nil {
		return nil, err
	}
	for i := 0; i < 2; i++ {
		cur := c.current()
		data, err := cur.ReceiveDatagram(ctx)
		if err == nil {
			return data, nil
		}
		if i == 1 || !c.isDead(cur) {
			return nil, err
		}
		c.ensureRedial()
	}
	return nil, errors.New("receive datagram failed")
}

// CloseWithError closes the connection with an application error.
func (c *Connection) CloseWithError(code uint64, desc string) error {
	c.once.Do(func() { close(c.closed) })
	return c.current().CloseWithError(quic.ApplicationErrorCode(code), desc)
}
