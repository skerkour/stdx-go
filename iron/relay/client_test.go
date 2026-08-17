package relay

import (
	"context"
	"net"
	"net/http"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/skerkour/stdx-go/iron/base"
	"github.com/skerkour/stdx-go/iron/proto"
)

// miniRelay runs a websocket server that authenticates blindly and echoes
// whatever follows, enough for RelayConn to connect.
func miniRelay(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/relay", func(w http.ResponseWriter, r *http.Request) {
		ws, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			Subprotocols:       []string{proto.RelayProtocolV2},
			InsecureSkipVerify: true,
		})
		if err != nil {
			return
		}
		ctx := r.Context()
		// ServerChallenge
		if err := ws.Write(ctx, websocket.MessageBinary, proto.EncodeServerChallenge([16]byte{1})); err != nil {
			return
		}
		if _, _, err := ws.Read(ctx); err != nil {
			return
		}
		ws.Write(ctx, websocket.MessageBinary, proto.EncodeServerConfirmsAuth())
		// ObservedAddr, as the real relay sends right after confirming auth.
		ws.Write(ctx, websocket.MessageBinary, proto.EncodeObservedAddr(net.IPv4(127, 0, 0, 1)))
		// echo binary messages back as-is
		for {
			typ, msg, err := ws.Read(ctx)
			if err != nil {
				return
			}
			if typ != websocket.MessageBinary {
				continue
			}
			if err := ws.Write(ctx, websocket.MessageBinary, msg); err != nil {
				return
			}
		}
	})
	srv := &http.Server{Handler: mux}
	go func() { _ = srv.Serve(ln) }()
	t.Cleanup(func() { srv.Close() })
	return "ws://" + ln.Addr().String()
}

// silentRelay authenticates the client and then goes completely silent (never
// pongs), so the client's keepalive watchdog should detect the dead relay.
func silentRelay(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/relay", func(w http.ResponseWriter, r *http.Request) {
		ws, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			Subprotocols:       []string{proto.RelayProtocolV2},
			InsecureSkipVerify: true,
		})
		if err != nil {
			return
		}
		ctx := r.Context()
		if err := ws.Write(ctx, websocket.MessageBinary, proto.EncodeServerChallenge([16]byte{1})); err != nil {
			return
		}
		if _, _, err := ws.Read(ctx); err != nil {
			return
		}
		ws.Write(ctx, websocket.MessageBinary, proto.EncodeServerConfirmsAuth())
		ws.Write(ctx, websocket.MessageBinary, proto.EncodeObservedAddr(net.IPv4(127, 0, 0, 1)))
		// Never read or write again.
		<-ctx.Done()
	})
	srv := &http.Server{Handler: mux}
	go func() { _ = srv.Serve(ln) }()
	t.Cleanup(func() { srv.Close() })
	return "ws://" + ln.Addr().String()
}

// TestWatchdogDetectsDeadRelay verifies the client's keepalive watchdog
// force-closes a relay that stops responding, surfacing it via Closed().
func TestWatchdogDetectsDeadRelay(t *testing.T) {
	url := silentRelay(t)
	secret, err := base.NewNodeSecret()
	if err != nil {
		t.Fatal(err)
	}
	rc, err := Dial(context.Background(), url, secret,
		WithKeepalive(20*time.Millisecond, 80*time.Millisecond))
	if err != nil {
		t.Fatal(err)
	}
	defer rc.Close()

	select {
	case <-rc.Closed():
		// detected: good
	case <-time.After(3 * time.Second):
		t.Fatal("watchdog did not close the dead relay connection")
	}
}

func TestReadFromDeadline(t *testing.T) {
	url := miniRelay(t)
	secret, err := base.NewNodeSecret()
	if err != nil {
		t.Fatal(err)
	}
	rc, err := Dial(context.Background(), url, secret)
	if err != nil {
		t.Fatal(err)
	}
	defer rc.Close()

	buf := make([]byte, 2048)

	if err := rc.SetReadDeadline(time.Now()); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		_, _, err := rc.ReadFrom(buf)
		done <- err
	}()
	select {
	case err := <-done:
		if err != os.ErrDeadlineExceeded {
			t.Fatalf("expected deadline exceeded, got %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("ReadFrom did not honor SetReadDeadline")
	}
}

func TestConcurrentWriteTo(t *testing.T) {
	url := miniRelay(t)
	secret, err := base.NewNodeSecret()
	if err != nil {
		t.Fatal(err)
	}
	rc, err := Dial(context.Background(), url, secret)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := rc.WriteTo([]byte("ping"), base.PeerAddr{ID: secret.Public()})
			if err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
}
