// Command echo demonstrates two iron endpoints exchanging a message over a
// QUIC connection established through a relay.
//
// Start the relay, then:
//
//	# listener
//	go run ./iron/example/echo -relay ws://127.0.0.1:3333 -mode listen
//
//	# dialer (use the node id printed by the listener)
//	go run ./iron/example/echo -relay ws://127.0.0.1:3333 -mode connect -peer <node-id>
//
// Both endpoints contact the same relay at startup; the dialer then opens a
// QUIC connection to the peer's NodeID, which is tunnelled through the relay
// as plain datagrams.
package main

import (
	"context"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"log"

	"github.com/quic-go/quic-go"

	"github.com/skerkour/stdx-go/iron"
	"github.com/skerkour/stdx-go/iron/base"
)

func main() {
	var (
		relayURL = flag.String("relay", "ws://127.0.0.1:3333", "relay websocket url")
		mode     = flag.String("mode", "listen", "listen or connect")
		peer     = flag.String("peer", "", "node id to dial (connect mode)")
		msg      = flag.String("msg", "hello from iron-go", "message to send (connect mode)")
		key      = flag.String("key", "", "64-byte hex ed25519 private key (optional, stable node id)")
	)
	flag.Parse()

	ctx := context.Background()

	var secret *base.NodeSecret
	if *key != "" {
		priv, err := hex.DecodeString(*key)
		if err != nil {
			log.Fatalf("invalid -key: %v", err)
		}
		secret, err = base.NewNodeSecretFromBytes(priv)
		if err != nil {
			log.Fatalf("identity: %v", err)
		}
	} else {
		var err error
		secret, err = base.NewNodeSecret()
		if err != nil {
			log.Fatalf("identity: %v", err)
		}
	}

	endpoint, err := iron.NewEndpoint(ctx, *relayURL, secret, "")
	if err != nil {
		log.Fatalf("endpoint: %v", err)
	}
	defer endpoint.Close()

	log.Printf("node id: %s", endpoint.NodeID())

	switch *mode {
	case "listen":
		runListen(ctx, endpoint)
	case "connect":
		if *peer == "" {
			log.Fatal("connect mode requires -peer")
		}
		id, err := base.NodeIDFromString(*peer)
		if err != nil {
			log.Fatalf("invalid -peer: %v", err)
		}
		runConnect(ctx, endpoint, id, *msg)
	default:
		log.Fatalf("unknown -mode %q", *mode)
	}
}

func runListen(ctx context.Context, endpoint *iron.Endpoint) {
	for {
		conn, err := endpoint.Accept(ctx)
		if err != nil {
			log.Fatalf("accept: %v", err)
		}
		remote, err := endpoint.PeerID(conn)
		if err != nil {
			log.Printf("accept: unknown peer: %v", err)
			conn.CloseWithError(0, "")
			continue
		}
		log.Printf("connection from %s (via %s)", remote, conn.Path())
		go handleConn(ctx, conn)
	}
}

func handleConn(ctx context.Context, conn *iron.Connection) {
	defer conn.CloseWithError(0, "")
	for {
		st, err := conn.AcceptStream(ctx)
		if err != nil {
			return
		}
		go echo(st)
	}
}

// echo reads everything the peer sends on the stream and sends it back.
func echo(st *quic.Stream) {
	defer st.Close()
	if _, err := io.Copy(st, st); err != nil {
		log.Printf("echo: %v", err)
	}
}

func runConnect(ctx context.Context, endpoint *iron.Endpoint, id base.NodeID, msg string) {
	conn, err := endpoint.Connect(ctx, id)
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer conn.CloseWithError(0, "")

	remote, err := endpoint.PeerID(conn)
	if err != nil {
		log.Fatalf("connected peer: %v", err)
	}
	log.Printf("connected to %s (via %s)", remote, conn.Path())

	st, err := conn.OpenStreamSync(ctx)
	if err != nil {
		log.Fatalf("open stream: %v", err)
	}
	if _, err := st.Write([]byte(msg)); err != nil {
		log.Fatalf("write: %v", err)
	}
	st.Close() // finish the send side; the read side stays open

	got, err := io.ReadAll(st)
	if err != nil {
		log.Fatalf("read: %v", err)
	}
	if string(got) != msg {
		log.Fatalf("echo mismatch: expected %q, got %q", msg, string(got))
	}
	fmt.Printf("echo round-trip OK (%d bytes)\n", len(got))
}
