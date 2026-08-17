// Command relay runs a minimal iron relay server.
//
//	go run ./iron/example/relay -addr :3333
package main

import (
	"flag"
	"log/slog"
	"os"

	"github.com/skerkour/stdx-go/iron/relayserver"
)

func main() {
	addr := flag.String("addr", ":3333", "listen address")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	s := relayserver.NewServer()
	s.Log = logger

	logger.Info("relay listening", "addr", *addr)
	if err := s.ListenAndServe(*addr); err != nil {
		logger.Error("relay failed", "err", err)
		os.Exit(1)
	}
}
