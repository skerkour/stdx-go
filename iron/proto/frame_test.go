package proto

import (
	"bytes"
	"testing"

	"github.com/skerkour/stdx-go/iron/base"
)

func TestDatagramBatchRoundTrip(t *testing.T) {
	var id base.NodeID
	id[0] = 0xab

	// Two same-sized packets.
	p1 := bytes.Repeat([]byte{0x01}, 1200)
	p2 := bytes.Repeat([]byte{0x02}, 1200)
	contents := append(append([]byte{}, p1...), p2...)

	frame := EncodeDatagramBatch(nil, ClientToRelayBatch, id, 0, 1200, contents)

	tag, payload, err := Parse(frame)
	if err != nil {
		t.Fatal(err)
	}
	if tag != ClientToRelayBatch {
		t.Fatalf("tag = %d, want %d", tag, ClientToRelayBatch)
	}
	b, err := ParseDatagramBatch(payload)
	if err != nil {
		t.Fatal(err)
	}
	if b.Remote != id {
		t.Fatalf("remote = %x, want %x", b.Remote, id)
	}
	if b.SegmentSize != 1200 {
		t.Fatalf("segment size = %d, want 1200", b.SegmentSize)
	}
	pkts := b.Packets()
	if len(pkts) != 2 {
		t.Fatalf("got %d packets, want 2", len(pkts))
	}
	if !bytes.Equal(pkts[0], p1) || !bytes.Equal(pkts[1], p2) {
		t.Fatalf("packet contents mismatch")
	}
}

func TestDatagramBatchInvalid(t *testing.T) {
	var id base.NodeID
	// contents not a multiple of segment size.
	frame := EncodeDatagramBatch(nil, ClientToRelayBatch, id, 0, 100, bytes.Repeat([]byte{0}, 250))
	_, payload, err := Parse(frame)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseDatagramBatch(payload); err == nil {
		t.Fatal("expected error for misaligned contents")
	}

	// zero segment size.
	frame = EncodeDatagramBatch(nil, ClientToRelayBatch, id, 0, 0, nil)
	_, payload, err = Parse(frame)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseDatagramBatch(payload); err == nil {
		t.Fatal("expected error for zero segment size")
	}
}
