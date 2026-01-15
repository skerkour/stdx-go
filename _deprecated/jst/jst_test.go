package jst_test

import (
	"crypto/rand"
	"testing"
	"time"

	"github.com/skerkour/stdx-go/jst"
)

func TestHelloWorld(t *testing.T) {
	type payloadHelloWorld struct {
		Message string `json:"message"`
	}

	key := make([]byte, 32)
	_, err := rand.Read(key)
	if err != nil {
		t.Error(err)
		return
	}

	keys := map[string][]byte{
		"default": key,
	}
	keysProvider := jst.NewKeyProviderMemory(keys)
	jstProvider, err := jst.NewProvider(keysProvider, "default")
	if err != nil {
		t.Error(err)
		return
	}

	payload := payloadHelloWorld{Message: "Hello World!"}
	expiresAt := time.Date(3000, 12, 31, 23, 59, 59, 0, time.UTC)
	token, err := jstProvider.IssueToken(payload, &expiresAt, nil)
	if err != nil {
		t.Error(err)
		return
	}

	var decodedPayload payloadHelloWorld
	decodedHeader, err := jstProvider.VerifyToken(token, &decodedPayload)
	if err != nil {
		t.Errorf("verifyToken (%s): %s", token, err)
		return
	}

	if decodedHeader.ExpiresAt == nil {
		t.Error("expires_at is null")
		return
	}
	if !decodedHeader.ExpiresAt.Equal(expiresAt) {
		t.Errorf("decoded_header.expires_at (%s) | expires_at (%s)", decodedHeader.ExpiresAt, expiresAt)
		return
	}

	if decodedPayload.Message != payload.Message {
		t.Errorf("decoded_payload.message (%s) | payload.message (%s)", decodedPayload.Message, payload.Message)
		return
	}
}
