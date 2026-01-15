package auth

import (
	"time"

	"github.com/skerkour/stdx-go/uuid"
)

type ApiKey struct {
	ID        uuid.UUID
	CreatedAt time.Time
	UpdatedAt time.Time

	Name string
	// ExpiresAt *time.Time
}
