package slogx

import (
	"log/slog"

	"github.com/skerkour/stdx-go/timex"
)

func Err(err error) slog.Attr {
	if err == nil {
		return slog.Any("error", nil)
	}

	return slog.String("error", err.Error())
}

func Time(key string, t timex.Time) slog.Attr {
	return slog.String(key, t.String())
}
