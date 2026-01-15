package middlewarex

import (
	"net/http"

	"github.com/skerkour/stdx-go/httpx"
)

func SetServerHeader(server string) func(next http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		fn := func(w http.ResponseWriter, req *http.Request) {
			w.Header().Set(httpx.HeaderServer, server)
			next.ServeHTTP(w, req)
		}
		return http.HandlerFunc(fn)
	}
}
