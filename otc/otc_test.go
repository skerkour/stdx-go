package otc_test

import (
	"testing"

	"github.com/skerkour/stdx-go/otc"
)

func TestNewAndVerify(t *testing.T) {
	length := 8

	for i := 0; i < 5; i += 1 {
		code, err := otc.New(uint16(length))
		if err != nil {
			t.Error(err)
			return
		}
		codeText := code.Code()
		token := code.Token()
		if !otc.Verify(codeText, token) {
			t.Errorf("code is not valid: code (%s) | token (%s)", codeText, token)
			return
		}
	}
}
