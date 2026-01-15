package jsonutil

import (
	"bufio"
	"bytes"
	"fmt"
	"strings"
)

func StripComments(input []byte) (output []byte, err error) {
	var jsonString string
	// Strip comments
	confScanner := bufio.NewScanner(bytes.NewReader(input))
	for confScanner.Scan() {
		line := confScanner.Text() // GET the line string
		if !strings.HasPrefix(strings.TrimSpace(line), "//") {
			jsonString += line + "\n"
		}
	}
	if err = confScanner.Err(); err != nil {
		err = fmt.Errorf("jsonutil.StripComments: scanning input: %w", err)
		return
	}

	output = []byte(jsonString)

	return
}
