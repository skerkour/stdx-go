package singleinstance_test

import (
	"os"
	"testing"

	"github.com/skerkour/stdx-go/singleinstance"
	"github.com/skerkour/stdx-go/testify/assert"
	"github.com/skerkour/stdx-go/testify/require"
)

func TestSingle(t *testing.T) {
	s, err := singleinstance.New("unittest")
	require.NoError(t, err)
	require.NotNil(t, s)

	t.Logf("Lockfile: %s", s.Lockfile())

	err = s.Lock()
	assert.NoError(t, err)

	assert.EqualError(t, checkLock(s), singleinstance.ErrAlreadyRunning.Error())

	err = s.Unlock()
	assert.NoError(t, err)
}

func checkLock(s *singleinstance.Single) error {
	if err := os.Remove(s.Lockfile()); err != nil && !os.IsNotExist(err) {
		return singleinstance.ErrAlreadyRunning
	}

	return nil
}
