//go:build !windows

package singleinstance_test

import (
	"os"
	"syscall"
	"testing"

	"github.com/skerkour/stdx-go/singleinstance"
	"github.com/skerkour/stdx-go/testify/assert"
	"github.com/skerkour/stdx-go/testify/require"
)

func TestSingle(t *testing.T) {
	s := singleinstance.New("unittest")
	require.NotNil(t, s)

	t.Logf("Lockfile: %s", s.Lockfile())

	err := s.Lock()
	assert.NoError(t, err)

	assert.EqualError(t, checkLock(s), singleinstance.ErrAlreadyRunning.Error())

	err = s.Unlock()
	assert.NoError(t, err)
}

func checkLock(s *singleinstance.SingleInstance) error {
	f, err := os.OpenFile(s.Lockfile(), os.O_RDONLY, 0600)
	if err != nil {
		return err
	}

	// try to obtain an exclusive lock with the PPID

	flock := syscall.Flock_t{
		Type: syscall.F_WRLCK,
		Pid:  int32(os.Getppid()),
	}

	if err := syscall.FcntlFlock(f.Fd(), syscall.F_SETLK, &flock); err != nil {
		return singleinstance.ErrAlreadyRunning
	}

	return nil
}
