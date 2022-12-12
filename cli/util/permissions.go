package util

import (
	"errors"
	"os/user"
)

var (
	ErrGetCurrentUser = errors.New("unable to get current user")
	ErrMissingAdmPriv = errors.New("you must have administrator privileges")
)

func UserVerifyRootPerm() error {
	user, err := user.Current()
	if err != nil {
		return ErrGetCurrentUser
	}
	if user.Uid != "0" {
		return ErrMissingAdmPriv
	}
	return nil
}
