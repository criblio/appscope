package util

import (
	"context"
	"errors"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/client"
)

var (
	ErrDockerNotAvailable = errors.New("docker daemon is not available")
)

// Get the List of PID(s) related to Docker container
func GetDockerPids() ([]int, error) {
	ctx := context.Background()
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, err
	}

	containers, err := cli.ContainerList(ctx, types.ContainerListOptions{})
	if err != nil {
		if client.IsErrConnectionFailed(err) {
			return nil, ErrDockerNotAvailable
		}
		return nil, err
	}

	pids := make([]int, len(containers))

	for indx, container := range containers {
		ins, err := cli.ContainerInspect(ctx, container.ID)
		if err != nil {
			return nil, err
		}
		pids[indx] = ins.State.Pid
	}
	return pids, nil
}
