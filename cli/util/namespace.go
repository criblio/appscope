package util

import (
	"context"
	"errors"
	"os"

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

/*
 * Detect whether or not scope was executed inside a container
 * What is a reasonable algorithm for determining if the current process is in a container?
 * Checking ~/.dockerenv doesn't work for non-docker containers
 * Checking hierarchies in /proc/1/cgroup == / works in some case. However, some hierarchies
 * on non-containers are /init.scope (Ubuntu 20.04 cgroup 0 and 1). Not reliable.
 * Checking for container=lxc | docker in /proc/1/environ works for lxc but not docker, and it
 * requires root privs.
 * Checking /proc/self/cgroup for content after cpuset doesn't work in all cases
 * So, we check for the presence of /proc/2/comm
 */
func InContainer() bool {
	if _, err := os.Stat("/proc/2/comm"); err != nil {
		return true
	}
	return false
}
