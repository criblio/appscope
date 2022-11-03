package util

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/namespaces"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/client"
	lxd "github.com/lxc/lxd/client"
)

var (
	ErrContainerDNotAvailable = errors.New("ContainerD is not available")
	ErrDockerNotAvailable     = errors.New("docker daemon is not available")
	ErrLXDSocketNotAvailable  = errors.New("LXD socket is not available")
)

// Get the LXD server unix socket
func getLXDServerSocket() string {
	socketPaths := []string{"/var/lib/lxd/unix.socket", "/var/snap/lxd/common/lxd/unix.socket"}
	for _, path := range socketPaths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}

// Get the List of PID(s) related to LXC container
func GetLXCPids() ([]int, error) {
	lxdUnixPath := getLXDServerSocket()
	if lxdUnixPath == "" {
		return nil, ErrLXDSocketNotAvailable
	}

	cli, err := lxd.ConnectLXDUnix(lxdUnixPath, nil)
	if err != nil {
		return nil, err
	}
	containers, err := cli.GetContainers()
	if err != nil {
		return nil, err
	}
	pids := make([]int, len(containers))

	for indx, container := range containers {
		state, _, err := cli.GetContainerState(container.Name)
		if err != nil {
			return nil, err
		}
		pids[indx] = int(state.Pid)
	}
	return pids, nil
}

// Get the List of PID(s) related to containerd container
func GetContainerDPids() ([]int, error) {
	ctx := context.Background()
	cli, err := containerd.New("/run/containerd/containerd.sock")
	if err != nil {
		return nil, ErrContainerDNotAvailable
	}
	defer cli.Close()

	// TODO: replace GetDockerPids with the API here and namespace "moby"
	cdctx := namespaces.WithNamespace(ctx, "default")
	containers, err := cli.Containers(cdctx)

	if err != nil {
		return nil, err
	}
	pids := make([]int, 0, len(containers))

	for _, container := range containers {
		task, err := container.Task(cdctx, nil)
		if err != nil {
			fmt.Printf("\nSkipped container %v, task error: %v", container, err)
			continue
		}
		status, err := task.Status(cdctx)
		if err != nil {
			fmt.Printf("\nSkipped task %v, container %v, status error: %v", task, container, err)
			continue
		}
		if status.Status == containerd.Running {
			pids = append(pids, int(task.Pid()))
		}
	}

	return pids, nil
}

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
