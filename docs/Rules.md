# Scope Rules

## Scenarios we intend to support

### Scenario: Edge running in a container

Start edge (as defined in cribl documentation):
```
docker run -it --rm -e CRIBL_EDGE=1  -p 9420:9420 -v /<path_to_code>/appscope:/opt/appscope  -v /var/run/appscope:/var/run/appscope  -v /var/run/docker.sock:/var/run/docker.sock  -v /:/hostfs:ro  --privileged  --name cribl-edge cribl/cribl:latest bash
/opt/cribl/bin/cribl start
```
Tests:
```
sudo touch /etc/ld.so.preload # for safety
sudo chmod ga+w /etc/ld.so.preload # for safety
<start edge container>
<run top on the host>
<start a container>
<run top in that container>
scope rules --add top --sourceid A --rootdir /hostfs --unixpath /var/run/appscope
scope rules --rootdir /hostfs
### Does the rules file contain an entry for top?
scope ps --rootdir /hostfs
### Are two top processes scoped by attach?
<run top on the host>
<start a new container>
<run top in the new container>
scope ps --rootdir /hostfs
### Are four top processes scoped (2 by attach, 2 by preload)?
### Is data flowing into edge from 3 processes (2 on host, 1 in new container)?
scope rules --remove top --sourceid A --rootdir /hostfs
scope rules --rootdir /hostfs
### Is the rules file empty?
scope ps --rootdir /hostfs
### Are 0 top processes scoped?
<run top on the host>
scope ps --rootdir /hostfs
### Are 0 top processes scoped?
### A unix sock path is supported on the rules add command line. it will place the unix path in the rules file where the config from Edge is placed. 
sudo scope rules --add top --unixpath /var/run/appscope
at the end of the rules file we will see this:
source:
  unixSocketPath: /var/run/appscope
  authToken: ""
the result is that /var/run/appscope is mounted in new containers.
```

### Scenario: Edge running on the host

Start edge (as defined in cribl documentation):
```
<switch the user to root>
curl https://cdn.cribl.io/dl/4.1.3/cribl-4.1.3-15457782-linux-x64.tgz -o ~/Downloads/cribl.tgz
cd /opt/
tar xvzf ~/Downloads/cribl.tgz
mv /opt/cribl/ /opt/cribl-edge
export CRIBL_HOME=/opt/cribl-edge # note: $CRIBL_HOME is set only in the cribl process (and cli children)
cd /opt/cribl-edge/bin
./cribl mode-edge
chown root:root /opt/cribl-edge/bin/cribl
./cribl start
```

Tests:
```
sudo touch /etc/ld.so.preload # for safety
sudo chmod ga+w /etc/ld.so.preload # for safety
<start edge on host>
<run top on the host>
<start a container>
<run top in that container>
scope rules --add top --sourceid A --unixpath /var/run/appscope
scope rules
### Does the rules file contain an entry for top?
scope ps
### Are two top processes scoped by attach?
<run top on the host>
<start a new container>
<run top in the new container>
scope ps
### Are four top processes scoped (2 by attach, 2 by preload)?
### Is data flowing into edge from three processes (2 on host, 1 in new container)?
scope rules --remove top --sourceid A
scope rules
### Is the rules file empty?
scope ps
### Are 0 top processes scoped?
<run top on the host>
scope ps
### Are 0 top processes scoped?
```

### Where files will be created

Host processes:
- libscope: should end up in `/usr/lib/appscope/<ver>/` on the host
- scope: should end up in `/usr/lib/appscope/<ver>/` on the host
- scope_rules: should end up in `/usr/lib/appscope/scope_rules` on the host
- unix socket: 
  - edge running in container: will be in `/var/run/appscope/` on the host by default (edge documentation describes that `/var/run/appscope` is mounted from the host into the container). 
  - edge running on host: will be in `$CRIBL_HOME/state/` by default

Existing container processes:
- libscope: installed into /usr/lib/appscope/<ver>/ in all existing containers (/etc/ld.so.preload points to this)
- scope: _not required_
- scope_rules: `/usr/lib/appscope` should be mounted into all existing containers into `/usr/lib/appscope/`
- unix socket: the dirpath defined in `scope_rules` should be mounted into all existing containers (`$CRIBL_HOME/state/` note that the env var will be resolved in the scope_rules file)

New container processes:
- libscope: extracted into `/opt/appscope` in all new containers
- scope: `/usr/lib/appscope` should be mounted into all new containers into `/usr/lib/appscope/`
- scope_rules: `/usr/lib/appscope` should be mounted into all new containers
- unix socket: the dirpath defined in `scope_rules` should be mounted into all new containers (default `/var/run/appscope/`)
