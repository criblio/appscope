# AppScope Bugathon

The included Dockerfile provides all tools necessary to run scope and perform bugathon testing.

## Download

```bash
git clone git@github.com:criblio/appscope.git
cd appscope/test/testContainers/bugathon
```
-or if you already have it locally-
```bash
git checkout master
git pull
cd test/testContainers/bugathon
```

## Build and Run

You must have docker installed in order to build and run the scope bugathon container.

From this directory, run:

```bash
docker build -t scope-bugathon .
docker run -it scope-bugathon
```

The scope product is installed in `/usr/local/bin`. You can start exploring your applications using the scope executable: Enter `scope` with no arguments to see help, or use `scope --help`.

To see detailed help for a given argument, use scope help <argument>.

## Additional Shells

You may connect to the docker container in another terminal window with:

```bash
docker exec -it <Container ID> /bin/bash
```

...where the Container ID can be found in:

```bash
docker ps
```


