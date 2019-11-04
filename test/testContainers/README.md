# Scope Test Containers

These represent containers used to test Cribl Scope. Each container is self-sufficent and self-executable. 
General strategy is to run set of tests/challenges against target application or syscall in order to determine that target is working correctly.
Then to run the same set of tests against the same target, but wrapped in Cribl Scope (scoped) and compare results with raw execution.
General strategy is to use one container per target. During container build Docker set up the environment, installs target application,
prepares environment for test-runner, ships runs the test-runner.

##Containers:
####cribl
Test the standalone version of Cribl LogStream. 

####splunk
Tests Splunk Enterprise. Splunk instance has Cribl LogStream as Splunk Application, but LogStream capabilities are not tested or used.

####interposed_func
Tests Linux syscalls supported by Scope using executable syscall unit tests. Some unit tests provided by [LTP Project](https://github.com/linux-test-project/ltp) 
the others are custom C test located in [interposed_func/altp folder](interposed_func/altp)

####nginx
Tests Nginx web server using Apache Benchmark tool.

##Execution

You can run test containers using docker-compose.
To run:
```
% cd scope/test/testContainers
% docker-compose up
```

To run individual container, for example nginx :
```
% docker-compose up nginx
```

You can also build and run containers individually using just Docker. Notice that Dockerfiles desinged to be build and run from `testContainers` folder.

####Verbose output
To see the verbose output in test runner output, run container with `-v` command:
```commandline
% docker-compose run nginx -v
```

####Results output

By default test runner will store logs and test results in `/opt/test-runner/logs`. You can mount this folder to your local folder using docker volume feature.

####Override default run mode
By default test runner would output only INFO level messages and won't store any logs in local system. 
If you need override this, you can use [docker-compose.override.template.yml](docker-compose.override.template.yml) file.
It will run the test execution in more verbose mode and store logs in `./logs` directory:
```commandline
cp docker-compose.override.template.yml docker.compose.override.yml
```
