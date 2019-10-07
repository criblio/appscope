# Scope Test Containers

These represent containers used to test Cribl Scope.

Containers:
cribl
The log stream product. Metric results are sent to log stream. One or more pipelines in log stream will maniplulate metrics and route them to splunk for evaluation.

splunk
A splunk instance that receives test metric data from log stream. A splunk app is/can be used to evaluate metric and data results fromnm tests.

interposed function test
This is a container used to create tests for interposed functions. It will start with tests from the Linux Test Program, LTP. Results are formatted in json and sent to log stream.

application containers
Various containers with application to test. The app is exercised without being scoped and then fully scoped. Results are sent to log stream.

exercise containers
Various containers used to exercise individual applications in containers.
Example: apache benchmark (ab) used to sent requests to nginx &/or apache.


To run:
% cd scope/test/testContainers
% docker-compose up -d

To stop:
% docker-compose down

To access a given container:
% docker-compose exec 'container-name' bash

To build the containers:
% docker-compose build

Files:
1) scope/test/testContainers/docker-compose.yml
defines which containers are included and configures ports, volumes, logs as needed

2) scope/test/testContainers/cribl/Dockerfile
defines the log stream container
Option: add additional packages to install when the container starts as needed

3) scope/test/testContainers/splunk/Dockerfile
defines the splunk container
Option: add additional packages to install when the container starts as needed
