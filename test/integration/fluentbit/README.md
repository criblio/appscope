
Add integration tests to discover compatibility between appscope and ootb fluentbit.

These discovery tests should eventually turn into fully integrated tests to validate continued fluentbit compatibility.

To include the tests in our integration workflow we need to uncomment the fluentbit section in testContainers/docker-compose.yml
