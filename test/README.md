# Interposed Functions
### Running tests
From project `src` folder run:
```
docker build -t scope-test test/
docker run -v $(pwd):/opt/scope -t scope-test python  /opt/scope/test/ltp_test_runner/runner.py --config /opt/scope/test/ltp_test_runner/.runner.conf
```
# Applications