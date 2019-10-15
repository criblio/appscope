import time
from ab import run_apache_benchmark

from common import Test, TestResult


class TestGetUrl(Test):

    def __init__(self, url, requests=100):
        self.requests = requests
        self.url = url

    def run(self):
        passed = True

        benchmark_results = run_apache_benchmark(url=self.url, requests=self.requests)

        self.check(benchmark_results.failed_requests == 0,
                   f"Failed requests detected {benchmark_results.failed_requests}", benchmark_results)
        self.check(benchmark_results.write_errors == 0, f"Write errors detected {benchmark_results.write_errors}",
                   benchmark_results)

        return TestResult(passed=passed, test_data=benchmark_results)

    @property
    def name(self):
        return "get " + self.url
