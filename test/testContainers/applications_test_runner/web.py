from ab import run_apache_benchmark

from common import Test
from validation import validate_all


class TestGetUrl(Test):

    def __init__(self, url, requests=100):
        self.requests = requests
        self.url = url

    def run(self):

        benchmark_results = run_apache_benchmark(url=self.url, requests=self.requests)

        test_result = validate_all(
            (benchmark_results.failed_requests == 0, f"Failed requests detected {benchmark_results.failed_requests}"),
            (benchmark_results.write_errors == 0, f"Write errors detected {benchmark_results.write_errors}")
        )

        return test_result, benchmark_results

    @property
    def name(self):
        return "get " + self.url
