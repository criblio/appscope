import time
from ab import run_apache_benchmark

from common import ApplicationTest, AppController
from validation import validate_all


class TestGetUrl(ApplicationTest):

    def __init__(self, url, app_controller: AppController, requests=100):
        super().__init__(app_controller)
        self.requests = requests
        self.url = url

    def do_run(self, scoped):
        time.sleep(5)
        benchmark_results = run_apache_benchmark(url=self.url, requests=self.requests)
        time.sleep(5)

        test_result = validate_ab(benchmark_results)

        return test_result, benchmark_results

    @property
    def name(self):
        return "get " + self.url


class TestPostToUrl(ApplicationTest):
    @property
    def name(self):
        return "POST file"

    def __init__(self, url, post_file, app_controller: AppController, requests=100):
        super().__init__(app_controller)
        self.requests = requests
        self.url = url
        self.post_file = post_file

    def do_run(self, scoped):
        time.sleep(5)
        benchmark_results = run_apache_benchmark(url=self.url, requests=self.requests, post_file=self.post_file)
        time.sleep(5)

        test_result = validate_ab(benchmark_results)

        return test_result, benchmark_results


def validate_ab(benchmark_results):
    return validate_all(
        (benchmark_results.failed_requests == 0, f"Failed requests detected {benchmark_results.failed_requests}"),
        (benchmark_results.write_errors == 0, f"Write errors detected {benchmark_results.write_errors}")
    )

