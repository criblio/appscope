import subprocess
import logging
from utils import extract_num


class BenchmarkResults:
    def __init__(self):
        self.completed_requests = 0
        self.failed_requests = 0
        self.write_errors = 0


def run_apache_benchmark(url, requests=10, concur=4, post_file=None):
    post_param = f"-p {post_file}" if post_file else ""

    command = f"ab -n {requests} -c {concur} {post_param} {url}"

    logging.debug("Running Apache Benchmark. Command " + command)

    out = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                         shell=True)

    logging.debug("Apache Benchmark finished. Out:" + out.stdout)

    output = __parse_ab_output(out.stdout)
    logging.debug("Parsed output: " + repr(output.__dict__))
    return output


def __parse_ab_output(stdout):
    results = BenchmarkResults()

    for line in stdout.split('\n'):
        if "Complete requests:" in line:
            results.completed_requests = extract_num(line)[0]
        if "Failed requests:" in line:
            results.failed_requests = extract_num(line)[0]
        if "Write errors:" in line:
            results.write_errors = extract_num(line)[0]

    return results
