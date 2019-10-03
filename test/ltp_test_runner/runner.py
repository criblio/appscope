import os
import json
import sys
import subprocess
import time
import watcher as w

help_text = '''options:
--help  display help text
--home  tests home dir (by default /opt/test/ltp/testcases/kernel/syscalls/)
--include_tests include tests array or dir
--exclude_tests exclude tests array or dir
--config config file for tests
'''


class Runner:
    def __init__(self, home, include_tests, exclude_tests):
        self.__home = home
        self.__include_tests = include_tests
        self.__exclude_tests = exclude_tests

    def run(self):
        print("Starting...")

        test_failed = 0

        for test in self.__include_tests:
            if test not in self.__exclude_tests:
                if os.path.isdir(test):
                    for d in os.listdir(test):
                        if d not in self.__exclude_tests:
                            if self.__execute_test(self.__home + d) != 0:
                                test_failed = 1;
                else:
                    if self.__execute_test(self.__home + test) != 0:
                        test_failed = 1;

        return test_failed

    def __execute_test(self, test_path):
        if not os.path.isdir(test_path):
            return 0

        print(test_path)

        for f in os.listdir(test_path):
            filename = test_path + '/' + f
            if os.path.isfile(filename) and os.access(filename, os.X_OK):
                print(" " + filename)
                start = time.time()
                proc = subprocess.Popen('./' + f, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=test_path)
                stdout, stderr = proc.communicate()
                end = time.time()
                rc = proc.returncode
                if rc == 0:
                    w.Watcher().test_passed(w.TestResult(f, False, stdout, stderr, end - start, rc))
                else:
                    w.Watcher().test_failed(w.TestResult(f, False, stdout, stderr, end - start, rc))


        return 0


def main(args):
    home = "/opt/test/ltp/testcases/kernel/syscalls/"
    include_tests = []
    exclude_tests = []

    print("!!!!!")
    # parse args
    for i in range(len(args)):
        if args[i] == '--help':
            print(help_text)

        if args[i] == '--home':
            home = args[i + 1]

        if args[i] == 'include_tests':
            include_tests = args[i + 1]

        if args[i] == 'exclude_tests':
            exclude_tests = args[i + 1]

        if args[i] == '--config':
            path = args[i + 1]
            with open(path, "rb") as f:
                config = json.loads(f.read().decode("utf8"))
                home = config["home"]
                include_tests = config["include_tests"]
                exclude_tests = config["exclude_tests"]

    if not home.endswith('/'):
        home += '/'

    if not os.path.exists(home):
        sys.stderr.write("tests home is not valid")
        return 1

    return Runner(home, include_tests, exclude_tests).run()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
