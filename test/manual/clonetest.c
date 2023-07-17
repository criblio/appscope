// gcc -g test/manual/clonetest.c -o clonetest
// Different behavior in case of:
// ./clonetest
// scope -z ./clonetest
//
// This test was written to show that we could see scope
// affecting threaded applications.
//
// This has since been understood and solved. (The behavior
// is now the same regardless of whether this test program is
// scoped or not.) See the writeup here:
//      https://github.com/criblio/appscope/issues/1170
//

#define _GNU_SOURCE
#include <linux/limits.h>
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <syscall.h>

pid_t gettid(void)
{
    return syscall(SYS_gettid);
}

#define CHILD_STACK_BYTES 5000
static char child_stack[CHILD_STACK_BYTES];

int child_fn(void* arg) {
  for (int i = 0 ; i < 100; ++i) {
      printf(" child_fn begin pid = %d tid = %d\n iteration = %d", getpid(), gettid(), i);
      sleep(1);
  }
  printf(" child_fn after iteration pid = %d tid = %d\n", getpid(), gettid());
  return 0;
}

int main() {
  int status;
  printf("main started pid = %d tid = %d\n", getpid(), gettid());
  pid_t retPid = clone(child_fn, child_stack+ CHILD_STACK_BYTES, CLONE_VFORK, NULL);

  printf("main after clone parent pid = %d tid = %d retPid = %d errno = %d\n", getpid(), gettid(), retPid, errno);

  waitpid(retPid, &status, 0);
  printf("main after after wait pid = %d tid = %d\n", getpid(), gettid());
  return 0;
}
