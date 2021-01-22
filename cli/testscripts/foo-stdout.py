#!/usr/bin/python

import time
import sys

i=0
while True:
    sys.stdout.write('foo %d\n' % i)
    sys.stdout.flush()
    i=i+1
    time.sleep(1)
