#!/bin/bash

# from scope src home run:
# docker build -t scope-test test/
# docker run -v $(pwd):/opt/scope -t scope-test /bin/bash '/opt/scope/test/test-ltp.sh'


source /opt/rh/devtoolset-7/enable
source /opt/rh/python27/enable

# ensure scope is linked against the right symbols
cd /opt/scope
make clean
make all

cd /opt/test/ltp
make autotools && ./configure 

### test cases 

cd testcases/kernel/syscalls/read
make -j4
./read01
./read02
./read03
./read04

echo
echo "####################################"
echo "### running scoped tests ..."
echo "####################################"
echo
LD_PRELOAD=/opt/scope/lib/linux/libwrap.so ./read01
LD_PRELOAD=/opt/scope/lib/linux/libwrap.so ./read02
LD_PRELOAD=/opt/scope/lib/linux/libwrap.so ./read03
LD_PRELOAD=/opt/scope/lib/linux/libwrap.so ./read04

