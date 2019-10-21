#!/bin/bash

make clean
make

TESTS=`find tests/ -perm /u=x,g=x,o=x -type f -exec ls {} \;`

echo ""
echo "Raw tests:"

for t in $TESTS; do
    ./$t
done

echo ""
echo "Scoped tests:"

for t in $TESTS; do
    LD_PRELOAD=/opt/scope/lib/linux/libwrap.so ./$t
done
