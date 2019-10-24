#!/bin/bash

make clean
make

TESTS=`find tests/ -perm /u=x,g=x,o=x -type f -exec ls {} \;`

echo ""
echo "Raw tests:"

counter=0
failed=0
passed=0

for t in $TESTS; do
    ./$t
    if [ $? -eq 0 ]; then
	let passed++
    else
	let failed++
    fi
    
    let counter++
done

echo ""
echo "Total tests: "$counter" Failed: "$failed" Passed: "$passed

echo ""
echo "Scoped tests:"

counter=0
failed=0
passed=0

for t in $TESTS; do
    LD_PRELOAD=/opt/scope/lib/linux/libwrap.so ./$t

    if [ $? -eq 0 ]; then
	let passed++
    else
	let failed++
    fi
    
    let counter++
done

echo ""
echo "Total tests: "$counter" Failed: "$failed" Passed: "$passed
