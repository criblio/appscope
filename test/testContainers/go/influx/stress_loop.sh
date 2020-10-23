#! /bin/bash

for i in {1..20}
do
    echo "                                      *************** Go stress $i times ***************"
    ./stress.sh
done

