#! /bin/bash

distro="/etc/os-release"
if [ -f "$distro" ]; then
    grep -i alpine $distro > /dev/null
    if [ $? = 0 ]; then
        echo "-D __ALPINE__"
        exit 0
    fi
fi
echo ""