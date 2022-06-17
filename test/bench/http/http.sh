#!/bin/bash
# Requires nginx, scope, ab
# Configure /etc/nginx/nginx.conf with: worker_processes 1

unscoped2_h2_nginx() {
    printf "Unscoped Nginx HTTP2:\n"
    sudo nginx -s stop
    sleep 3
    sudo nginx
    PID=$(ps -ef | grep www-data | grep -v grep | awk '{ print $2 }')
    START=$(cat /proc/$PID/schedstat | awk '{ print $1 }')
    h2load -c 1 -n 100 https://127.0.0.1:443/ #> /dev/null 2>&1
    END=$(cat /proc/$PID/schedstat | awk '{ print $1 }')
    printf "Start:\t%s\n" $START 
    printf "End:\t%s\n" $END 
    printf "Diff:\t%s\n" $((($END-$START) / 100))
}

scoped2_h2_nginx() {
    printf "\nScoped Nginx HTTP2:\n"
    sudo nginx -s stop
    sudo -E ../../../bin/linux/x86_64/ldscope nginx
    sleep 2
    PID=$(ps -ef | grep www-data | grep -v grep | awk '{ print $2 }')
    START=$(cat /proc/$PID/schedstat | awk '{ print $1 }')
    h2load -c 1 -n 1 https://127.0.0.1:443/ #> /dev/null 2>&1
    END=$(cat /proc/$PID/schedstat | awk '{ print $1 }')
    printf "Start:\t%s\n" $START 
    printf "End:\t%s\n" $END 
    printf "Diff:\t%s\n" $((($END-$START) / 100))
}

unscoped2_w2_nginx() {
    printf "Unscoped Nginx HTTP2:\n"
    nn=0
    sys=0.00
    usr=0.00
    val=0
    uv=0
    sudo nginx -s stop
    sudo nginx
    sleep 1

    PID=$(ps -ef | grep www-data | grep -v grep | awk '{ print $2 }')
    START=$(cat /proc/$PID/schedstat | awk '{ print $1 }')

    while [[ $nn -lt "1" ]]; do
        /usr/bin/time -f "%S %U" -o temp wget2 --no-check-certificate --http2 https://localhost/
        sleep 1
        nn=$((nn+1))

        val=`cat temp | cut -d " " -f 1`
        x=$sys
        y=$val
        sys="$x + $y"
        echo "sys = `bc <<< $sys`"

        val=`cat temp | cut -d " " -f 2`
        x=$usr
        y=$val
        usr="$x + $y"
        echo $usr
        echo "usr = `bc <<< $usr`"
    done

    END=$(cat /proc/$PID/schedstat | awk '{ print $1 }')
    printf "Start:\t%s\n" $START
    printf "End:\t%s\n" $END
    printf "Diff:\t%s\n\n" $(($END-$START))
    printf "system time: %s \n" `bc <<< $sys`
    printf "user time: %s\n" `bc <<< $usr`

    x="$sys + $usr"
    echo "Total time:  `bc <<< $x`"

    rm index.html*
    rm temp
}

scoped2_w2_nginx() {
    printf "\nScoped Nginx HTTP2:\n"
    nn=0
    sys=0.00
    usr=0.00
    val=0
    uv=0
    sudo nginx -s stop
    sudo -E ../../../bin/linux/x86_64/ldscope nginx
    sleep 1

    PID=$(ps -ef | grep www-data | grep -v grep | awk '{ print $2 }')
    START=$(cat /proc/$PID/schedstat | awk '{ print $1 }')

    while [[ $nn -lt "1" ]]; do
        /usr/bin/time -f "%S %U" -o temp wget2 --no-check-certificate --http2 https://localhost/
        sleep 1
        nn=$((nn+1))

        val=`cat temp | cut -d " " -f 1`
        x=$sys
        y=$val
        sys="$x + $y"
        echo "sys = `bc <<< $sys`"

        val=`cat temp | cut -d " " -f 2`
        x=$usr
        y=$val
        usr="$x + $y"
        echo $usr
        echo "usr = `bc <<< $usr`"
    done

    END=$(cat /proc/$PID/schedstat | awk '{ print $1 }')
    printf "Start:\t%s\n" $START
    printf "End:\t%s\n" $END
    printf "Diff:\t%s\n\n" $(($END-$START))
    printf "system time: %s \n" `bc <<< $sys`
    printf "user time: %s\n" `bc <<< $usr`

    x="$sys + $usr"
    echo "Total time:  `bc <<< $x`"

    rm index.html*
    rm temp
}

unscoped1_nginx() {
    printf "Unscoped Nginx HTTP1:\n"
    sudo nginx -s stop
    sudo nginx
    PID=$(ps -ef | grep www-data | grep -v grep | awk '{ print $2 }')
    START=$(cat /proc/$PID/schedstat | awk '{ print $1 }')
    ab -c 1 -n 100 http://127.0.0.1:80/ > /dev/null 2>&1
    END=$(cat /proc/$PID/schedstat | awk '{ print $1 }')
    printf "Start:\t%s\n" $START
    printf "End:\t%s\n" $END
    printf "Diff:\t%s\n" $((($END-$START) / 100))
}

scoped1_nginx() {
    printf "\nScoped Nginx HTTP1:\n"
    sudo nginx -s stop
    sudo -E ../../../bin/linux/x86_64/ldscope nginx
    PID=$(ps -ef | grep www-data | grep -v grep | awk '{ print $2 }')
    START=$(cat /proc/$PID/schedstat | awk '{ print $1 }')
    ab -c 1 -n 100 http://127.0.0.1:80/ > /dev/null 2>&1
    END=$(cat /proc/$PID/schedstat | awk '{ print $1 }')
    printf "Start:\t%s\n" $START
    printf "End:\t%s\n" $END
    printf "Diff:\t%s\n" $((($END-$START) / 100))
}

unscoped1_nginx
scoped1_nginx
#unscoped2_h2_nginx
#scoped2_h2_nginx
#unscoped2_w2_nginx
#scoped2_w2_nginx
