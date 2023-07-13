#!/bin/bash
#
# Cleanup the environment after scope start 
#

PRELOAD_PATH="/etc/ld.so.preload"
LIBSCOPE_PATH="/usr/lib/libscope.so"
PROFILE_SCOPE_SCRIPT="/etc/profile.d/scope.sh"
USR_APPSCOPE_DIR="/usr/lib/appscope/"
TMP_APPSCOPE_DIR="/tmp/appscope/"
CRIBL_APPSCOPE_DIR="$CRIBL_HOME/appscope/"

echo "Following script will try to remove following files:"
echo "- $PRELOAD_PATH"
echo "- $LIBSCOPE_PATH"
echo "- $PROFILE_SCOPE_SCRIPT"
echo "- $USR_APPSCOPE_DIR"
echo "- $TMP_APPSCOPE_DIR"
echo "- \$CRIBL_HOME/appscope/"

read -p "Continue (y/n)?" choice
case "$choice" in 
  y|Y ) echo "Yes selected - Continuing";;
  n|N ) echo "No selected - Exiting"; exit;;
  * ) echo "Unknown choice - Exiting"; exit;;
esac

if [ "$EUID" -ne 0 ]
  then echo "Please run script with sudo"
  exit
fi

if [ -f $PRELOAD_PATH ] ; then
    rm $PRELOAD_PATH
    echo "$PRELOAD_PATH file was removed"
else
    echo "$PRELOAD_PATH file was missing. Continue..."
fi

# This one is a symbolic link
if [ -L $LIBSCOPE_PATH ] ; then
    rm $LIBSCOPE_PATH
    echo "$LIBSCOPE_PATH file was removed"
else
    echo "$LIBSCOPE_PATH file was missing. Continue..."
fi

if [ -f $PROFILE_SCOPE_SCRIPT ] ; then
    rm $PROFILE_SCOPE_SCRIPT
    echo "$PROFILE_SCOPE_SCRIPT file was removed"
else
    echo "$PROFILE_SCOPE_SCRIPT file was missing. Continue..."
fi

if [ -d "$USR_APPSCOPE_DIR" ] ; then
    rm -r $USR_APPSCOPE_DIR
    echo "$USR_APPSCOPE_DIR directory was removed"
else
    echo "$USR_APPSCOPE_DIR directory was missing. Continue..."
fi

if [ -d "$TMP_APPSCOPE_DIR" ] ; then
    rm -r $TMP_APPSCOPE_DIR
    echo "$TMP_APPSCOPE_DIR directory was removed"
else
    echo "$TMP_APPSCOPE_DIR directory was missing."
fi

if [ -f $CRIBL_HOME ] && [ -d "$CRIBL_APPSCOPE_DIR" ] ; then
    rm -r $CRIBL_APPSCOPE_DIR
    echo "$CRIBL_APPSCOPE_DIR directory was removed"
else
    echo "\$CRIBL_HOME was not set or \$CRIBL_HOME/appscope directory was missing."
fi
