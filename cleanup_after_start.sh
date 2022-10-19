#!/bin/bash
#
# Cleanup the environment after scope start 
#

PROFILE_SCOPE_SCRIPT="/etc/profile.d/scope.sh"
USR_APPSCOPE_DIR="/usr/lib/appscope/"
TMP_APPSCOPE_DIR="/tmp/appscope/"

echo "Following script will try to remove following files:"
echo "- $PROFILE_SCOPE_SCRIPT"
echo "- $USR_APPSCOPE_DIR"
echo "- $TMP_APPSCOPE_DIR"
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
