#!/bin/bash

#
# Idempotence...
#
# This script was made to run as often as you like; it will only make changes
# the first time it's run.
#

echo "Running install_build_tools.sh."

PLATFORM=$(./scope_env.sh platform)
if [ ${PLATFORM} = "Error" ]; then
    echo "Exiting.  install_build_tools.sh does not support this platform type."
    exit 1
else
    echo "Determined platform type is : ${PLATFORM}."
fi

FAILED=0

########################################
##  macOS section
########################################
#
#  This was originally developed and tested on MacOS with:
#      System Version: macOS 10.14.5 (18F203)
#      Homebrew 2.1.6
#      autoconf (GNU Autoconf) 2.69
#      automake (GNU automake) 1.16.1
#      glibtool (GNU libtool) 2.4.6
#      cmake version 3.14.5
#      lcov: LCOV version 1.14
#

macOS_brew_exists() {
    if brew --version &>/dev/null; then
        echo "homebrew is already installed; doing nothing for homebrew."
    else
        echo "homebrew is not already installed."
        return 1
    fi
}

macOS_brew_install() {
    echo "Installing homebrew."
    # The initial "echo" is to pass the "hit enter to continue" prompt.
    echo | /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    if [ $? = 0 ]; then
        echo "Installation of homebrew successful."
    else
        echo "Installation of homebrew failed."
        FAILED=1
    fi
}

macOS_autoconf_exists() {
    if autoconf --version &>/dev/null; then
        echo "autoconf is already installed; doing nothing for autoconf."
    else
        echo "autoconf is not already installed."
        return 1
    fi
}

macOS_autoconf_install() {
    echo "Installing autoconf."
    brew install autoconf
    if [ $? = 0 ]; then
        echo "Installation of autoconf successful."
    else
        echo "Installation of autoconf failed."
        FAILED=1
    fi
}

macOS_automake_exists() {
    if automake --version &>/dev/null; then
        echo "automake is already installed; doing nothing for automake."
    else
        echo "automake is not already installed."
        return 1
    fi
}

macOS_automake_install() {
    echo "Installing automake."
    brew install automake
    # We had a new failure with macOS 10.14 on azure on 8-dec-2020.
    # Work around it by ignoring the return code
    if automake --version &>/dev/null; then
        echo "Installation of automake successful."
    else
        echo "Installation of automake failed."
        FAILED=1
    fi
}

macOS_glibtool_exists() {
    if glibtool --version &>/dev/null; then
        echo "glibtool is already installed; doing nothing for glibtool."
    else
        echo "glibtool is not already installed."
        return 1
    fi
}

macOS_glibtool_install() {
    echo "Installing glibtool."
    brew install libtool
    if [ $? = 0 ]; then
        echo "Installation of glibtool successful."
    else
        echo "Installation of glibtool failed."
        FAILED=1
    fi
}

macOS_cmake_exists() {
    if cmake --version &>/dev/null; then
        echo "cmake is already installed; doing nothing for cmake."
    else
        echo "cmake is not already installed."
        return 1
    fi
}

macOS_cmake_install() {
    echo "Installing cmake."
    brew install cmake
    if [ $? = 0 ]; then
        echo "Installation of cmake successful."
    else
        echo "Installation of cmake failed."
        FAILED=1
    fi
}

macOS_lcov_exists() {
    if lcov --version &>/dev/null; then
        echo "lcov is already installed; doing nothing for lcov."
    else
        echo "lcov is not already installed."
        return 1
    fi
}

macOS_lcov_install() {
    echo "Installing lcov."
    brew install lcov
    if [ $? = 0 ]; then
        echo "Installation of lcov successful."
    else
        echo "Installation of lcov failed."
        FAILED=1
    fi
}

macOS_dump_versions() {
    # The crazy sed stuff at the end of each just provides indention.
    system_profiler SPSoftwareDataType | grep "System Version" | sed 's/^[ ]*/      /'
    brew --version | head -n1 | sed 's/^/      /'
    autoconf --version | head -n1 | sed 's/^/      /'
    automake --version | head -n1 | sed 's/^/      /'
    glibtool --version | head -n1 | sed 's/^/      /'
    cmake --version | head -n1 | sed 's/^/      /'
    lcov --version | head -n1 | sed 's/^/      /'
}

if [ ${PLATFORM} = "macOS" ]; then
    if ! macOS_brew_exists; then
        macOS_brew_install
    fi
    if ! macOS_autoconf_exists; then
        macOS_autoconf_install
    fi
    if ! macOS_automake_exists; then
        macOS_automake_install
    fi
    if ! macOS_glibtool_exists; then
        macOS_glibtool_install
    fi
    if ! macOS_cmake_exists; then
        macOS_cmake_install
    fi
    if ! macOS_lcov_exists; then
        macOS_lcov_install
    fi

    if (( FAILED )); then
        echo "Installation failure detected."
        echo "Pertinent version info:"
        macOS_dump_versions
        exit 1
    fi
    exit 0
fi

########################################
##  linux yum section
########################################
#
#  This was originally developed and tested on CentOS with:
#      CentOS Linux release 7.6.1810 (Core) 
#      yum 3.4.3
#      Sudo version 1.8.23
#      autoconf (GNU Autoconf) 2.69
#      automake (GNU automake) 1.13.4
#      libtool (GNU libtool) 2.4.2
#      cmake version 3.13.5
#      lcov: LCOV version 1.13
#

PKG_MGR="unknown"

linux_determine_pkg_mgr() {
    if yum --version &>/dev/null; then
        PKG_MGR="yum"
    elif apt-get --version &>/dev/null; then
        PKG_MGR="apt-get"
    else
        return 1
    fi
    return 0
}

yum_sudo_exists() {
    if sudo --version &>/dev/null; then
        echo "sudo is already installed; doing nothing for sudo."
    else
        echo "sudo is not already installed."
        return 1
    fi
}

yum_sudo_install() {
    echo "Installing sudo."
    yum install sudo
    if [ $? = 0 ]; then
        echo "Installation of sudo successful."
    else
        echo "Installation of sudo failed."
        FAILED=1
    fi
}

yum_autoconf_exists() {
    if autoconf --version &>/dev/null; then
        echo "autoconf is already installed; doing nothing for autoconf."
    else
        echo "autoconf is not already installed."
        return 1
    fi
}

yum_autoconf_install() {
    echo "Installing autoconf."
    sudo yum install autoconf
    if [ $? = 0 ]; then
        echo "Installation of autoconf successful."
    else
        echo "Installation of autoconf failed."
        FAILED=1
    fi
}

yum_automake_exists() {
    if automake --version &>/dev/null; then
        echo "automake is already installed; doing nothing for automake."
    else
        echo "automake is not already installed."
        return 1
    fi
}

yum_automake_install() {
    echo "Installing automake."
    sudo yum install automake
    if [ $? = 0 ]; then
        echo "Installation of automake successful."
    else
        echo "Installation of automake failed."
        FAILED=1
    fi
}

yum_libtool_exists() {
    if libtool --version &>/dev/null; then
        echo "libtool is already installed; doing nothing for libtool."
    else
        echo "libtool is not already installed."
        return 1
    fi
}

yum_libtool_install() {
    echo "Installing libtool."
    sudo yum install libtool
    if [ $? = 0 ]; then
        echo "Installation of libtool successful."
    else
        echo "Installation of libtool failed."
        FAILED=1
    fi
}

yum_cmake_exists() {
    if cmake --version &>/dev/null; then
        echo "cmake is already installed; doing nothing for cmake."
    else
        echo "cmake is not already installed."
        return 1
    fi
}

yum_cmake_install() {
    echo "Installing cmake."
    # As a note, cmake3 is provided by epel-release, so if this doesn't
    # work, we probably just need to add sudo yum install epel-release first.
    sudo yum install cmake3
    if [ $? = 0 ]; then
        sudo alternatives --install /usr/local/bin/cmake cmake /usr/bin/cmake3 10 \
             --slave /usr/local/bin/ctest ctest /usr/bin/ctest3 \
             --slave /usr/local/bin/cpack cpack /usr/bin/cpack3 \
             --slave /usr/local/bin/ccmake ccmake /usr/bin/ccmake3 \
             --family cmake
    fi
    if [ $? = 0 ]; then
        echo "Installation of cmake successful."
    else
        echo "Installation of cmake failed."
        FAILED=1
    fi
}

yum_lcov_exists() {
    if lcov --version &>/dev/null; then
        echo "lcov is already installed; doing nothing for lcov."
    else
        echo "lcov is not already installed."
        return 1
    fi
}

yum_lcov_install() {
    echo "Installing lcov."
    sudo yum install lcov
    if [ $? = 0 ]; then
        echo "Installation of lcov successful."
    else
        echo "Installation of lcov failed."
        FAILED=1
    fi
}

yum_go_exists() {
    if go version &>/dev/null; then
        echo "go is already installed; doing nothing for go."
    else
        echo "go is not already installed."
        return 1
    fi
}

yum_go_install() {
    echo "Installing go."
    sudo yum -y install epel-release
    sudo yum -y install golang
    if [ $? = 0 ]; then
        echo "Installation of go successful."
    else
        echo "Installation of go failed."
        FAILED=1
    fi
}

yum_upx_exists() {
    if upx --version &>/dev/null; then
        echo "upx is already installed; doing nothing for upx."
    else
        echo "upx is not already installed."
        return 1
    fi
}

yum_upx_install() {
    echo "Installing upx."
    sudo yum -y install epel-release
    sudo yum -y install upx
    if [ $? = 0 ]; then
        echo "Installation of upx successful."
    else
        echo "Installation of upx failed."
        FAILED=1
    fi
}




yum_dump_versions() {
    # The crazy sed stuff at the end of each just provides indention.
    if [ -f /etc/centos-release ]; then
        cat /etc/centos-release | head -n1 | sed 's/^[ ]*/      /'
    elif [ -f /etc/redhat-release ]; then
        cat /etc/redhat-release | head -n1 | sed 's/^[ ]*/      /'
    elif [ -f /etc/os-release ]; then
        grep PRETTY_NAME /etc/os-release | cut -d \" -f2 |  sed 's/^/      /'
    fi
    yum --version | head -n1 | sed 's/^/      yum /'
    sudo --version | head -n1 | sed 's/^/      /'
    autoconf --version | head -n1 | sed 's/^/      /'
    automake --version | head -n1 | sed 's/^/      /'
    libtool --version | head -n1 | sed 's/^/      /'
    cmake --version | head -n1 | sed 's/^/      /'
    lcov --version | head -n1 | sed 's/^/      /'
    go version | head -n1 | sed 's/^/      /'
    upx --version | head -n1 | sed 's/^/      /'
}

yum_install() {
    echo "Running the yum install."

    if ! yum_sudo_exists; then
        yum_sudo_install
    fi
    if ! yum_autoconf_exists; then
        yum_autoconf_install
    fi
    if ! yum_automake_exists; then
        yum_automake_install
    fi
    if ! yum_libtool_exists; then
        yum_libtool_install
    fi
    if ! yum_cmake_exists; then
        yum_cmake_install
    fi
    if ! yum_lcov_exists; then
        yum_lcov_install
    fi
    if ! yum_go_exists; then
        yum_go_install
    fi
    if ! yum_upx_exists; then
        yum_upx_install
    fi


    if (( FAILED )); then
        echo "Installation failure detected."
        echo "Pertinent version info:"
        yum_dump_versions
        exit 1
    fi
    exit 0
}

########################################
##  linux apt section
########################################
#
#  This was originally developed and tested on ubuntu with:
#      Ubuntu 18.04.2 LTS
#      apt 1.6.11 (amd64)
#      Sudo version 1.8.21p2
#      GNU Make 4.1
#      autoconf (GNU Autoconf) 2.69
#      libtool (GNU libtool) 2.4.6
#      cmake version 3.10.2
#      lcov: LCOV version 1.13
#

apt_sudo_exists() {
    if sudo --version &>/dev/null; then
        echo "sudo is already installed; doing nothing for sudo."
    else
        echo "sudo is not already installed."
        return 1
    fi
}

apt_sudo_install() {
    echo "Installing sudo."
    apt-get update && apt-get install sudo
    if [ $? = 0 ]; then
        echo "Installation of sudo successful."
    else
        echo "Installation of sudo failed."
        FAILED=1
    fi
}

apt_make_exists() {
    if make --version &>/dev/null; then
        echo "make is already installed; doing nothing for make."
    else
        echo "make is not already installed."
        return 1
    fi
}

apt_make_install() {
    echo "Installing make."
    sudo apt-get install -y make
    if [ $? = 0 ]; then
        echo "Installation of make successful."
    else
        echo "Installation of make failed."
        FAILED=1
    fi
}

apt_autoconf_exists() {
    if autoconf --version &>/dev/null; then
        echo "autoconf is already installed; doing nothing for autoconf."
    else
        echo "autoconf is not already installed."
        return 1
    fi
}

apt_autoconf_install() {
    echo "Installing autoconf."
    sudo apt-get install -y autoconf
    if [ $? = 0 ]; then
        echo "Installation of autoconf successful."
    else
        echo "Installation of autoconf failed."
        FAILED=1
    fi
}

apt_libtool_exists() {
    if libtool --version &>/dev/null; then
        echo "libtool is already installed; doing nothing for libtool."
    else
        echo "libtool is not already installed."
        return 1
    fi
}

apt_libtool_install() {
    echo "Installing libtool."
    sudo apt-get install -y libtool-bin
    if [ $? = 0 ]; then
        echo "Installation of libtool successful."
    else
        echo "Installation of libtool failed."
        FAILED=1
    fi
}

apt_cmake_exists() {
    if cmake --version &>/dev/null; then
        echo "cmake is already installed; doing nothing for cmake."
    else
        echo "cmake is not already installed."
        return 1
    fi
}

apt_cmake_install() {
    echo "Installing cmake."
    sudo apt-get install -y cmake
    if [ $? = 0 ]; then
        echo "Installation of cmake successful."
    else
        echo "Installation of cmake failed."
        FAILED=1
    fi
}

apt_lcov_exists() {
    if lcov --version &>/dev/null; then
        echo "lcov is already installed; doing nothing for lcov."
    else
        echo "lcov is not already installed."
        return 1
    fi
}

apt_lcov_install() {
    echo "Installing lcov."
    sudo apt-get install -y lcov
    if [ $? = 0 ]; then
        echo "Installation of lcov successful."
    else
        echo "Installation of lcov failed."
        FAILED=1
    fi
}

apt_go_exists() {
    if go version &>/dev/null; then
        echo "go is already installed; doing nothing for go."
    else
        echo "go is not already installed."
        return 1
    fi
}

apt_go_install() {
    echo "Installing go."
    sudo apt-get install -y golang-go
    if [ $? = 0 ]; then
        echo "Installation of go successful."
    else
        echo "Installation of go failed."
        FAILED=1
    fi
}

apt_upx_exists() {
    if upx --version &>/dev/null; then
        echo "upx is already installed; doing nothing for upx."
    else
        echo "upx is not already installed."
        return 1
    fi
}

apt_upx_install() {
    echo "Installing upx."
    sudo apt-get install -y upx
    if [ $? = 0 ]; then
        echo "Installation of upx successful."
    else
        echo "Installation of upx failed."
        FAILED=1
    fi
}

apt_dump_versions() {
    # The crazy sed stuff at the end of each just provides indention.
    if lsb_release -d &>/dev/null; then
        lsb_release -d | sed 's/^Description:\s*/      /'
    elif [ -f /etc/lsb-release ]; then
        grep DISTRIB_DESCRIPTION /etc/lsb-release | cut -d \" -f2 | sed 's/^/      /'
    fi
    apt-get --version | head -n1 | sed 's/^/      /'
    sudo --version | head -n1 | sed 's/^/      /'
    make --version | head -n1 | sed 's/^/      /'
    autoconf --version | head -n1 | sed 's/^/      /'
    libtool --version | head -n1 | sed 's/^/      /'
    cmake --version | head -n1 | sed 's/^/      /'
    lcov --version | head -n1 | sed 's/^/      /'
    go version | head -n1 | sed 's/^/      /'
    upx --version | head -n1 | sed 's/^/      /'
}

apt_install() {
    echo "Running the apt-get install."

    if ! apt_sudo_exists; then
        apt_sudo_install
    fi
    if ! apt_make_exists; then
        apt_make_install
    fi
    if ! apt_autoconf_exists; then
        apt_autoconf_install
    fi
    if ! apt_libtool_exists; then
        apt_libtool_install
    fi
    if ! apt_cmake_exists; then
        apt_cmake_install
    fi
    if ! apt_lcov_exists; then
        apt_lcov_install
    fi
    if ! apt_go_exists; then
        apt_go_install
    fi
    if ! apt_upx_exists; then
        apt_upx_install
    fi


    if (( FAILED )); then
        echo "Installation failure detected."
	echo "Pertinent version info:"
	apt_dump_versions
	exit 1
    fi
    exit 0
}

if [ ${PLATFORM} = "Linux" ]; then
    if ! linux_determine_pkg_mgr; then
        echo "Exiting.  install_build_tools.sh does not support this package manager."
        exit 1
    else
        echo "Determined package manager is : ${PKG_MGR}."
    fi

    if [ ${PKG_MGR} = "yum" ]; then
        yum_install
    elif [ ${PKG_MGR} = "apt-get" ]; then
        apt_install
    fi
    exit 1
fi
