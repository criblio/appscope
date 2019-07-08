#!/bin/bash

print_help() {
    echo "Usage Examples:"
    echo "    ./scope.sh platform      (returns Linux or macOS)"
    echo "    ./scope.sh pkg_mgr       (returns yum, apt-get, or brew)"
    echo "    ./scope.sh build         (clones, installs tools, and builds scope)"
    echo "    ./scope.sh help          (prints this message)"
}

determine_platform() {
    if uname -s 2> /dev/null | grep -i "linux" > /dev/null
    then
        echo "Linux"
    elif uname -s 2> /dev/null | grep -i darwin > /dev/null
    then
        echo "macOS"
    else
        echo "Error"
    fi
}

determine_pkg_mgr() {
    if yum --version &>/dev/null; then
        echo "yum"
    elif apt-get --version &>/dev/null; then
        echo "apt-get"
    elif brew --version &>/dev/null; then
        echo "brew"
    else
        echo "Error"
    fi
}

install_brew() {
    echo "Installing homebrew."
    # The initial "echo" is to pass the "hit enter to continue" prompt.
    echo | /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    if [ $? = 0 ]; then
        echo "Installation of homebrew successful."
    else
        echo "Installation of homebrew failed."
        return 1
    fi
}

platform_and_pkgmgr_defined() {
    if [ ${PLATFORM} = "Error" ]; then
        echo "scope.sh does not support this platform type.  Exiting."
        exit 1
    elif [ ${PKG_MGR} = "Error" ]; then
        echo "scope.sh does not support this package menager.  Exiting."
        exit 1
    else
        echo "Platform type is ${PLATFORM} with package manager ${PKG_MGR}."
    fi
    return 0
}

install_sudo() {
    case $PKG_MGR in
        "yum")
            if ! sudo --version &>/dev/null; then
                echo "Install of sudo is required."
                if su -c "yum install sudo"; then
                    echo "Install of sudo was successful."
                else
                    echo "Install of sudo was unsuccessful.  Exiting."
                    exit 1
                fi
            fi
            ;;
        "apt-get")
            if ! sudo --version &>/dev/null; then
                echo "Install of sudo is required."
                if su -c "apt-get update && apt-get install sudo"; then
                    echo "Install of sudo was successful."
                else
                    echo "Install of sudo was unsuccesful.  Exiting."
                    exit 1
                fi
            fi
            ;;
        "brew")
            ;;
    esac
    return 0
}

install_git() {
    case $PKG_MGR in
        "yum")
            if ! git --version &>/dev/null; then
                echo "Install of git is required."
                if sudo yum install git; then
                    echo "Install of git was successful."
                else
                    echo "Install of git was unsuccessful.  Exiting."
                    exit 1
                fi
            fi
            ;;
        "apt-get")
            if ! git --version &>/dev/null; then
                echo "Install of git is required."
                if sudo apt-get install git; then
                    echo "Install of git was successful."
                else
                    echo "Install of git was unsuccesful.  Exiting."
                    exit 1
                fi
            fi
            ;; 
        "brew")
            if ! git --version &>/dev/null; then
                echo "Install of git is required."
                if brew install git; then
                    echo "Install of git was successful."
                else
                    echo "Install of git was unsuccessful.  Exiting."
                    exit 1
                fi
            fi
            ;;
    esac
    return 0
}

clone_scope() {
    if git remote -v 2>/dev/null | grep "scope.git" &>/dev/null; then
        echo "We're in an existing repo now.  No need to clone."
        return 0 
    fi

    if [ -d "./scope" ]; then
        echo "switching to existing ./scope directory."
        cd ./scope;
        clone_scope
        return 0
    fi

    echo "Clone of scope.git is required."
    if git clone git@bitbucket.org:cribl/scope.git; then
        echo "Clone of scope.git was successful."
        cd scope
        #rm ../scope.sh
    else
        echo "Clone of scope.git was unsuccessful.  Exiting."
        exit 1
    fi
    return 0
}

install_build_tools() {
    if ./install_build_tools.sh; then
        echo "install_build_tools.sh was successful."
    else 
        echo "install_build_tools.sh was unsuccessful.  Exiting."
        exit 1
    fi
    return 0
}

run_make() {
    echo "Running make all."
    if make all; then
        echo "make all was successful."
    else 
        echo "make all failed.  Exiting."
        exit 1
    fi
    echo "Running make test."
    if make test; then
        echo "make test was successful."
    else
        echo "make test was unsuccessful.  Exiting."
        exit 1
    fi
    return 0
}


build_scope() {
    echo "Running scope.sh."

    PLATFORM=$(determine_platform)
    PKG_MGR=$(determine_pkg_mgr)

    # Mac's brew package manager may not be there unless we put it there...
    if [ ${PLATFORM} = "macOS" ] && [ ${PKG_MGR} = "Error" ]; then
        if install_brew; then
            PKG_MGR=$(determine_pkg_mgr)
        fi
    fi
    platform_and_pkgmgr_defined
    install_sudo
    install_git
    clone_scope
    install_build_tools
    run_make
}


if [ "$#" = 0 ]; then
    echo "Expected at least one argument.  Try './scope.sh help; for more info."
    exit 1
fi

for arg in "$@"; do 
    case $arg in
    "platform")
        determine_platform
        ;;
    "pkg_mgr")
        determine_pkg_mgr
        ;;
    "build")
        build_scope
        ;;
    "help")
        print_help
        ;;
    *)
        echo "Argument $arg is not expected.  Try './scope.sh help' for expected arguments."
        ;;
    esac
done

exit 0
