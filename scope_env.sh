#!/bin/bash

print_help() {
    echo "Usage Examples:"
    echo "    ./scope_env.sh run <cmd and args>     (executes cmd with scope library)"
    echo "    ./scope_env.sh version                (identifies the scope library version)"
    echo "    ./scope_env.sh platform               (returns Linux or macOS)"
    echo "    ./scope_env.sh pkg_mgr                (returns yum, apt-get, or brew)"
    echo "    ./scope_env.sh build                  (clones, installs tools, and builds scope)"
    echo "    ./scope_env.sh help                   (prints this message)"
}

determine_lib_path() {
    if [ -z "$SCOPE_HOME" ]; then
       export SCOPE_HOME=`pwd`
    fi

    PLATFORM=$(determine_platform)
    ARCH=$(determine_arch)

    if [ $PLATFORM == "macOS" ]; then
        echo "$SCOPE_HOME/lib/macOS/libscope.so"
    elif [ $PLATFORM == "Linux" ]; then
        echo "$SCOPE_HOME/lib/linux/$ARCH/libscope.so"
    else
        echo "ERROR"
    fi
}

run_scoped_cmd() {
    if [ -z "$SCOPE_HOME" ]; then
       export SCOPE_HOME=`pwd`
    fi

    PLATFORM=$(determine_platform)
    LIBRARY=$(determine_lib_path)

    if [ $PLATFORM == "macOS" ]; then
         DYLD_INSERT_LIBRARIES=$LIBRARY DYLD_FORCE_FLAT_NAMESPACE=1 "$@"
    elif [ $PLATFORM == "Linux" ]; then
         LD_PRELOAD=$LIBRARY "$@"
    else
        echo "ERROR: bad platform: $PLATFORM"
    fi

    exit $?
}

print_version() {
    local ok=1
    if ! strings --version &>/dev/null; then
        echo "Error: can't determine version without 'strings' command."
        ok=0
    fi
    if ! grep --version &>/dev/null; then
        echo "Error: can't determine version without 'grep' command."
        ok=0
    fi
    if ! sed --version &>/dev/null; then
        echo "Error: can't determine version without 'sed' command."
        ok=0
    fi

    if (( ! ok )); then
        exit 1
    fi

    LIBRARY=$(determine_lib_path)

    # sed deletes 'Constructor (' from the beginning, and ')' from the end
    strings $LIBRARY | grep "Constructor (Scope Version" | sed 's/.*(//' | sed 's/.$//'
    return $?
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

determine_arch() {
    echo $(uname -m)
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
    if [ "${PLATFORM}" = "Error" ]; then
        echo "scope_env.sh does not support this platform type.  Exiting."
        exit 1
    elif [ "${PKG_MGR}" = "Error" ]; then
        echo "scope_env.sh does not support this package menager.  Exiting."
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
                if su -c "apt-get update && apt-get install -y sudo"; then
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
                if sudo apt-get install -y git; then
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
        #rm ../scope_env.sh
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


prep() {
    echo "Running scope_env.sh prep"

    PLATFORM=$(determine_platform)
    PKG_MGR=$(determine_pkg_mgr)

    platform_and_pkgmgr_defined
    install_sudo
    install_git
    clone_scope
    install_build_tools
}

build_scope() {
    echo "Running scope_env.sh."

    PLATFORM=$(determine_platform)
    PKG_MGR=$(determine_pkg_mgr)

    # Mac's brew package manager may not be there unless we put it there...
    if [ "${PLATFORM}" = "macOS" ] && [ "${PKG_MGR}" = "Error" ]; then
        if install_brew; then
            PKG_MGR=$(determine_pkg_mgr)
        fi
    fi
    prep
    run_make
    print_version
}


if [ "$#" = 0 ]; then
    echo "Expected at least one argument.  Try './scope_env.sh help; for more info."
    exit 1
fi

for arg in "$@"; do 
    case $arg in
    "run")
        shift;
        run_scoped_cmd "$@"
        ;;
    "version")
        print_version
        ;;
    "platform")
        determine_platform
        ;;
    "pkg_mgr")
        determine_pkg_mgr
        ;;
    "build")
        build_scope
        ;;
    "prep")
        prep
        ;;
    "help")
        print_help
        ;;
    *)
        echo "Argument $arg is not expected.  Try './scope_env.sh help' for expected arguments."
        ;;
    esac
done

exit 0
