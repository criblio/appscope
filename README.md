**Cribl Scope**

# Introduction
Scope is part of the Deep Collect project.  Scope can be used to extract details from within a running process.  Information is extracted by loading a shared library into the address space of a process as the process is started. The library is loaded by the dynamic linker as part of the normal process of loading libraries in response to an exec system call in Linux/Unix/macOS, CreateProcess in Windows.  The mechanism to configure the loader to load our library varies with each OS. In Linux we start with LD_PRELOAD. In macOS we start with DYLD_INSERT_LIBRARIES and DYLD_FORCE_FLAT_NAMESPACE. In Windows we start with AppInitDLL.

# Requirements
In order to develop code for Scope you will need either a Linux, macOS, or Windows environment, and access to the scope.git repository.

# Install and Build
## macOS, Linux
(Only linux distros with yum or apt-get package managers are supported)

Run these commands to do everything: checkout the scope codebase, install necessary build tools, build, and run tests.

    git clone git@bitbucket.org:cribl/scope.git
    cd scope
    ./scope.sh build

`./scope.sh build` is a good command to know, it does everything needed to get a brand new environment up and running, but can also be run on an existing environment to rebuild, and rerun tests.  If desired, individual parts of `./scope.sh build` can be run individually from the scope directory.

    ./install_build_tools.sh
    make all
    make test

And finally, to clean out files that were manually added or the result of running build commands:

    git clean -f -d


## Windows
TBD

# Run
In order to run any given process in a mode where detailed information about the process can be extraced, without changes to code or the executable, you simply need to run the process with one or more environment variables set. You can set the envirnment variables defined below as needed.  Or, just start the executable with the Scope wrapper. Here's how: From the top level Scope directory execute the command:

    bin/scopeRun your_command params
Where "your_command" is any executable and "params" are any parameters needed for the executable.  
 
# Developer notes
- The top-level Makefile checks to see what platform you are on and then includes the specific Makefile for the platform from the OS specific directory.
- The run script will determine which platform you are using and will set environment variables accordingly. 

# Environment variables 
## CRIBL_HOME
This envirinment variable defines the base for a Scope install. For a development environment it should be set to the top level dir in the repository.
This variable is set by the run command if a definition does not exist or if it is null. 

## macOS
DYLD_INSERT_LIBRARIES
DYLD_FORCE_FLAT_NAMESPACE
These are the variables that are needed in order for the Scope library to be loaded and enable functions in new processes to be interposed on macOS.

## Linux
LD_PRELOAD
This is the variable that is needed in order for the Scope library to be loaded and enable functions in new processes to be interposed on Linux.

