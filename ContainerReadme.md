# Purpose
​
The cribl/scope container is the Docker-based distribution of Cribl's AppScope product. For details about AppScope, see [our documentation](https://appscope.dev/docs).
​

AppScope is licensed under the ​[Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0).
​
​
# Run
​
To run the container, enter:
​

`docker run cribl/scope:<tag>`

example:

`docker run -it cribl/scope:0.5.1 bash`

The scope product is installed in `/usr/local/bin`. You can start exploring your applications using the scope executable. Enter `scope` with no arguments to see help or use `scope --help`. To see detailed help for a given argument use `scope help <argument>`.

# Tagging
​
The current release of scope is 0.5.1 and `:0.5.1` is the official tag for this release.
