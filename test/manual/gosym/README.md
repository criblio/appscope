The intention of this setup is to retrieve information useful for adding support for new Go version.

Action performed by the script:

- validate presence of supported pclntab in example go binary
- generate disassmbly for http2 functions

TODO:
- retrieve information about offsets both cases
- compare offsets between Go versions?

### start.sh

Usage: 

```
./start.sh <GO_VERSION> [bash]
```

<!-- Will generate data for Go 1.20 -->
```bash
./start.sh 1.20 
```
<!-- Will allows you to login into container -->
```bash
./start.sh 1.20 bash
```
### cleanup.sh
Will cleanup the generated files