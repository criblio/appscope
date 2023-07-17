# Adding new libary

Below are short description of required steps to extend AppScope with a new library.

- The new added library must be build as a statically-linked library.
- Please add new library to the `contrib` directory, as an example see following [change](https://github.com/criblio/appscope/commit/99f2f249e9dfd2d325d58bacae03a8882ec73a07) for details.
- Extend the AppScope build system (Makefile) with new library and extend the `extract_symbols.sh` with the new library, as an example see following [change](https://github.com/criblio/appscope/commit/72dfc4abe914a3e7a10a680792e7209cacd124de) for details.
- Extend the GitHub Action caching layer with new library, as an example see following [change](https://github.com/criblio/appscope/commit/f405c947e4f1bcf6f9bcf21276569249fe5f3ebd) for details.
- To determine if additional symbols should be added in symbol redefinition please run `extract_symbols.sh` script before and after adding new library comparing outputs and if necessary extend the list of symbols to redefine (`redefine_syms.lst`), as an example see following [change](https://github.com/criblio/appscope/commit/4cc38f23cddfd8ca86e5686a6d929317bee67b6d) for details.
