add_executable(encode-qif encode-qif.c ../deps/xxhash/xxhash.c)
target_link_libraries(encode-qif PRIVATE ls-hpack)

add_executable(gen-fast-dec-table gen-fast-dec-table.c)

add_executable(gen-fast-enc-table gen-fast-enc-table.c)

add_executable(huff-decode huff-decode.c ../deps/xxhash/xxhash.c)
target_link_libraries(huff-decode PRIVATE ls-hpack)

add_executable(huff-encode huff-encode.c ../deps/xxhash/xxhash.c)
target_link_libraries(huff-encode PRIVATE ls-hpack)

add_executable(calc-xxh calc-xxh.c ../deps/xxhash/xxhash.c)

add_executable(find-xxh find-xxh.c ../deps/xxhash/xxhash.c)
