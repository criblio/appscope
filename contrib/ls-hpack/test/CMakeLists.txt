ENABLE_TESTING()

FOREACH(HASH 0 1)
    FOREACH(HTTP 0 1)
        ADD_EXECUTABLE(test_hpack_hash${HASH}_http${HTTP} test_hpack.c ../lshpack.c ../deps/xxhash/xxhash.c)
        SET_TARGET_PROPERTIES(test_hpack_hash${HASH}_http${HTTP}
            PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -DLSHPACK_DEC_HTTP1X_OUTPUT=${HTTP} -DLSHPACK_DEC_CALC_HASH=${HASH}")
        ADD_TEST(hpack-hash${HASH}-http${HTTP} test_hpack_hash${HASH}_http${HTTP})
    ENDFOREACH(HTTP)
ENDFOREACH(HASH)

ADD_EXECUTABLE(test_int test_int.c ../deps/xxhash/xxhash.c)
TARGET_LINK_LIBRARIES(test_int ls-hpack)
ADD_TEST(int test_int)
