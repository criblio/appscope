#!/bin/bash
#
# Perform a check if in contrib code has not translated symbols.
#

# List of symbols which are ignored in replacement process
declare -a stdlib_ignore_syms=('/_GLOBAL_OFFSET_TABLE_/d'
'/^\./d'
'/^sha256_init/d'
'/^sha256_update/d'
'/^sha512_init/d'
'/^sha512_update/d'
'/^md5_init/d'
'/^md5_update/d'
'/^default_malloc/d'
'/^BF_set_key/d'
'/^BF_encrypt/d'
'/^getenv/d'
'/^secure_getenv/d'
'/^signal/d'
'/^sigaction/d'
'/^syscall/d'
'/^pthread_/d'
'/^dlopen/d'
'/^dladdr/d'
'/^dlclose/d'
'/^dlerror/d'
'/^dlsym/d'
)

# List of contrib libraries used by the libscope.so
declare -a conrib_libs=(
"./contrib/cJSON/libcjson.a" 
"./contrib/build/funchook/libfunchook.a"
"./contrib/build/funchook/capstone_src-prefix/src/capstone_src-build/libcapstone.a"
"./contrib/build/libyaml/src/.libs/libyaml.a"
"./contrib/build/openssl/libcrypto.a"
"./contrib/build/openssl/libssl.a"
"./contrib/build/pcre2/libpcre2-8.a"
"./contrib/build/ls-hpack/libls-hpack.a"
"./contrib/build/libunwind/src/.libs/libunwind.a"
"./contrib/build/musl/lib/libc_orig.a" ## must be last
)


#######################################
# Translate absolute path to file name: 
# "symbols<lib_name>.txt"
# Arguments:
#   Absolute path to the file (library)
# Outputs:
#   Writes name of the file name to stdout
#######################################
path_to_output_file () {
    local lib_path=$1

    echo symbols_"$(basename "$lib_path" .a)".txt
}


#######################################
# Extract symbols from the library and save in separate file
# Arguments:
#   Absolute path to the file (library)
#######################################
extract_symbols () {
    local lib_path=$1
    local output_file

    output_file="$(path_to_output_file "$lib_path")"
    nm "$lib_path" | awk 'NF{print $NF}' | sort | uniq > "$output_file"
    for ignore_sym in "${stdlib_ignore_syms[@]}"
    do
        sed -i "$ignore_sym" "$output_file"
    done
}

#######################################
# Print common symbols
# Arguments:
#   Absolute path to the file (library)
# Outputs:
#   Writes the common symbols to stdout
#######################################
print_common_symbols () {
    local contrib_lib
    local stdlib

    contrib_lib=$(path_to_output_file "$1")
    stdlib=$(path_to_output_file "$2")
    comm -12 "$contrib_lib" "$stdlib"
}

# Extract symbols from all static libraries
for lib in "${conrib_libs[@]}"
do
   extract_symbols "$lib"
done

# Print symbols which should be replaced
for ((i = 0; i < ${#conrib_libs[@]}-1; ++i)); do
    print_common_symbols "${conrib_libs[$i]}" "${conrib_libs[-1]}"
done
