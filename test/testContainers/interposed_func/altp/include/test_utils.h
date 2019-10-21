

#define CREATE_TMP_DIR() \
    char tmp_dir_template[] = "/tmp/tmpdir.XXXXXX"; \
    char *tmp_dir_name = mkdtemp(tmp_dir_template); \
    if(tmp_dir_name == NULL) { \
        perror("mkdtemp failed: "); \
        return EXIT_FAILURE; \
    } 
    
#define REMOVE_TMP_DIR() \
    if(rmdir(tmp_dir_name) == -1) { \
        perror("rmdir failed: "); \
        return EXIT_FAILURE; \
    } 

