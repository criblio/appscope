#ifndef __COMMON_H__
#define __COMMON_H__

#ifdef __APPLE__                                                                                                                                                                
#ifndef off64_t                                                                                                                                                                 
typedef uint64_t off64_t;                                                                                                                                                       
#endif                                                                                                                                                                          
#ifndef fpos64_t                                                                                                                                                                
typedef uint64_t fpos64_t;                                                                                                                                                      
#endif                                                                                                                                                                          
#ifndef statvfs64                                                                                                                                                               
struct statvfs64 {                                                                                                                                                            
    uint64_t x;                                                                                                                                                                 
};                                                                                                                                                                              
#endif                                                                                                                                                                          
#endif // __APPLE__  

#define TEST_MSG "test"
#define TEST_MSGW L"test"
#define TEST_MSG_N "test\n"

#define TEST_CHAR 'A'
#define TEST_CHARW L'А'

#define TEST_COUNT 100
#define SEND_MSG_COUNT 100

int do_test();

#endif /* __COMMON_H__ */