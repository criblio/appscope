#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define READ_ONLY_VAR  0
#define NOT_MAPPED_VAR 1
#define BUS_ERROR_VAR  2

void read_only_area_error(void) {
    char *s = "Hello goat";
    *s = 'a';
}

void not_mapped_address_error(void) {
   char *test = NULL;
   fprintf(stderr, "%s", test);
}

void bus_error(void) {
    int fd;
    int *map;
    int size = sizeof(int);
    char *name = "/a";
    shm_unlink(name);
    fd = shm_open(name, O_RDWR | O_CREAT, (mode_t)0600);
    map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    *map = 0;
}

void test_function(int variant) {
   switch (variant){
      case READ_ONLY_VAR:
         read_only_area_error();
         break;
      case NOT_MAPPED_VAR:
         not_mapped_address_error();
         break;
      case BUS_ERROR_VAR:
         bus_error();
         break;
      default:
         exit(EXIT_SUCCESS);
         break;
   }
}

int main(int argc, char *argv[])
{
   int variant;
   if (argc != 2) {
      return 0;
   }
   variant = atoi(argv[1]);
   test_function(variant);
   return 0;
}
