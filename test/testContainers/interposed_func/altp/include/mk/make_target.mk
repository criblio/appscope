test_list := $(foreach wrd,$(TESTS),$(wrd))

all:
	@for test in $(test_list); do \
		gcc -Wall -Wextra -I../../include ../../common/main.c $$test.c -o $$test $(AFLAGS); \
	done
clean:
	@rm -vf *.o
	@for test in $(test_list); do \
		rm -vf $$test; \
	done
