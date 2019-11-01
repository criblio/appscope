list := $(filter-out Makefile,$(wildcard *))

all:
	@for dir in $(list); do \
		make -C $$dir $@; \
	done
	
clean:
	@for dir in $(list); do \
		make -C $$dir $@; \
	done

