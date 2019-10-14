tests := fclose fopen fwrite
list := $(foreach wrd,$(tests),$(wrd))

all:
	@for dir in $(list); do \
	    make -C $$dir $@; \
	done
	
clean:
	@for dir in $(list); do \
	    make -C $$dir $@; \
	done

