
CWD=$(shell pwd)
PLATFORM=$(shell "$(CWD)"/scope_env.sh platform)

ifeq ($(PLATFORM),Linux)
	include os/linux/Makefile
	include cli/Makefile
else
	ifeq ($(PLATFORM),macOS)
		include os/macOS/Makefile
	else
$(info ERROR not a valid platform: "$(PLATFORM)")
endif
endif

ifeq ($(PLATFORM),Linux)
clean:
	$(MAKE) -f os/linux/Makefile core$@
	$(MAKE) -f cli/Makefile cli$@

test:
	$(MAKE) -f os/linux/Makefile core$@
	$(MAKE) -f cli/Makefile cli$@
endif
