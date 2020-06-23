
CWD=$(shell pwd)
PLATFORM=$(shell "$(CWD)"/scope_env.sh platform)

ifeq ($(PLATFORM),Linux)
	include os/linux/Makefile
else
	ifeq ($(PLATFORM),macOS)
		include os/macOS/Makefile
	else
$(info ERROR not a valid platform: "$(PLATFORM)")
endif
endif


