all:
	@echo "TODO: ..."

clean:
	@$(MAKE) -C cJSON clean
	@$(RM) -r cmocka/build
	@$(RM) -r funchook/build
	@$(MAKE) -C libyaml clean
	@$(MAKE) -C openssl/ clean
	@$(RM) -r pcre2/build

.PHONY: all clean
