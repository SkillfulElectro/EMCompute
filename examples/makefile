test: cleanup
	@gcc main.c libEMCompute.so EMCompute.h -Wl,-rpath=. -o test; \
	echo "Start time: $$(date)"; \
	./test; \
	echo "End time: $$(date)"

cleanup:
	@rm -f test

