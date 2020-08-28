.PHONY: cleanDynamic cleanStatic

all: cgoStatic cgoDynamic

cleanDynamic:
	rm -f libmyc.so myc.o cgoDynamic

cleanStatic:
	rm -f libmyc.a myc.o cgoStatic

cgoDynamic:
	make cleanDynamic
	gcc -c -fpic myc.c
	gcc -shared -o libmyc.so myc.o
	go build -o cgoDynamic mygo.go
	#LD_LIBRARY_PATH=. ./cgoDynamic

cgoStatic:
	make cleanStatic
	gcc -c -fPIC myc.c
	ar rcs libmyc.a myc.o
	go build -ldflags "-linkmode external -extldflags -static" -o cgoStatic mygo.go
	#./cgoStatic
