source = src/*.c
lib = lib/*.o
include = include/*.h
build = build/

bash = bash -c
cc = gcc
ccflags = -Wall

main: ${source}
	${cc} ${ccflags} -o $(build)$@ $^

run:
	$(build)main

clean:
	${bash} "rm -rf ./build/main"