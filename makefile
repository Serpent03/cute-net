source = src/*.c
lib = -lm
include = include/*.h
build = build/

bash = bash -c
cc = gcc
ccflags = -Wall

main: $(source)
	$(cc) $(ccflags) -o $(build)$@ $^ $(lib)

run:
	$(build)main

clean:
	$(bash) "rm -rf ./build/main"
