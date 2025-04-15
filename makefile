source = src/*.c
lib = -lm
include = include/*.h
build = build/

bash = bash -c 
cc = gcc
ccflags = -Wall

main: $(source)
	$(cc) -O3 $(ccflags) -o $(build)$@ $^ $(lib)

debug: $(source)
	$(cc) -g $(ccflags) -o $(build)$@ $^ $(lib)

run:
	$(build)main

clean:
	$(bash) "rm -rf ./build/*"
