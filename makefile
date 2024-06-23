cc = gcc
ccflags = -Wall
sources = main.c nn.c
bash = bash -c

main: ${sources}
	${cc} ${ccflags} -o $@ $^

clean:
	${bash} "rm -rf ./main"