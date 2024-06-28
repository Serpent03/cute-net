cc = gcc
ccflags = -Wall
sources = main.c nn.c
bash = bash -c

main: ${sources}
	${cc} ${ccflags} -o $@ $(sources)

clean:
	${bash} "rm -rf ./main"