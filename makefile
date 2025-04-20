SRC_DIR = src
BLD_DIR = build
INC_DIR = include

C_SOURCES = $(wildcard $(SRC_DIR)/*.c)
C_INCLUDES = $(wildcard $(INC_DIR)/*.h)
OBJ_SOURCES = $(patsubst %.c, %.o, $(C_SOURCES))
LIBS = -lm

bash = bash -c 
cc = gcc
ccflags = -Wall -O2

all: cn

cn: $(OBJ_SOURCES)
	ar rcs $(BLD_DIR)/libcutenet.a $(BLD_DIR)/*.o
	rm -rf $(BLD_DIR)/*.o
	
cn_debug: $(source)
	$(cc) -g $(ccflags) -c $(build)$@ $^ $(lib)

example: cn
	$(cc) $(ccflags) -o $(BLD_DIR)/example main.c -L. $(BLD_DIR)/libcutenet.a $(LIBS)

clean:
	$(bash) "rm -rf ./build/*"

# generic compilation rule for converting .c => .o
%.o: %.c
	$(cc) $(ccflags) -c $< -o $@
	mv $@ $(BLD_DIR)/
