#include "../include/fileops.h"
#include <stdio.h>

void write_new_line(void *data, size_t size, size_t count, FILE *fptr) {
  uint32 magic = 0x55AA;
  fwrite(data, size, count, fptr);
  fwrite(&magic, sizeof(uint32), 1, fptr); /* append a new line */
}

void read_new_line(void *buffer, size_t size, size_t count, FILE *fptr) {

}