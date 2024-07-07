#include "common.h"
#include <stdio.h>

void write_section(void *data, size_t size, size_t count, FILE *fptr);

void *read_section(void *buffer, size_t size, FILE *fptr, uint32 *cursor, uint32 *length);