#include "../include/fileops.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

uint8 magic[4] = { 0x55, 0xAA, 0x00, 0x00 };
uint8 cmp[4]; /* we'll use this to compare the magic number if we find it. */

void write_section(void *data, size_t size, size_t count, FILE *fptr) {
  fwrite(data, size, count, fptr);
  fwrite(&magic, sizeof(uint8), 4, fptr); /* end of section demarcation */
}

void *read_section(void *buffer, size_t size, FILE *fptr, uint32 *cursor, uint32 *length) {
  uint32 old_cursor = *cursor;
  fseek(fptr, (*cursor) * 4, SEEK_SET);
  /* parse over until we find 0xAA55000 or 0x000055AA in the file, and then
  initialize an array and return it. */

  while (memcmp(cmp, magic, sizeof(magic)) != 0) {
    /* the length of the section is determined by difference in the starting position of the
    cursor(old_cursor) and when it finally reaches a 4-byte block of value 55|AA|00|00 */
    fread(cmp, sizeof(uint8), 4, fptr);
    (*cursor)++;
  }
  cmp[0] = 0; /* reset the cmp array so that on the next run it will again try to find end of section. */
  uint32 size_of_section = (((*cursor) - 1) - old_cursor) * 4;
  *length = size_of_section / size;
  /* the difference in the cursor position gives us the number of bytes that a specific section
  encases. after we get this, we can get the size of the buffer itself by dividing that with the
  <<size>> of the data, which we are conveniently providing! */

  buffer = (void*)malloc((*length) * size); 
  fseek(fptr, - (size_of_section + 4), SEEK_CUR); /* go to start of section */
  fread(buffer, size, *length, fptr); /* copy data */
  fseek(fptr, 4, SEEK_CUR); /* go past the end of section */

  return buffer;
}