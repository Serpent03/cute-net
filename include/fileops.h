#include "common.h"
#include <stdio.h>

/**
 * @brief Write a section of data to file. After the data has been written, a magic
  integer is inserted into the file as well, to mark the end of the section.
 * @param data The buffer to write.
 * @param size The size of each element of the buffer.
 * @param count The amount of elements in the buffer.
 * @param fptr The FILE pointer directing to the file on the disk.
*/
void write_section(void *data, size_t size, size_t count, FILE *fptr);

/**
 * @brief Read a section of data from file.
 * @param size The size of each element in the buffer.
 * @param fptr The FILE pointer directing to the file on the disk.
 * @param cursor The location of the cursor currently in the open file.
 * @param length The length of the array. This is required to allocate memory to the returned buffer.
 * @return An allocated array containing the bytes read from the file.
*/
void *read_section(size_t size, FILE *fptr, uint32 *cursor, uint32 *length);
