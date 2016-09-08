#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cuda.h>
#include "cub/cub.cuh"
#include "cub/device/device_radix_sort.cuh"

#define BLOCK_SIZE 1024
#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/denominator)

typedef struct Graph
{
	unsigned int *destination, *weight, *first_edge, *out_degree;
	unsigned int nodes, edges;
} Graph;

Graph *create_graph(char *input_file);
void destroy_graph(Graph *gr);
