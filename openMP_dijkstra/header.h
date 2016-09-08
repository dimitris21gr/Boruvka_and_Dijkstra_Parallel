#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>

#define THREADS 4


typedef struct Graph
{
	unsigned int *destination, *weight, *first_edge, *out_degree;
	unsigned int nodes, edges;
} Graph;

typedef struct Dijkstra
{
	unsigned int id, cost, prev;
	char flag;
} Dijkstra;

Graph *create_graph(char *input_file);
void destroy_graph(Graph *gr);
