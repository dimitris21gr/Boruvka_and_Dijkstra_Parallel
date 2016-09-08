#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

#define THREADS 4

typedef struct Graph
{
	unsigned int *destination, *weight, *first_edge, *out_degree;
	unsigned int nodes, edges;
} Graph;

Graph *create_graph(char *input_file);
void destroy_graph(Graph *gr);


void find_min(Graph *, unsigned int*, unsigned int);
void mirrors_edge(Graph *, unsigned int*, unsigned int);
void count_edges(Graph *, Graph *, unsigned int *, unsigned int *, unsigned int);
void insert_new_edges(Graph *, Graph *, unsigned int *, unsigned int *, unsigned int *, unsigned int);
void prefix_sum(unsigned int *, unsigned int *, unsigned int);
