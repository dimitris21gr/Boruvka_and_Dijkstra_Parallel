#include "header.h"
#include "boruvka_header.h"

__global__
void find_min(Graph *gr, unsigned int *p)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if(v >= gr->nodes) return;
	unsigned int degree = gr->out_degree[v];
	unsigned int edge = gr->first_edge[v];
	unsigned int min_weight = gr->weight[edge];
	unsigned int min_edge = edge;
	for (unsigned int i = 1; i < degree; i++)
	{
		if (gr->weight[edge + i] < min_weight)
		{
			min_weight = gr->weight[edge + i];
			min_edge = edge + i;
		}
		else if (gr->weight[edge + i] == min_weight)
		{
			if (gr->destination[edge + i] < gr->destination[min_edge])
				min_edge = edge + i;
		}
	}
	p[v] =  min_edge;
}

__global__
void mirrors_edge(Graph *gr, unsigned int *p)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if(v >= gr->nodes) return;
	if (p[v] == gr->edges) return;
	unsigned int successorA = gr->destination[p[v]];			//vertex successor
	if (p[successorA] == gr->edges) return;
	unsigned int successorB = gr->destination[p[successorA]];	//successor of successor
	if (successorB == v && v > successorA)	//mirrors edges check
			p[successorA] = gr->edges;	//invalid value that defines a mirrors edge
}

__global__
void initialize_colors(Graph *gr, unsigned int *m, unsigned int *c)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if(v >= gr->nodes) return;
	if (m[v] == gr->edges)
		c[v] = v;
	else
		c[v] = gr->destination[m[v]];
}

__global__
void propagate_colors(Graph *gr, unsigned int *color, unsigned int *change)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if(v >= gr->nodes) return;
	unsigned int current = color[v];
	unsigned int other = color[current];
	if (current != other)
	{
		color[v] = other;
		*change = 1;
	}
}

__global__
void create_new_vertex_ids(Graph *gr, unsigned int *color, unsigned int *flag)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if(v >= gr->nodes) return;
	if (color[v] == v)
		flag[v] = 1;
	else
		flag[v] = 0;
}

__global__
void count_edges(Graph *gr, Graph *gr2, unsigned int *color, unsigned int *EPS)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if(v >= gr->nodes) return;
	unsigned int sum = 0, start = gr->first_edge[v], last = gr->out_degree[v], my_color = color[v];
	for (unsigned int i = start; i < start + last; i++)
	{
		if (color[gr->destination[i]] != my_color)
			sum++;
	}
	atomicAdd(&(gr2->out_degree[EPS[my_color]]), sum);
}

__global__
void insert_new_edges(Graph *gr, Graph *gr2, unsigned int *color, unsigned int *first_edge_copy, unsigned int *EPS)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if(v >= gr->nodes) return;
	unsigned int start = gr->first_edge[v], last = gr->out_degree[v], my_color = color[v], super_vertex_id = EPS[my_color];
	for (unsigned int i = start; i < start + last; i++)
	{
		unsigned int other_color = color[gr->destination[i]];
		if (other_color != my_color)
		{
			unsigned int edge_id = atomicInc(&(first_edge_copy[super_vertex_id]), UINT_MAX);
			gr2->destination[edge_id] = EPS[other_color];
			gr2->weight[edge_id] = gr->weight[i];
		}
			
	}
}

__global__
void test(Graph *gr, int flag)
{	
	// int stop = gr->nodes;
	// if (flag == 1)
	// {
	// 	for (unsigned int i = 0; i < stop; i++)
	// 		printf("kernel %u\n", gr->out_degree[i]);
	// }
	printf("nodes: %u\n", gr->nodes);
		
}
