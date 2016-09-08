#include "header.h"
#include "dijkstra_header.h"

__global__ 
void init(Dijkstra *dijkstra, Graph *gr, unsigned int *startNode)
{
	unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
	if(v >= gr->nodes) return;
	if (v == *startNode)
	{
		dijkstra[v].flag = 1;
		dijkstra[v].prev = *startNode;
		dijkstra[v].cost = 0;
	}
	else
	{
		dijkstra[v].flag = 0;
		dijkstra[v].cost = UINT_MAX;
	}
	dijkstra[v].id = v;
}

__global__ 
void update(Graph *gr, Dijkstra *dijkstra, unsigned int *currentNode)
{
	unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
	if(v >= gr->out_degree[*currentNode]) return;
	unsigned int start_edge = gr->first_edge[*currentNode];
	unsigned int edge  = start_edge + v;
	unsigned int dest = gr->destination[edge];
	unsigned int weight = gr->weight[edge];

	if (dijkstra[dest].flag == 0)
	{
		if (dijkstra[dest].cost == UINT_MAX)	//if cost is infinite
		{
			dijkstra[dest].cost = dijkstra[*currentNode].cost + weight;
			dijkstra[dest].prev = *currentNode;
		}
		else
		{
			if (dijkstra[*currentNode].cost + weight < dijkstra[dest].cost)
			{
				dijkstra[dest].cost = dijkstra[*currentNode].cost + weight;
				dijkstra[dest].prev = *currentNode;
			}
		}
		
	}
}

__global__ 
void test(Dijkstra *dijkstra, unsigned int size)
{
	unsigned int i;
	for (i = 0; i < size; i++)
		printf("%u ", dijkstra[i].cost);
	printf("\n");
	for (i = 0; i < size; i++)
		printf("%u ", dijkstra[i].flag);
	printf("\n");
	for (i = 0; i < size; i++)
		printf("%u ", dijkstra[i].prev);
	printf("\n");
}
