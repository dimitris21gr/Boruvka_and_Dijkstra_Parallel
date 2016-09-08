#include "header.h"

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		printf("Input file not found\n");
		return 0;
	}
	Graph *gr = create_graph(argv[1]);
	printf("File Loaded!\n");

	unsigned int startNode, currentNodeOutdegree, currentNode;
	
	printf("Set start node id (1-%u): ", gr->nodes);
	scanf("%u", &startNode);
	startNode--;
	Dijkstra *dijkstra;

	if ( (dijkstra = malloc(gr->nodes * sizeof(Dijkstra) ) ) == NULL )
	{
		perror("malloc");
		destroy_graph(gr);
		return 1;
	}

	printf("Running Dijkstra parallel on %d CPU threads...\n", THREADS);

	unsigned int v;
	//init
	#pragma omp parallel for num_threads(THREADS)
	for ( v = 0; v < gr->nodes; v++ )
	{
		if (v == startNode)
		{
			dijkstra[v].flag = 1;
			dijkstra[v].prev = startNode;
			dijkstra[v].cost = 0;
		}
		else
		{
			dijkstra[v].flag = 0;
			dijkstra[v].cost = UINT_MAX;
		}
		dijkstra[v].id = v;
	}

	currentNodeOutdegree = gr->out_degree[startNode];
	currentNode = startNode;
	unsigned int start_edge, edge, dest, weight;

	struct timeval start, end;
	gettimeofday(&start, NULL);
	while(1)
	{
		//update
		#pragma omp parallel for private(start_edge, edge, dest, weight) num_threads(THREADS)
		for ( v = 0; v < currentNodeOutdegree; v++ )
		{
			start_edge = gr->first_edge[currentNode];
			edge  = start_edge + v;
			dest = gr->destination[edge];
			weight = gr->weight[edge];

			if (dijkstra[dest].flag == 0)
			{
				if (dijkstra[dest].cost == UINT_MAX)	//if cost is infinite
				{
					dijkstra[dest].cost = dijkstra[currentNode].cost + weight;
					dijkstra[dest].prev = currentNode;
				}
				else
				{
					if (dijkstra[currentNode].cost + weight < dijkstra[dest].cost)
					{
						dijkstra[dest].cost = dijkstra[currentNode].cost + weight;
						dijkstra[dest].prev = currentNode;
					}
				}
				
			}
		}

		//find min
		unsigned int minVal = UINT_MAX, min_id = gr->nodes;
		#pragma omp parallel for reduction(min: minVal) num_threads(THREADS)
		for ( v = 0; v < gr->nodes; v++ )
		{
			if (dijkstra[v].flag == 0)
			{
				if ( dijkstra[v].cost < minVal )
				{
					minVal = dijkstra[v].cost;
					min_id = v;
				}
			}
		}
		
		if (minVal == UINT_MAX)
			break;

		currentNode = min_id;
		dijkstra[currentNode].flag = 1;
		currentNodeOutdegree = gr->out_degree[currentNode];
	}
	gettimeofday(&end, NULL);
	double delta = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / (double)1000000;
	printf("Time: %.4fs\n", delta);

	// unsigned int i;
	// for (i = 0; i < gr->nodes; i++)
	// 	printf("%u ", dijkstra[i].cost);
	// printf("\n");
	// for (i = 0; i < gr->nodes; i++)
	// 	printf("%u ", dijkstra[i].flag);
	// printf("\n");
	// for (i = 0; i < gr->nodes; i++)
	// 	printf("%u ", dijkstra[i].prev);
	// printf("\n");

	free(dijkstra);
	destroy_graph(gr);
}
