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
	Graph *gr2;
	unsigned int *minedge, *color, *flag, *EPS, *first_edge_copy, *temp;
	unsigned int change, s1, s2, i;

	printf("Running parallel Boruvka on %d CPU threads...\n", THREADS);
	
	struct timeval start, end;
	gettimeofday(&start, NULL);
	do
	{
		minedge = malloc(gr->nodes * sizeof(unsigned int));
		color = malloc(gr->nodes * sizeof(unsigned int));
		flag = malloc(gr->nodes * sizeof(unsigned int));
		EPS = malloc(gr->nodes * sizeof(unsigned int));
		
		#pragma omp parralel for num_threads(THREADS)
		for(i = 0; i < gr->nodes; i++)
		{
			find_min(gr, minedge, i);
		}

		#pragma omp parallel for num_threads(THREADS)
		for(i = 0; i < gr->nodes; i++)
		{
			mirrors_edge(gr, minedge, i);
		}

		#pragma omp parallel for num_threads(THREADS)
		for(i = 0; i < gr->nodes; i++)		//initialize colors
		{
			unsigned int edge = minedge[i];
			if(edge == gr->edges) color[i] = i;
			else color[i] = gr->destination[edge];
		}

		do{
			change = 0;		//propagate colors

			#pragma omp parallel private(i) num_threads(THREADS)
			{
				unsigned int my_changed = 0;

				#pragma omp for
				for(i = 0; i < gr->nodes; i++)
				{
					unsigned int my_color = color[i];
					unsigned int other_color = color[my_color];

					if(my_color != other_color)
					{
						color[i] = other_color;
						my_changed = 1;
					}				
				}

				if(my_changed) change = 1;	
			}

		} while(change);

		flag = memset(flag, 0, gr->nodes * sizeof(unsigned int));
		#pragma omp parallel for num_threads(THREADS)
		for(i = 0; i < gr->nodes; i++)		//create super vertex ids
		{
			if (color[i] == i && gr->out_degree[i] > 0)
				flag[i] = 1;
		}

		temp = malloc(gr->nodes * sizeof(unsigned int));
		memcpy(temp, flag, gr->nodes * sizeof(unsigned int));
		prefix_sum(EPS, temp, gr->nodes);

		memcpy(&s1, &EPS[gr->nodes-1], sizeof(unsigned int));

		printf("nodes: %u\n", gr->nodes);
		if (s1 <= 1)
		{
			break;
		}

		gr2 = malloc(sizeof(Graph));
		gr2->nodes = s1;
		gr2->first_edge = malloc(gr2->nodes * sizeof(unsigned int));
		gr2->out_degree = malloc(gr2->nodes * sizeof(unsigned int));
		memset(&gr2->out_degree[0], 0, gr2->nodes * sizeof(unsigned int));
		
		#pragma omp parallel for num_threads(THREADS)
		for(i = 0; i < gr->nodes; i++)
		{
			count_edges(gr, gr2, color, EPS, i);
		}
		
		temp = malloc(gr2->nodes * sizeof(unsigned int));
		memcpy(temp, gr2->out_degree, gr2->nodes * sizeof(unsigned int));
		prefix_sum(gr2->first_edge, temp, gr2->nodes);

		memcpy(&s1, &gr2->out_degree[gr2->nodes-1], sizeof(unsigned int));
		memcpy(&s2, &gr2->first_edge[gr2->nodes-1], sizeof(unsigned int));
		gr2->edges = s1 + s2;
		gr2->destination = malloc(gr2->edges * sizeof(unsigned int));
		gr2->weight = malloc(gr2->edges * sizeof(unsigned int));
		
		first_edge_copy = malloc(gr2->nodes * sizeof(unsigned int));
		memcpy(&first_edge_copy[0], &gr2->first_edge[0], gr2->nodes * sizeof(unsigned int));
		
		#pragma omp parallel for num_threads(THREADS)
		for(i = 0; i < gr->nodes; i++)
		{
			insert_new_edges(gr, gr2, color, first_edge_copy, EPS, i);
		}

		free(first_edge_copy);
		free(minedge);
		free(color);
		free(flag);
		free(EPS);

		//swap start
		destroy_graph(gr);
		gr = gr2;
		//swap end
	} while(1);

	gettimeofday(&end, NULL);
	double delta = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / (double)1000000;
	printf("Time: %.4fs\n", delta);
	
	free(minedge);
	free(color);
	free(flag);
	free(EPS);
	
	destroy_graph(gr);
}
