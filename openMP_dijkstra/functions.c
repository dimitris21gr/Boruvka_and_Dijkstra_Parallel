#include "header.h"

Graph *create_graph(char *input_file)
{
	Graph *gr = (Graph *)malloc(sizeof(Graph));
	FILE *fp = fopen(input_file, "r");
	unsigned int src, dest, weight, j = 0, sum, temp_src; 
	//error check
	fscanf(fp, "%u %u", &(gr->nodes), &(gr->edges));
	
	gr->destination = (unsigned int*)malloc(gr->edges * sizeof(unsigned int));
	gr->weight = (unsigned int*)malloc(gr->edges * sizeof(unsigned int));
	gr->first_edge = (unsigned int*)malloc(gr->nodes * sizeof(unsigned int));
	gr->out_degree = (unsigned int*)malloc(gr->nodes * sizeof(unsigned int));
	
	fscanf(fp, "%u %u %u", &src, &dest, &weight);
	temp_src = src;
	sum = 1;
	gr->destination[0]  = dest;
	gr->weight[0]  = weight;
	gr->first_edge[0] = 0;
	unsigned int i;
	
	for (i = 1; i < gr->edges; i++)
	{
		fscanf(fp, "%u %u %u", &src, &dest, &weight);
		gr->destination[i]  = dest;
		gr->weight[i]  = weight;
		
		if (temp_src == src)
			sum++;
		else
		{
			temp_src = src;
			gr->out_degree[j] = sum;
			j++;
			gr->first_edge[j] = i;
			sum = 1;
		}
		
	}
	gr->out_degree[j] = sum;	//store last node's degree
	
	fclose(fp);
	return gr;
}

void destroy_graph(Graph *gr)
{
	free(gr->destination);
	free(gr->weight);
	free(gr->first_edge);
	free(gr->out_degree);
	free(gr);
}
