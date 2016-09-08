#include "header.h"

void find_min(Graph *gr, unsigned int *p, unsigned int v)
{
	unsigned int degree = gr->out_degree[v];
	unsigned int edge = gr->first_edge[v];
	unsigned int min_weight = gr->weight[edge];
	unsigned int min_edge = edge;
	unsigned int i;
	for (i = 1; i < degree; i++)
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

void mirrors_edge(Graph *gr, unsigned int *p, unsigned int v)
{
	if (p[v] == gr->edges)
		return;
	unsigned int successorA = gr->destination[p[v]];			//vertex successor
	if (p[successorA] == gr->edges)
		return;
	unsigned int successorB = gr->destination[p[successorA]];	//successor of successor
	if (successorB == v && v > successorA)	//mirrors edges check
	{
			p[successorA] = gr->edges;	//invalid value that defines a mirrors edge
	}
}

void count_edges(Graph *gr, Graph *gr2, unsigned int *color, unsigned int *EPS, unsigned int v)
{
	
	unsigned int i, sum = 0, start = gr->first_edge[v], last = gr->out_degree[v], my_color = color[v];
	for (i = start; i < start + last; i++)
	{
		if (color[gr->destination[i]] != my_color)
			sum++;
	}
	__atomic_fetch_add(&(gr2->out_degree[EPS[my_color]]), sum, __ATOMIC_SEQ_CST);
}

void insert_new_edges(Graph *gr, Graph *gr2, unsigned int *color, unsigned int *first_edge_copy, unsigned int *EPS, unsigned int v)
{
	unsigned int i, start = gr->first_edge[v], last = gr->out_degree[v], my_color = color[v], super_vertex_id = EPS[my_color];
	unsigned int edge_id, other_color;
	for (i = start; i < start + last; i++)
	{
		other_color = color[gr->destination[i]];
		if (other_color != my_color)
		{
			edge_id = __atomic_fetch_add(&(first_edge_copy[super_vertex_id]), 1, __ATOMIC_SEQ_CST);
			gr2->destination[edge_id] = EPS[other_color];
			gr2->weight[edge_id] = gr->weight[i];
		}	
	}
}

void prefix_sum(unsigned int *result, unsigned int *arr, unsigned int n)
{
	unsigned int *partial, *temp;
	unsigned int threads = THREADS, work;
	unsigned int i, mynum, last;
		
	#pragma omp parallel default(none) private(i, mynum, last) shared(arr, partial, temp, threads, work, n) num_threads(THREADS)
	{
		#pragma omp single
		{
			if(!(partial = (unsigned int *) malloc (sizeof (unsigned int) * threads))) exit(-1);
			if(!(temp = (unsigned int *) malloc (sizeof (unsigned int) * threads))) exit(-1);
			work = n / threads + 1; /*sets length of sub-arrays*/
		}
		mynum = omp_get_thread_num();
		/*calculate prefix-sum for each subarray*/
		for(i = work * mynum + 1; i < work * mynum + work && i < n; i++)
			arr[i] += arr[i - 1];
		partial[mynum] = arr[i - 1];
		#pragma omp barrier
		/*calculate prefix sum for the array that was made from last elements of each of the previous sub-arrays*/
		for(i = 1; i < threads; i <<= 1) {
			if(mynum >= i)
				temp[mynum] = partial[mynum] + partial[mynum - i];
			#pragma omp barrier
			#pragma omp single
			memcpy(partial + 1, temp + 1, sizeof(unsigned int) * (threads - 1));
		}
		/*update original array*/
		for(i = work * mynum; i < (last = work * mynum + work < n ? work * mynum + work : n); i++)
		  arr[i] += partial[mynum] - arr[last - 1];
	}
	result[0] = 0;
	memcpy(&result[1], arr, (n-1) * sizeof(unsigned int));
	free(arr);
}

// void prefix_sum(unsigned int *result, unsigned int *arr, unsigned int n)
// {
// 	result[0] = 0;
// 	unsigned int i;
// 	for (i = 1; i < n; i++)
// 	{
// 		result[i] = result[i-1] + arr[i-1];
// 	}
// }
