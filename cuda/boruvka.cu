#include "header.h"
#include "boruvka_header.h"

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		printf("Input file not found\n");
		return 0;
	}
	Graph *gr = create_graph(argv[1]);
	printf("File Loaded!\n");
	Graph *d_gr, temp, *d_gr2, temp2;
	unsigned int *d_minedge, *d_color, *d_flag, *d_EPS, *d_first_edge_copy;
	unsigned int change, *d_change, s1, s2;
	temp.nodes = gr->nodes;
	temp.edges = gr->edges;
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	cudaMalloc(&d_change, sizeof(unsigned int));
	cudaMalloc(&d_gr, sizeof(Graph));
	cudaMalloc(&temp.destination, gr->edges * sizeof(unsigned int));
	cudaMalloc(&temp.weight, gr->edges * sizeof(unsigned int));
	cudaMalloc(&temp.first_edge, gr->nodes * sizeof(unsigned int));
	cudaMalloc(&temp.out_degree, gr->nodes * sizeof(unsigned int));
	
	cudaMemcpy(temp.destination, gr->destination, gr->edges * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(temp.weight, gr->weight, gr->edges * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(temp.first_edge, gr->first_edge, gr->nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(temp.out_degree, gr->out_degree, gr->nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gr, &temp, sizeof(Graph), cudaMemcpyHostToDevice);
	
	do
	{
		cudaMalloc(&d_minedge, temp.nodes * sizeof(unsigned int));
		cudaMalloc(&d_color, temp.nodes * sizeof(unsigned int));
		cudaMalloc(&d_flag, temp.nodes * sizeof(unsigned int));
		cudaMalloc(&d_EPS, temp.nodes * sizeof(unsigned int));

	    dim3 block(BLOCK_SIZE);  
    	dim3 grid(FRACTION_CEILING(temp.nodes, BLOCK_SIZE));
		
		find_min<<<grid,block>>>(d_gr, d_minedge);
		mirrors_edge<<<grid,block>>>(d_gr, d_minedge);
		initialize_colors<<<grid,block>>>(d_gr, d_minedge, d_color);
		do
		{
			cudaMemset(d_change, 0, sizeof(unsigned int));
			propagate_colors<<<grid,block>>>(d_gr, d_color, d_change);
			cudaMemcpy(&change, d_change, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		} while (change);
		create_new_vertex_ids<<<grid,block>>>(d_gr, d_color, d_flag);
		
		// EPS on flag array using CUB Library
		// Determine temporary device storage requirements
		void     *d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;
		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_flag, d_EPS, temp.nodes);
		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Run exclusive prefix sum
		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_flag, d_EPS, temp.nodes);
		cudaFree(d_temp_storage);
		
		cudaMemcpy(&s1, &d_EPS[temp.nodes-1], sizeof(unsigned int), cudaMemcpyDeviceToHost);

		test<<<1,1>>>(d_gr, 1);
		if (s1 <= 1)
		{
			break;
		}

		cudaMemcpy(&temp2.nodes, &d_EPS[temp.nodes-1], sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMalloc(&d_gr2, sizeof(Graph));
		cudaMalloc(&temp2.first_edge, temp2.nodes * sizeof(unsigned int));
		cudaMalloc(&temp2.out_degree, temp2.nodes * sizeof(unsigned int));
		cudaMemset(temp2.out_degree, 0, temp2.nodes * sizeof(unsigned int));
		cudaMemcpy(d_gr2, &temp2, sizeof(Graph), cudaMemcpyHostToDevice);
		
		count_edges<<<grid,block>>>(d_gr, d_gr2, d_color, d_EPS);
		
		// EPS on outdegree array, gives firstedge array using CUB Library
		// Determine temporary device storage requirements
		d_temp_storage = NULL;
		temp_storage_bytes = 0;
		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp2.out_degree, temp2.first_edge, temp2.nodes);
		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Run exclusive prefix sum
		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp2.out_degree, temp2.first_edge, temp2.nodes);
		cudaFree(d_temp_storage);
		
		cudaMemcpy(&s1, &temp2.out_degree[temp2.nodes-1], sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&s2, &temp2.first_edge[temp2.nodes-1], sizeof(unsigned int), cudaMemcpyDeviceToHost);
		temp2.edges = s1 + s2;
		cudaMalloc(&temp2.destination, temp2.edges * sizeof(unsigned int));
		cudaMalloc(&temp2.weight, temp2.edges * sizeof(unsigned int));
		cudaMemcpy(d_gr2, &temp2, sizeof(Graph), cudaMemcpyHostToDevice);
		
		//test<<<1,1>>>(d_gr2, 1);
		cudaMalloc(&d_first_edge_copy, temp2.nodes * sizeof(unsigned int));
		cudaMemcpy(d_first_edge_copy, temp2.first_edge, temp2.nodes * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

		insert_new_edges<<<grid,block>>>(d_gr, d_gr2, d_color, d_first_edge_copy, d_EPS);
		
		cudaFree(d_first_edge_copy);
		cudaFree(d_minedge);
		cudaFree(d_color);
		cudaFree(d_flag);
		cudaFree(d_EPS);
		//swap start
		cudaFree(temp.destination);
		cudaFree(temp.weight);
		cudaFree(temp.first_edge);
		cudaFree(temp.out_degree);
		temp.nodes = temp2.nodes;
		temp.edges = temp2.edges;
		temp.destination = temp2.destination;
		temp.weight = temp2.weight;
		temp.first_edge = temp2.first_edge;
		temp.out_degree = temp2.out_degree;
		cudaMemcpy(d_gr, &temp, sizeof(Graph), cudaMemcpyHostToDevice);
		temp2.destination = NULL;
		temp2.weight = NULL;
		temp2.first_edge = NULL;
		temp2.out_degree = NULL;
		cudaFree(d_gr2);
		//swap end
	} while(1);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for kernels: %.4f ms\n", time);
	
	cudaFree(temp.destination);
	cudaFree(temp.weight);
	cudaFree(temp.first_edge);
	cudaFree(temp.out_degree);
	cudaFree(d_minedge);
	cudaFree(d_color);
	cudaFree(d_change);
	cudaFree(d_flag);
	cudaFree(d_EPS);
	cudaFree(d_gr);
	
	destroy_graph(gr);
}
