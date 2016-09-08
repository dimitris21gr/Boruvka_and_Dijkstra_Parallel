#include "header.h"
#include "dijkstra_header.h"

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		printf("Input file not found\n");
		return 0;
	}
	Graph *gr = create_graph(argv[1]);
	printf("File Loaded!\n");
	Graph *d_gr, temp;
	temp.nodes = gr->nodes;
	temp.edges = gr->edges;
	
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
	
	unsigned int startNode, *d_startNode, *d_currentNode, currentNodeOutdegree;	//to be fixed
	
	printf("Set start node id (1-%u): ", gr->nodes);
	scanf("%u", &startNode);
	startNode--;

	cudaMalloc(&d_startNode, sizeof(unsigned int));
	cudaMalloc(&d_currentNode, sizeof(unsigned int));
	cudaMemcpy(d_startNode, &startNode, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_currentNode, &startNode, sizeof(unsigned int), cudaMemcpyHostToDevice);

	Dijkstra *dijkstra;
	cudaMalloc(&dijkstra, gr->nodes * sizeof(Dijkstra));

	dim3 block(BLOCK_SIZE);  
    dim3 grid(FRACTION_CEILING(gr->nodes, BLOCK_SIZE));	

	init<<<grid, block>>>(dijkstra, d_gr, d_startNode);

	currentNodeOutdegree = gr->out_degree[startNode];

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	while(1)
	{
    	dim3 block2(BLOCK_SIZE);  
    	dim3 grid2(FRACTION_CEILING(currentNodeOutdegree, BLOCK_SIZE));
    	update<<<grid2, block2>>>(d_gr, dijkstra, d_currentNode);

    	//Find min from unfixed nodes using Cub
		void     *d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;
		Dijkstra *d_out, out;
		out.cost = UINT_MAX;
		out.flag = 0;
		cudaMalloc(&d_out, sizeof(Dijkstra));
		struct CustomMin min_op;
		cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, dijkstra, d_out, gr->nodes, min_op, out);
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, dijkstra, d_out, gr->nodes, min_op, out);
		cudaMemcpy(&out, d_out, sizeof(Dijkstra), cudaMemcpyDeviceToHost);

		cudaMemset(&dijkstra[out.id].flag, 1, sizeof(char));	//fix new node
		cudaMemcpy(d_currentNode, &out.id, sizeof(unsigned int), cudaMemcpyHostToDevice);
		
		if (out.cost == UINT_MAX)
			break;

    	currentNodeOutdegree = gr->out_degree[out.id];
    	cudaFree(d_out);
    	cudaFree(d_temp_storage);
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for kernels: %.4f ms\n", time);

	cudaFree(dijkstra);
	cudaFree(d_startNode);
	cudaFree(d_currentNode);
	cudaFree(temp.destination);
	cudaFree(temp.weight);
	cudaFree(temp.first_edge);
	cudaFree(temp.out_degree);
	destroy_graph(gr);
}
