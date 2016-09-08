typedef struct Dijkstra
{
	unsigned int id, cost, prev;
	char flag;
} Dijkstra;

struct CustomMin
{
    template<typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
    	Dijkstra temp;
    	temp.cost = UINT_MAX;
    	temp.flag = 0;
    	if (b.flag == 1 && a.flag == 0)
    		return a;
    	else if (a.flag == 1 && b.flag == 0)
    		return b;
    	else if (a.flag == 0 && b.flag == 0)
        	return (b.cost < a.cost) ? b : a;
        else
        	return temp;
    }
};

__global__ void init(Dijkstra *dijkstra, Graph *gr, unsigned int *startNode);

__global__ void update(Graph *gr, Dijkstra *dijkstra, unsigned int *currentNode);

__global__ void test(Dijkstra *dijkstra, unsigned int size);

