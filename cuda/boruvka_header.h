__global__ void find_min(Graph *gr, unsigned int *p);
__global__ void mirrors_edge(Graph *gr, unsigned int *p);
__global__ void initialize_colors(Graph *gr, unsigned int *m, unsigned int *f);
__global__ void propagate_colors(Graph *gr, unsigned int *color, unsigned int *change);
__global__ void create_new_vertex_ids(Graph *gr, unsigned int *color, unsigned int *flag);
__global__ void count_edges(Graph *gr, Graph *gr2, unsigned int *color, unsigned int *EPS);
__global__ void insert_new_edges(Graph *gr, Graph *gr2, unsigned int *color, unsigned int *first_edge_copy, unsigned int *EPS);
__global__ void EPS(unsigned int *p1, unsigned int *p2, Graph *gr);

__global__ void test(Graph *gr, int flag);
