boruvka: boruvka.cu boruvka_operators.cu functions.cu header.h boruvka_header.h
	nvcc -o boruvka boruvka.cu boruvka_operators.cu functions.cu -arch=sm_20
	
dijkstra: dijkstra.cu functions.cu header.h dijkstra_header.h dijkstra_operators.cu
	nvcc -o dijkstra dijkstra.cu dijkstra_operators.cu functions.cu -arch=sm_20
	
clean:
	rm -f *~ boruvka dijkstra
