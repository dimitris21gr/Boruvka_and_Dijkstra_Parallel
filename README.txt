INPUT FILES CAN BE DOWNLOADED FROM HERE: http://www.dis.uniroma1.it/challenge9/download.shtml

EVERY FILE HAS TO BE BE CONVERTED FROM .gr FORMAT TO .edges FORMAT USING OUR CONVERTER

gr_to_edges_converter.c
COMPILE WITH COMMAND: gcc -o <executable_name> gr_to_edges_converter.c
RUN WITH COMMAND: ./<executable_name> <input_file.gr>
THE APPLICATION WILL CREATE A FILE WITH THE SAME NAME AS input_file 

CUDA IMPLEMENTATIONS:
	BORUVKA
		COMPILE WITH COMMAND: make boruvka
		RUN WITH COMMAND: ./boruvka <input_file.edges>
	DIJKSTRA
		COMPILE WITH COMMAND: make dijkstra
		RUN WITH COMMAND: ./dijkstra <input_file.edges>

openMP IMPLEMENTATIONS:
	BORUVKA
		COMPILE WITH COMMAND: make
		RUN WITH COMMAND: ./openMP_boruvka <input_file.edges>
	DIJKSTRA
		COMPILE WITH COMMAND: make
		RUN WITH COMMAND: ./openMP_dijkstra <input_file.edges>
TO SET NUMBER OF THREADS, OPEN FILE header.h OF EACH ALGORITHM AND SET
THE #define THREADS VALUE TO YOU PREFERENCE