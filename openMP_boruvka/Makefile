all : openMP_boruvka.c functions.c boruvka_operators.c header.h
	gcc -fopenmp -o openMP_boruvka openMP_boruvka.c functions.c boruvka_operators.c
	
clean:
	rm -f *~ openMP_boruvka
