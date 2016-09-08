#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct record
{
	unsigned int src, dest, w;
} Record;

int partition(Record *, unsigned int, unsigned int);

void quickSort(Record *a, unsigned int l, unsigned int r)
{
   unsigned int j;

   if(l < r) 
   {
       j = partition(a, l, r);
       quickSort(a, l, j-1);
       quickSort(a, j+1, r);
   }
	
}

int partition(Record *a, unsigned int l, unsigned int r)
{
   unsigned int pivot, i, j;
   Record t;
   pivot = a[l].src;
   i = l; j = r + 1;
		
   while(1)
   {
		do ++i; while(a[i].src <= pivot && i <= r);
		do --j; while(a[j].src > pivot);
		if(i >= j) break;
		t = a[i]; a[i] = a[j]; a[j] = t;
   }
   t = a[l]; a[l] = a[j]; a[j] = t;
   return j;
}

int main(int argc, char *argv[])
{
	if (argc != 2)
		printf("Wrong inline parameters\n");
	clock_t start = clock();
	FILE *fp = fopen(argv[1], "r");
	char line[256], first_letter[1];
	unsigned long int nodes, arcs, i;
	while (fgets(line, 80, fp) != NULL)
	{
		char *token = strtok(line, " ");
		if (strcmp(token, "p") == 0)	//find number of arcs for input graph
		{
			token = strtok(NULL, " ");
			token = strtok(NULL, " ");
			nodes = atoi(token);
			token = strtok(NULL, " ");
			arcs = atoi(token);
			break;
		}
	}
	fclose(fp);
	Record *rd;
	if ((rd = malloc(arcs * sizeof(Record))) == NULL)	//allocate space for data
	{
		printf("malloc error\n");
		return 0;
	}
	fp = fopen(argv[1], "r");	//read the file for a second time to get the data
	while (fgets(line, 80, fp) != NULL)
	{
		char *token = strtok(line, " ");
		if (strcmp(token, "a") == 0)	//store first "a" line to data
		{
			token = strtok(NULL, " ");
			rd[0].src = atoi(token) - 1;
			token = strtok(NULL, " ");
			rd[0].dest = atoi(token) - 1;
			token = strtok(NULL, " ");
			rd[0].w = atoi(token);
			break;
		}
	}
	for (i = 1; i < arcs; i++)	//store the rest of the data
	{
		fscanf(fp, " %c %u %u %u ", first_letter, &rd[i].src, &rd[i].dest, &rd[i].w);
		rd[i].src--;
		rd[i].dest--;
	}
	fclose(fp);
	printf("File Loaded!\nSorting data...\n");
	quickSort(rd, 0, arcs - 1);
	char *new_file = malloc((strlen(argv[1]) + 4) * sizeof(char));
	strncpy(new_file, argv[1], (strlen(argv[1]) - 2) * sizeof(char));
	strncpy(new_file + strlen(argv[1]) - 2, "edges", 5 * sizeof(char));
	new_file[strlen(argv[1]) + 3] = '\0';
	printf("Writing output file...\n");
	fp = fopen(new_file, "w");
	fprintf(fp, "%lu %lu\n", nodes, arcs);
	for (i = 0; i < arcs; i++)
		fprintf(fp, "%u %u %u\n", rd[i].src, rd[i].dest, rd[i].w);
	fclose(fp);
	clock_t stop = clock();
	float seconds = (float)(stop - start) / CLOCKS_PER_SEC;
	printf("Time to convert data: %.2fs\n", seconds);
	free(new_file);
	free(rd);
}