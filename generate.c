#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void checkCommandArgs(int, char **, int *, int *);
void printUsage();
void createPPMImage(FILE *, int, int);

int main(int argc, char * argv[])
{
    int width, height;
    checkCommandArgs(argc, argv, &width, &height);
    FILE * fp = fopen(argv[1], "wb");
    if (fp == NULL)
    {
        printf("\nFile open failed.\n\n");
        printUsage();
    }
    createPPMImage(fp, width, height);
    return EXIT_SUCCESS;
}

void  createPPMImage(FILE * fp, int width, int height)
{
  int i, j;
  (void) fprintf(fp, "P6\n%d %d\n255\n", width, height);
  for (j = 0; j < height; ++j)
  {
    for (i = 0; i < width; ++i)
    {
      static unsigned char color[3];
      color[0] = i % 256;  // red
      color[1] = j % 256;  // green
      color[2] = (i * j) % 256;  // blue
      (void) fwrite(color, sizeof(unsigned char), 3, fp);
    }
  }
  (void) fclose(fp);
}

void checkCommandArgs(int argc, char * argv[], int * width, int * height)
{
    if (argc != 4)
    {
        printUsage();
    }
    int len = strlen(argv[1]);
    if (len < 5) printUsage();
    if (strncmp(".ppm", &argv[1][len - 4], 4) != 0) printUsage();
    (*width) = atoi(argv[2]);
    (*height) = atoi(argv[3]);
    if ((*width) < 100 || (*height) < 100) printUsage();
}

void printUsage()
{
    printf("\nThis program creates a ppm file of the txt file passed as an argument.\n\n");
    printf("usage: generate <name>.txt <name>.ppm width height\n");
    printf("       <name>.ppm is the name of the created ppm file\n");
    printf("       both width and height must be = 32\n\n");
    exit(EXIT_FAILURE);
}

