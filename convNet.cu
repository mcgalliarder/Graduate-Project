#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "wrappers.h"
#include "d_forwardPropagation.h"

#define CHANNELS 1    

void readPGMImage(char *, unsigned char **, int *, int *, int *);
void parseCommandArgs(int, char **, char **);
void printUsage();
void fillArray(float *, int);
void printFloatArray(float * array, int width, int length);
void printCharArray(unsigned char * array, int width, int length);

int main(int argc, char * argv[])
{
    unsigned char * Pin;
    char * fileName;
    int width, height, color;
    parseCommandArgs(argc, argv, &fileName);
    readPGMImage(fileName, &Pin, &width, &height, &color);
    //Print original
    printCharArray(Pin, width, width*height);    

    //use the GPU to perform the convoluted neural network
    float * d_Pout;
    float * weights;
    int weightLen = 2;
    weights = (float *) Malloc((sizeof(float) * weightLen * weightLen));
    fillArray(weights, weightLen);
    d_Pout = (float *) Malloc((sizeof(float) * width * height));
    d_convLayerForward(Pin, d_Pout, weights, 1, weightLen);

    //Print result
    printFloatArray(d_Pout, width-4, width*height);
}

void readPGMImage(char * filename, unsigned char ** Pin,
               int * width, int * height, int * color)
{
    int ht, wd, colr;
    char P3[3];
    FILE * fp = fopen(filename, "rb"); //read binary
    if(fp == NULL) {
        printf("File not found.\n");
        printUsage();
    }
    int count = fscanf(fp, "%s\n%d %d\n%d\n", P3, &wd, &ht, &colr);
    //should have read four values
    //first value is the string "P3"
    //color value must be less than 256 and greater than 0
    //printf("Width: %d\n",wd);
    //printf("Height: %d\n",ht);
    //printf("Color: %d\n",colr);
    //printf("P3: %d %d %d\n",P3[0],P3[1],P3[2]);
    if (count != 4 || strncmp(P3, "P5", CHANNELS) || colr <= 0 || colr > 255)
    {
        printf("\nInvalid file format.\n\n");
        printUsage();
    }
    (*Pin) = (unsigned char *) Malloc(sizeof(unsigned char) * wd * ht * CHANNELS);
    unsigned char * ptr = *Pin;
    int bytes;
    if ((bytes = fread(ptr, sizeof(unsigned char) * wd, ht, fp)) != ht)
    {
       printf("Invalid file format.\n\n");
       printf("\nExpected rows: %d, Read rows: %d\n", ht, bytes);
       printUsage();
    } 

    (*width) = wd;
    (*height) = ht;
    (*color) = colr;
    fclose(fp);
}

void fillArray(float * array, int len) 
{
   for (int i = 0; i < len; i++)
   {
       array[i] = (float)(rand() % 100);
   } 
}


/*
    parseCommandArgs
    This function parses the command line arguments. The program can be executed in
    one of two ways:
    ./greyscalar <file>.ppm
    or
    ./greyscalar -w <blkWidth> -h <blkHeight> <file>.ppm
    This function parses the command line arguments, setting block width and block
    height to the command line argument values or to 16 if no command line arguments
    are provided.  In addition, it checks to see if the last command line argument
    is a ppm file and sets (*fileNm) to argv[i] where argv[i] is the name of the ppm
    file.
*/
void parseCommandArgs(int argc, char * argv[], char ** fileNm)
{
    int fileIdx = 1;
    struct stat buffer; 
    fileIdx = 1;

    //stat function returns 1 if file does not exist
    if (stat(argv[fileIdx], &buffer)) printUsage();
    (*fileNm) = argv[fileIdx];
}

void printFloatArray(float * array, int width, int length) 
{
    for (int i = 0; i < length; i++)
    {   
        if (!(i%width)) printf("\n%2d:", i/width);
        printf("%2.1f ", array[i]);
    } 
}

void printCharArray(unsigned char * array, int width, int length)
{
    for (int i = 0; i < length; i++)
    {
        if (!(i%width)) printf("\n%2d:", i/width);
        printf("%3x", array[i]);
    }
    printf("\n");
}



/*
    printUsage
    This function is called if there is an error in the command line
    arguments or if the .ppm file that is provided by the command line
    argument is improperly formatted.  It prints usage information and
    exits.
*/
void printUsage()
{
    printf("This application takes as input the name of a .pgm\n");
    printf("\nusage: convNet  <name>.pgm\n");
    printf("         <name>.pgm is the name of the input pgm file\n");
    printf("Examples:\n");
    printf("./convNet 1.pgm\n");
    exit(EXIT_FAILURE);
}

   
