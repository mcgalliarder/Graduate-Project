/*  
In addition to implementing the device code, answer the following questions:
1)  If you run the program like this:
   ./greyscalar -w 16 -h 16 file.ppm
   how many data elements are calculated by a single internal block? 

  Each block has a total of (width x height) threads. Each thread calculates two
data elements. So a single internal block would calculate 16x16x2 data elements, 
so 512.

2)  If you run the program like this:
   ./greyscalar -w 8 -h 16 file.ppm
   how many data elements are calculated by a single internal block? 

  For this problem a single internal block would calculate 8x16x2 data elements,
so 256.

3) What is an advantage of running the program as specified by 1)?   

  The advantage of running the program with a width and height of 16 is that the
speedup is much greater for larger data sets. Anything over ~200x200 runs faster
with a larger height and width from my observations.  

4) What is an advantage of running the program as specified by 2)?   

  The advantage of running the program with a width of 8 and height of 16 is that
the program runs faster with smaller data sets, i.e. those less than 200x200. 
*/

#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include "wrappers.h"
#include "h_colorToGreyscale.h"
#include "d_colorToGreyscale.h"

#define CHANNELS 3

//prototypes for functions in this file 
void parseCommandArgs(int, char **, int *, int *, char **);
void printUsage();
void readPPMImage(char *, unsigned char **, int *, int *, int *);
void writePPMImage(char *, unsigned char *, int, int, int);
char * buildFilename(char *, const char *);
void compare(unsigned char * d_Pout, unsigned char * h_Pout, int size);

/*
    main 
    Opens the ppm file and reads the contents.  Uses the CPU
    and the GPU to perform the greyscale.  Compares the CPU and GPU
    results.  Writes the results to output files.  Outputs the
    time of each.
*/
int main(int argc, char * argv[])
{
    unsigned char * Pin;
    char * fileName;
    int width, height, blkWidth, blkHeight, color;
    parseCommandArgs(argc, argv, &blkWidth, &blkHeight, &fileName);
    readPPMImage(fileName, &Pin, &width, &height, &color);

    //use the CPU to perform the greyscale
    unsigned char * h_Pout; 
    h_Pout = (unsigned char *) Malloc(sizeof(unsigned char) * width * height);
    float cpuTime = h_colorToGreyscale(h_Pout, Pin, width, height);
    char * h_outfile = buildFilename(fileName, "h_grey");
    writePPMImage(h_outfile, h_Pout, width, height, color);

    //use the GPU to perform the greyscale 
    unsigned char * d_Pout; 
    d_Pout = (unsigned char *) Malloc((sizeof(unsigned char) * width * height));
    float gpuTime = d_colorToGreyscale(d_Pout, Pin, width, height, blkWidth, blkHeight);
    char * d_outfile = buildFilename(fileName, "d_grey");
    writePPMImage(d_outfile, d_Pout, width, height, color);

    //compare the CPU and GPU results
    compare(d_Pout, h_Pout, width * height);

    printf("CPU time: %f msec\n", cpuTime);
    printf("GPU time: %f msec\n", gpuTime);
    printf("Speedup: %f\n", cpuTime/gpuTime);
    return EXIT_SUCCESS;
}

/* 
    compare
    This function takes two arrays of greyscale pixel values.  One array
    contains pixel values calculated  by the GPU.  The other array contains
    greyscale pixel values calculated by the CPU.  This function checks to
    see that the values are the same within a slight margin of error.

    d_Pout - pixel values calculated by GPU
    h_Pout - pixel values calculated by CPU
    size - size in elements of both arrays
    
    Outputs an error message and exits program if the arrays differ.
*/
void compare(unsigned char * d_Pout, unsigned char * h_Pout, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        //GPU and CPU have different floating point standards so
        //the results could be slightly different
        int diff = d_Pout[i] - h_Pout[i];
        if (abs(diff) > 1)
        {
            printf("Greyscale results don't match.\n");
            printf("CPU pixel %d: %d\n", i, h_Pout[i]);
            printf("GPU pixel %d: %d\n", i, d_Pout[i]);
            exit(EXIT_FAILURE);
        }
    }
}

/* 
    writePPMImage
    Writes a greyscale ppm image to an output file.

    outfile - name of ppm file (ends with a .ppm extension)
    Pout - array of pixels
    width - width (x-dimension) of image
    height - height (y-dimension) of image
    color - maximum color value 
*/
void writePPMImage(char * outfile, unsigned char * Pout, int width, 
                int height, int color)
{
    int i, j, k = 0;
    FILE *fp = fopen(outfile, "wb"); //write binary 
    //output "P6", width, height, and color separated by whitespace
    fprintf(fp, "P6\n%d %d\n%d\n", width, height, color);
    for (j = 0; j < height; j++)
    {
        for (i = 0; i < width; i++, k++)
        {
           //write same byte three times for red, blue and green
           static unsigned char color[3];
           color[0] = color[1] = color[2] = Pout[k];
           fwrite(color, sizeof(unsigned char), 3, fp); 
        }
    }
    fclose(fp);
}

/*
    buildFilename
    This function returns the concatenation of two strings by
    first allocating enough space to hold both strings and then
    copying the two strings into the allocated space.  
    It is used by the program to build the output file names.
*/    
char * buildFilename(char * infile, const char * prefix)
{
   int len = strlen(infile) + strlen(prefix) + 1;
   char * outfile = (char *) Malloc(sizeof(char *) * len);
   strncpy(outfile, prefix, strlen(prefix));
   strncpy(&outfile[strlen(prefix)], infile, strlen(infile) + 1);
   return outfile;
}
   
/*
    readPPMImage
    This function opens a ppm file and reads the contents.  A ppm file
    is of the following format:
    P6
    width  height
    color
    pixels

    Each pixel consists of bytes for red, green, and blue.  If color
    is less than 256 then each color is encoded in 1 byte.  Otherwise,
    each color is encoded in 2 bytes. This function fails if the color
    is encoded in 2 bytes.
    
    The array Pin is initialized to the pixel bytes.  width, height,
    and color are pointers to ints that are set to those values.
    filename - name of the .ppm file
*/
void readPPMImage(char * filename, unsigned char ** Pin, 
               int * width, int * height, int * color)
{
    int ht, wd, colr;
    char P6[3];
    FILE * fp = fopen(filename, "rb"); //read binary
    int count = fscanf(fp, "%s\n%d %d\n%d\n", P6, &wd, &ht, &colr);
    //should have read four values
    //first value is the string "P6"
    //color value must be less than 256 and greater than 0
    if (count != 4 || strncmp(P6, "P6", CHANNELS) || colr <= 0 || colr > 255)
    {
        printf("\nInvalid file format.\n\n");
        printUsage();
    }
    (*Pin) = (unsigned char *) Malloc(sizeof(unsigned char) * wd * ht * CHANNELS);
    unsigned char * ptr = *Pin;
    int bytes;
    if ((bytes = fread(ptr, sizeof(unsigned char) * wd * CHANNELS, ht, fp)) != ht)
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
void parseCommandArgs(int argc, char * argv[], int * blkWidth, int * blkHeight, char ** fileNm)
{
    int fileIdx = 1, blkW = 16, blkH = 16;
    struct stat buffer;
    if (argc != 2 && argc != 6) printUsage();
    if (argc == 6) 
    {
        fileIdx = 5;
        if (strncmp("-w", argv[1], 2) != 0) printUsage();
        if (strncmp("-h", argv[3], 2) != 0) printUsage();
        blkW = atoi(argv[2]);
        blkH = atoi(argv[4]);
        if (blkW <= 0 || blkH <= 0) printUsage();
    }

    int len = strlen(argv[fileIdx]);
    if (len < 5) printUsage();
    if (strncmp(".ppm", &argv[fileIdx][len - 4], 4) != 0) printUsage();

    //stat function returns 1 if file does not exist
    if (stat(argv[fileIdx], &buffer)) printUsage();
    (*blkWidth) = blkW;
    (*blkHeight) = blkH;
    (*fileNm) = argv[fileIdx];
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
    printf("This application takes as input the name of a .ppm\n");
    printf("file containing a color image and creates a file\n");
    printf("containing a greyscale version of the file.\n");
    printf("\nusage: greyscaler [-w <blkWidth> -h <blkHeight>] <name>.ppm\n");
    printf("         <blkWidth> is the width of the blocks created for GPU\n");
    printf("         <blkHeight> is the height of the blocks created for GPU\n");
    printf("         If the -w and -h arguments are omitted, the block size\n");
    printf("         defaults to 16 by 16.\n");
    printf("         <name>.ppm is the name of the input ppm file\n");
    printf("Examples:\n");
    printf("./greyscaler color1200by800.ppm\n");
    printf("./greyscaler -w 8 -h 16 color1200by800.ppm\n");
    exit(EXIT_FAILURE);
}
