#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include "wrappers.h"

#define CHANNELS 3    

int main(int argc, char * argv[])
{
    unsigned char * Pin;
    char * fileName;
    int width, height, blkWidth, blkHeight, color;
    parseCommandArgs(argc, argv, &blkWidth, &blkHeight, &fileName);
    readPPMImage(fileName, &Pin, &width, &height, &color);

    //use the GPU to perform the greyscale
    unsigned char * d_Pout;
    d_Pout = (unsigned char *) Malloc((sizeof(unsigned char) * blkWidth * blkHeight));
    d_colorToGreyscale(d_Pout, Pin, width, height, blkWidth, blkHeight);
    char * d_outfile = buildFilename(fileName, "d_grey");
    writePPMImage(d_outfile, d_Pout, width, height, color);

}

void readPPMImage(char * filename, unsigned char ** Pin,
               int * width, int * height, int * color)
{
    int ht, wd, colr;
    char P3[3];
    FILE * fp = fopen(filename, "rb"); //read binary
    int count = fscanf(fp, "%s\n%d %d\n%d\n", P6, &wd, &ht, &colr);
    //should have read four values
    //first value is the string "P6"
    //color value must be less than 256 and greater than 0
    if (count != 4 || strncmp(P3, "P3", CHANNELS) || colr <= 0 || colr > 255)
    {
        printf("\nInvalid file format.\n\n");
        printUsage();
    }
    (*Pin) = (unsigned char *) Malloc(sizeof(unsigned char) * wd * ht * CHANNELS);
    unsigned char * ptr = *Pin;

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
void parseCommandArgs(int argc, char * argv[], char ** fileNm)
{
    int fileIdx = 1, blkW = 32, blkH = 32;
    struct stat buffer; 
    fileIdx = 1;

    //stat function returns 1 if file does not exist
    if (stat(argv[fileIdx], &buffer)) printUsage();
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
    printf("\nusage: greyscaler  <name>.ppm\n");
    printf("         <name>.ppm is the name of the input ppm file\n");
    printf("Examples:\n");
    printf("./convolution 1.ppm\n");
    exit(EXIT_FAILURE);
}

   
