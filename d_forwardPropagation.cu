#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "d_colorToGreyscale.h"
#include "CHECK.h"

#define CHANNELS 3
#define MASKSIZE 5
#define SUBSAMPLESIZE 2
#define TILEWIDTH 16

__global__ void d_convLayerForwardKernel(int, int, int, float *, float *, float *);

/**
 * Performs one forward run through the network
 * @param Pin input image
 * @param resulting vector
 * @param size of input image
 */
float * d_convLayerForward(unsigned char * inputMap, int numInput, int size, unsigned char * result)
{
    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

    //Use cuda functions to do the timing
    //create event objects
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    //record the starting time
    CHECK(cudaEventRecord(start_cpu));

    //Create device vectors 
    unsigned char * d_inputMap;
    unsigned char * d_outputMap;
    unsigned char * d_result;
    int inSize = sizeof(unsigned char)*size*size*numInput;
    int outSize = sizeof(unsigned char)*(size-MASKSIZE-1)*(size-MASKSIZE-1);
    int resultSize = sizeof(float)*10;
    CHECK(cudaMalloc((void **)&d_outputMap, outSize));
    CHECK(cudaMalloc((void **)&d_inputMap, resultSize));

    CHECK(cudaMemcpy(d_inputMap, inputMap, inSize, cudaMemcpyHostToDevice));

    //Prepare Convolution Kernel
    int outSize = size - (MASKSIZE-1);
    int gridSize = outSize/TILEWIDTH;
    int gridZ = gridSize * gridSize;
    dim3 blockDim(TILEWIDTH, TILEWIDTH, 1);
    dim3 gridDim(gridSize, gridSize, gridZ);
    size_t shmemSize = sizeof(float) * ((TILEWIDTH + gridSize-1)*(TILEWIDTH + gridSize-1) + gridSize*gridSize);
    //Launch
    //((Might need shmemSize as third parameter on kernel launch))
    d_convLayerForwardKernel<<<gridDim, blockDim>>>(gridSize, gridSize, Pin, weights, outputMap);

}



/**
 * Convolutes a set of input feature maps into a set
 * of output feature maps
 * @param W_grid width of grid
 * @param numOutput number of output elements
 * @param numInput number of input elements
 * @param inputMap input feature maps
 * @param weights to apply to each input map
 * @param outputMap
 */
__global__ void d_convLayerForwardKernel(int gridWidth, int numInput, int numOutput, float * inputMap, 
                                                          float * weights, float * outputMap)
{
    int n, m, h_base, w_base, h, w;
    int xTileWidth = TILEWIDTH + numOutput-1;
    int weightLen = xTileWidth * xTileWidth;
    int inputLen = numOutput*numOutput; //((Might need more space))
    __shared__ float inputShared[inputLen];
    __shared__ float weightLen[weightLen];    

    n = blockIdx.x;
    m = blockIdx.y;
    h_base = (blockIdx.z / gridWidth) * TILEWIDTH; //vertical base out data index for the block
    w_base = (blockIdx.z % gridWidth) * TILEWIDTH; // horizontal base out data index for the block
    h = h_base + threadIdx.x;
    w = w_base + threadIdx.y;

    float acc = 0.;
    int c, i, j, p, q;
    //Add over all channels
    for (c = 0; c < numInput; c++) {
        //Load weight vector into shared memory
        if ((threadIdx.x < numOutput) && (threadIdx.y < numOutput))                           
            weightShared[threadIdx.y*blockDim.x+threadIdx.x] = 
                           weights[m+numOutput*(c+numInput*(threadIdx.x+blockDim.x*threadIdx.y))]; //m,c,tIdx,tIdy
        __syncthreads();                    
        
        //Load input map into shared memory
        for (i = h; i < h_base + xTileWidth; i += TILEWIDTH) {
            for (j = w; j < w_base + xTileWidth; j += TILEWIDTH)
                inputShared[(i-h_base)*inputLen+(j-w_base)] = inputMap[n+numOutput*(c+CHANNELS*(h+gridWidth*w))]; //n,c,h,w
        }                                       

        __syncthreads();
        for (p = 0; p < numOutput; p++) {
            for (q = 0; q < numOutput; q++)
                acc += inputShared[(h+p)*inputLen+(w+q)] * weightShared[p*weightLen+q];
        }
        __syncthreads();
    }
    outputMap[n+numOutput*(m+numOutput*(h+gridWidth*w))] = acc; //n,m,h,w
}
