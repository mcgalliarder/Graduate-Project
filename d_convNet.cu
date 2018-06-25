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
 * @param Pout output vector
 * @param Pin input image
 * @param size of input image
 * @param samples to take
 * @param number of output feature maps
 */
float d_convNet(unsigned char * Pout, unsigned char * Pin, int size)
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
    unsigned char * d_Pin;
    unsigned char * d_Pout;
    int size = sizeof(unsigned char)*size*size;
    //TODO: do something with output sometime
    CHECK(cudaMalloc((void **)&d_Pin, size*CHANNELS));

    //Prepare Convolution Kernel
    int outSize = size - (MASKSIZE-1);
    int gridSize = outSize/TILEWIDTH;
    int gridZ = gridSize * gridSize;
    dim3 blockDim(TILEWIDTH, TILEWIDTH, 1);
    dim3 gridDim(gridSize, gridSize, gridZ);
    size_t shmemSize = sizeof(float) * ((TILEWIDTH + gridSize-1)*(TILEWIDTH + gridSize-1) + gridSize*gridSize);
    //Launch
    d_convLayerForwardKernel<<<gridDim, blockDim, shmemSize>>>(gridSize, gridSize, Pin, weights, outputMap);

}

__global__ void d_convLayerForwardKernel(int W_grid, int numOutput, float * inputMap, 
                                                          float * weights, float * outputMap)
{
    int n, m, h_base, w_base, h, w;
    int X_tile_width = TILEWIDTH + numOutput-1;
    extern __shared__ float shmem[];
    float* inputShared = &shmem[0];
    float* weightShared = &shmem[X_tile_width * X_tile_width];
    n = blockIdx.x;
    m = blockIdx.y;
    h_base = (blockIdx.z / W_grid) * TILEWIDTH; //vertical base out data index for the block
    w_base = (blockIdx.z % W_grid) * TILEWIDTH; // horizontal base out data index for the block
    h = h_base + threadIdx.x;
    w = w_base + threadIdx.y;

    float acc = 0.;
    int c, i, j, p, q;
    //Add over all channels
    for (c = 0; c < CHANNELS; c++) {
        //Load weight vector into shared memory
        if ((threadIdx.x < numOutput) && (threadIdx.y < numOutput))
            weightShared[threadIdx.x, threadIdx.y] = weights[m,c,threadIdx.x,threadIdx.y];
        __syncthreads();                    
        
        //Load input map into shared memory
        for (i = h; i < h_base + X_tile_width; i += TILEWIDTH) {
            for (j = w; j < w_base + X_tile_width; j += TILEWIDTH)
                inputShared[i - h_base, j - w_base] = inputMap[n,c,h,w];
        }                                       

        __syncthreads();
        for (p = 0; p < numOutput; p++) {
            for (q = 0; q < numOutput; q++)
                acc = acc + inputShared[h + p, w + q] * weightShared[p,q];
        }
        __syncthreads();
    }
    outputMap[n,m,h,w] = acc;
}
