#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
//#include "d_colorToGreyscale.h"
#include "CHECK.h"

#define CHANNELS 3
#define MASKSIZE 5
#define SUBSAMPLESIZE 2
#define TILEWIDTH 16


__global__ void d_convLayerForwardKernel(int, int, int, unsigned char *, float *, float *);

/**
 * Performs one forward run through the network
 * @param Pin input image
 * @param resulting vector
 * @param size of input image
 */
void d_convLayerForward(unsigned char * inputMap, float * outputMap, float * weights, 
                                                                  int numInput, int weightLen)
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
    float * d_weights;
    float * d_outputMap;
    
    int inSize = sizeof(unsigned char)*numInput*numInput;
    int outputSize = sizeof(float)*(numInput-MASKSIZE-1)*(numInput-MASKSIZE-1);
    int weightSize = sizeof(float)*weightLen*weightLen;
    CHECK(cudaMalloc((void **)&d_outputMap, outputSize));
    CHECK(cudaMalloc((void **)&d_weights, weightSize));
    CHECK(cudaMalloc((void **)&d_inputMap, inSize));

    CHECK(cudaMemcpy(d_inputMap, inputMap, inSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weights, weights, weightSize, cudaMemcpyHostToDevice));

    //Prepare Convolution Kernel
    int outSize = numInput - (MASKSIZE-1);
    int gridSize = outSize/TILEWIDTH;
    int gridZ = gridSize * gridSize;
    dim3 blockDim(TILEWIDTH, TILEWIDTH, 1);
    dim3 gridDim(gridSize, gridSize, gridZ);
    size_t shmemSize = sizeof(float) * ((TILEWIDTH + gridSize-1)*(TILEWIDTH + gridSize-1) + gridSize*gridSize);
    //Launch
    d_convLayerForwardKernel<<<gridDim, blockDim, shmemSize>>>(gridSize, numInput, weightLen, d_inputMap, 
										d_weights, d_outputMap);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(outputMap, d_outputMap, outputSize, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_outputMap));
    CHECK(cudaFree(d_inputMap));
    CHECK(cudaFree(d_weights));
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
__global__ void d_convLayerForwardKernel(int gridWidth, int numInput, int weightLen, unsigned char * inputMap, 
                                                          float * weights, float * outputMap)
{
    int n, m, h_base, w_base, h, w;
    int xTileWidth = TILEWIDTH + weightLen-1;
    int iWeight = xTileWidth * xTileWidth;
    int inputLen = numInput*numInput; 
    extern __shared__ float shmem[];
    float * inputShared = &shmem[0];
    float * weightShared = &shmem[iWeight];    

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
        if ((threadIdx.x < weightLen) && (threadIdx.y < weightLen))                           
            weightShared[threadIdx.y*blockDim.x+threadIdx.x] = 
                           weights[m+weightLen*(c+numInput*(threadIdx.x+blockDim.x*threadIdx.y))]; //m,c,tIdx,tIdy
        __syncthreads();                    
        
        //Load input map into shared memory
        for (i = h; i < h_base + xTileWidth; i += TILEWIDTH) {
            for (j = w; j < w_base + xTileWidth; j += TILEWIDTH)
                inputShared[(i-h_base)*inputLen+(j-w_base)] = (float) inputMap[n+numInput*(c+CHANNELS*(h+gridWidth*w))]; //n,c,h,w
        }                                       

        __syncthreads();
        for (p = 0; p < weightLen; p++) {
            for (q = 0; q < weightLen; q++)
                acc += inputShared[(h+p)*inputLen+(w+q)] * weightShared[p*weightLen+q];
        }
        __syncthreads();
    }
    outputMap[n+numInput*(m+numInput*(h+gridWidth*w))] = acc; //n,m,h,w
}
