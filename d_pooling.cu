#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "config.h"
#include "d_pooling.h"

//prototypes for kernels in this file
__global__ void d_poolingKernel(float *, float *, int, int, int);
__device__ int d_sigmoid(float x); 

void d_pooling(float * inputMap, float * outputMap, int numInput, int weightLen, int width)
{
    float * d_inputMap;
    float * d_outputMap;
    int size = sizeof(float) * (width * width);

    CHECK(cudaMalloc((void **)&d_inputMap,size));
    CHECK(cudaMalloc((void **)&d_outputMap,size));

    CHECK(cudaMemcpy(d_inputMap, inputMap, size, cudaMemcpyHostToDevice));

    dim3 dimGrid(ceil(width/TILEWIDTH),ceil(width/TILEWIDTH),1);
    dim3 dimBlock(TILEWIDTH,TILEWIDTH,1);
    
    d_poolingKernel<<<dimGrid, dimBlock>>>(inputMap, outputMap, numInput, width, weightLen);

    CHECK(cudaMemcpy(outputMap, d_outputMap, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_inputMap));
    CHECK(cudaFree(d_outputMap));
}


/**
 * Creates an averaged subsample of the input map
 * @param d_inputMap
 * @param d_outputMap
 * @param numInput number of inputs
 * @param width of input
 * @param weightLen length of weight vector
 */
__global__ void d_poolingKernel(float * d_inputMap, float * d_outputMap, int numInput, int width, int weightLen)
{

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
   
        //Go over all input feature maps
        for (int c = 0; c < numInput; c++){        
            int index = c+numInput*(row*blockDim.x+col);
            for (int p = 0; p < weightLen; p++) 
                for (int q = 0; q < weightLen; q++) {
                    d_outputMap[index] = d_outputMap[index] + 
                         d_inputMap[c+numInput*((width*col+p)*width+(width*row+q))]
                               /(weightLen*weightLen);
            }
            //Add bias and sigmoid
            d_outputMap[index] = d_sigmoid(d_outputMap[index]);
        }

}

__device__ int d_sigmoid(float x) {
    return x/(1 + abs(x));
}

