#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
//config.h defines the TILE_WIDTH
//and the constants: SIMPLE, TILED, TILED2
//that indicate which kernel to launch
#include "config.h"
#include "d_pooling.h"

//prototypes for kernels in this file


void d_pooling(float * d_inputMap, float * d_outputMap, int width)
{
    cudaEvent_t start_gpu, stop_gpu;
    float gpuMsecTime = -1;
    float * d_inputMap;
    float * d_outputMap;
    int size = sizeof(float) * (width * width);

    //time the sum
    CHECK(cudaEventCreate(&start_gpu));
    CHECK(cudaEventCreate(&stop_gpu));
    CHECK(cudaEventRecord(start_gpu));

    //Your work goes here
    //kernel calls provided but you need to write the code for the
    //memory allocations, etc. and define the grid and the block
    //Use TILE_SIZE (defined in config.h)
    CHECK(cudaMalloc((void **)&d_inputMap,size));
    CHECK(cudaMalloc((void **)&d_outputMap,size));

    CHECK(cudaMemcpy(d_inputMap, inputMap, size, cudaMemcpyHostToDevice));

    dim3 dimGrid(ceil(width/TILE_WIDTH),ceil(width/TILE_WIDTH),1);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
    dim3 dimHalfBlock(TILE_WIDTH,TILE_WIDTH/2,1);

    
    d_poolingKernel<<<dimGrid, dimBlock>>>(d_matrixM, d_matrixN, d_result, width);

    CHECK(cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_inputMap));
    CHECK(cudaFree(d_outputMap));
}


__global__ void d_poolingKernel(float * d_inputMap, float * d_outputMap, float * bias, int width)
{

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int acc = 0.0;
        int index = c+numInput*(row*blockDim.x+col);
        //Go over all input feature maps
        for (int c = 0; i < numInput; c++) {
	    //definirely not finished yo            
            for (p = 0; p < numOutput; p++) 
                for (q = 0; q < numOutput; q++) {
                    d_outputMap[index] = d_outputMap[index] + 
                         d_inputMap[c+numInput*((numInput*col+p)*width+(numInput*row+q))]
                               /(numOutput*numOutput);
            }
            //Add bias and sigmoid
            d_output[index] = sigmoid(d_output[index] + bias[c]);
        }


        if ((row < width) && (col < width)) {
                float pValue = 0;
                for (int k = 0; k < width; ++k) {
                        pValue += d_matrixM[row * width + k]; 
                }
	    pValue /= (width * width); //Get the average of the current pValue
            d_result[row * width + col] = d_sigmoid(pValue);
            
        }
}

__global__ int d_sigmoid(float x) {
    return x/(1 + abs(x));
}

