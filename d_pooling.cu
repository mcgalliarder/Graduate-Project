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


void d_pooling(float * matrixM, float * matrixN, float * result,
                    int width)
{
    cudaEvent_t start_gpu, stop_gpu;
    float gpuMsecTime = -1;
    float * d_matrixM;
    float * d_matrixN;
    float * d_result;
    int size = sizeof(float) * (width * width);

    //time the sum
    CHECK(cudaEventCreate(&start_gpu));
    CHECK(cudaEventCreate(&stop_gpu));
    CHECK(cudaEventRecord(start_gpu));

    //Your work goes here
    //kernel calls provided but you need to write the code for the
    //memory allocations, etc. and define the grid and the block
    //Use TILE_SIZE (defined in config.h)
    CHECK(cudaMalloc((void **)&d_matrixM,size));
    CHECK(cudaMalloc((void **)&d_matrixN,size));
    CHECK(cudaMalloc((void **)&d_result,size));

    CHECK(cudaMemcpy(d_matrixM, matrixM, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_matrixN, matrixN, size, cudaMemcpyHostToDevice));

    dim3 dimGrid(ceil(width/TILE_WIDTH),ceil(width/TILE_WIDTH),1);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
    
    d_poolingKernel<<<dimGrid, dimBlock>>>(d_matrixM, d_matrixN, d_result, width);

    CHECK(cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_matrixM));
    CHECK(cudaFree(d_matrixN));
    CHECK(cudaFree(d_result));
}


__global__ void d_poolingKernel(float * d_matrixM, float * d_matrixN,
                                          float * d_result, int width)
{

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if ((row < width) && (col < width)) {
                float pValue = 0;
                for (int k = 0; k < width; ++k) {
                        pValue += d_matrixM[row * width + k] * d_matrixN[k * width + col];
                }
	pValue /= (width * width); //Get the average of the current pValue
        d_result[row * width + col] = pValue;
        }
}

