#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
//#include "d_colorToGreyscale.h"
#include "CHECK.h"
#include "config.h"


__global__ void d_convLayerForwardKernel(int, int, int, unsigned char *, int, float *, float *);
__device__ void printVector(float * array, int width);
__device__ void printCharVector(unsigned char * array, int width);

/**
 * Performs one forward run through the network
 * @param Pin input image
 * @param resulting vector
 * @param size of input image
 */
void d_convLayerForward(unsigned char * inputMap, float * outputMap, float * weights, 
                                                          int inputLen, int numInput, int weightLen)
{
    //Create device vectors 
    unsigned char * d_inputMap;
    float * d_weights;
    float * d_outputMap;
    

    int inSize = sizeof(unsigned char)*inputLen*inputLen*numInput;
    int outputSize = sizeof(float)*(inputLen-weightLen-1)*(inputLen-weightLen-1);
    int weightSize = sizeof(float)*weightLen*weightLen;
    CHECK(cudaMalloc((void **)&d_outputMap, outputSize));
    CHECK(cudaMalloc((void **)&d_weights, weightSize));
    CHECK(cudaMalloc((void **)&d_inputMap, inSize));

    CHECK(cudaMemcpy(d_inputMap, inputMap, inSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weights, weights, weightSize, cudaMemcpyHostToDevice));

    //Launch
    int outSize = inputLen - (weightLen-1);
    int gridSize = ceil(outSize/TILEWIDTH);
    int gridZ = gridSize * gridSize;
    dim3 blockDim(TILEWIDTH, TILEWIDTH, 1);
    dim3 gridDim(gridSize, gridSize, gridZ);
    size_t shmemSize = sizeof(float) * ((TILEWIDTH + weightLen-1)*(TILEWIDTH + weightLen-1) + weightLen*weightLen);
    //Launch
    d_convLayerForwardKernel<<<gridDim, blockDim, shmemSize>>>(gridSize, numInput, weightLen, d_inputMap, 
									inputLen, d_weights, d_outputMap);
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
                                                          int inputLen, float * weights, float * outputMap)
{
    int n, m, h_base, w_base, h, w;
    int xTileWidth = TILEWIDTH + weightLen-1;
    int iWeight = xTileWidth * xTileWidth;
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
        int wIndex = threadIdx.x*TILEWIDTH+threadIdx.y;
        if (wIndex < weightLen*weightLen) {                           
            weightShared[threadIdx.x*TILEWIDTH+threadIdx.y] = 
                           weights[(c+numInput*(threadIdx.x*TILEWIDTH+threadIdx.y))]; //m,c,tIdx,tIdy
        }
        __syncthreads();                   
        //Load input map into shared memory
        for (i = h; i < h_base + xTileWidth; i += TILEWIDTH) {
            for (j = w; j < w_base + xTileWidth; j += TILEWIDTH){
                inputShared[(i-h_base)*TILEWIDTH+(j-w_base)] = 
                           ((float) inputMap[n+gridWidth*(c+numInput*(h+TILEWIDTH*w))]); //n,c,h,w
                
            }
        }                                       

        __syncthreads();
        //Accumulate input and weight vectors
        for (p = 0; p < weightLen; p++) {
            for (q = 0; q < weightLen; q++) {
                acc += inputShared[(h+p)*TILEWIDTH+(w+q)] * weightShared[p*weightLen+q];
            }
        }
        __syncthreads();
    }
    
    //Load into output
    outputMap[n+gridWidth*(m+gridWidth*(h+TILEWIDTH*w))] = acc; //n,m,h,w
}

__device__ void printVector(float * array, int width)
{
    for (int i = 0; i < width*width; i++) 
    {
        if (!(i%width)) printf("\n%2d:", i/width);
        printf("%6.1f", array[i]);
    }
    printf("\n");
}

__device__ void printCharVector(unsigned char * array, int width)
{
    for (int i = 0; i < width*width; i++)
    {
        if (!(i%width)) printf("\n%2d:", i/width);
        printf("%4.1d", array[i]);
    }
    printf("\n");
}
