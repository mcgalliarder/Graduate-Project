
__global__ void d_convLayerForwardKernel(int, int, int, unsigned char *, float *, float *);
void  d_convLayerForward(unsigned char * inputMap, unsigned char * weights,
                                            int numInput, int size, unsigned char * result)

