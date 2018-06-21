__global__ void
ConvLayerForward_Kernel(int C, int W_grid, int K, float* X, float* W, float* Y) {
int n, m, hO, wO, h_base, w_base, h, w;
int X_tile,_width = TILE_WIDTH + K-1;
extern __shared__ float shmem[];
float* X_shared = &shmem[0];
float* W_shared = &shmem[X_tile_width * X_tile_width];
n = blockIdx.x;
m = blockIdx.y;
hO = threadIdx.x; //hO and wO used as shorthand for threadIdx.x and threadIdx.y
wO = threadIdx.y;
h_base = (blockIdx.z / W_grid) * TILE_SIZE; //vertical base out data index for the block
w_base = (blockIdx.z % W_grid) * TILE_SIZE; // horizontal base out data index for the block
h = h_base + hO;
w = w_base + wO;

float acc = 0.;
int c, i, j, p, q;
for (c = 0; c < C; c++) {  			// sum over all input channels
	if ((hO < K) && (wO < K))
		W_shared[hO,wO] = W[m,c,hO,wO]; // load weights for W[m,c,..]
	__syncthreads();			// hO and wO used as shorthand for threadIdx.x and Idx.y
	
	for (i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
		for (j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
			X-shared[i - h_base, j - w_base] = X[n,c,h,w];
	}					// load tile form X[n,c,...] into shared memory
	
	__syncthreads();
	for (p = 0; p < K; p++) {
		for (q = 0; q < K; q++)
		acc = acc + X_shared[h + p, w + q] * W_shared[p,q];
	}
	__syncthreads();
 }
Y[n,m,h,w] = acc;
}
