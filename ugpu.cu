void unroll_gpu(int C, int H, int W, int K, float* X, float* X_unroll) {
 int H_out = H - K + 1;
 int W_out = W - K + 1;
int num_threads = C * H_out * W_out;
int num_blocks = ceil((C * H_out * W_out) / CUDA MAX_NUM_THREADS);
unroll_kernel<<<num_blocks,CUDA MAX_NUM_THREADS>>>();
}
