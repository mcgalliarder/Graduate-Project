__global__ void unroll_Kernel(int C, int H, int W, int K, float* X, float* X_unroll)
{
int c,s,h_out,w_out,h_unroll,w_base,p,q;
int t = blockIdx.x * CUDA MAX_NUM_THREADS + threadIdx.x;
int H_out = H-K+1;
int W_out = W-K+1;
int W_unroll = H_out*W_out;

if (t<C*W_unroll) {
c = t/W_unroll;
s = t%W_unroll;
h_out = s/W_out;
w_out = s%W_out;
h_unroll = h_out * W_out + w_out;
w_base = c * K * K;
for(p=0; p < K; p++)
 for (q=0; q<K; q++) {
  w_unroll = w_base + p * K + q;
  X_unroll(h_unroll, w_unroll) = X(c,h_out+p,w_out+q);
 }
}
}
