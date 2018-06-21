void convLayer_forward(int N, int M, int C, int H, int W, int K, float* X, float*W_unroll, float* Y)
{
int W_out = W-K+1;
int H_out = H-K+1;
int W_unroll = C*K*K;
int H_unroll = H_out * W_out;
float* X_unrolled = malloc(W_unroll * H_unroll * sizeof(float));
for (int n = 0; n < N; n++) {
 unroll(C,H,W,K,n,X,X_unrolled);
 gemm(H_unroll,M,W_unroll,X_unrolled,W,Y[n]);
}
}
