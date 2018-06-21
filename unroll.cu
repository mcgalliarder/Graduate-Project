void unroll(int C, int H, int W, int K, float* X, float* X_unroll) {
int c,h,w,p,q,w_base,w_unroll.h_unroll;
int H_out = H-K+1;
int W_out = W-K+1;
for (c = 0; c < C; c++) {
 w_base = c*(K*K);
 for (p = 0; p < K; p++)
  for (q = 0; q < K; q++){
   for (h = 0; h < H_out; h++)
    for (w = 0; w < W_out; w++){
  	w_unroll = w_base + p*K+q;
	h_unroll = h*W_out + w;
	X_unroll(h_unroll,w_unroll) = X(c,h+p,w+q);
}
}
}
