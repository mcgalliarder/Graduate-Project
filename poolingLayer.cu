void poolingLayer_forward(int M, int H, int W, int K, foat* Y, float* S) {
int m,h,w,p,q;
for(m = 0; m < M; m++)
 for(h = 0; x < H/K; h++)
  for(w = 0; y < W/K; y++) {
   S[m,x,y] = 0.;
   for (p = 0; p < K; p++) {
    for (q = 0; q < K; q++)
     S[m,h,w] = S[m,h,w] + Y[m, K * k + p, K * y + q]/(K*K);
    }
    // add bias and apply non-linear activation
    S[m,h,w] = sigmoid(S[m,h,w] + b[m]); // Probably need to change this
    }
}
