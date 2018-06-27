void convLayer_backward_wgrad(int M, int C, int H, int W, int K, float* dE_dY, float * X, float * dE_dW) 
{
 int m,c,h,w,p,q;
 int H_out = H-K+1;
 int W_out = W-K+1;
 for(m = 0; m < M; m++)
  for(c = 0; c < C; c++)
   for(p = 0; p < K; p++)
    for(q = 0; q < K; q++)
     dE_dW[m,c,p,q] = 0.;

for(m = 0; m < M; m++)
 for(h = 0; h < H_out; h++)
  for(w = 0; w < W_out; w++)
   for(c = 0; c < C; c++)
    for(p = 0; p < K; p++)
     for(q = 0; q < K; q++)
      dE_dW[m,c,p,q] += X[c,h+p,w+q] * dE_dY[m,c,h,w];
}
