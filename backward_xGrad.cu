void convLayer_backward_xgrad(int M, int C, in H_in, int W_in, int K, float * dE_dY, float* W, float * dE_dX)
{
 int m,c,h,w,p,q;
 int H_out = H_in - K+1;
 int W_out = W_in - K + 1;
 for(c=0;c<C;c++)
  for(h=0;h<H_in;h++)
   for(w=0;w<W_in;w++)
    dE_dX[c,h,w] = 0.;

 for(m=0;m<M;m++)
  for(h=0;h<H_out;h++)
   for(w=0;w<W_out;w++)
    for(c=0;c<C;c++)
     for(p=0;p<K;p++)
      for(q=0;q<K;q++)
       dE_dX[c,h+p,w+q] += dE_dY[m,h,w]*W[m,c,p,q];
}
