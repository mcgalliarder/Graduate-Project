#Sam Barr and Eli McGalliard
#6/20/2018

NVCC = /usr/local/cuda-8.0/bin/nvcc
CC = g++
GENCODE_FLAGS = -arch=sm_30
CXXFLAGS = -Wall -g -std=c++0x
CC_FLAGS = -c
#NVCCFLAGS = -m64 -O3 -Xptxas -v
NVCCFLAGS = -g -G -m64
.SUFFIXES: .cu .o .h

OBJS =  d_forwardPropagation.o d_pooling.o convNet.o wrappers.o
.cu.o:
	$(NVCC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

#all: convNet

convNet: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -o convNet

d_forwardPropagation.o: d_forwardPropagation.cu d_forwardPropagation.h CHECK.h

convNet.o: wrappers.h 

wrappers.o: wrappers.cu wrappers.h

d_pooling.o: d_pooling.cu d_pooling.h

clean:
	rm -f convNet *.o
