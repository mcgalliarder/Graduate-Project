#Sam Barr and Eli McGalliard
#6/20/2018
############  NOT COMPLETE  ############
CC = g++
CXXFLAGS = -Wall -g -std=c++0x
.SUFFIXES: .C .o
OBJS = generate.o d_convNet.o convNet.o
.C.o:
	g++ -c -g $<

d_convNet.o: d_convNet.cu d_convNet.h config.h CHECK.h

wrappers.o: wrappers.cu wrappers.h


clean:
	rm -f *.o
