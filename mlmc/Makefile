
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math
# NVCCFLAGS	:= -lineinfo -arch=sm_50 --ptxas-options=-v --use_fast_math

all:	mlmc 

main.o:	main.cpp Makefile
	g++ -c -fPIC -o main.o main.cpp

mlmc.o:	mlmc.cu Makefile
	nvcc mlmc.cu -c -o mlmc.o $(INC) $(NVCCFLAGS) $(LIB)	

mlmc:	main.o mlmc.o Makefile
	g++ -fPIC -o mlmc main.o mlmc.o $(LIB)

clean:
	rm -f *.o *.a mlmc
