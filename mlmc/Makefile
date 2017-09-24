
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

GCCFLAGS 	:= -std=c++11
NVCCFLAGS	:= -lineinfo -arch=sm_53 --ptxas-options=-v --use_fast_math

all:	mlmc 

main.o:	main.cpp Makefile
	g++ -c -fPIC -o main.o main.cpp  $(GCCFLAGS)

mlmc_gpu.o: mlmc_first.cu Makefile
	nvcc mlmc_first.cu -c -o mlmc_gpu.o $(INC) $(NVCCFLAGS) $(LIB)	

mlmc.o:	first_mlmc.cpp Makefile
	g++ -c -fPIC first_mlmc.cpp mlmc_cpu_estimator.cpp $(GCCFLAGS)

mlmc:	main.o mlmc.o mlmc_gpu.o Makefile
	g++ -fPIC -o mlmc *.o $(LIB) $(GCCFLAGS)

clean:
	rm -f *.o *.a mlmc
