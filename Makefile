# cuda compiler
NVCC = nvcc
# compile options
NVCCFLAGS =  -Xcompiler "-fopenmp" -Xcompiler -Wall -rdc=true -lmpi

# C compiler
CC = g++
# compile options
CFLAGS = -Wall -O3 -std=c++11 -fopenmp -lm

# MPI compiler
MPICC = mpicc
# MPI compile options
MPICFLAGS = -Wall -O3 -fopenmp


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda11/lib64

# link options
LDFLAGS =  -L/usr/local/cuda/lib64 -lcudart -lcudadevrt  

# source files
CFILES = $(wildcard *.c)
CUFILES = $(wildcard *.cu)
OBJECTS = $(CFILES:.c=.o) $(CUFILES:.cu=.o)
EXECUTABLE = mymoc


# target
all: 
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) $(CUFILES)
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) $(CFILES)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(OBJECTS) -o $(EXECUTABLE)
# clean
clean:
	rm -f ./*.o
	rm -f $(EXECUTABLE)

# rebuild
rebuild:
	@make clean
	@make all

.PHONY: all clean rebuild