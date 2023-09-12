CUDA_PATH = /usr/local/cuda

CFLAGS = -g -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib
LDLIBS = -lcuda -lnvidia-ml

.SUFFIXES: .ptx .cu

.cu.ptx:
	nvcc -g -ptx -o $@ $<

all: mig slicing reciprocate.ptx read_clock.ptx

mig: mig.c kernel.c

slicing: slicing.c

clean:
	rm -f mig slicing *.ptx
