CUDA_PATH = /usr/local/cuda

CFLAGS = -g -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib
LDLIBS = -lcuda -lnvidia-ml

.SUFFIXES: .ptx .cu

.cu.ptx:
	nvcc -g -ptx -o $@ $<

all: green_ctx_create mig slicing reciprocate.ptx read_clock.ptx

bench.o: bench.h
green_ctx_create.o: bench.h

green_ctx_create: green_ctx_create.o bench.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o green_ctx_create green_ctx_create.o \
	    bench.o $(LDLIBS)

mig.o: kernel.h

mig: mig.o kernel.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o mig mig.o kernel.o $(LDLIBS)

clean:
	rm -f *.o
