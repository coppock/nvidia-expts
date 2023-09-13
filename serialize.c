#define _GNU_SOURCE

#include <cuda.h>

#include <err.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#define N 40
#define NBLOCKS 64 * 1024

void cu_error(int status, int errnum, const char *format, ...)
{
	const char *errstr;
	int written;
	char fmt[BUFSIZ], str[BUFSIZ];
	va_list args;

	(void)fflush(stdout);
	if (errnum) {
		errstr = "";
		(void)cuGetErrorString(errnum, &errstr);
		(void)snprintf(fmt, BUFSIZ, "%s: %s: %s\n",
			       program_invocation_name, format, errstr);
	} else {
		(void)snprintf(fmt, BUFSIZ, "%s: %s\n",
			       program_invocation_name, format);
	}
	va_start(args, format);
	(void)vsnprintf(str, BUFSIZ, fmt, args);
	va_end(args);
	(void)fputs(str, stderr);
	if (status)
		exit(status);
}

int initialize_kernel(CUfunction *f, void ***args)
{
	CUresult errnum;
	CUdevice dev;
	CUcontext ctx;
	CUmodule mod;
	float *a;
	int i;
	struct timespec t_i;

	errnum = cuInit(0);
	if (errnum != 0)
		cu_error(1, errnum, "cuInit()");
	errnum = cuDeviceGet(&dev, 0);
	if (errnum != 0)
		cu_error(1, errnum, "cuDeviceGet()");
	errnum = cuCtxCreate(&ctx, 0, dev);
	if (errnum != 0)
		cu_error(1, errnum, "cuCtxCreate()");
	errnum = cuModuleLoad(&mod, "reciprocate.ptx");
	if (errnum != 0)
		cu_error(1, errnum, "cuModuleLoad()");
	errnum = cuModuleGetFunction(f, mod, "reciprocate");
	if (errnum != 0)
		cu_error(1, errnum, "cuModuleGetFunction()");
	*args = malloc(2 * sizeof(**args));
	if (*args == NULL)
		err(1, "malloc(2 * sizeof(*args))");
	(*args)[0] = malloc(sizeof(CUdeviceptr));
	if ((*args)[0] == NULL)
		err(1, "malloc(sizeof(CUdeviceptr))");
	errnum = cuMemAlloc((*args)[0], NBLOCKS * sizeof(float));
	if (errnum != 0)
		cu_error(1, errnum, "cuMemAlloc()");
	a = malloc(NBLOCKS * sizeof(*a));
	if (a == NULL)
		err(1, "malloc()");
	for (i = 0; i < NBLOCKS; ++i)
		a[i] = 3.14159;
	errnum = cuMemcpyHtoD(*(CUdeviceptr *)(*args)[0], a,
			      NBLOCKS * sizeof(*a));
	if (errnum != 0)
		cu_error(1, errnum, "cuMemcpyHtoD()");
	(*args)[1] = malloc(sizeof(int));
	if ((*args)[1] == NULL)
		err(1, "malloc(sizeof(int))");
	*(int *)(*args)[1] = 1;
	free(a);
}

int execute_kernel(CUfunction f, void **args, int nchunks)
{
	CUresult errnum;
	CUdeviceptr old;
	int i;

	old = *(CUdeviceptr *)args[0];
	for (i = 0; i < nchunks; ++i) {
		errnum = cuLaunchKernel(f, NBLOCKS / nchunks, 1, 1, 1, 1, 1, 0,
					NULL, args, NULL);
		if (errnum != 0)
			cu_error(1, errnum, "cuLaunchKernel()");
		*(CUdeviceptr *)args[0] = (CUdeviceptr)((float *)*(CUdeviceptr *)args[0]
							+ NBLOCKS / nchunks);
	}
	errnum = cuCtxSynchronize();
	if (errnum != 0)
		cu_error(1, errnum, "cuCtxSynchronize()");
	*(CUdeviceptr *)args[0] = old;
}

int check_kernel(void **args)
{
	CUresult errnum;
	float *a;
	int i;

	a = malloc(NBLOCKS * sizeof(*a));
	if (a == NULL)
		err(1, "malloc()");
	errnum = cuMemcpyDtoH(a, *(CUdeviceptr *)args[0],
			      NBLOCKS * sizeof(*a));
	if (errnum != 0)
		cu_error(1, errnum, "cuMemcpyDtoH()");
	for (i = 0; i < NBLOCKS; ++i)
		if (a[i] != 1.f / 3.14159f)
			errx(1, "a[%d] (%f) != 1. / 3.14159", i, a[i]);
	free(a);
}

void deinitialize_kernel(void **args)
{
	CUresult errnum;

	errnum = cuMemFree(*(CUdeviceptr *)args[0]);
	if (errnum != 0)
		cu_error(1, errnum, "cuMemFree()");
}

void printtsdelta(struct timespec *t_i)
{
	struct timespec t_f;

	clock_gettime(CLOCK_MONOTONIC, &t_f);
	t_f.tv_sec -= t_i->tv_sec;
	t_f.tv_nsec -= t_i->tv_nsec;
	if (t_f.tv_nsec < 0) {
		t_f.tv_sec -= 1;
		t_f.tv_nsec += 1000 * 1000 * 1000;
	}
	printf("%ld.%09ld\n", t_f.tv_sec, t_f.tv_nsec);
}

int main(void)
{
	int nchunks;
	struct timespec t_i;
	void **args;
	CUfunction f;

	initialize_kernel(&f, &args);
	for (nchunks = 1; nchunks <= NBLOCKS; nchunks <<= 1) {
		clock_gettime(CLOCK_MONOTONIC, &t_i);
		execute_kernel(f, args, nchunks);
		printtsdelta(&t_i);
		check_kernel(args);
		execute_kernel(f, args, 1);
	}
	deinitialize_kernel(args);
}
