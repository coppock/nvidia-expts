#define _GNU_SOURCE

#include <cuda.h>

#include <err.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#define N 40

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
	float a;
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
	errnum = cuMemAlloc((*args)[0], sizeof(float));
	if (errnum != 0)
		cu_error(1, errnum, "cuMemAlloc()");
	a = 3.14159;
	errnum = cuMemcpyHtoD(*(CUdeviceptr *)(*args)[0], &a, sizeof(a));
	if (errnum != 0)
		cu_error(1, errnum, "cuMemcpyHtoD()");
	(*args)[1] = malloc(sizeof(int));
	if ((*args)[1] == NULL)
		err(1, "malloc(sizeof(int))");
	*(int *)(*args)[1] = 1;
}

int execute_kernel(CUfunction f, void **args)
{
	CUresult errnum;

	errnum = cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
	if (errnum != 0)
		cu_error(1, errnum, "cuLaunchKernel()");
	errnum = cuCtxSynchronize();
	if (errnum != 0)
		cu_error(1, errnum, "cuCtxSynchronize()");
}

int check_kernel(void **args)
{
	CUresult errnum;
	float a;

	errnum = cuMemcpyDtoH(&a, *(CUdeviceptr *)args[0], sizeof(a));
	if (errnum != 0)
		cu_error(1, errnum, "cuMemcpyDtoH()");
	errnum = cuMemFree(*(CUdeviceptr *)args[0]);
	if (errnum != 0)
		cu_error(1, errnum, "cuMemFree()");
	printf("%f\n", a);
}
