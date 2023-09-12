#define _GNU_SOURCE

#include <cuda.h>
#include <nvml.h>

#include <err.h>
#include <errno.h>
#include <error.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>

#define NTHREADS 2

CUdevice device;
size_t nstamps = 1024 * 1024 * 1024;
pthread_barrier_t barrier;

struct execution {
	CUcontext ctx;
	CUmodule mod;
	CUfunction f;
	CUdeviceptr p;
};

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

void nvml_error(int status, int errnum, const char *format, ...)
{
	fflush(stdout);
	fprintf(stderr, "%s: %s: %s\n", program_invocation_name, format,
		nvmlErrorString(errnum));
	if (status)
		exit(status);
}

void *start(void *_)
{
	CUresult cu_errnum;
	CUcontext ctx;
	CUmodule mod;
	CUfunction f;
	void **args;
	CUdeviceptr p;
	int errnum;
	clock_t *stamps;

	cu_errnum = cuCtxCreate(&ctx, 0, device);
	if (cu_errnum != 0)
		cu_error(1, cu_errnum, "cuCtxCreate()");
	cu_errnum = cuModuleLoad(&mod, "read_clock.ptx");
	if (cu_errnum != 0)
		cu_error(1, cu_errnum, "cuModuleLoad()");
	cu_errnum = cuModuleGetFunction(&f, mod, "read_clock");
	if (cu_errnum != 0)
		cu_error(1, cu_errnum, "cuModuleGetFunction()");
	args = malloc(2 * sizeof(*args));
	if (args == NULL)
		err(1, "malloc(2 * sizeof(*args))");
	args[0] = &p;
	args[1] = &nstamps;
	cu_errnum = cuMemAlloc(&p, nstamps * sizeof(*stamps));
	errnum = pthread_barrier_wait(&barrier);
	if (errnum != 0 && errnum != PTHREAD_BARRIER_SERIAL_THREAD)
		error(1, cu_errnum, "pthread_barrier_wait()");
	cu_errnum = cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
	if (cu_errnum != 0)
		cu_error(1, cu_errnum, "cuLaunchKernel()");
	cu_errnum = cuCtxSynchronize();
	if (cu_errnum != 0)
		cu_error(1, cu_errnum, "cuCtxSynchronize()");
	stamps = malloc(nstamps * sizeof(*stamps));
	if (stamps == NULL)
		err(1, "malloc(%zu * sizeof(*stamps)", nstamps);
	cu_errnum = cuMemcpyDtoH(stamps, p, nstamps * sizeof(*stamps));
	if (cu_errnum != 0)
		cu_error(1, cu_errnum, "cuMemcpyDtoH()");
	cu_errnum = cuMemFree(p);
	if (cu_errnum != 0)
		cu_error(1, cu_errnum, "cuMemFree()");
	return stamps;
}

int main(void)
{
	CUresult cu_errnum;
	size_t i, j;
	int errnum;
	pthread_t threads[NTHREADS];
	clock_t *stamps[NTHREADS], t, end, last0, last1;
	nvmlReturn_t nvml_errnum;
	nvmlDevice_t nvml_dev;
	unsigned int freq;
	float now, then, last_other;

	errnum = cuInit(0);
	if (errnum != 0)
		cu_error(1, errnum, "cuInit()");
	errnum = cuDeviceGet(&device, 0);
	if (errnum != 0)
		cu_error(1, errnum, "cuDeviceGet()");

	errnum = pthread_barrier_init(&barrier, NULL, NTHREADS);
	if (errnum != 0)
		error(1, errnum, "pthread_barrier_init()");
	for (i = 0; i < NTHREADS; ++i) {
		errnum = pthread_create(&threads[i], NULL, start, NULL);
		if (errnum != 0)
			error(1, errnum, "pthread_create()");
	}
	nvml_errnum = nvmlInit();
	if (nvml_errnum != 0)
		nvml_error(1, nvml_errnum, "nvmlInit()");
	nvml_errnum = nvmlDeviceGetHandleByIndex(0, &nvml_dev);
	if (nvml_errnum != 0)
		nvml_error(1, nvml_errnum, "nvmlDeviceGetHandleByIndex()");
	nvml_errnum = nvmlDeviceGetClock(nvml_dev, NVML_CLOCK_SM,
					 NVML_CLOCK_ID_CURRENT, &freq);
	if (errnum != 0)
		nvml_error(1, nvml_errnum, "nvmlDeviceGetClock()");
	for (i = 0; i < NTHREADS; ++i) {
		errnum = pthread_join(threads[i], (void **)&stamps[i]);
		if (errnum != 0)
			error(1, errnum, "pthread_join()");
	}
	printf("Thread 0 did%s start before Thread 1 (%ld, %ld)\n",
	       stamps[0][0] < stamps[1][0] ? "" : "n't", stamps[0][0],
	       stamps[1][0]);
	t = stamps[0][0] < stamps[1][0] ? stamps[1][0] : stamps[0][0];
	printf("Thread 0 did%s end before Thread 1 (%ld, %ld)\n",
	       stamps[0][nstamps - 1] < stamps[1][nstamps - 1] ? "" : "n't",
	       stamps[0][nstamps - 1], stamps[1][nstamps - 1]);
	end = stamps[0][nstamps - 1] < stamps[1][nstamps - 1] ?
	    stamps[0][nstamps - 1] : stamps[1][nstamps - 1];
	if (t > end)
		errx(1, "Kernel executions didn't overlap in time!");
	for (i = 0; stamps[0][i] < t; ++i);
	last0 = stamps[0][i++];
	for (j = 0; stamps[1][j] < t; ++j);
	last1 = stamps[1][j++];
	while (t++ <= end) {
		if (stamps[0][i] < t) {
			/*
			 * Filter out low values.
			 *
			 * When running one kernel on its own, differences
			 * follow an apparently deterministic pattern: 20, 20,
			 * 20, 51, 20, 20, 20, 51, ....
			 */
			if (stamps[0][i] - last0 > 300) {
				now = (float)stamps[0][i] / freq / 1000 / 1000;
				then = (float)last0 / freq / 1000 / 1000;
				last_other = (float)last1 / freq / 1000 / 1000;
				printf("0\t%f\t%f\t%f\t%ld\t%ld\n", now, now - then,
				       now - last_other, stamps[0][i], last1);
			}
			last0 = stamps[0][i++];
		}
		if (stamps[1][j] < t) {
			if (stamps[1][j] - last1 > 300) {
				now = (float)stamps[1][j] / freq / 1000 / 1000;
				then = (float)last1 / freq / 1000 / 1000;
				last_other = (float)last0 / freq / 1000 / 1000;
				printf("1\t%f\t%f\t%f\t%ld\t%ld\n", now, now - then,
				       now - last_other, stamps[1][j], last0);
			}
			last1 = stamps[1][j++];
		}
	}
	return 0;
}
