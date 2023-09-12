#define _GNU_SOURCE

#include <cuda.h>
#include <nvml.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "kernel.h"

#define N 40

void nvml_error(int status, int errnum, const char *format, ...)
{
	fflush(stdout);
	fprintf(stderr, "%s: %s: %s\n", program_invocation_name, format,
		nvmlErrorString(errnum));
	if (status)
		exit(status);
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

int initialize_mig(nvmlDevice_t *device, unsigned int *id)
{
	nvmlReturn_t errnum;
	nvmlGpuInstanceProfileInfo_t profile;

	errnum = nvmlInit();
	if (errnum)
		nvml_error(1, errnum, "nvmlInit");
	errnum = nvmlDeviceGetHandleByIndex(1, device);
	if (errnum)
		nvml_error(1, errnum, "nvmlDeviceGetHandleByIndex");
	errnum = nvmlDeviceGetGpuInstanceProfileInfo(*device,
						     NVML_GPU_INSTANCE_PROFILE_1_SLICE,
						     &profile);
	if (errnum)
		nvml_error(1, errnum, "nvmlDeviceGetGpuInstanceProfileInfo");
	*id = profile.id;
}

int setup_mig(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t *gi,
	      nvmlComputeInstance_t *ci)
{
	nvmlReturn_t errnum;
	nvmlComputeInstanceProfileInfo_t profile;

	errnum = nvmlDeviceCreateGpuInstance(device, id, gi);
	if (errnum)
		nvml_error(1, errnum, "nvmlDeviceCreateGpuInstance");
	errnum = nvmlGpuInstanceGetComputeInstanceProfileInfo(*gi,
							      NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE,
							      NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED,
							      &profile);
	if (errnum)
		nvml_error(1, errnum,
			   "nvmlGpuInstanceGetComputeInstanceProfileInfo");
	errnum = nvmlGpuInstanceCreateComputeInstance(*gi, profile.id, ci);
	if (errnum)
		nvml_error(1, errnum, "nvmlGpuInstanceCreateComputeInstance");
}

int teardown_mig(nvmlGpuInstance_t gi, nvmlComputeInstance_t ci)
{
	nvmlReturn_t errnum;

	errnum = nvmlComputeInstanceDestroy(ci);
	if (errnum)
		nvml_error(1, errnum, "nvmlComputeInstanceDestroy");
	errnum = nvmlGpuInstanceDestroy(gi);
	if (errnum)
		nvml_error(1, errnum, "nvmlGpuInstanceDestroy");
}

int shutdown_mig(void)
{
	nvmlReturn_t errnum;

	errnum = nvmlShutdown();
	if (errnum)
		nvml_error(1, errnum, "nvmlShutdown");
}

int main(void)
{
	nvmlDevice_t device;
	unsigned int id;
	CUfunction f;
	void **args;
	nvmlGpuInstance_t gi;
	nvmlComputeInstance_t ci;
	int i;
	struct timespec t_i;

	initialize_mig(&device, &id);
	initialize_kernel(&f, &args);
	for (i = 0; i < N; ++i) {
		clock_gettime(CLOCK_MONOTONIC, &t_i);
		setup_mig(device, id, &gi, &ci);
		teardown_mig(gi, ci);
		execute_kernel(f, args);
		printtsdelta(&t_i);
	}
	check_kernel(args);
	return 0;
}
