#include <time.h>
#include <nvml.h>
#include "bench.h"

int main(void)
{
	nvmlDevice_t device;

	CHECK_NVML(nvmlInit());
	CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &device));
	for (int f = 210; f <= 1440; f += 15) {
		struct timespec t;

		CHECK(clock_gettime(CLOCK_MONOTONIC, &t));
		CHECK_NVML(nvmlDeviceSetGpuLockedClocks(device, f, f));
		put_durations(&t, 1);
	}
	CHECK_NVML(nvmlShutdown());
	return 0;
}
