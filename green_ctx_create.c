#include <time.h>
#include <cuda.h>
#include "bench.h"

#define N 2

int main(void)
{
	CUdevice device;
	CUdevResource resource, result[N];
	unsigned n;
	struct timespec t;
	CUdevResourceDesc desc;
	CUgreenCtx ctx;

	CHECK_CU(cuInit(0));
	CHECK_CU(cuDeviceGet(&device, 0));
	CHECK_CU(cuDeviceGetDevResource(device, &resource,
					CU_DEV_RESOURCE_TYPE_SM));
	n = N;
	CHECK_CU(cuDevSmResourceSplitByCount(result, &n, &resource, NULL, 0,
					     0));
	CHECK(clock_gettime(CLOCK_MONOTONIC, &t));
	CHECK_CU(cuDevResourceGenerateDesc(&desc, result, n));
	CHECK_CU(cuGreenCtxCreate(&ctx, desc, device,
				  CU_GREEN_CTX_DEFAULT_STREAM));
	put_duration(t);
	return 0;
}
