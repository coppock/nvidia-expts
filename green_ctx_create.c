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
	size_t before, after;
	CUgreenCtx ctx;
	CUcontext context;

	CHECK_CU(cuInit(0));
	CHECK_CU(cuDeviceGet(&device, 0));
	CHECK_CU(cuDeviceGetDevResource(device, &resource,
					CU_DEV_RESOURCE_TYPE_SM));
	n = N;
	CHECK_CU(cuDevSmResourceSplitByCount(result, &n, &resource, NULL, 0,
					     0));
	CHECK_CU(cuDevResourceGenerateDesc(&desc, result, n));
	CHECK_CU(cuDevicePrimaryCtxRetain(&context, device));
	CHECK_CU(cuCtxSetCurrent(context));
	CHECK_CU(cuMemGetInfo(&before, NULL));
	CHECK_CU(cuDevicePrimaryCtxRelease(device));
	CHECK(clock_gettime(CLOCK_MONOTONIC, &t));
	for (int i = 0; i < 100; ++i)
		CHECK_CU(cuGreenCtxCreate(&ctx, desc, device,
					  CU_GREEN_CTX_DEFAULT_STREAM));
	put_durations(&t, 1);
	CHECK_CU(cuMemGetInfo(&after, NULL));
	CHECK(printf("%zu\n", before - after));
	CHECK_CU(cuCtxFromGreenCtx(&context, ctx));
	CHECK_CU(cuCtxSetCurrent(context));
	return 0;
}
