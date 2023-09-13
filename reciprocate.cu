extern "C" {
	__global__ void reciprocate(float *a, int n)
	{
		int i;

		for (i = 0; i < n; ++i)
			a[blockIdx.x] = 1. / a[blockIdx.x];
	}
}
