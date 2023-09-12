extern "C" {
	__global__ void read_clock(clock_t *stamps, size_t n)
	{
		size_t i;

		for (i = 0; i < n; ++i)
			stamps[i] = clock();
	}
}
