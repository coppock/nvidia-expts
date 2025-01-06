#ifndef _BENCH_H_
#define _BENCH_H_

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>

#define CHECK(call) do if (call < 0) { \
	(void)fprintf(stderr, \
		      __FILE__ ":%d: %s: Call `" #call "` failed: %s\n", \
		      __LINE__, __func__, strerror(errno)); \
	abort(); \
} while (0)

#define CHECK_CU(call) do { \
	CUresult result; \
	\
	if ((result = call) != CUDA_SUCCESS) { \
		const char *s; \
		\
		(void)fprintf(stderr, \
			      __FILE__ ":%d: %s: CUDA call `" #call \
			      "` failed", \
			      __LINE__, __func__); \
		if (cuGetErrorString(result, &s) == CUDA_SUCCESS) \
			(void)fprintf(stderr, ": %s", s); \
		else (void)fprintf(stderr, " with error %d", result); \
		putchar('\n'); \
		abort(); \
	} \
} while (0)

void put_duration(struct timespec);

#endif
