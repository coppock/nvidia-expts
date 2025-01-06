#ifndef _BENCH_H_
#define _BENCH_H_

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda.h>

#define CHECK(call) do if (call < 0) { \
	(void)fprintf(stderr, \
		      __FILE__ ":%d: %s: Call `%s` failed: %s\n", \
		      __LINE__, __func__, #call, strerror(errno)); \
	abort(); \
} while (0)

#define CHECK_CU(call) do { \
	CUresult _check_cu_result; \
	\
	if ((_check_cu_result = call) != CUDA_SUCCESS) { \
		const char *s; \
		\
		(void)fprintf(stderr, \
			      __FILE__ ":%d: %s: CUDA call `" #call \
			      "` failed", \
			      __LINE__, __func__); \
		if (cuGetErrorString(_check_cu_result, &s) == CUDA_SUCCESS) \
			(void)fprintf(stderr, ": %s", s); \
		else \
			(void)fprintf(stderr, " with error %d", \
				      _check_cu_result); \
		(void)putchar('\n'); \
		abort(); \
	} \
} while (0)

void put_durations(struct timespec *, int);

#endif
