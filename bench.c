#include <time.h>

#include "bench.h"

void put_duration(struct timespec t_i)
{
	struct timespec t_f;

	CHECK(clock_gettime(CLOCK_MONOTONIC, &t_f));
	t_f.tv_nsec -= t_i.tv_nsec;
	if (t_f.tv_nsec < 0) {
		t_f.tv_sec -= 1;
		t_f.tv_nsec += 1000 * 1000 * 1000;
	}
	t_f.tv_sec -= t_i.tv_sec;
	CHECK(printf("%d.%09d\n", t_f.tv_sec, t_f.tv_nsec));
}
