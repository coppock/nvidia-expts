#include <time.h>

#include "bench.h"

void put_durations(struct timespec *t, int n)
{
	struct timespec t_f;

	CHECK(clock_gettime(CLOCK_MONOTONIC, &t_f));
	for (int i = 0; i < n; ++i) {
		t[i].tv_nsec = (i + 1 < n ? t[i + 1].tv_nsec : t_f.tv_nsec)
			       - t[i].tv_nsec;
		if (t[i].tv_nsec < 0) {
			t[i].tv_sec -= 1;
			t[i].tv_nsec += 1000 * 1000 * 1000;
		}
		t[i].tv_sec = (i + 1 < n ? t[i + 1].tv_sec : t_f.tv_sec)
			      - t[i].tv_sec;
		CHECK(printf(i ? "\t%ld.%09ld" : "%ld.%09ld", t[i].tv_sec,
			     t[i].tv_nsec));
	}
	CHECK(putchar('\n'));
}
