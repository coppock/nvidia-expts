#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <cuda.h>

int initialize_kernel(CUfunction *, void ***args);
int execute_kernel(CUfunction, void **args);
int check_kernel(void **args);

#endif /* !defined(_KERNEL_H_) */
