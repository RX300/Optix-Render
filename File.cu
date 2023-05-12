#include<cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void init_rng(curandState* state, unsigned long long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

extern "C" void initOneState(curandState* devStates) {
	init_rng << <1, 1 >> > (devStates, 1);
}