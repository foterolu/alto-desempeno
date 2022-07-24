#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* en un solo kernel se hace todo*/
__global__ void reduccion_atomica(int* conejos, int* suma_conejos, int* lobos, int* suma_lobos, int array_size) {
    extern __shared__ int sdata[];
    int* conejos_shared = sdata;
    int* lobos_shared = (int*)&sdata[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    int sum_conejos = 0;
    int sum_lobos = 0;


    for (int preSumIndex = i; preSumIndex < array_size; preSumIndex += gridDim.x * blockDim.x) {
        sum_conejos += conejos[preSumIndex];
        sum_lobos += lobos[preSumIndex];
    }

    conejos_shared[tid] = sum_conejos;
    lobos_shared[tid] = sum_lobos;
    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            conejos_shared[tid] += conejos_shared[tid + s];
            lobos_shared[tid] += lobos_shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(suma_conejos, conejos_shared[0]);
        atomicAdd(suma_lobos, conejos_shared[0]);
    }
}