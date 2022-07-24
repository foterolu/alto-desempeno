#pragma once

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define MIN 0
#define MAX 4


__global__ void generar_direcciones(curandState* state, int* direcciones_c, int* direcciones_l, int M, int N) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < M * N) {
		direcciones_c[tid] = (int)((curand_uniform(&state[tid]) - 1e-14) * (MAX - MIN + 1));
		direcciones_l[tid] = (int)((curand_uniform(&state[tid]) - 1e-14) * (MAX - MIN + 1));
	}
}

__global__ void movimiento(int* conejos, int* lobos, int* buffer1_c, int* buffer1_l, int M, int N, int* direcciones_c, int* direcciones_l) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < M * N) {
		int y = tid % N;
		int x = tid / N;
		buffer1_c[tid] =
			// Conejos que estaban aqui antes
			conejos[tid] * (direcciones_c[tid] == 0) +
			// Conejos que vienen de la derecha
			conejos[x * N + ((y + 1) % N)] * (direcciones_c[x * N + ((y + 1) % N)] == 3) +
			// Conejos que vienen de arriba
			conejos[N * ((x - 1 + M) % M) + y] * (direcciones_c[N * ((x - 1 + M) % M) + y] == 4) +
			// Conejos que vienen de la izquierda
			conejos[x * N + ((y - 1 + N) % N)] * (direcciones_c[x * N + ((y - 1 + N) % N)] == 1) +
			// Conejos que vienen de abajo
			conejos[N * ((x + 1) % M) + y] * (direcciones_c[N * ((x + 1) % M) + y] == 2);

		buffer1_l[tid] =
			// Lobos que estaban aqui antes
			lobos[tid] * (direcciones_l[tid] == 0) +
			// Lobos que vienen de la derecha
			lobos[x * N + ((y + 1) % N)] * (direcciones_l[x * N + ((y + 1) % N)] == 3) +
			// Lobos que vienen de arriba
			lobos[N * ((x - 1 + M) % M) + y] * (direcciones_l[N * ((x - 1 + M) % M) + y] == 4) +
			// Lobos que vienen de la izquierda
			lobos[x * N + ((y - 1 + N) % N)] * (direcciones_l[x * N + ((y - 1 + N) % N)] == 1) +
			// Lobos que vienen de abajo
			lobos[N * ((x + 1) % M) + y] * (direcciones_l[N * ((x + 1) % M) + y] == 2);
	}
}

__global__ void reproduccion_depredacion(int* conejos, int* lobos, int* buffer1_c, int* buffer1_l, int M, int N) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < M * N) {
		int y = tid % N;
		int x = tid / N;
		lobos[tid] =
			// Lobos desde la izquierda
			(buffer1_l[x * N + ((y - 1 + N) % N)] >= 2) ||
			// Lobos desde la derecha
			(buffer1_l[x * N + ((y + 1) % N)] >= 2) ||
			// Lobos ya en la posicion
			(buffer1_l[tid] >= 1);

		conejos[tid] = 
			// Conejos desde arriba
			((buffer1_c[N * ((x - 1 + M) % M) + y] >=2) ||
			// Conejos desde abajo
			(buffer1_c[N * ((x + 1) % M) + y] >= 2) ||
			// Conejos ya en la posicion
			(buffer1_c[tid] >=1)) && 
			// No puede haber conejo si existe un lobo
			(lobos[tid] != 1);
	}
}