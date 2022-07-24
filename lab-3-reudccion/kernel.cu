
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "Header1.h"
#include "Header2.h"
#include "Header3.h"
#include "Header4.h"

#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;



// GENERACION DE SEEDS PARA CADA HEBRA
__global__ void random_init(curandState* state) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(0, tid, 0, &state[tid]);
}


int main()
{

	// Leyendo archivos de entrada
    int M;
    int N;

    fstream newfile;
    newfile.open("initial.txt", ios::in);

    string tp;
    getline(newfile, tp);
    char* token = strtok(const_cast<char*>(tp.c_str()), " ");
    M = stoi(token);
    token = strtok(NULL, " ");
    N = stoi(token);

    int size = M * N * sizeof(int);

    int* Conejos = (int*)malloc(size);
    int* Lobos = (int*)malloc(size);



    int threads_block = 256;
    int number_of_blocks = (int)ceil((float)M*N/threads_block);

    curandState* states;
    cudaMalloc(&states, M * N * sizeof(curandState));
    random_init<<<number_of_blocks, threads_block>>>(states);

    cudaEvent_t e1, e2;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);

    clock_t start;
    clock_t finish;
    double tiempo_cpu;


    int M_actual = 0;
    printf("Creando matrices\n");
    if (newfile.is_open()) {
        while (getline(newfile, tp)) {
            for (int i = 0; i < N; i++) {
                Conejos[i + M_actual * N] = 0;
                Lobos[i + M_actual * N] = 0;
                if ((int)tp[2 * i] == (1+48)) {
                    Conejos[i + M_actual * N] = 1;
                }
                if ((int)tp[2 * i] == (2+48)) {
                    Lobos[i + M_actual * N] = 1;
                }
            }
            M_actual++;
        }
        newfile.close();
    }

    int suma_inicial_c;
    int suma_inicial_l;



    sequential_sum(Conejos, &suma_inicial_c, Lobos, &suma_inicial_l, M * N);
    

	// Creando variables para el problema

    /* variables CPU*/
    int *copia_conejos, *copia_lobos;
    int *buffer1_c, *buffer2_c, *buffer1_l, *buffer2_l;
    int suma_conejos, suma_lobos;

    buffer1_c = (int*)malloc(M * N * sizeof(int));
    buffer2_c = (int*)malloc(M * N * sizeof(int));
    buffer1_l = (int*)malloc(M * N * sizeof(int));
    buffer2_l = (int*)malloc(M * N * sizeof(int));
    copia_conejos = (int*)malloc(M * N * sizeof(int));
    copia_lobos = (int*)malloc(M * N * sizeof(int));
    /*variables GPU para iteracion*/
    int* d_direcciones_c, *d_direcciones_l;
    int* d_buffer_c, *d_buffer_l;
    int* d_conejos, *d_lobos;
    float time_iteracion_gpu;
    float dt;
    int* guardar_conejos, *guardar_lobos;

    guardar_conejos = (int*)malloc(M * N * sizeof(int));
    guardar_lobos = (int*)malloc(M * N * sizeof(int));
    cudaMalloc(&d_conejos, size);
    cudaMalloc(&d_lobos, size);
    cudaMalloc(&d_buffer_c, size);
    cudaMalloc(&d_buffer_l, size);
    cudaMalloc(&d_direcciones_c, size);
    cudaMalloc(&d_direcciones_l, size);
    /*variables GPU para reduccion y reduccion atomica*/
    int* d_conejos2, *d_lobos2;
    int* d_suma_conejos, *d_suma_lobos;

    cudaMalloc(&d_conejos2, size);
    cudaMalloc(&d_lobos2, size);
    cudaMalloc(&d_suma_conejos, size);
    cudaMalloc(&d_suma_lobos, size);
    /*variables GPU para reduccion atomica*/
    int *d_suma_conejos2;
    int *d_suma_lobos2;

    cudaMalloc(&d_suma_conejos2, sizeof(int));
    cudaMalloc(&d_suma_lobos2, sizeof(int));

    // FUNCIONES CPU

    /* copiando los valores originales a otro arreglo*/
    for (int i = 0; i < M * N; i++) {
        copia_conejos[i] = Conejos[i];
        copia_lobos[i] = Lobos[i];
    }
    int* temp;

    start = clock();

    /* ejecutando 1000 iteraciones del algoritmo*/
    for (int i = 0; i < 1000; i++) {
        Iteracion_CPU(copia_conejos, copia_lobos, buffer1_c, buffer1_l, buffer2_c, buffer2_l, M, N);
        /* invirtiendo los indices*/
        temp = copia_conejos;
        copia_conejos = buffer2_c;
        buffer2_c = temp;

        temp = copia_lobos;
        copia_lobos = buffer2_l;
        buffer2_l = temp;
    }

    finish = clock();
    tiempo_cpu = 1000*(double)(finish - start) / CLOCKS_PER_SEC;
    printf("Tiempo de 1000 iteraciones en CPU : %f \n", tiempo_cpu);

    /* calculando la suma final de conejos y lobos*/


    start = clock();
    sequential_sum(copia_conejos, &suma_conejos, copia_lobos, &suma_lobos, M * N);
    finish = clock();
    tiempo_cpu = 1000 * (double)(finish - start) / CLOCKS_PER_SEC;
    printf("Tiempo de reducir en CPU : %f \n", tiempo_cpu);

    printf("cantidad de animales final calculada por CPU : %d conejos y %d lobos \n", suma_conejos, suma_lobos);
    

    // ITERACIONES EN GPU

    /* copiando los arreglos originales a GPU*/
    cudaMemcpy(d_conejos, Conejos, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lobos, Lobos, size, cudaMemcpyHostToDevice);

    cudaEventRecord(e1);

    /* ejecutando 1000 iteraciones del algoritmo en GPU*/
    for (int i = 0; i < 1000; i++){
        generar_direcciones<<< number_of_blocks, threads_block>>>(states, d_direcciones_c, d_direcciones_l, M, N);
        movimiento<<< number_of_blocks, threads_block>>>(d_conejos, d_lobos, d_buffer_c, d_buffer_l, M, N, d_direcciones_c, d_direcciones_l);
        reproduccion_depredacion<<< number_of_blocks, threads_block>>>(d_conejos, d_lobos, d_buffer_c, d_buffer_l, M, N);
    }

    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&dt, e1, e2);
    time_iteracion_gpu = dt;
    printf("Tiempo de 1000 iteraciones en GPU : %f \n", time_iteracion_gpu);

    /* copiando a memoria los arreglos finales*/
    cudaMemcpy(guardar_conejos, d_conejos, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(guardar_lobos, d_lobos, size, cudaMemcpyDeviceToHost);


    /* calculando la suma total de conejos y lobos*/
    sequential_sum(guardar_conejos, &suma_conejos, guardar_lobos, &suma_lobos, M* N);
    printf("cantidad de animales final calculada por GPU (ITERACION EN GPU TAMBIEN) : %d conejos y %d lobos \n", suma_conejos, suma_lobos);


    // REDUCCION EN GPU
    
    // 2 * SM = 2 * 46 = 92
    int gridSize = 92;  
    
    /* copiando los arreglos originales a GPU*/
    cudaMemcpy(d_conejos2, copia_conejos, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lobos2, copia_lobos, size, cudaMemcpyHostToDevice);


    cudaEventRecord(e1);
    /* ejecutando la reduccion en GPU de dos pasos*/
    reduccion<<<gridSize, threads_block, (2 * threads_block * sizeof(int)) >>>(d_conejos2, d_suma_conejos, d_lobos2, d_suma_lobos, M*N);
    reduccion<<<1, threads_block, (2 * threads_block * sizeof(int))>>>(d_suma_conejos, d_suma_conejos, d_suma_lobos, d_suma_lobos, gridSize);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&dt, e1, e2);
    time_iteracion_gpu = dt;
    printf("Tiempo de reduccion en GPU : %f \n", time_iteracion_gpu);


    int out1;
    int out2;
    /* copiando a memoria el resultado final de la reduccion en GPU*/
    cudaMemcpy(&out1, d_suma_conejos, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out2, d_suma_lobos, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d \n", out1);
    printf("%d \n", out2);


    // REDUCCION ATOMICA EN GPU

    /* copiando los arreglos originales a GPU*/
    cudaMemcpy(d_conejos2, copia_conejos, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lobos2, copia_lobos, size, cudaMemcpyHostToDevice);
    
    
    int valor_inicial = 0;
    /* para realizar la suma atomica necesito que el valor inicial sea 0*/
    cudaMemcpy(d_suma_conejos2, &valor_inicial, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suma_lobos2, &valor_inicial, sizeof(int), cudaMemcpyHostToDevice);


    cudaEventRecord(e1);
    /* ejecutando reduccion atomica en GPU de un solo paso*/
    reduccion_atomica<<<gridSize, threads_block, (2 * threads_block * sizeof(int))>>>(d_conejos2, d_suma_conejos2, d_lobos2, d_suma_lobos2, M * N);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&dt, e1, e2);
    time_iteracion_gpu = dt;
    printf("Tiempo de reduccion atomica en GPU : %f \n", time_iteracion_gpu);

    cudaMemcpy(&out1, d_suma_conejos, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out2, d_suma_lobos, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d \n", out1);
    printf("%d \n", out2);

    

    /* LIBERA MEMORIA*/
    cudaFree(d_conejos);
    cudaFree(d_lobos);
    cudaFree(d_buffer_c);
    cudaFree(d_buffer_l);
    cudaFree(d_direcciones_c);
    cudaFree(d_direcciones_l);
    cudaFree(d_suma_conejos);
    cudaFree(d_suma_lobos);
    cudaFree(d_conejos2);
    cudaFree(d_lobos2);
    cudaFree(d_suma_conejos2);
    cudaFree(d_suma_lobos2);
    cudaFree(states);

    free(buffer1_c);
    free(buffer2_c);
    free(buffer1_l);
    free(buffer2_l);
    free(Conejos);
    free(Lobos);
    free(guardar_conejos);
    free(guardar_lobos);
}


