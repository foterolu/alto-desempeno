
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



__global__ void imageCutter (int* fotoOriginal, int* fotoNueva, int posFotoNueva, int fotoSize, int N){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < fotoSize*fotoSize){
    	fotoNueva[tid] = fotoOriginal[posFotoNueva+tid%fotoSize+(tid/fotoSize)*N];
    }
}




__global__ void promedioTemplate(int* array, float* promedio, int array_size) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    int pre_sum = 0;

    for (int preSumIndex = i; preSumIndex < array_size; preSumIndex += gridDim.x * blockDim.x) {
        pre_sum += array[preSumIndex];
        
    }

    sdata[tid] = pre_sum;
    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        
        atomicAdd(promedio, (float)sdata[0]/array_size);
    }
}





__global__ void sumaDiferenciasCuadradas(int* array, float* sum, int array_size, float *promedio){
    extern __shared__ float sdata1[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float pre_sum = 0;

    for (int preSumIndex = i; preSumIndex < array_size; preSumIndex += gridDim.x * blockDim.x) {
        pre_sum += pow(array[preSumIndex] - *promedio, 2);
    }

    sdata1[tid] = pre_sum;
    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata1[tid] += sdata1[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, sdata1[0]);
    }
}





__global__ void promedioFoto(int* foto, float* promedio, int Foto_size, int Template_size, int posXTemplate, int posYTemplate) {
    extern __shared__ int floatedMemory[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_sum = 0;
    for (int preSumIndex = i; preSumIndex < pow(Template_size,2); preSumIndex += gridDim.x * blockDim.x) {
        pre_sum += foto[(posXTemplate*Foto_size+posYTemplate)+preSumIndex%Template_size+(preSumIndex/Template_size)*Foto_size];
    }

   floatedMemory[tid] = pre_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            floatedMemory[tid] += floatedMemory[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(promedio+posXTemplate*(Template_size+1)+posYTemplate, (float)floatedMemory[0]/(float)pow(Template_size,2));
    }
}




__global__ void PasoFinal(int* foto, int* template_h, float* numerador, float* denominador, int Foto_size, int Template_size, int posXTemplate, int posYTemplate, float* promedioFoto, float* promedioTemplate) {
    extern __shared__ float floatedMemory2[];
    float* suma_varianza = floatedMemory2;
    float* suma_multiplicaciones = (float*)&floatedMemory2[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float varianza = 0;
    float multiplicaciones = 0;

    for (int preSumIndex = i; preSumIndex < pow(Template_size,2); preSumIndex += gridDim.x * blockDim.x) {
        varianza += pow(foto[(posXTemplate*Foto_size+posYTemplate)+preSumIndex%Template_size+(preSumIndex/Template_size)*Foto_size]-promedioFoto[posXTemplate*(1+Template_size) + posYTemplate],2);
        multiplicaciones += (foto[(posXTemplate*Foto_size+posYTemplate)+preSumIndex%Template_size+(preSumIndex/Template_size)*Foto_size]-promedioFoto[posXTemplate*(1+Template_size) + posYTemplate])*(template_h[preSumIndex]-*promedioTemplate);
    }

    suma_varianza[tid] = varianza;
    suma_multiplicaciones[tid] = multiplicaciones;
    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            suma_varianza[tid] += suma_varianza[tid + s];
            suma_multiplicaciones[tid] += suma_multiplicaciones[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&numerador[posXTemplate*(1+Template_size) + posYTemplate], suma_multiplicaciones[0]);
        atomicAdd(&denominador[posXTemplate*(1+Template_size)+posYTemplate], suma_varianza[0]/pow(Template_size,2));
    }
}

__global__ void PasoFinalOptimizado(int* foto, int* template_d, float* numerador, float* denominador, int Foto_size, int Template_size, int posXTemplate, int posYTemplate, float* promedioFoto, float* promedioTemplate) {
    extern __shared__ float floatedMemory2[];
    float* suma_varianza = floatedMemory2;
    float* suma_multiplicaciones = (float*)&floatedMemory2[blockDim.x];
    int* suma_promedio_next = (int*)&floatedMemory2[2*blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float varianza = 0;
    float multiplicaciones = 0;
    int promedio_siguiente = 0;

    int posXTemplateSiguiente = posXTemplate+(posYTemplate==Template_size)*1;
    int posYTemplateSiguiente = (posYTemplate+1)%(Template_size+1);

    for (int preSumIndex = i; preSumIndex < pow(Template_size, 2); preSumIndex += gridDim.x * blockDim.x) {
        varianza += pow(foto[(posXTemplate * Foto_size + posYTemplate) + preSumIndex % Template_size + (preSumIndex / Template_size) * Foto_size] - promedioFoto[posXTemplate * (1 + Template_size) + posYTemplate], 2);
        multiplicaciones += (foto[(posXTemplate * Foto_size + posYTemplate) + preSumIndex % Template_size + (preSumIndex / Template_size) * Foto_size] - promedioFoto[posXTemplate * (1 + Template_size) + posYTemplate]) * (template_d[preSumIndex] - *promedioTemplate);
        promedio_siguiente += foto[(posXTemplateSiguiente * Foto_size + posYTemplateSiguiente) + preSumIndex % Template_size + (preSumIndex / Template_size) * Foto_size];
    }

    suma_varianza[tid] = varianza;
    suma_multiplicaciones[tid] = multiplicaciones;
    suma_promedio_next[tid] = promedio_siguiente;
    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            suma_varianza[tid] += suma_varianza[tid + s];
            suma_multiplicaciones[tid] += suma_multiplicaciones[tid + s];
            suma_promedio_next[tid] += suma_promedio_next[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&numerador[posXTemplate * (1 + Template_size) + posYTemplate], suma_multiplicaciones[0]);
        atomicAdd(&denominador[posXTemplate * (1 + Template_size) + posYTemplate], suma_varianza[0] / pow(Template_size, 2));
        atomicAdd(promedioFoto + posXTemplateSiguiente * (Template_size + 1) + posYTemplateSiguiente, ((float)suma_promedio_next[0])/((float)pow(Template_size, 2)));
    }
}


__global__ void CalculoFinal(float desviacionTemplate, float* desviacionFoto, float* sumaMultiplicaciones, int numIteraciones, int templateArea, float* valorFinal){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<numIteraciones){
        float desviacionFotoRaiz = pow(desviacionFoto[tid], 0.5);
        valorFinal[tid] = sumaMultiplicaciones[tid]/(desviacionTemplate*desviacionFotoRaiz*templateArea);
    }
}
