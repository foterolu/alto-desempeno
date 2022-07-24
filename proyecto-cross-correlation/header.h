#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/**
 * @brief 
 * Funcion que genera al azar dos posiciones en las que tomar las fotos, asegurando que el area en las que coincidan sea un cierto porcentaje del area total
 * en la que podrian coincidir
 * @param setArea porcentaje del area en el que coinciden ambas fotos (valor flotante de 0 a 1)
 * @param foto1 direccion de memoria donde se guarda la posicion del primer valor del primer cuadrado
 * @param foto2 direccion de memoria donde se guarda la posicion del primer valor del segundo cuadrado
 * @param fotoSize tamaño de un lado de la foto (las fotos son cuadradas)
 * @param M tamaño vertical de la mesa
 * @param N tamaño horizontal de la mesa
 */
void escoger_fotos(float setArea, int* foto1, int*foto2, int fotoSize, int M, int N){

    // Escogemos una posicion de foto 1 tal que la segunda foto pueda estar en contacto con la foto1 y desde cualquier direccion sin superar los maximos de la mesa
    int x1;
    int y1;
    x1 = fotoSize+1+rand()%(M-3*fotoSize-2);
    y1 = fotoSize+1+rand()%(N-3*fotoSize-2);

    // Escogemos un x2 de tal forma que sea posible seleccionar un y2 que cumpla con setArea
    int x2;
    x2 = rand()%((int)(x1 + fotoSize*(1 - setArea) - 1 - x1 - fotoSize*(setArea-1) - 1)) + x1 + fotoSize*(setArea-1) + 1;

    // Existen dos posibles valores de y2 que cumplen que el area se ajuste a setArea, por lo que escogeremos uno al azar
    int y2;
    if(rand()%2 == 1)
        y2 = y1 + fotoSize - (setArea*fotoSize*fotoSize)/(fotoSize-abs(x2-x1));
    else
        y2 = y1 - fotoSize + (setArea*fotoSize*fotoSize)/(fotoSize-abs(x2-x1));

    // Entregamos a foto 1 y foto 2 la direccion en la cual comenzaran sus cuadrados
    *foto1 = N * x1 + y1;
    *foto2 = N * x2 + y2;
}

void Iteracion(int* Foto, int* Template, int Foto_size, int Template_size, int posXTemplate, int posYTemplate, float promedioTemplate, float desviacionTemplate, float* valorFinal) {
    float promedio = 0;
    for (int preSumIndex = 0; preSumIndex < pow(Template_size, 2); preSumIndex ++) {
        promedio += Foto[(posXTemplate * Foto_size + posYTemplate) + preSumIndex % Template_size + (preSumIndex / Template_size) * Foto_size];
    }
    promedio = promedio/pow(Template_size, 2);

    float varianza = 0;
    float multiplicaciones = 0;

    for (int preSumIndex = 0; preSumIndex < pow(Template_size, 2); preSumIndex ++) {
        varianza += pow(Foto[(posXTemplate * Foto_size + posYTemplate) + preSumIndex % Template_size + (preSumIndex / Template_size) * Foto_size] - promedio, 2);
        multiplicaciones += (Foto[(posXTemplate * Foto_size + posYTemplate) + preSumIndex % Template_size + (preSumIndex / Template_size) * Foto_size] - promedio) * (Template[preSumIndex] - promedioTemplate);
    }

    valorFinal[posXTemplate * (1 + Template_size) + posYTemplate] = multiplicaciones / (pow(varianza / pow(Template_size, 2), 0.5) * desviacionTemplate * pow(Template_size, 2));
}