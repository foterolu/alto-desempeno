#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>
#include "kernels.h"
#include "ruido.h"
#include "header.h"

#include <opencv2\opencv.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;



int main() {
    srand(time(NULL));

    // Variables que puede cambiar el usuario
    int square_size = 1000; // Tamaño de los lados de la mesa
    int fotoSize = 256; //Tamaño de las fotos del mouse (Debe ser potencia de 2)
    int gridSize = 23; // Cantidad de bloques (Se recomienda 2*SM)
    int threadsBlock = 512; // Cantidad de threads por bloque
    int showOPENCV = 0; 
    int doCPU = 0;
    
    int templateSize = fotoSize / 2;
    int templateArea = pow(templateSize, 2);
    int fotoArea = pow(fotoSize, 2);
    int posTemplate = (fotoSize / 4) * fotoSize + fotoSize / 4;
    int posTemplateX = fotoSize / 4;
    int posTemplateY = fotoSize / 4;

    int numIteraciones = pow(templateSize + 1, 2);

    int gridSizeCutterFotos = (fotoArea + threadsBlock - 1) / threadsBlock;
    int gridSizeCutterTemplate = (templateArea + threadsBlock - 1) / threadsBlock;
    int gridSizeFinal = (numIteraciones + threadsBlock - 1) / threadsBlock;
    // Variables CPU
    float* valorFinal_h = (float*)malloc(numIteraciones * sizeof(float));
    int* foto1_h = (int*)malloc(fotoArea * sizeof(int));
    int* template_h = (int*)malloc(templateArea * sizeof(int));

    // Variables GPU
    int* foto1_d;
    int* foto2_d;
    int* template_d;
    float* promedioTemplate_d;
    float* desviacionEstandarTemplate_d;
    float* desviacionesFoto_d;
    float* sumasMultiplicaciones_d;
    float* promedioFoto_d;
    float* valorFinal_d;

    //Crear espacio en memoria GPU
    cudaMalloc(&foto1_d, fotoArea * sizeof(int));
    cudaMalloc(&foto2_d, fotoArea * sizeof(int));
    cudaMalloc(&template_d, templateArea * sizeof(int));
    cudaMalloc(&promedioTemplate_d, sizeof(float));
    cudaMalloc(&desviacionEstandarTemplate_d, sizeof(float));
    cudaMalloc(&desviacionesFoto_d, numIteraciones * sizeof(float));
    cudaMalloc(&sumasMultiplicaciones_d, numIteraciones * sizeof(float));
    cudaMalloc(&promedioFoto_d, numIteraciones * sizeof(float));
    cudaMalloc(&valorFinal_d, numIteraciones * sizeof(float));

    // Inicializando arreglos en 0
    cudaMemset(sumasMultiplicaciones_d, 0, sizeof(float) * numIteraciones);
    cudaMemset(desviacionesFoto_d, 0, sizeof(float) * numIteraciones);
    cudaMemset(promedioFoto_d, 0, sizeof(float) * numIteraciones);

    // Generar mesa
    printf("Generamos la mesa en CPU \n");
    float* ruido = (float*)malloc(square_size * square_size * sizeof(float));
    int* ruido_int = (int*)malloc(square_size * square_size * sizeof(int));

    /* Usamos ruido de perlin para generar una mesa */
    for (int x = 0; x < square_size; x++) {
        for (int y = 0; y < square_size; y++)
            ruido[square_size * x + y] = perlin((float)x / (0.03 * square_size), (float)y / (0.03 * square_size));
    }

    /* Modificamos los valores de tal manera que vayan desde 0 a 255 */
    for (int x = 0; x < square_size; x++) {
        for (int y = 0; y < square_size; y++) {
            ruido_int[x * square_size + y] = (int)(184 * (0.69 + ruido[square_size * x + y]));
        }
    }

    // Cargar mesa a GPU
    printf("Cargamos la mesa a la GPU\n");
    int* mesa_d;
    cudaMalloc(&mesa_d, sizeof(int) * square_size * square_size);
    cudaMemcpy(mesa_d, ruido_int, sizeof(int) * square_size * square_size, cudaMemcpyHostToDevice);

    // Escoger dos puntos en la mesa para las fotos
    printf("Escogemos los dos puntos en los cuales tomar las fotos \n\n");
    int posFoto1 = 0.5 * (square_size * square_size);
    int posFoto2 = 0.5 * (square_size * square_size);
    escoger_fotos(0.75, &posFoto1, &posFoto2, fotoSize, square_size, square_size);
    int posFoto1X = (posFoto1) / square_size;
    int posFoto2X = (posFoto2) / square_size;
    int posFoto1Y = (posFoto1) % square_size;
    int posFoto2Y = (posFoto2) % square_size;

    printf("posicion de la foto1 (esquina superior izquierda) es : (%d, %d)\n", posFoto1X, posFoto1Y);
    printf("posicion de la foto2 (esquina superior izquierda) es : (%d, %d)\n\n", posFoto2X, posFoto2Y);
    // Cortar las dos fotos
    printf("Cortamos las dos fotos \n");
    imageCutter << <gridSizeCutterFotos, threadsBlock >> > (mesa_d, foto1_d, posFoto1, fotoSize, square_size);
    imageCutter << <gridSizeCutterFotos, threadsBlock >> > (mesa_d, foto2_d, posFoto2, fotoSize, square_size);

    cudaMemcpy(foto1_h, foto1_d, fotoArea * sizeof(int), cudaMemcpyDeviceToHost);
    // Cortar el template
    printf("Cortamos el template desde la foto2 \n");
    imageCutter << <gridSizeCutterTemplate, threadsBlock >> > (foto2_d, template_d, posTemplate, templateSize, fotoSize);

    cudaMemcpy(template_h, template_d, templateArea * sizeof(int), cudaMemcpyDeviceToHost);


    // Calcular promedio del template
    printf("Calcularemos valores necesarios del template \n");
    float promedioTemplate_h = 0;

    cudaMemcpy(promedioTemplate_d, &promedioTemplate_h, sizeof(float), cudaMemcpyHostToDevice);
    promedioTemplate << <gridSize, threadsBlock, threadsBlock * sizeof(float) >> > (template_d, promedioTemplate_d, templateArea);
    cudaMemcpy(&promedioTemplate_h, promedioTemplate_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Promedio del template : %f \n", promedioTemplate_h);

    // Calcular varianza del template
    float desviacionEstandarTemplate_h = 0;
    cudaMemcpy(desviacionEstandarTemplate_d, &desviacionEstandarTemplate_h, sizeof(float), cudaMemcpyHostToDevice);
    sumaDiferenciasCuadradas << <gridSize, threadsBlock, threadsBlock * sizeof(float) >> > (template_d, desviacionEstandarTemplate_d, templateArea, promedioTemplate_d);
    cudaMemcpy(&desviacionEstandarTemplate_h, desviacionEstandarTemplate_d, sizeof(float), cudaMemcpyDeviceToHost);
    desviacionEstandarTemplate_h = pow(desviacionEstandarTemplate_h / (templateArea), 0.5);
    printf("Desviacion estandar del template : %f \n", desviacionEstandarTemplate_h);

    // Ejecutar algoritmo en CPU
    float max = -1;
    int XIndex = 0;
    int YIndex = 0;
    int DiffX;
    int DiffY;


    clock_t t1;
    double time_taken;

    if (doCPU == 1) {
        printf("Ejecutaremos el algoritmo en todas las posibles posiciones del template en CPU \n");
        t1 = clock();
        for (int x = 0; x <= templateSize; x++) {
            for (int y = 0; y <= templateSize; y++) {
                Iteracion(foto1_h, template_h, fotoSize, templateSize, x, y, promedioTemplate_h, desviacionEstandarTemplate_h, valorFinal_h);
            }
        }
        t1 = clock() - t1;
        time_taken = 1000 * ((double)t1) / (CLOCKS_PER_SEC);
        printf("Tiempo CPU: %f \n", time_taken);


        for (int i = 0; i < numIteraciones; i++) {
            if (valorFinal_h[i] > max) {
                max = valorFinal_h[i];
                XIndex = i / (templateSize + 1);
                YIndex = i % (templateSize + 1);
            }
        }
        DiffX = XIndex - posTemplateX;
        DiffY = YIndex - posTemplateY;
        printf("(CPU) Desplazamiento entre foto1 y foto2 (%d, %d) con una coincidencia de : %f", DiffX, DiffY, max);
    }
    // Ejecutar algoritmo en GPU
    cudaEvent_t e1, e2;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    float dt;
    float timeGPU1;

    printf("Ejecutaremos el algoritmo en todas las posibles posiciones del template en GPU \n");

    cudaEventRecord(e1);
    for (int x = 0; x <= templateSize; x++) {
        for (int y = 0; y <= templateSize; y++) {
            /* Calcular promedio del area en la foto, en el que chocan la foto y el template*/
            promedioFoto << <gridSize, threadsBlock, threadsBlock * sizeof(int) >> > (foto1_d, promedioFoto_d, fotoSize, templateSize, x, y);
            /* Calcular la desviacion estandar en la foto y calcular las multiplicaciones en cada punto*/
            PasoFinal << <gridSize, threadsBlock, 2 * threadsBlock * sizeof(float) >> > (foto1_d, template_d, sumasMultiplicaciones_d, desviacionesFoto_d, fotoSize, templateSize, x, y, promedioFoto_d, promedioTemplate_d);
        }
    }

    CalculoFinal << <gridSizeFinal, threadsBlock >> > (desviacionEstandarTemplate_h, desviacionesFoto_d, sumasMultiplicaciones_d, numIteraciones, templateArea, valorFinal_d);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&dt, e1, e2);
    timeGPU1 = dt;
    printf("Tiempo en GPU %f \n\n", timeGPU1);

    cudaMemcpy(valorFinal_h, valorFinal_d, numIteraciones * sizeof(float), cudaMemcpyDeviceToHost);

    max = -1;
    for (int i = 0; i < numIteraciones; i++) {
        if (valorFinal_h[i] > max) {
            max = valorFinal_h[i];
            XIndex = i / (templateSize + 1);
            YIndex = i % (templateSize + 1);
        }
    }
    DiffX = XIndex - posTemplateX;
    DiffY = YIndex - posTemplateY;
    printf("(GPU) Desplazamiento entre foto1 y foto2 (%d, %d) con una coincidencia de : %f", DiffX, DiffY, max);


    // Ejecutar algoritmo en GPU optimizado
    printf("Ejecutaremos el algoritmo en todas las posibles posiciones del template en GPU optimizado \n");
    cudaMemset(sumasMultiplicaciones_d, 0, sizeof(float) * numIteraciones);
    cudaMemset(desviacionesFoto_d, 0, sizeof(float) * numIteraciones);
    cudaMemset(promedioFoto_d, 0, sizeof(float) * numIteraciones);

    /* calculamos primer promedio*/
    promedioFoto << <gridSize, threadsBlock, threadsBlock * sizeof(int) >> > (foto1_d, promedioFoto_d, fotoSize, templateSize, 0, 0);
    cudaEventRecord(e1);
    /* corremos algoritmo hasta la penultima iteracion*/
    for (int x = 0; x <= templateSize; x++) {
        for (int y = 0; y <= templateSize; y++) {
            if ((x == templateSize) && (y == templateSize)) break;
            PasoFinalOptimizado << <gridSize, threadsBlock, 2 * threadsBlock * sizeof(float) + threadsBlock* sizeof(int)>> > (foto1_d, template_d, sumasMultiplicaciones_d, desviacionesFoto_d, fotoSize, templateSize, x, y, promedioFoto_d, promedioTemplate_d);
        }
    }
    /* corremos la ultima iteracion*/
    PasoFinal << <gridSize, threadsBlock, 2 * threadsBlock * sizeof(float) >> > (foto1_d, template_d, sumasMultiplicaciones_d, desviacionesFoto_d, fotoSize, templateSize, templateSize, templateSize, promedioFoto_d, promedioTemplate_d);

    /* calculo final*/
    CalculoFinal << <gridSizeFinal, threadsBlock >> > (desviacionEstandarTemplate_h, desviacionesFoto_d, sumasMultiplicaciones_d, numIteraciones, templateArea, valorFinal_d);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&dt, e1, e2);
    timeGPU1 = dt;
    printf("Tiempo en GPU optimizado %f \n\n", timeGPU1);

    cudaMemcpy(valorFinal_h, valorFinal_d, numIteraciones * sizeof(float), cudaMemcpyDeviceToHost);

    max = -1;
    for (int i = 0; i < numIteraciones; i++) {
        if (valorFinal_h[i] > max) {
            max = valorFinal_h[i];
            XIndex = i / (templateSize + 1);
            YIndex = i % (templateSize + 1);
        }
    }
    DiffX = XIndex - posTemplateX;
    DiffY = YIndex - posTemplateY;
    printf("(GPU optimizado) Desplazamiento entre foto1 y foto2 (%d, %d) con una coincidencia de : %f", DiffX, DiffY, max);

    // Open CV
    if (showOPENCV == 1) {

        Mat image = Mat(square_size, square_size, CV_8UC3, Scalar(0, 0, 255));
        namedWindow("Mesa"); // Create a window for display
        for (int x = 0; x < square_size; x++) {
            for (int y = 0; y < square_size; y++) {
                image.at<Vec3b>(Point(y, x))[0] = ruido_int[square_size * x + y];
                image.at<Vec3b>(Point(y, x))[1] = ruido_int[square_size * x + y];
                image.at<Vec3b>(Point(y, x))[2] = ruido_int[square_size * x + y];
            }
        }

        rectangle(image, Point(posFoto1Y, posFoto1X), Point(posFoto1Y + fotoSize, posFoto1X + fotoSize), Scalar(255, 255, 255), 2); // Foto1
        rectangle(image, Point(posFoto2Y, posFoto2X), Point(posFoto2Y + fotoSize, posFoto2X + fotoSize), Scalar(0, 255, 0), 2); // Foto2


        Mat templateSobreFoto = Mat(fotoSize, fotoSize, CV_8UC3, Scalar(255, 255, 255));
        namedWindow("templateMoviendose");
        moveWindow("templateMoviendose", 1300, 0);


        Mat valorFinalImagen = Mat(templateSize + 1, templateSize + 1, CV_8UC3, Scalar(0, 0, 0));
        namedWindow("valorFinal");
        moveWindow("valorFinal", 1050, 430);


        Mat templateSobreFotoEscalado;
        Mat valorFinalImagenEscalado;

        Mat mesaEscalado;
        resize(image, mesaEscalado, Size(1000, 1000), INTER_LINEAR);
        imshow("Mesa", mesaEscalado); // Show our image inside it
        for (int x = 0;  x<= templateSize; x++) {
            for (int y = 0; y <= templateSize; y++) {
                rectangle(templateSobreFoto, Point(0, 0), Point(fotoSize - 1, fotoSize - 1), Scalar(255, 255, 255), FILLED);
                rectangle(templateSobreFoto, Point(y, x), Point(y + templateSize - 1, x + templateSize - 1), Scalar(126, 126, 126), 1);

                valorFinalImagen.at<Vec3b>(Point(y, x))[0] = 255 * valorFinal_h[x * (templateSize + 1)+y];
                valorFinalImagen.at<Vec3b>(Point(y, x))[1] = 255 * valorFinal_h[x * (templateSize + 1)+y];
                valorFinalImagen.at<Vec3b>(Point(y, x))[2] = 255 * valorFinal_h[x * (templateSize + 1)+y];

                if (valorFinal_h[x * (templateSize + 1)+y] >= 0.995) {
                    valorFinalImagen.at<Vec3b>(Point(y, x))[0] = 0;
                    valorFinalImagen.at<Vec3b>(Point(y, x))[1] = 255;
                    valorFinalImagen.at<Vec3b>(Point(y, x))[2] = 0;
                }

                if (valorFinal_h[x * (templateSize + 1)+y] >= 0.999999) {
                    valorFinalImagen.at<Vec3b>(Point(y, x))[0] = 0;
                    valorFinalImagen.at<Vec3b>(Point(y, x))[1] = 0;
                    valorFinalImagen.at<Vec3b>(Point(y, x))[2] = 255;
                }

                resize(templateSobreFoto, templateSobreFotoEscalado, Size(8 * 64, 8 * 64), INTER_LINEAR);
                resize(valorFinalImagen, valorFinalImagenEscalado, Size(16 * (32 + 1), 16 * (32 + 1)), INTER_LINEAR);

                int delay = 1 + (valorFinal_h[x * (templateSize + 1)+y] >= 0.99) * 250;
                waitKey(delay);
                imshow("templateMoviendose", templateSobreFotoEscalado);
                imshow("valorFinal", valorFinalImagenEscalado);
            }
        }

    }

    waitKey(100000);
    // Liberar memoria
    cudaFree(foto1_d);
    cudaFree(foto2_d);
    cudaFree(template_d);
    cudaFree(promedioTemplate_d);
    cudaFree(desviacionEstandarTemplate_d);
    cudaFree(desviacionesFoto_d);
    cudaFree(sumasMultiplicaciones_d);
    cudaFree(promedioFoto_d);
    cudaFree(valorFinal_d);
    cudaFree(mesa_d);

    free(valorFinal_h);
    free(foto1_h);
    free(template_h);
    free(ruido);
    free(ruido_int);

    return 0;
}