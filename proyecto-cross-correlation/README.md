## Compilar código
 -  se necesita opencv y cuda.
 -  teniendo las librerias instaladas sólo queda compilar y correr.
## Variables para testeo
-   int square_size : tamaño de los lados de la mesa (NxN)
-   int fotoSize    : tamaño de las fotos del mouse(debe ser potencia de 2)
-   int gridSize    : cantidad de bloques (se recomienda 2*SM)
-   int threadsBlock: cantidad de threads por bloque
-   int showOPENCV  : puede ser 0 o 1 y sirve para mostrar visualmente  el recorrido del template através de la foto
-   int doCPU       : puede ser 0 o 1 y sirve para correr la versión CPU del código.