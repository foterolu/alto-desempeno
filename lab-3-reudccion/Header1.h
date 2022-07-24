#pragma once

void Iteracion_CPU(int* conejos, int* lobos, int* buffer1_c, int* buffer1_l, int* buffer2_c, int* buffer2_l, int M, int N) {
	srand(time(NULL));
	int direccion;
	// Vaciamos buffers
	for (int i = 0; i < M * N; i++) {
		buffer1_c[i] = 0;
		buffer1_l[i] = 0;
		buffer2_c[i] = 0;
		buffer2_l[i] = 0;
	}

	// Movimiento
	for (int x = 0; x < M; x++) {
		for (int y = 0; y < N; y++){
			if (conejos[x * N + y] == 1) {
				direccion = rand() % 5;
				switch (direccion) {
					case 0:
						buffer1_c[x * N + y]++;
						break;
					case 1:
						buffer1_c[x * N + ((y + 1) % N)]++;
						break;
					case 2:
						buffer1_c[N * ((x - 1 + M) % M) + y]++;
						break;
					case 3:
						buffer1_c[x * N + ((y - 1 + N) % N)]++;
						break;
					case 4:
						buffer1_c[N * ((x + 1) % M) + y]++;
						break;
				}
			}
			if (lobos[x * N + y] == 1) {
				direccion = rand() % 5;
				switch (direccion) {
					case 0:
						buffer1_l[x * N + y]++;
						break;
					case 1:
						buffer1_l[x * N + ((y + 1) % N)]++;
						break;
					case 2:
						buffer1_l[N * ((x - 1 + M) % M) + y]++;
						break;
					case 3:
						buffer1_l[x * N + ((y - 1 + N) % N)]++;
						break;
					case 4:
						buffer1_l[N * ((x + 1) % M) + y]++;
						break;
				}
			}
		}
	}

	// Reproduccion (se asume que siempre quedara solo 1 conejo y lobo por casilla)
	for (int x = 0; x < M; x++) {
		for (int y = 0; y < N; y++) {
			// Rep. conejos
			if(buffer1_c[x * N + y] == 1)
				buffer2_c[x * N + y] = 1;

			if (buffer1_c[x * N + y] >= 2) {
				buffer2_c[x * N + y] = 1;
				buffer2_c[N * ((x - 1 + M) % M) + y] = 1;
				buffer2_c[N * ((x + 1) % M) + y] = 1;
			}
			// Rep. lobos
			if (buffer1_l[x * N + y] == 1)
				buffer2_l[x * N + y] = 1;

			if (buffer1_l[x * N + y] >= 2) {
				buffer2_l[x * N + y] = 1;
				buffer2_l[x * N + ((y + 1) % N)] = 1;
				buffer2_l[x * N + ((y - 1 + N) % N)] = 1;
			}
		}
	}

	// Depredacion
	for (int x = 0; x < M; x++) {
		for (int y = 0; y < N; y++) {
			if (buffer2_l[x * N + y] == 1) {
				buffer2_c[x * N + y] = 0;
			}
		}
	}
}

void sequential_sum(int* conejos, int* sum_conejos, int* lobos, int* sum_lobos, int size) {
	// Sequentially sum conejos/lobos
	*sum_conejos = 0;
	*sum_lobos = 0;
	for (int i = 0; i < size; i++)
	{
		*sum_conejos += conejos[i];
		*sum_lobos += lobos[i];
	}
}