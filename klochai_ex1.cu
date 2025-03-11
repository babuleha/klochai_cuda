#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define VECTOR_SIZE (2048*2048)
#define THREADS_PER_BLOCK 512

// CUDA-функция для сложения векторов
__global__ void vector_add(double *a, double *b, double *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < VECTOR_SIZE) {
        c[idx] = a[idx] + b[idx];
    }
}

// Функция сложения на CPU
void vector_add_cpu(double *a, double *b, double *c) {
    for (int i = 0; i < VECTOR_SIZE; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    double *host_a, *host_b, *host_c_cpu, *host_c_gpu;
    double *dev_a, *dev_b, *dev_c;
    int error_count = 0;
    
    // Выделяем память на хосте
    host_a = (double*)malloc(VECTOR_SIZE * sizeof(double));
    host_b = (double*)malloc(VECTOR_SIZE * sizeof(double));
    host_c_cpu = (double*)malloc(VECTOR_SIZE * sizeof(double));
    host_c_gpu = (double*)malloc(VECTOR_SIZE * sizeof(double));
    
    // Заполняем массивы случайными числами
    srand(time(NULL));
    for (int i = 0; i < VECTOR_SIZE; i++) {
        host_a[i] = (double)(rand() % 100);
        host_b[i] = (double)(rand() % 100);
    }
    
    // Выделяем память на устройстве
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&dev_c, VECTOR_SIZE * sizeof(double));
    
    // Копируем данные с хоста на устройство
    cudaMemcpy(dev_a, host_a, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    
    // Запуск вычислений на GPU
    int blocks = (VECTOR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_add<<<blocks, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();
    
    // Копируем результат обратно на хост
    cudaMemcpy(host_c_gpu, dev_c, VECTOR_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Вычисления на CPU
    vector_add_cpu(host_a, host_b, host_c_cpu);
    
    // Проверка корректности
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (host_c_cpu[i] != host_c_gpu[i]) {
            error_count++;
        }
    }
    
    // Вывод результата
    if (error_count == 0) {
        printf("Вычисления совпадают\n");
    } else {
        printf("Вычисления не совпадают\n", error_count);
    }
    
    // Освобождаем память
    free(host_a);
    free(host_b);
    free(host_c_cpu);
    free(host_c_gpu);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    return 0;
}
