#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Вводим размеры блоков и радиус --- определяем параметры
#define BLOCK_SIZE 8
#define R 4
#define ARR_SIZE 12

// Глобальное ядро GPU
__global__ void stencil_kernel(const double *in, double *out) {
    __shared__ double tile[BLOCK_SIZE + 2 * R];
    
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int local_idx = threadIdx.x + R;
    
    tile[local_idx] = in[global_idx];
    if (threadIdx.x < R) {
        if (global_idx >= R)
            tile[local_idx - R] = in[global_idx - R];
        if (global_idx + BLOCK_SIZE < ARR_SIZE)
            tile[local_idx + BLOCK_SIZE] = in[global_idx + BLOCK_SIZE];
    }
    __syncthreads();
    
    if (global_idx >= R && global_idx < ARR_SIZE - R) {
        double sum = 0.0;
        for (int i = -R; i <= R; i++)
            sum += tile[local_idx + i];
        out[global_idx - R] = sum;
    }
}

// Запуск ядра на GPU
void run_stencil(const double *host_in, double *host_out) {
    double *dev_in, *dev_out;
    size_t bytes = ARR_SIZE * sizeof(double);
    
    cudaMalloc(&dev_in, bytes);
    cudaMalloc(&dev_out, bytes);
    cudaMemcpy(dev_in, host_in, bytes, cudaMemcpyHostToDevice);
    
    stencil_kernel<<<(ARR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dev_in, dev_out);
    cudaMemcpy(host_out, dev_out, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(dev_in);
    cudaFree(dev_out);
}

// Реализация на CPU
void stencil_cpu(const double *in, double *out) {
    for (int i = R; i < ARR_SIZE - R; i++) {
        double sum = 0.0;
        for (int j = -R; j <= R; j++)
            sum += in[i + j];
        out[i - R] = sum;
    }
}

// Запускаем программу
int main() {
    double *host_in = (double *)malloc(ARR_SIZE * sizeof(double));
    double *host_out_cpu = (double *)malloc(ARR_SIZE * sizeof(double));
    double *host_out_gpu = (double *)malloc(ARR_SIZE * sizeof(double));
    int comparison = 1;
    
    for (int i = 0; i < ARR_SIZE; i++) {
        host_in[i] = rand() % 100;
        host_out_cpu[i] = 0;
    }
    
    stencil_cpu(host_in, host_out_cpu);
    run_stencil(host_in, host_out_gpu);
    
    // Проверка на равенство
    for (int i = 0; i < ARR_SIZE - 2 * R; i++) {
        if (host_out_cpu[i] != host_out_gpu[i]) {
            comparison = 0;
            break;
        }
    }
    
    // Сравнение результатов
    if (comparison == 0) 
    {
        printf("Вычисления не совпадают\n");
    } 
    else 
    {
        printf("Вычисления совпадают\n");
    }
    
    // Освобождаем память
    free(host_in);
    free(host_out_cpu);
    free(host_out_gpu);
    return 0;
}
