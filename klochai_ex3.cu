#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32 // Размер блока для разбиения матрицы
// Размер матрицы вводим в блоке main

// Наивный алгоритм умножения матриц на GPU
__global__ void naive_kernel(double *A, double *B, double *C, size_t N) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (y < N && x < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[y * N + k] * B[k * N + x];
        }
        C[y * N + x] = sum;
    }
}

// Блоковый алгоритм умножения матриц
__global__ void blocked_kernel(double *A, double *B, double *C, size_t N) {
    __shared__ double Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bsub[BLOCK_SIZE][BLOCK_SIZE];
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    double sum = 0.0;
    for (int tile = 0; tile < N / BLOCK_SIZE; ++tile) {
        // Загружаем подматрицы в shared memory
        Asub[threadIdx.y][threadIdx.x] = A[y * N + (tile * BLOCK_SIZE + threadIdx.x)];
        Bsub[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * N + x];
        __syncthreads();
        
        // Выполняем умножение части строки на часть столбца
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (y < N && x < N) {
        C[y * N + x] = sum;
    }
}

// Функция для сравнения двух массивов на равенство
__global__ void check_equal(double *X, double *Y, size_t size, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && X[idx] != Y[idx]) {
        *result = 0; // Если найдено различие, устанавливаем флаг
    }
}

// Функция запускает вычисления на GPU
void launch_kernels(double *A, double *B, double *C1, double *C2, size_t N, int *match) {
    double *d_A, *d_B, *d_C1, *d_C2;
    size_t bytes = N * N * sizeof(double);
    
    // Выделяем память на устройстве (GPU)
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C1, bytes);
    cudaMalloc(&d_C2, bytes);
    
    // Копируем матрицы A и B из хоста (CPU) на устройство (GPU)
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);
    
    // Определяем конфигурацию сетки и блоков
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Запускаем два ядра: наивное и блоковое
    naive_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C1, N);
    blocked_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C2, N);
    
    // Выделяем память для результата сравнения
    int *d_match;
    cudaMalloc(&d_match, sizeof(int));
    cudaMemcpy(d_match, match, sizeof(int), cudaMemcpyHostToDevice);
    
    // Проверяем равенство результатов
    check_equal<<<(N * N + 255) / 256, 256>>>(d_C1, d_C2, N * N, d_match);
    
    // Копируем результаты с устройства обратно на хост
    cudaMemcpy(match, d_match, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C1, d_C1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(C2, d_C2, bytes, cudaMemcpyDeviceToHost);
    
    // Освобождаем память на устройстве
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C1); cudaFree(d_C2);
    cudaFree(d_match);
}

int main() {
    const size_t N = 2048; // Вводим размер матрицы N x N
    size_t numElements = N * N;
    size_t bytes = numElements * sizeof(double);
    
    // Выделяем память на хосте (CPU)
    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C1 = (double *)malloc(bytes);
    double *C2 = (double *)malloc(bytes);
    
    // Заполняем матрицы случайными числами
    for (size_t i = 0; i < numElements; ++i) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }
    
    // Запускаем вычисления
    int comparison = 1;
    launch_kernels(A, B, C1, C2, N, &comparison);
    
    // Выводим результат сравнения
    if (comparison) {
        printf("Вычисления совпадают");
    } else {
        printf("Вычисления не совпадают");
    }
    
    // Освобождаем память
    free(A); 
    free(B); 
    free(C1); 
    free(C2);
    return 0;
}
