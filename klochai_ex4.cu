#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_SIZE 32  // Размер блока для shared memory

// Простое умножение матриц на GPU (каждый поток вычисляет один элемент C)
__global__ void kernel_naive(const double* A, const double* B, double* C, size_t N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // координата по столбцу
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // координата по строке

    if (x >= N || y >= N) return;

    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        sum += A[y * N + k] * B[k * N + x];
    }
    C[y * N + x] = sum;
}

// Оптимизированное tiled-умножение с использованием shared memory
__global__ void kernel_tiled(const double* A, const double* B, double* C, size_t N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // координата по столбцу
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // координата по строке

    if (x >= N || y >= N) return;

    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];

    double sum = 0.0;

    for (int t = 0; t < N / TILE_SIZE; ++t) {
        // Загружаем блоки из A и B в shared memory
        tileA[threadIdx.y][threadIdx.x] = A[y * N + t * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + x];

        __syncthreads(); // Синхронизация всех потоков в блоке

        // Перемножаем текущие блоки
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads(); // Синхронизация перед переходом к следующему блоку
    }

    C[y * N + x] = sum;
}

// Проверка, что матрицы совпадают
bool matrices_match(const double* A, const double* B, size_t N) {
    double eps = 1e-6;
    for (size_t i = 0; i < N * N; ++i) {
        if (fabs(A[i] - B[i]) > eps) {
            return false;
        }
    }
    return true;
}

// Основная функция, которая вызывает ядра и cuBLAS
void run_multiplication(const double* A, const double* B, double* C1, double* C2, double* C3, size_t N) {
    double *d_A, *d_B, *d_C1, *d_C2, *d_C3;
    size_t bytes = N * N * sizeof(double);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C1, bytes);
    cudaMalloc(&d_C2, bytes);
    cudaMalloc(&d_C3, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Таймеры CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_naive, time_tiled, time_cublas;

    // Запуск наивной версии
    cudaEventRecord(start);
    kernel_naive<<<blocks, threads>>>(d_A, d_B, d_C1, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_naive, start, stop);

    // Запуск блоковой версии
    cudaEventRecord(start);
    kernel_tiled<<<blocks, threads>>>(d_A, d_B, d_C2, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tiled, start, stop);

    // cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0, beta = 0.0;

    cudaEventRecord(start);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C3, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_cublas, start, stop);

    cublasDestroy(handle);

    // Считываем результаты обратно на CPU
    cudaMemcpy(C1, d_C1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(C2, d_C2, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(C3, d_C3, bytes, cudaMemcpyDeviceToHost);

    // Вывод времени выполнения
    printf("Наивный подход:  %.3f мс\n", time_naive);
    printf("Блоковый подход:  %.3f мс\n", time_tiled);
    printf("cuBLAS:        %.3f мс\n", time_cublas);

    // Проверка совпадения результатов
    bool ok1 = matrices_match(C1, C2, N);
    bool ok2 = matrices_match(C1, C3, N);
    bool ok3 = matrices_match(C2, C3, N);

    printf("Наивный метод == Блоковый метод: %s\n", ok1 ? "Совпало" : "Не совпало");
    printf("Наивный метод == cuBLAS: %s\n", ok2 ? "Совпало" : "Не совпало");
    printf("Блоковый метод == cuBLAS: %s\n", ok3 ? "Совпало" : "Не совпало");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(d_C3);
}

// Главная точка входа
int main() {
    const size_t N = 2048;  // Размер квадратной матрицы
    size_t bytes = N * N * sizeof(double);

    // Выделение памяти под матрицы
    double* A = (double*)malloc(bytes);
    double* B = (double*)malloc(bytes);
    double* C1 = (double*)malloc(bytes);
    double* C2 = (double*)malloc(bytes);
    double* C3 = (double*)malloc(bytes);

    // Заполнение матриц случайными значениями
    for (size_t i = 0; i < N * N; ++i) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Запуск умножения и сравнение
    run_multiplication(A, B, C1, C2, C3, N);

    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C3);

    return 0;
}