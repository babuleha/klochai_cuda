#include <iostream>
#include <cstddef>
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "assert.h"
#include <cmath>

using namespace std;

// Функция для проверки равенства двух матриц
bool matrices_match(const double* A, const double* B, size_t N) {
    double eps = 1e-6;
    for (size_t i = 0; i < N * N; ++i) {
        if (fabs(A[i] - B[i]) > eps) {
            return false;
        }
    }
    return true;
}

// Функция для вывода матрицы на экран
void print_matrix(string array_name, double *a, int N, int M) {
    for (size_t i = 0; i != N; i++) {
        for (size_t j = 0; j != M; j++) {
            cout << a[j * N + i] << " ";
        }
        cout << endl;
    }
}

int main() {
    double *A, *B_long, *C_long, *C_long_async;
    double *A_device, *B_long_device, *C_long_device, *C_long_async_device;
    const size_t matrix_dim = 1024; // Размерность матрицы A
    const size_t array_dim = matrix_dim * matrix_dim; // Число элементов в матрице A
    const size_t N = 2; // Количество раз, сколько матрица A будет в длинной матрице
    double alpha = 1.0;
    double beta = 0.0;
    cublasStatus_t status;
    cudaError_t cudaerr;
    cublasHandle_t handle;
    cudaEvent_t start;
    cudaStream_t stream[N];
    float gpuTime = 0.0f;

    size_t size = array_dim * sizeof(double);

    // Выделение памяти для хост-данных
    A = (double *)malloc(size);
    B_long = (double *)malloc(size * N);
    C_long = (double *)malloc(size * N);
    C_long_async = (double *)malloc(size * N);

    // Выделение памяти на устройстве
    cudaMalloc((void **)&A_device, size);
    cudaMalloc((void **)&B_long_device, size * N);
    cudaMalloc((void **)&C_long_device, size * N);
    cudaMalloc((void **)&C_long_async_device, size * N);

    // Заполнение матриц случайными значениями
    for (size_t i = 0; i < matrix_dim * matrix_dim; ++i) {
        A[i] = rand() % 100;
    }
    for (size_t i = 0; i < matrix_dim * matrix_dim * N; ++i) {
        B_long[i] = rand() % 100;
    }

    // Сериализация вычислений
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    cublasCreate(&handle);

    status = cublasSetVector(array_dim, sizeof(double), A, 1, A_device, 1);
    assert(status == CUBLAS_STATUS_SUCCESS);
    status = cublasSetVector(array_dim * N, sizeof(double), B_long, 1, B_long_device, 1);
    assert(status == CUBLAS_STATUS_SUCCESS);

    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_dim, matrix_dim * N, matrix_dim,
                         &alpha, A_device, matrix_dim, B_long_device, matrix_dim, &beta, C_long_device, matrix_dim);
    assert(status == CUBLAS_STATUS_SUCCESS);

    status = cublasGetVector(array_dim * N, sizeof(double), C_long_device, 1, C_long, 1);
    assert(status == CUBLAS_STATUS_SUCCESS);

    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("Serial cuBLAS: %.2f мс\n", gpuTime);

    // Асинхронное выполнение
    cudaEventRecord(start, 0);
    cublasHandle_t handle_asyn;
    cublasCreate(&handle_asyn);
    status = cublasSetVector(array_dim, sizeof(double), A, 1, A_device, 1);
    assert(status == CUBLAS_STATUS_SUCCESS);

    for (int i = 0; i != N; i++) {
        cudaerr = cudaStreamCreate(&stream[i]);
        assert(cudaerr == cudaSuccess);
    }

    for (int istream = 0; istream != N; istream++) {
        status = cublasSetVectorAsync(array_dim, sizeof(double), B_long + array_dim * istream,
                                      1, B_long_device + array_dim * istream, 1, stream[istream]);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }

    for (int istream = 0; istream != N; istream++) {
        status = cublasSetStream(handle_asyn, stream[istream]);
        assert(status == CUBLAS_STATUS_SUCCESS);

        status = cublasDgemm(handle_asyn, CUBLAS_OP_N, CUBLAS_OP_N, matrix_dim, matrix_dim, matrix_dim,
                             &alpha, A_device, matrix_dim, B_long_device + array_dim * istream, matrix_dim,
                             &beta, C_long_async_device + array_dim * istream, matrix_dim);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }

    for (int istream = 0; istream != N; istream++) {
        status = cublasGetVectorAsync(array_dim, sizeof(double), C_long_async_device + array_dim * istream,
                                      1, C_long_async + array_dim * istream, 1, stream[istream]);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }

    for (int i = 0; i != N; i++) {
        cudaerr = cudaStreamDestroy(stream[i]);
        assert(cudaerr == cudaSuccess);
    }

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("Async cuBLAS: %.2f мс\n", gpuTime);

    // Проверка на равенство матриц
    if (matrices_match(C_long, C_long_async, matrix_dim)) {
        cout << "Вычисления совпадают" << endl;
    } else {
        cout << "Вычисления не совпадают" << endl;
    }

    // Освобождение памяти
    cudaFree(A_device);
    cudaFree(B_long_device);
    cudaFree(C_long_device);
    free(A);
    free(B_long);
    free(C_long);
    free(C_long_async);

    return 0;
}
