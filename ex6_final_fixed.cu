#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <math.h>
#include <time.h>

#define THREADS_PER_BLOCK 32

// Генерация случайной матрицы (на GPU)
void random_doubles(double* d_a, int size, curandGenerator_t gen) {
    curandGenerateUniformDouble(gen, d_a, size);
}

// Генерация положительно определённой матрицы B = Aᵗ·A
void gen_dpm(cublasHandle_t cublasH, double *d_out, double *d_tmp, int n) {
    const double alpha = 1.0, beta = 0.0;
    cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, d_tmp, n, d_tmp, n, &beta, d_out, n);
}

// Генерация матриц A и B
void gen_matrix(cublasHandle_t cublasH, curandGenerator_t gen, double *d_a, double *d_b, double *d_tmp, int n) {
    random_doubles(d_a, n * n, gen); // A — просто случайная
    random_doubles(d_tmp, n * n, gen);
    gen_dpm(cublasH, d_b, d_tmp, n); // B = tmpᵗ * tmp — положительно определённая
}

// Диагонализация на GPU
void eig_gpu(cusolverDnHandle_t solverH, double *d_a, double *d_b, double *d_eig, int n) {
    int *d_info;
    double *work;
    int lwork;

    cudaMalloc(&d_info, sizeof(int));
    cusolverDnDsygvd_bufferSize(solverH, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
                                CUBLAS_FILL_MODE_UPPER, n, d_a, n, d_b, n, d_eig, &lwork);

    cudaMalloc(&work, lwork * sizeof(double));

    cusolverDnDsygvd(solverH, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
                     CUBLAS_FILL_MODE_UPPER, n, d_a, n, d_b, n, d_eig, work, lwork, d_info);

    cudaFree(work);
    cudaFree(d_info);
}

int main() {
    int dim[] = {500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000};
    int len = sizeof(dim) / sizeof(int);
    int trial = 5, execute = 20;
    float times[execute];
    float mean, stddev, elapsed;

    double *d_a, *d_b, *d_eig, *d_tmp;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasHandle_t cublasH;
    cusolverDnHandle_t solverH;
    cublasCreate(&cublasH);
    cusolverDnCreate(&solverH);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)time(NULL));

    printf("|   N   |   GPU (ms)  |  σ GPU (ms) |\n");
    printf("+-------+------------+-------------+\n");

    for (int index = 0; index < len; index++) {
        int n = dim[index];
        cudaMalloc(&d_a, n * n * sizeof(double));
        cudaMalloc(&d_b, n * n * sizeof(double));
        cudaMalloc(&d_tmp, n * n * sizeof(double));
        cudaMalloc(&d_eig, n * sizeof(double));

        for (int i = 0; i < trial + execute; i++) {
            gen_matrix(cublasH, gen, d_a, d_b, d_tmp, n);

            if (i >= trial) {
                cudaDeviceSynchronize();
                cudaEventRecord(start, 0);

                eig_gpu(solverH, d_a, d_b, d_eig, n);

                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsed, start, stop);

                times[i - trial] = elapsed;
            } else {
                eig_gpu(solverH, d_a, d_b, d_eig, n); // прогрев
            }
        }

        mean = 0.0f;
        for (int i = 0; i < execute; i++) mean += times[i];
        mean /= execute;

        stddev = 0.0f;
        for (int i = 0; i < execute; i++) stddev += (times[i] - mean) * (times[i] - mean);
        stddev = sqrtf(stddev / execute);

        printf("| %5d | %10.4f | %11.4f |\n", n, mean, stddev);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_tmp);
        cudaFree(d_eig);
    }

    curandDestroyGenerator(gen);
    cublasDestroy(cublasH);
    cusolverDnDestroy(solverH);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
