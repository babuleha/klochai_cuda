#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <math.h>

#define THREADS_PER_BLOCK 32

// Функция для генерации случайной матрицы
void random_doubles(double* d_a, int size) {
    static curandGenerator_t gen = NULL;
    if (gen == NULL) {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)time(NULL));
    }
    curandGenerateUniformDouble(gen, d_a, size);
}

// Кернел для симметризации матрицы
__global__ void gen_symm(double *d_a, double *d_tmp, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) { 
        d_a[y * n + x] = d_tmp[y * n + x] + d_tmp[x * n + y];
    }
}

// Функция для генерации положительно определённой матрицы
void gen_dpm(double *d_a, double *d_tmp, int n) {
    double alpha = 1.0;
    double beta = 0.0;
    static cublasHandle_t handle = NULL;
    if (handle == NULL) {
        cublasCreate(&handle);
    }
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_tmp, n, d_tmp, n, &beta, d_a, n);
}

// Генерация матриц A и B
void gen_matrix(double *d_a, double *d_b, int n) {
    double *d_tmp;
    cudaMalloc((void **)&d_tmp, n * n * sizeof(double));

    dim3 gridDim((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    random_doubles(d_tmp, n * n);
    gen_symm<<<gridDim, blockDim>>>(d_a, d_tmp, n);
    cudaDeviceSynchronize();

    random_doubles(d_tmp, n * n);
    gen_symm<<<gridDim, blockDim>>>(d_tmp, d_b, n);
    cudaDeviceSynchronize();
    gen_dpm(d_b, d_tmp, n);

    cudaFree(d_tmp);
}

// Диагонализация на GPU
void eig_gpu(cusolverDnHandle_t cusolverH, double *d_a, double *d_b, double *d_eig, int n) {
    int *d_info;
    double *work;
    int lwork = 0;

    cudaMalloc((void **)&d_info, sizeof(int));

    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    cusolverDnDsygvd_bufferSize(cusolverH, itype, jobz, uplo, n, d_a, n, d_b, n, d_eig, &lwork);
    cudaMalloc((void **)&work, sizeof(double) * lwork);

    cusolverDnDsygvd(cusolverH, itype, jobz, uplo, n, d_a, n, d_b, n, d_eig, work, lwork, d_info);

    cudaFree(work);
    cudaFree(d_info);
}

int main() {
    double *d_a, *d_b, *d_eig;

    int dim[] = {500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000};
    int len = sizeof(dim) / sizeof(int);
    int trial = 5, execute = 20;
    float elapsed, mean, stddev;
    float times[execute];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    printf("|   N   |   GPU (ms)  |  σ GPU (ms) |\n");
    printf("+-------+------------+-------------+\n");

    for (int index = 0; index < len; index++) {
        int n = dim[index];

        cudaMalloc((void **)&d_a, n * n * sizeof(double));
        cudaMalloc((void **)&d_b, n * n * sizeof(double));
        cudaMalloc((void **)&d_eig, n * sizeof(double));

        for (int i = 0; i < trial + execute; i++) {
            gen_matrix(d_a, d_b, n);

            if (i >= trial) { // Только для execute замеров считаем время
                cudaDeviceSynchronize();
                cudaEventRecord(start, 0);

                eig_gpu(cusolverH, d_a, d_b, d_eig, n);

                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsed, start, stop);

                times[i - trial] = elapsed;
            } else {
                eig_gpu(cusolverH, d_a, d_b, d_eig, n); // Прогрев
            }
        }

        // Считаем среднее и СКО
        mean = 0.0f;
        for (int i = 0; i < execute; i++) mean += times[i];
        mean /= execute;

        stddev = 0.0f;
        for (int i = 0; i < execute; i++) stddev += (times[i] - mean) * (times[i] - mean);
        stddev = sqrtf(stddev / execute);

        printf("| %5d | %10.4f | %11.4f |\n", n, mean, stddev);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_eig);
    }

    cusolverDnDestroy(cusolverH);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
