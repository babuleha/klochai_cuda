#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <lapacke.h>

#define THREADS_PER_BLOCK 32

void print_matrix(double *a, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%e ", a[i * n + j]);
        }
        printf("\n");
    }
}

void random_doubles(double* d_a, int size) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, (int) time(NULL));
    curandGenerateUniformDouble(gen, d_a, size);
    curandDestroyGenerator(gen);
}

global void gen_symm(double *d_a, double *d_tmp, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) { 
        d_a[y * n + x] = d_tmp[y * n + x] + d_tmp[x * n + y];
    }
}


void gen_dpm(double *d_a, double *d_tmp, int n) {
    double alpha = 1.0;
    double beta = 0.0;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_tmp, n, d_tmp, n, &beta, d_a, n);
    cublasDestroy(handle);
    
}

void gen_matrix(double *a, double *b, double *d_a, double *d_b, int n) {
    double *d_tmp;
    cudaMalloc((void ) &d_tmp, n * n * sizeof(double));

    dim3 gridDim((n + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, (n + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, 1);
    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    random_doubles(d_tmp, n * n);
    gen_symm<<<gridDim,blockDim>>>(d_a, d_tmp, n);
    cudaMemcpy(a, d_a, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    random_doubles(d_b, n * n);
    gen_symm<<<gridDim,blockDim>>>(d_tmp, d_b, n);
    gen_dpm(d_b, d_tmp, n);
    cudaMemcpy(b, d_b, n * n * sizeof(double), cudaMemcpyDeviceToHost);
 
    cudaFree(d_tmp);
}

void eig_cpu(double *a, double *b, double *eig, int n) {
    int info;
    info = LAPACKE_dsygvd(LAPACK_ROW_MAJOR, 1, 'V', 'U', n, a, n, b, n, eig);
    if (info != 0) {
        printf("Ошибка LAPACK: %d\n", info);
    }
}

void eig_gpu(double *d_a, double *d_b, double *d_eig, int n) {
    cusolverDnHandle_t cusolverH;
    cudaStream_t stream;
    int *d_info, info;
    double *work;
    int lwork = 0;

    cudaMalloc((void )&d_info, sizeof(int));

    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;


    cusolverDnCreate(&cusolverH);
    cudaStreamCreate(&stream);
    cusolverDnSetStream(cusolverH, stream);
    cusolverDnDsygvd_bufferSize(cusolverH, itype, jobz, uplo, n, d_a, n, d_b, n, d_eig, &lwork);
    cudaMalloc((void )&work, sizeof(double) * lwork);

    cusolverDnDsygvd(cusolverH, itype, jobz, uplo, n, d_a, n, d_b, n, d_eig, work, lwork, d_info);

    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    if (info != 0) {
        printf("Ошибка CuSOLVER: %d\n", info);
    }

    cudaFree(work);
    cudaFree(d_info);
    cusolverDnDestroy(cusolverH);
    cudaStreamDestroy(stream);
}

int main() {
    double *a, *b, *eig;
    double *d_a, *d_b, *d_eig;

    int dim[] = {100, 1000, 2000, 3000, 5000, 6000, 8000, 10000, 12000};
    int len = (int) sizeof(dim) / sizeof(int);
    int trial = 5, execute = 100;
    clock_t start, stop;
    double cpu_time, gpu_time;

    printf("    Размер матрицы   ||   Время на CPU   ||   Время на GPU \n");

    for (int index = 0; index < len; index++) {

        a = (double *) malloc(dim[index] * dim[index] * sizeof(double));
        b = (double *) malloc(dim[index] * dim[index] * sizeof(double));
        eig = (double *) malloc(dim[index] * sizeof(double));

        cudaMalloc((void ) &d_a, dim[index] * dim[index] * sizeof(double));
        cudaMalloc((void ) &d_b, dim[index] * dim[index] * sizeof(double));
        cudaMalloc((void ) &d_eig, dim[index] * sizeof(double));



        cpu_time = 0.;
        gpu_time = 0.;

        for (int i = 0; i < trial + execute; i++) {
            if (i < trial) {
                gen_matrix(a, b, d_a, d_b, dim[index]);

                eig_cpu(a, b, eig, dim[index]);
                eig_gpu(d_a, d_b, d_eig, dim[index]);
            } else {
                gen_matrix(a, b, d_a, d_b, dim[index]);

                start = clock();
                eig_cpu(a, b, eig, dim[index]);
                stop = clock();
                cpu_time += (double)(stop - start) / CLOCKS_PER_SEC;

                start = clock();
                eig_gpu(d_a, d_b, d_eig, dim[index]);
                stop = clock();
                gpu_time += (double)(stop - start) / CLOCKS_PER_SEC;
            }
        }
        
        printf("%5d      %2.4f        %2.4f \n", dim[index], cpu_time / execute, gpu_time / execute);

        free(a); cudaFree(d_a);
        free(b); cudaFree(d_b);
        free(eig); cudaFree(d_eig);


    }
    
    return 0;

}
