#include <stdio.h>
#include <cuda_runtime.h>

const int N = 65536;

__global__ void f(int *a, int *b, int *c) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < N) c[i] = a[i] + b[i];
}

int main(int argc, char **argv) {
    int host_a[N], host_b[N], host_c[N];
    int *dev_a, *dev_b, *dev_c;

    for(int i = 0 ; i < N ; i++) {
        host_a[i] = i;
        host_b[i] = i + 1;
    }

    cudaMalloc((void**)&dev_a, sizeof(int)*N);
    cudaMalloc((void**)&dev_b, sizeof(int)*N);
    cudaMalloc((void**)&dev_c, sizeof(int)*N);

    cudaMemcpy(dev_a, host_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, host_c, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads_block = 256;
    int blocks_grid = (N + threads_block - 1) / threads_block;
    f<<<blocks_grid, threads_block>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(host_c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    /*
    for(int i = 0 ; i < N ; i++) {
        printf("%d ", host_c[i]);
    }
    */
}