#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
 * Function to perform error checking in CUDA.
 * CUDA function calls must be passed as a parameter to gpuErrchk to print Error strings, if any.
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

const int N = 128;

/*
 * CUDA Kernel
 * __global__ allows function to be called from Host (CPU), and executed in the Device (GPU).
 */
__global__ void f(int *dev_a) {
    unsigned int tid = threadIdx.x; //
    
    if(tid < N) {
        dev_a[tid] = tid * tid;
    }
}

int main(int argc, char **argv) {
    int host_a[N];
    int *dev_a;
    gpuErrchk( cudaMalloc((void**)&dev_a, N * sizeof(int)));
    for(int i = 0 ; i < N ; i++) {
        host_a[i] = i;
    }
    gpuErrchk(cudaMemcpy(dev_a, host_a, N * sizeof(int), cudaMemcpyHostToDevice));

    /*
     * Kernel function call
     * <<< X, Y >>> : X = Number of blocks per Grid,
     *                Y = Number of threads per block
     */
    f<<<1, N>>>(dev_a);
    
    gpuErrchk(cudaMemcpy(host_a, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    for(int i = 0 ; i < N ; i++) {
        printf("%d ", host_a[i]);
    }
}
