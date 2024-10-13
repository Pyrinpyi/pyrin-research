#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

__device__ void calc_heavy_hash_kernel(const uint8_t* mat, const uint8_t* hash, uint8_t* product) {
    uint8_t vector[64];
    for (int i = 0; i < 32; i++) {
        vector[2 * i] = hash[i] >> 4;
        vector[2 * i + 1] = hash[i] & 0x0F;
    }

    for (int i = threadIdx.x; i < 32; i += blockDim.x) {
        uint32_t sum1 = 0, sum2 = 0;
        for (int j = 0; j < 64; j++) {
            sum1 += mat[128 * i + j] * vector[j];
            sum2 += mat[128 * i + 64 + j] * vector[j];
        }
        product[i] = (((sum1 & 0xF) ^ ((sum1 >> 4) & 0xF) ^ ((sum1 >> 8) & 0xF)) << 4) | ((sum2 & 0xF) ^ ((sum2 >> 4) & 0xF) ^ ((sum2 >> 8) & 0xF));
    }

    __syncthreads();

    for (int i = threadIdx.x; i < 32; i += blockDim.x) {
        product[i] ^= hash[i];
    }
}

extern "C" {
    __global__ void calc_heavy_hash_cuda_kernel(const uint8_t* mat, const uint8_t* hash, uint8_t* product) {
        calc_heavy_hash_kernel(mat, hash, product);
    }
}

// Host function to call the kernel
extern "C" void calc_heavy_hash_cuda(const uint8_t* mat, const uint8_t* hash, uint8_t* product) {
    uint8_t *d_mat, *d_hash, *d_product;

    // Allocate memory on the GPU
    cudaMalloc(&d_mat, 128 * 32 * sizeof(uint8_t));
    cudaMalloc(&d_hash, 32 * sizeof(uint8_t));
    cudaMalloc(&d_product, 32 * sizeof(uint8_t));

    // Copy data to the GPU
    cudaMemcpy(d_mat, mat, 128 * 32 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash, hash, 32 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch the kernel
    calc_heavy_hash_cuda_kernel<<<1, 256>>>(d_mat, d_hash, d_product);

    // Copy the result back to the host
    cudaMemcpy(product, d_product, 32 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_mat);
    cudaFree(d_hash);
    cudaFree(d_product);
}