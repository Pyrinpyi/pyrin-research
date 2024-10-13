import numpy as np
import pycuda.driver as drv

def uint256_to_array(value):
    return np.array([(value >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(4)], dtype=np.uint64)

def array_to_uint256(arr):
    return sum(arr[i] << (64 * i) for i in range(4))

def cuda_compare_uint256(kernels, a, b) -> bool:
    # Combine arrays
    input_array = np.concatenate([a, b])

    # Allocate memory on GPU
    input_gpu = drv.mem_alloc(input_array.nbytes)
    output_gpu = drv.mem_alloc(2 * 4 * 8)  # 2 Uint256 structures
    result_gpu = drv.mem_alloc(1)  # 1 boolean result

    # Copy input data to GPU
    drv.memcpy_htod(input_gpu, input_array)

    # Run kernel
    init_and_compare_uint256 = kernels["init_and_compare_uint256"].function
    init_and_compare_uint256(
        input_gpu, output_gpu, result_gpu,
        block=(1, 1, 1), grid=(1, 1)
    )

    # Retrieve results
    output = np.zeros(8, dtype=np.uint64)
    result = np.zeros(1, dtype=bool)
    drv.memcpy_dtoh(output, output_gpu)
    drv.memcpy_dtoh(result, result_gpu)

    # Convert back to Python integers
    a = array_to_uint256(output[:4])
    b = array_to_uint256(output[4:])

    return a, b, result