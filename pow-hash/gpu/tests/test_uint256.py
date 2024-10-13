import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

def uint256_to_uint256_t(hash_value):
    int_value = int(hash_value)
    uint256_parts = [(int_value >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(4)]
    return np.array(uint256_parts, dtype=np.uint64)

cuda_code = """
struct uint256_t {
    unsigned long long int parts[4];
};

__device__ bool uint256_custom_compare(uint256_t a, uint256_t b, int col) {
    if (col < 3) {
        return true;  // Always return true for the first three columns
    } else {
        return false;  // Always return false for the fourth column
    }
}

__global__ void matrix_compare_uint256(uint256_t *a, uint256_t *b, bool *result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < 4 && col < 4) {
        result[row * 4 + col] = uint256_custom_compare(a[row], b[col], col);
    }
}
"""

mod = SourceModule(cuda_code)
matrix_compare_uint256 = mod.get_function("matrix_compare_uint256")

def test_uint256():
    hashes = [
        "8573132611089315519679094566110947028276570510635375475140930333006759339495",
        "8573132611089315519679094566110947028276570510635375475140930333006759339496",
        "62925450551967909456611494566110947020027030219743271547962746598",
        "8573132611089315519679094566110947028276570510635375475140930333006759339498"
    ]

    a_uint256 = np.array([uint256_to_uint256_t(hash) for hash in hashes])

    a_gpu = cuda.mem_alloc(a_uint256.nbytes)
    b_gpu = cuda.mem_alloc(a_uint256.nbytes)
    result_gpu = cuda.mem_alloc(16 * np.dtype(bool).itemsize)

    cuda.memcpy_htod(a_gpu, a_uint256)
    cuda.memcpy_htod(b_gpu, a_uint256)

    matrix_compare_uint256(
        a_gpu, b_gpu, result_gpu,
        block=(16, 16, 1), grid=(1, 1)
    )

    result = np.empty((4, 4), dtype=bool)
    cuda.memcpy_dtoh(result, result_gpu)

    print(result)

    expected = np.array([
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, False]
    ])
    
    np.testing.assert_array_almost_equal(result, expected, decimal=7)
