import os
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# CUDA code string (contents of your CUDA file)
cuda_code = ""

with open("heavy_hash.cu", "r") as f:
    cuda_code = f.read()

# Compile the CUDA code
# mod = SourceModule(cuda_code, options=['--expt-relaxed-constexpr'])
# mod = SourceModule(cuda_code, options=['--expt-relaxed-constexpr', '-rdc=true'])
mod = SourceModule(cuda_code)

calc_heavy_hash_cuda = mod.get_function("calc_heavy_hash_cuda_kernel")

def calc_heavy_hash(mat, hash_input):
    # Ensure inputs are numpy arrays
    mat = np.asarray(mat, dtype=np.uint8)
    hash_input = np.frombuffer(hash_input, dtype=np.uint8)

    # Prepare output array
    product = np.zeros(32, dtype=np.uint8)

    # Call the CUDA function
    calc_heavy_hash_cuda(
        cuda.In(mat),
        cuda.In(hash_input),
        cuda.Out(product),
        block=(256, 1, 1),
        grid=(1, 1)
    )

    return product

def load_matrix(hash: str):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    matrix_dump_file_path = f"{root_dir}/miner/tests/data/{hash}.bin"

    with open(matrix_dump_file_path, "rb") as f:
        raw_data = f.read()

    mat = np.frombuffer(raw_data, dtype=np.uint16)
    return mat.reshape((64, 64))

# Example usage
if __name__ == "__main__":
    # mat = np.random.randint(0, 256, size=(32, 128), dtype=np.uint8)
    # hash_input = np.random.bytes(32)

    mat_pre_pow_hash_str = "a33db6a616a3fb1e13097b515664eecd7a00e5c5d4bd67b705bc83928bb60c7a"
    hash_input = bytes.fromhex(mat_pre_pow_hash_str)
    mat = load_matrix(mat_pre_pow_hash_str)

    result = calc_heavy_hash(mat, hash_input)

    assert bytes(result).hex() == "900e85952580b82d203a48636557ddfe5933d6f6e78f5594368fb0a1b8853f59"
    
    print("Heavy Hash Result:", result)
    print("Heavy Hash Result:", bytes(result).hex())