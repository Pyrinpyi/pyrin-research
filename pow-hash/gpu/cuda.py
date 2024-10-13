from typing import Dict, NamedTuple
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np

import cuda_uint256

"""

https://documen.tician.de/pycuda/tutorial.html#

"""

# # WINDOWS_CL_PATH = r"D:\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx86\x86"
# # os.environ["PATH"] += os.pathsep + WINDOWS_CL_PATH 

class CUDAKernel(NamedTuple):
    function: drv.Function
    block: tuple
    grid: tuple

def _init_kernels(vector_size: int) -> Dict[str, CUDAKernel]:
    with open("kernel.cu", "r") as f:
        cuda_kernel = f.read()

    # Compile the CUDA kernel
    mod = SourceModule(cuda_kernel)

    # Calculate grid and block dimensions
    block_size = 256
    grid_size = (vector_size + block_size - 1) // block_size

    kernels = {
        "init_and_compare_uint256": CUDAKernel(
            function=mod.get_function("init_and_compare_uint256"),
            # block=(block_size, 1, 1),
            block=(1, 1, 1),
            # grid=(grid_size, 1)
            grid=(1, 1, 1)
        ),
    } \
        | _init_blake3_kernels(vector_size) \
        | _init_heavy_hash_kernels(vector_size) \
        | _init_state_kernels(vector_size)
    
    return kernels

def _init_blake3_kernels(vector_size: int) -> Dict[str, CUDAKernel]:
    with open("blake3.cu", "r") as f:
        cuda_kernel = f.read()

    # Compile the CUDA kernel
    mod = SourceModule(cuda_kernel)

    # Calculate grid and block dimensions
    block_size = 256
    grid_size = (vector_size + block_size - 1) // block_size

    kernels = {
        "blake3_hash_kernel": CUDAKernel(
            function=mod.get_function("blake3_hash_kernel"),
            # block=(block_size, 1, 1),
            block=(1, 1, 1),
            # grid=(grid_size, 1)
            grid=(1, 1, 1)
        )
    }

    return kernels

def _init_heavy_hash_kernels(vector_size: int) -> Dict[str, CUDAKernel]:
    with open("heavy_hash.cu", "r") as f:
        cuda_kernel = f.read()

    mod = SourceModule(cuda_kernel)

    block_size = 256
    grid_size = (vector_size + block_size - 1) // block_size

    kernels = {
        "calc_heavy_hash_cuda_kernel": CUDAKernel(
            function=mod.get_function("calc_heavy_hash_cuda_kernel"),
            # block=(block_size, 1, 1),
            block=(256, 1, 1),
            # grid=(grid_size, 1)
            grid=(1, 1, 1)
        )
    }

    return kernels

# TODO: Remove everything but the state_kernels

def _init_state_kernels(vector_size: int) -> Dict[str, CUDAKernel]:
    with open("state.cu", "r") as f:
        cuda_kernel = f.read()

    mod = SourceModule(cuda_kernel)

    block_size = 256
    grid_size = (vector_size + block_size - 1) // block_size

    kernels = {
        "update_state": CUDAKernel(
            function=mod.get_function("update_state"),
            block=(1, 1, 1),
            grid=(1, 1)
        ),
        "get_state": CUDAKernel(
            function=mod.get_function("get_state"),
            block=(1, 1, 1),
            grid=(1, 1)
        ),
        "update_block_header": CUDAKernel(
            function=mod.get_function("update_block_header"),
            block=(1, 1, 1),
            grid=(1, 1)
        ),
        "calculate_block_header_hash": CUDAKernel(
            function=mod.get_function("calculate_block_header_hash"),
            block=(1, 1, 1),
            grid=(1, 1)
        ),
    }

    return kernels

def uint256_to_uint256_t(hash_value):
    # Convert the decimal string to an integer
    int_value = int(hash_value)
    
    # Create an array of 4 uint64 values
    uint256_parts = [(int_value >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(4)]
    
    # Return as a numpy array
    return np.array(uint256_parts, dtype=np.uint64)

class GPUKernel():
    def __init__(self, vector_size: int):
        self.vector_size = vector_size
        self.kernels  = _init_kernels(vector_size)

    def check_pow(self, kheavy_hash: int, target: int):
        # n = self.vector_size

        kheavy_hash = cuda_uint256.uint256_to_array(kheavy_hash)
        target = cuda_uint256.uint256_to_array(target)

        kheavy_hash, target, result = cuda_uint256.cuda_compare_uint256(self.kernels, kheavy_hash, target)

        # kheavy_hash <= target
        return result[0]
    
    def blake3_hash(self, input_data):
        if isinstance(input_data, str):
            input_data = input_data.encode()
        
        # Prepare input and output arrays
        input_array = np.frombuffer(input_data, dtype=np.uint8)
        output_array = np.zeros(32, dtype=np.uint8)

        # Allocate memory on the GPU
        input_gpu = drv.mem_alloc(input_array.nbytes)
        output_gpu = drv.mem_alloc(output_array.nbytes)

        # Copy input data to the GPU
        drv.memcpy_htod(input_gpu, input_array)

        # Call the CUDA kernel
        self.kernels["blake3_hash_kernel"].function(
            input_gpu,
            np.int32(len(input_array)),
            output_gpu,
            block=(1, 1, 1),
            grid=(1, 1)
        )

        # Copy the result back to the host
        drv.memcpy_dtoh(output_array, output_gpu)

        return output_array.tobytes()
    
    def calc_heavy_hash(self, mat, hash_input):
        mat = np.asarray(mat, dtype=np.uint8)
        hash_input = np.frombuffer(hash_input, dtype=np.uint8)

        # Prepare output array
        product = np.zeros(32, dtype=np.uint8)

        # Call the CUDA function
        self.kernels["calc_heavy_hash_cuda_kernel"].function(
            drv.In(mat),
            drv.In(hash_input),
            drv.Out(product),
            block=(256, 1, 1),
            grid=(1, 1)
        )

        return product

def get_cuda_info(log_info=False):
    device = drv.Device(0)
    attrs = device.get_attributes()

    if log_info:
        print("\n\n")
        print("-" * 80)
        print(f"\tDevice: {device.name()}")
        print(f"\tCompute Capability: {device.compute_capability()[0]}.{device.compute_capability()[1]}")
        print(f"\tTotal Memory: {device.total_memory() // (1024**2)} MB")

    # Calculate CUDA cores
    cores_per_mp = {
        (2, 0): 32,  # Fermi
        (2, 1): 48,  # Fermi
        (3, 0): 192, # Kepler
        (3, 5): 192, # Kepler
        (3, 7): 192, # Kepler
        (5, 0): 128, # Maxwell
        (5, 2): 128, # Maxwell
        (6, 0): 64,  # Pascal
        (6, 1): 128, # Pascal
        (7, 0): 64,  # Volta
        (7, 5): 64,  # Turing
        (8, 0): 64,  # Ampere
        (8, 6): 128, # Ampere
        (8, 9): 128, # Lovelace
        (9, 0): 128, # Hopper
    }

    cc = device.compute_capability()
    num_mp = attrs[drv.device_attribute.MULTIPROCESSOR_COUNT]
    cores_per_mp_for_cc = cores_per_mp.get(cc, 0)
    cuda_cores = cores_per_mp_for_cc * num_mp

    # Block is a group of threads, this is the limit per block (e.g. <<<blocksPerGrid, threadsPerBlock>>>)
    max_threads_per_block = attrs[drv.device_attribute.MAX_THREADS_PER_BLOCK]

    # Max memory can be used inside a block shared by all the threads (in bytes)
    max_shared_memory_per_block = attrs[drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]

    memory_clock_rate = attrs[drv.device_attribute.CLOCK_RATE] / 1000 # MHz
    memory_bus_width = attrs[drv.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]

    # Calculate memory bandwidth: bus width * clock rate
    # e.g. 256-bit bus at 1000 MHz = 256,000 Mbps = 32 GB/s
    memory_bandwidth = memory_bus_width * memory_clock_rate / 8000  # GB/s (/8000 is Mbits/s to GB/s)

    if log_info:
        print(f"\tCUDA Cores: {cuda_cores}")
        print(f"\tMax threads per block: {max_threads_per_block}")
        print(f"\tMax shared memory per block: {max_shared_memory_per_block // 1024} KB")
        print(f"\tClock rate: {attrs[drv.device_attribute.CLOCK_RATE] / 1000} MHz")
        print(f"\tMemory Clock rate: {memory_clock_rate} MHz")
        print(f"\tMemory Bus Width: {attrs[drv.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]} bits")
        print(f"\tMemory Bandwidth: {memory_bandwidth:.2f} GB/s")
        print("-" * 80)
        print("\n\n")

    return cuda_cores, max_threads_per_block, max_shared_memory_per_block,

# Helper function to convert Python int to uint256_t
def int_to_uint256(x):
    return np.array([(x >> (64*i)) & 0xFFFFFFFFFFFFFFFF for i in range(4)][::-1], dtype=np.uint64)

# Helper function to convert uint256_t to Python int
def uint256_to_int(x):
    return sum(int(part) << (64*i) for i, part in enumerate(reversed(x)))