import struct
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import blake3

# Load the CUDA code
with open('blake3.cu', 'r') as f:
    cuda_code = f.read()

mod = SourceModule(cuda_code)

# Get the CUDA functions
blake3_init_kernel = mod.get_function("blake3_init_kernel")
blake3_update_kernel = mod.get_function("blake3_update_kernel")
blake3_finalize_kernel = mod.get_function("blake3_finalize_kernel")
blake3_hash_kernel = mod.get_function("blake3_hash_kernel")

# Define the Blake3State structure using NumPy dtype
blake3_state_dtype = np.dtype([
    ('key', np.uint32, 8),
    ('chunk', np.uint32, 16),
    ('chunk_length', np.uint64),
    ('total_length', np.uint64),
    ('flags', np.uint32)
])

def blake3_hash_no_key(input_data):
    # Ensure input is a bytes object
    if isinstance(input_data, str):
        input_data = input_data.encode()
    
    # Prepare input and output arrays
    input_array = np.frombuffer(input_data, dtype=np.uint8)
    output_array = np.zeros(32, dtype=np.uint8)

    # Allocate memory on the GPU
    input_gpu = cuda.mem_alloc(input_array.nbytes)
    output_gpu = cuda.mem_alloc(output_array.nbytes)

    # Copy input data to the GPU
    cuda.memcpy_htod(input_gpu, input_array)

    # Call the CUDA kernel
    blake3_hash_kernel(
        input_gpu,
        np.int32(len(input_array)),
        output_gpu,
        block=(1, 1, 1),
        grid=(1, 1)
    )

    # Copy the result back to the host
    cuda.memcpy_dtoh(output_array, output_gpu)

    return output_array.tobytes()

def blake3_hash(data, key=None, data_list=[]):
    # Allocate memory for the state
    state = cuda.mem_alloc(blake3_state_dtype.itemsize)

    # Initialize the state
    if key is None:
        key = np.zeros(32, dtype=np.uint8)
        flags = 0  # No flags for non-keyed hash
    else:
        key = np.frombuffer(key, dtype=np.uint8)
        flags = 1 << 4  # KEYED_HASH flag
    key_gpu = cuda.mem_alloc(key.nbytes)
    cuda.memcpy_htod(key_gpu, key)
    blake3_init_kernel(state, key_gpu, np.uint32(flags), block=(1,1,1), grid=(1,1))

    # Update the state with the input data
    data_np = np.frombuffer(data, dtype=np.uint8)
    data_gpu = cuda.mem_alloc(data_np.nbytes)
    cuda.memcpy_htod(data_gpu, data_np)
    blake3_update_kernel(state, data_gpu, np.int64(len(data)), block=(1,1,1), grid=(1,1))

    for data in data_list:
        data_np = np.frombuffer(data, dtype=np.uint8)
        data_gpu = cuda.mem_alloc(data_np.nbytes)
        cuda.memcpy_htod(data_gpu, data_np)
        blake3_update_kernel(state, data_gpu, np.int64(len(data)), block=(1,1,1), grid=(1,1))

    # Finalize and get the output
    output = np.zeros(32, dtype=np.uint8)
    output_gpu = cuda.mem_alloc(output.nbytes)
    blake3_finalize_kernel(state, output_gpu, block=(1,1,1), grid=(1,1))
    cuda.memcpy_dtoh(output, output_gpu)

    # Clean up
    state.free()
    key_gpu.free()
    data_gpu.free()
    output_gpu.free()

    return bytes(output)

def test_blake3():
    data = b"Hello, BLAKE3!"
    key = b"BlockHash\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
    # print("key", key, len(key))

    data = struct.pack("<Q", 1)

    assert blake3_hash(data).hex() == blake3.blake3().update(data).hexdigest()

    hash_result = blake3_hash(data, data_list=[data])
    assert hash_result.hex() == blake3.blake3().update(data).update(data).hexdigest()

    hash_result = blake3_hash(data, key=key)
    assert hash_result.hex() == blake3.blake3(key=key).update(data).hexdigest()

    hash_result = blake3_hash(data, key=key, data_list=[data])
    assert hash_result.hex() == blake3.blake3(key=key).update(data).update(data).hexdigest()

    # key = np.zeros(32, dtype=np.uint8)
    # Hash without key
    hash_result = blake3_hash(data, key=key)
    hash_result = blake3_hash(data, data_list=[data])
    # hash_result = blake3_hash(data, key=key, data_list=[data])
    print("hash_result", hash_result.hex())

    # Python
    hasher = blake3.blake3(key=key)
    # hasher = blake3.blake3()
    # hasher.update(data)
    hasher.update(data)

    # print("data:", data)
    print("Hash:", hasher.hexdigest())

    # assert hasher.hexdigest() == hash_result.hex()

    # # Hash without key
    # hash_result = blake3_hash(data)
    # print("BLAKE3 Hash: 1", hash_result.hex())

    # hash_result_no_key = blake3_hash_no_key(data)
    # print("BLAKE3 Hash: 2", hash_result_no_key.hex())

    # assert hash_result.hex() == hash_result_no_key.hex()

    # # Python
    # hasher = blake3.blake3()
    # hasher.update(data)

    # print("BLAKE3 Hash: 3", hasher.hexdigest())
    # assert hasher.hexdigest() == hash_result.hex()

    # # Update the state
    # hash_result = blake3_hash(data, data_list=[b"1"])
    # print("BLAKE3 With updated State 1:", hash_result.hex())

    # hasher = blake3.blake3()
    # hasher.update(data)
    # hasher.update(b"1")
    # print("BLAKE3 With updated State 2:", hasher.hexdigest())

    # # Hash with key
    # keyed_hash_result = blake3_hash(data, key)
    # print("BLAKE3 Keyed Hash:", keyed_hash_result.hex())

    # hasher = blake3.blake3(key=key)
    # hasher.update(data)

    # print("hasher.hexdigest()", hasher.hexdigest())
    # # assert hasher.hexdigest() == keyed_hash_result.hex()