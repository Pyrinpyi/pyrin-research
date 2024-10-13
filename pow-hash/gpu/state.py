from typing import List

import struct
import asyncio
import binascii
import pycuda.driver as drv
import pycuda.autoinit
import numpy as np

BlockHeaderData = np.dtype([
    ('version', np.uint16),
    ('parents_by_level_count', np.uint64),
    ('parents_by_level_lengths', np.uintp),
    ('parents_by_level_data', np.uintp),
    ('hash_merkle_root', np.uint8, 32),
    ('accepted_id_merkle_root', np.uint8, 32),
    ('utxo_commitment', np.uint8, 32),
    ('timestamp', np.uint64),
    ('bits', np.uint32),
    ('nonce', np.uint64),
    ('daa_score', np.uint64),
    ('blue_score', np.uint64),
    ('blue_work_length', np.uint64),
    ('blue_work', np.uintp),
    ('pruning_point', np.uint8, 32)
])

def calc_block_header_hash_cuda(
        kernels,
        hash_merkle_root: str, accepted_id_merkle_root: str, utxo_commitment: str, bits: int,
        timestamp: int, nonce: int, version: int, daa_score: int, blue_score: int, parents_by_level: List,
        blue_work: str, pruning_point: str
) -> bytes:
    # Prepare the data
    parents_by_level_count = np.uint64(len(parents_by_level))
    parents_by_level_lengths = np.array([len(level) for level in parents_by_level], dtype=np.uint64)
    parents_by_level_data = np.concatenate([np.frombuffer(bytes.fromhex(''.join(level)), dtype=np.uint8) for level in parents_by_level])

    blue_work_bytes = decode_blue_work(blue_work)
    blue_work_length = np.uint64(len(blue_work_bytes))

    # Create the BlockHeaderData structure
    block_header_data = np.array([
        (np.uint16(version),
         parents_by_level_count,
         drv.to_device(parents_by_level_lengths),
         drv.to_device(parents_by_level_data),
         np.frombuffer(bytes.fromhex(hash_merkle_root), dtype=np.uint8),
         np.frombuffer(bytes.fromhex(accepted_id_merkle_root), dtype=np.uint8),
         np.frombuffer(bytes.fromhex(utxo_commitment), dtype=np.uint8),
         np.uint64(timestamp),
         np.uint32(bits),
         np.uint64(nonce),
         np.uint64(daa_score),
         np.uint64(blue_score),
         blue_work_length,
         drv.to_device(np.frombuffer(blue_work_bytes, dtype=np.uint8)),
         np.frombuffer(bytes.fromhex(pruning_point), dtype=np.uint8))
    ], dtype=BlockHeaderData)

    # Allocate output memory
    output_hash = drv.mem_alloc(32)

    print("block_header_data", block_header_data)

    kernels["calculate_block_header_hash"].function(
        drv.In(block_header_data), 
        output_hash,
        block=(1, 1, 1),
        grid=(1, 1)
    )

    # Copy the result back to host
    result = np.empty(32, dtype=np.uint8)
    drv.memcpy_dtoh(result, output_hash)

    return bytes(result)

def calculate_block_hash(kernels, header):
    # Convert parents to the required format
    parents_by_level = [
        [parent for parent in level]
        for level in header.parents_by_level
    ]
    
    # Call the CUDA function
    block_hash = calc_block_header_hash_cuda(
        kernels,
        hash_merkle_root=header.hash_merkle_root,
        accepted_id_merkle_root=header.accepted_id_merkle_root,
        utxo_commitment=header.utxo_commitment,
        bits=header.bits,
        timestamp=header.timestamp,
        nonce=header.nonce,
        version=header.version,
        daa_score=header.daa_score,
        blue_score=header.blue_score,
        parents_by_level=parents_by_level,
        blue_work=header.blue_work,
        pruning_point=header.pruning_point
    )
    
    return block_hash.hex()

block_header_dtype = np.dtype([
    ('version', np.uint16),
    # ('parents_by_level_count', np.uint64),
    # ('parents_by_level', np.uint64),  # Pointer, so 64-bit on most systems
    # ('hash_merkle_root', 'S32'),
    # ('accepted_id_merkle_root', 'S32'),
    # ('utxo_commitment', 'S32'),
    ('hash_merkle_root', (np.uint8, 32)),
    ('accepted_id_merkle_root', (np.uint8, 32)),
    ('utxo_commitment', (np.uint8, 32)),
    ('timestamp', np.uint64),
    ('bits', np.uint32),
    ('nonce', np.uint64),
    ('daa_score', np.uint64),
    ('blue_score', np.uint64),
    ('blue_work_length', np.uint64),
    ('blue_work', np.uint64),  # Pointer, so 64-bit on most systems
    ('pruning_point', (np.uint8, 32)),
])

class GPUState:
    def __init__(self, kernel):
        self.is_synced = False
        self.kernels = kernel.kernels
        # self.state_gpu = drv.mem_alloc(np.dtype([('is_synced', np.bool_), ('header', np.void, 32)]).itemsize)
        self.state_gpu = drv.mem_alloc(block_header_dtype.itemsize)
        
        # init_state = np.array([(False, b'\0' * 32)], dtype=[('is_synced', np.bool_), ('header', np.void, 32)])
        init_state = np.zeros(1, dtype=block_header_dtype)
        drv.memcpy_htod(self.state_gpu, init_state)

    def __del__(self):
        # Free GPU memory when the object is destroyed
        self.state_gpu.free()

    def sync(self, is_synced=False, block=None):
        self.is_synced = is_synced
        self._update_state(is_synced)

        if block:
            self._update_block_header(block.header)

    def _update_state(self, new_state):
        return
        try:
            self.kernels["update_state"].function(
                self.state_gpu,
                np.array([new_state], dtype=np.bool_),
                block=(1, 1, 1),
                grid=(1, 1)
            )
        except drv.Error as e:
            print(f"CUDA error in _update_state: {e}")

    def _get_state(self):
        result = np.array([False], dtype=np.bool_)
        result_gpu = drv.mem_alloc(result.nbytes)
        
        try:
            self.kernels["get_state"].function(
                self.state_gpu,
                result_gpu,
                block=(1, 1, 1),
                grid=(1, 1)
            )
            
            drv.memcpy_dtoh(result, result_gpu)
        except drv.Error as e:
            print(f"CUDA error in _get_state: {e}")
        finally:
            result_gpu.free()
        
        return result[0]

    def _update_block_header(self, header):
        """
        # TODO: https://documen.tician.de/pycuda/tutorial.html
        grid = (1, 1)
        block = (4, 4, 1)
        func.prepare("P")
        func.prepared_call(grid, block, a_gpu)

        a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
        a_doubled = (2*a_gpu).get()
        print(a_doubled)
        print(a_gpu)
        """

        output = calculate_block_hash(self.kernels, header)
        print("output", output)

        # try:
        #     self.kernels["double_array"].function(
        #         struct_arr,
        #         block=(1, 1, 1),
        #         grid=(1, 1)
        #     )
        # except drv.Error as e:
        #     print(f"CUDA error in _update_block_header: {e}")

        # try:
        #     self.kernels["double_array"].function(
        #         np.intp(do2_ptr),
        #         block=(1, 1, 1),
        #         grid=(1, 1)
        #     )
        # except drv.Error as e:
        #     print(f"CUDA error in _update_block_header: {e}")

        blue_work = decode_blue_work(header.blue_work)
        blue_work = np.frombuffer(blue_work, dtype=np.uint8)
        blue_work_gpu = drv.mem_alloc(blue_work.nbytes)
        drv.memcpy_htod(blue_work_gpu, blue_work)

        header_data = np.array([
            (header.version,
            #  len(header.parents_by_level),
            #  0,  # We'll update this pointer later
            # header.hash_merkle_root.encode(),
            # header.accepted_id_merkle_root.encode(),
            # header.utxo_commitment.encode(),
            np.frombuffer(bytes.fromhex(header.hash_merkle_root), dtype=np.uint8),
            np.frombuffer(bytes.fromhex(header.accepted_id_merkle_root), dtype=np.uint8),
            np.frombuffer(bytes.fromhex(header.utxo_commitment), dtype=np.uint8),
            header.timestamp,
            header.bits,
            header.nonce,
            header.daa_score,
            header.blue_score,
            len(blue_work),
            int(blue_work_gpu),
            np.frombuffer(bytes.fromhex(header.pruning_point), dtype=np.uint8),
            )
        ], dtype=block_header_dtype)

        print("header_data", header_data)

        header_gpu = drv.mem_alloc(header_data.nbytes)
        drv.memcpy_htod(header_gpu, header_data)

        try:
            self.kernels["update_block_header"].function(
                # self.state_gpu,
                header_gpu,
                block=(1, 1, 1),
                grid=(1, 1)
            )
        except drv.Error as e:
            print(f"CUDA error in _update_block_header: {e}")
        finally:
            header_gpu.free()

    # def _update_block_header(self, header):
    #     block_header_dtype = np.dtype([
    #         ('version', np.uint16),
    #         # ('parents_by_level_count', np.uint64),
    #         # ('parents_by_level', np.uint64),  # Pointer, so 64-bit on most systems
    #         # ('hash_merkle_root', 'S32'),
    #         # ('accepted_id_merkle_root', 'S32'),
    #         # ('utxo_commitment', 'S32'),
    #         ('timestamp', np.uint64),
    #         ('bits', np.uint32),
    #         ('nonce', np.uint64),
    #         ('daa_score', np.uint64),
    #         # ('blue_score', np.uint64),
    #         # ('blue_work_length', np.uint64),
    #         # ('blue_work', np.uint64),  # Pointer, so 64-bit on most systems
    #         # ('pruning_point', 'S32')
    #     ])

    #     blue_work = decode_blue_work(header.blue_work)

    #     # Create a numpy array with our structure
    #     header_data = np.array([
    #         (header.version,
    #         #  len(header.parents_by_level),
    #         #  0,  # We'll update this pointer later
    #         #  header.hash_merkle_root.encode(),
    #         #  header.accepted_id_merkle_root.encode(),
    #         #  header.utxo_commitment.encode(),
    #         header.timestamp,
    #         header.bits,
    #         header.nonce,
    #         header.daa_score)
    #         #  header.blue_score,
    #         #  len(blue_work),
    #         #  0,  # We'll update this pointer later
    #         #  header.pruning_point.encode())
    #     ], dtype=block_header_dtype)

    #     # Allocate memory on the GPU for our structure
    #     header_gpu = drv.mem_alloc(header_data.nbytes)
    #     drv.memcpy_htod(header_gpu, header_data)

    #     # # Allocate and copy parents_by_level
    #     # parents_flat = np.concatenate([np.frombuffer(parent.encode(), dtype=np.uint8) for level in header.parents_by_level for parent in level])
    #     # parents_gpu = drv.mem_alloc(parents_flat.nbytes)
    #     # drv.memcpy_htod(parents_gpu, parents_flat)

    #     # # Update the parents_by_level pointer in our structure
    #     # header_data["parents_by_level"] = int(parents_gpu)
        
    #     # # Allocate and copy blue_work
    #     # blue_work = np.frombuffer(blue_work, dtype=np.uint8)
    #     # blue_work_gpu = drv.mem_alloc(blue_work.nbytes)
    #     # drv.memcpy_htod(blue_work_gpu, blue_work)

    #     # # Update the blue_work pointer in our structure
    #     # header_data["blue_work"] = int(blue_work_gpu)

    #     # # Copy the updated structure back to the GPU
    #     # drv.memcpy_htod(header_gpu, header_data)
    
    #     # Call the CUDA kernel
    #     self.kernels["update_block_header"].function(
    #         self.state_gpu,
    #         header_gpu,
    #         block=(1, 1, 1),
    #         grid=(1, 1)
    #     )

    #     # TODO: Remove
    #     # To ensure all printf output is flushed
    #     # drv.Context.synchronize()

    #     # TODO:
    #     # Free GPU memory
    #     header_gpu.free()
    #     # fixed_size_gpu.free()
    #     # hash_fields_gpu.free()
    #     # parents_gpu.free()
    #     # blue_work_gpu.free()


# TODO: Duplicated
def decode_blue_work(blue_work: str) -> bytes:
    # Ensure even number of characters by prepending '0' if necessary
    if len(blue_work) % 2 != 0:
        blue_work = '0' + blue_work
    
    # Calculate the length of the resulting byte array
    blue_work_len = (len(blue_work) + 1) // 2
    
    # Decode the hexadecimal string to bytes
    try:
        decoded = binascii.unhexlify(blue_work)
    except binascii.Error as e:
        raise ValueError(f"Invalid hexadecimal string: {e}")
    
    # Ensure the decoded bytes are the correct length
    if len(decoded) != blue_work_len:
        raise ValueError(f"Decoded length {len(decoded)} doesn't match expected length {blue_work_len}")
    
    return decoded.lstrip(b'\x00')