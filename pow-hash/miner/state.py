

import binascii
import struct
from typing import List

import blake3
import blake3.blake3
import numpy as np
from uint256 import Uint256
from matrix import Matrix

class State():
    def __init__(self) -> None:
        self.is_synced = False
        self.block = None
        self.pre_pow_hash = None
        self.target = None
        self.pow_hasher = None

    """
        On each new block template we will prepare the current state for it
        We can cache the pre pow hash, target, matrix and a copy of part of the pow hasher (last part is nonce on each block attempt)
    """
    def sync(self, block):
        self.block = block

        # Save the current state of the PoW
        self.pre_pow_hash = self.calc_block_hash(True)
        self.target = Uint256.from_compact_target_bits(block.header.bits)

        # Update the matrix with the new pre_pow_hash
        self.generate_matrix()

        # Save the current PoW Hash state, we will later finalize it with the nonce for each block hash we search
        self.pow_hasher = create_pow_hasher(self.pre_pow_hash, block.header.timestamp)


    def check_pow(self, nonce: int) -> bool:
        # Calculate the PoW Hash with the pre made hasher finalize  with nonce
        pow_hash = pow_hasher_finalize(self.pow_hasher.copy(), nonce)
        heavy_hash = calc_heavy_hash(self.mat, pow_hash)
        kheavy_hash = calc_kheavy_hash(heavy_hash)

        return Uint256(kheavy_hash) <= self.target

    def generate_matrix(self):
        self.mat = Matrix.generate(self.pre_pow_hash).data

    def calc_block_hash(self, pre_pow: bool = False) -> bytes:
        header = self.block.header
        return calc_block_header_hash(
            hash_merkle_root=header.hash_merkle_root,
            accepted_id_merkle_root=header.accepted_id_merkle_root,
            utxo_commitment=header.utxo_commitment,
            bits=header.bits,
            timestamp=0 if pre_pow else header.timestamp,
            nonce=0 if pre_pow else header.nonce,
            version=header.version,
            daa_score=header.daa_score,
            blue_score=header.blue_score,
            parents_by_level=header.parents_by_level,
            blue_work=header.blue_work,
            pruning_point=header.pruning_point
        )

def pow_hasher_finalize(hasher: blake3.blake3, nonce: int) -> bytes:
    hasher.update(struct.pack("<Q", nonce))
    return hasher.digest()

def create_pow_hasher(pre_pow_hash: bytes, timestamp: int) -> blake3.blake3:
    hasher = blake3.blake3()
    hasher.update(pre_pow_hash)
    hasher.update(struct.pack("<Q", timestamp))
    hasher.update(bytearray(32)) # 32 bytes
    return hasher

def calc_block_header_hash(
        hash_merkle_root: str, accepted_id_merkle_root: str, utxo_commitment: str, bits: int,
        timestamp: int, nonce: int, version: int, daa_score: int, blue_score: int, parents_by_level: List,
        blue_work: str, pruning_point: str
) -> bytes:
    hasher = blake3.blake3(key=b"BlockHash\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")
    hasher.update(struct.pack("<H", version)) # uint16
    hasher.update(struct.pack("<Q", len(parents_by_level))) # uint64

    for parent in parents_by_level:
        hasher.update(struct.pack("<Q", len(parent)))
        for hash_string in parent:
            hasher.update(bytes.fromhex(hash_string))

    hasher.update(bytes.fromhex(hash_merkle_root))
    hasher.update(bytes.fromhex(accepted_id_merkle_root))
    hasher.update(bytes.fromhex(utxo_commitment))
    hasher.update(struct.pack("<Q", timestamp))
    hasher.update(struct.pack("<I", bits)) # uint32
    hasher.update(struct.pack("<Q", nonce)) # uint64
    hasher.update(struct.pack("<Q", daa_score))
    hasher.update(struct.pack("<Q", blue_score))

    blue_work = decode_blue_work(blue_work)
    hasher.update(struct.pack("<Q", len(blue_work)))
    hasher.update(blue_work)

    hasher.update(bytes.fromhex(pruning_point))

    return hasher.digest()

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

def calc_heavy_hash(mat, hash: bytes):
    vector = np.array([0]*64, dtype=np.uint8)
    for i in range(32):
        vector[2 * i] = hash[i] >> 4
        vector[2 * i + 1] = hash[i] & 0x0F

    product = np.zeros(32, dtype=np.uint8)
    
    for i in range(32):
        # Compute sum1 and sum2 using dot product
        sum1 = np.dot(mat[2*i], vector)
        sum2 = np.dot(mat[2*i + 1], vector)
        
        # Combine the results with bit operations
        # product[i] = ((sum1 >> 10) << 4) | (sum2 >> 10)
        product[i] = (((sum1 & 0xF) ^ ((sum1 >> 4) & 0xF) ^ ((sum1 >> 8) & 0xF)) << 4) | ((sum2 & 0xF) ^ ((sum2 >> 4) & 0xF) ^ ((sum2 >> 8) &0xF))
    
    # # Product
    # product = np.array([0]*64, dtype=np.uint16)

    # # Perform matrix-vector multiplication
    # product = np.dot(mat, vector)

    #  # Right-shift by 10 bits and ensure uint16 type
    # product = (product >> 10).astype(np.uint16)

    # for i in range(64):
    #     sum = 0 # uint16
    #     for j in range(64):
    #         # sum += (mat[i][j] * vector[j]) & 0xffff
    #         sum += (mat[i][j] * vector[j])
    #     product[i] = sum >> 10

    # XOR with original hash
    product ^= np.frombuffer(hash, dtype=np.uint8)

    # Convert each byte to two 4-bit values
    # vec = np.array([(b >> 4, b & 0x0F) for b in hash_bytes], dtype=np.uint16).flatten()

    # Matrix-vector multiplication and conversion back to 8-bit values
    # product = np.zeros(32, dtype=np.uint16)
    # for i in range(32):
    #     sum1 = np.sum(mat[2*i] * vec)
    #     sum2 = np.sum(mat[2*i + 1] * vec)
    #     product[i] = ((sum1 >> 10) << 4) | (sum2 >> 10)

    # print(vec)
    # print(product)

    # # XOR with original hash
    # product ^= np.frombuffer(hash_bytes, dtype=np.uint8)

    return product.tobytes()

def calc_kheavy_hash(heavy_hash: bytes) -> bytes:
    hasher = blake3.blake3()
    hasher.update(heavy_hash)
    return hasher.digest()