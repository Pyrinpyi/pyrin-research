import json
import struct

import blake3
import numpy as np
from uint256 import Uint256
from state import *

def test_hashers():
    hasher = blake3.blake3(key=b"BlockHash\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")

    # Empty blake3 with just the key
    assert hasher.hexdigest() == "b5276fb2a7776cbcf5e57e329a840ac2951bc711231ccab04deae865d9519208"

def test_calc_block_header_hash():
    block = json.loads(open("tests/test.block.json").read())
    header = block["header"]

    hash = calc_block_header_hash(
        hash_merkle_root=header["hash_merkle_root"],
        accepted_id_merkle_root=header["accepted_id_merkle_root"],
        utxo_commitment=header["utxo_commitment"],
        bits=header["bits"],
        timestamp=header["timestamp"],
        nonce=header["nonce"],
        version=header["version"],
        daa_score=header["daa_score"],
        blue_score=header["blue_score"],
        parents_by_level=header["parents_by_level"],
        blue_work=header["blue_work"],
        pruning_point=header["pruning_point"]
    )

    assert hash.hex() == "52e80693ea920cbadcb5182553e9b14b48b88f1343b2d30450ac7e22d9c19a65"
    assert hash.hex() == header["hash"]

# def test_calc_block_header_hash_inside_state():
#     block = json.loads(open("tests/test.block.json").read())

#     state = )
#     state.sync(block)
#     state.debug()

#     pre_pow_hash = state.calc_block_hash(zeros=True)

#     assert pre_pow_hash.hex() == "8d930982795d6f6c71e0c65226b2f58512f14827eaa25d323e4fc3f3c6ea3cdb"

#     target = [0, 0, 0, 140737471578112]

#     # State of Pre Pow Hash
#     pre_pow_hash = state.calc_block_hash(True)
#     state.target = pyrin.uint256.compact_to_big(state.bits)

#     # pow_hash = state.calc_pow_hash(nonce=nonce, zeros=False)

#     # POW HASH = real nonce and tiemstamp, (pre is zero for both? )
#     hasher = blake3.blake3()
#     hasher.update(pre_pow_hash)
#     hasher.update(struct.pack("<Q", state.timestamp))
#     hasher.update(struct.pack("<Q", 0).rjust(32, b'\x00'))
#     hasher.update(struct.pack("<Q", state.nonce))
#     pow_hash = hasher.digest()

#     heavy_hash = state.heavy_hash(pow_hash)

#     a = int.from_bytes(heavy_hash, byteorder="little", signed=False)
#     b = state.target

def test_todo2():
    block = json.loads(open("tests/test.block.json").read())
    header = block["header"]

    pre_pow_hash = calc_block_header_hash(
        hash_merkle_root=header["hash_merkle_root"],
        accepted_id_merkle_root=header["accepted_id_merkle_root"],
        utxo_commitment=header["utxo_commitment"],
        bits=header["bits"],
        timestamp=0,
        nonce=0,
        version=header["version"],
        daa_score=header["daa_score"],
        blue_score=header["blue_score"],
        parents_by_level=header["parents_by_level"],
        blue_work=header["blue_work"],
        pruning_point=header["pruning_point"]
    )

    assert pre_pow_hash.hex() == "8d930982795d6f6c71e0c65226b2f58512f14827eaa25d323e4fc3f3c6ea3cdb"

def test_pow_hash():
    pre_pow_hash = "1aad5f2d82611fbc12f2ead2274191ad3509f4003b3d53ac1b0669366b103026"
    timestamp = 1724363826172
    nonce = 13789334910599599310
    
    hasher = blake3.blake3()
    hasher.update(bytes.fromhex(pre_pow_hash))
    hasher.update(struct.pack("<Q", timestamp))
    hasher.update(bytearray(32)) # 32 bytes
    hasher.update(struct.pack("<Q", nonce))
    pow_hash = hasher.digest()

    assert pow_hash.hex() == "25d98097b64293712b2a4dc444c934077ccb2b7799a318ffd7a1e217094c5169"

def test_check_pow():
    nonce = 4003789453554927071
    pow_hash = bytes.fromhex("810a10bbe4e20439f3128434422a5ec3260f957580b5d791ce0521bbc3f3cdb2")
    kheavy_hash = bytes.fromhex("e60f16e65f9c5357c5f5bf1e7e617a1305d7db06d4dae3ccd708fc9b6a4ec003")

    pow_uint256 = Uint256(kheavy_hash)
    assert pow_uint256 == Uint256([6292545040062746598, 1403541428189001157, 14763884607633610501, 270302197432715479])
    assert pow_uint256.hex() == "03c04e6a9bfc08d7cce3dad406dbd705137a617e1ebff5c557539c5fe6160fe6"

"""
    Check first and last 3 of first/third/1 from last/last
    [[ 2 13  4 ... 14  2  4]
    [ X X   X ... X   X  X]
    [ 7 11  8 ...  6  9 15]
    ...
    [ X  X  X ... XX  X  X]
    [15  2  1 ... 14  3 13]
    [15 13  3 ...  7  5  5]]
"""
def assert_matrix_values(mat, expected_values):
    for row_index, row_data in expected_values.items():
        for col_index, expected_value in row_data.items():
            assert mat[row_index][col_index] == expected_value, \
                f"Mismatch at mat[{row_index}][{col_index}]: " \
                f"Expected {expected_value}, got {mat[row_index][col_index]}"

def test_generate_matrix():
    pre_pow_hash = bytes.fromhex("1aad5f2d82611fbc12f2ead2274191ad3509f4003b3d53ac1b0669366b103026")
    mat = Matrix.generate(pre_pow_hash).data

    expected = {
        0:  {0: 2,  1: 13, 2: 4,  -3: 14, -2: 2,  -1: 4},
        2:  {0: 7,  1: 11, 2: 8,  -3: 6,  -2: 9,  -1: 15},
        -2: {0: 15, 1: 2,  2: 1,  -3: 14, -2: 3,  -1: 13},
        -1: {0: 15, 1: 13, 2: 3,  -3: 7,  -2: 5,  -1: 5}
    }

    assert_matrix_values(mat, expected)

def test_heavyhash():
    mat_pre_pow_hash_str = "a33db6a616a3fb1e13097b515664eecd7a00e5c5d4bd67b705bc83928bb60c7a"
    mat_pre_pow_hash = bytes.fromhex(mat_pre_pow_hash_str)
    mat = load_matrix(mat_pre_pow_hash_str)

    expected = mat 
    generated_mat = Matrix.generate(mat_pre_pow_hash).data

    np.testing.assert_array_equal(generated_mat, expected)


    # pow_hash = bytes.fromhex("810a10bbe4e20439f3128434422a5ec3260f957580b5d791ce0521bbc3f3cdb2")
    heavy_hash = calc_heavy_hash(mat, mat_pre_pow_hash)
    assert heavy_hash.hex() == "900e85952580b82d203a48636557ddfe5933d6f6e78f5594368fb0a1b8853f59"

    # assert heavy_hash.hex() == "233559b22c7174bdd68786721ec3de5b6b8cd67fdeda4b88856ece7ad21fd10c"
    # assert kheavy_hash(product).hex() == "233559b22c7174bdd68786721ec3de5b6b8cd67fdeda4b88856ece7ad21fd10c"

def test_kheavyhash():
    product = bytes([178, 56, 35, 152, 199, 193, 54, 10, 192, 32, 183, 6, 96, 8, 125, 224, 5, 45, 166, 86, 178, 151, 245, 178, 252, 38, 2, 137, 240, 193, 239, 145])
    heavy_hash = product.hex()
    assert heavy_hash == "b2382398c7c1360ac020b70660087de0052da656b297f5b2fc260289f0c1ef91"
    assert calc_kheavy_hash(product).hex() == "e60f16e65f9c5357c5f5bf1e7e617a1305d7db06d4dae3ccd708fc9b6a4ec003"

def test_heavyhash_breakdown():
    hash = bytes.fromhex("1f9c3313aed61df15e978d845514ae8f3992d7aeb56c6e2e79df6b23028cc981")
    mat = load_matrix("1f9c3313aed61df15e978d845514ae8f3992d7aeb56c6e2e79df6b23028cc981")

    # Vector
    vector = np.array([0]*64, dtype=np.uint8)
    for i in range(32):
        vector[2 * i] = hash[i] >> 4
        vector[2 * i + 1] = hash[i] & 0x0F

    expected = np.array(
    [
        0x01, 0x0f, 0x09, 0x0c,   0x03, 0x03, 0x01, 0x03,   0x0a, 0x0e, 0x0d, 0x06,   0x01, 0x0d, 0x0f, 0x01,
        0x05, 0x0e, 0x09, 0x07,   0x08, 0x0d, 0x08, 0x04,   0x05, 0x05, 0x01, 0x04,   0x0a, 0x0e, 0x08, 0x0f,
        0x03, 0x09, 0x09, 0x02,   0x0d, 0x07, 0x0a, 0x0e,   0x0b, 0x05, 0x06, 0x0c,   0x06, 0x0e, 0x02, 0x0e,
        0x07, 0x09, 0x0d, 0x0f,   0x06, 0x0b, 0x02, 0x03,   0x00, 0x02, 0x08, 0x0c,   0x0c, 0x09, 0x08, 0x01,
    ], dtype=np.uint8)

    np.testing.assert_array_equal(vector, expected, 
                              err_msg="The vector does not match the expected values")

    product = np.zeros(32, dtype=np.uint8)
    
    for i in range(32):
        # Compute sum1 and sum2 using dot product
        sum1 = np.dot(mat[2*i], vector)
        sum2 = np.dot(mat[2*i + 1], vector)
        
        # Combine the results with bit operations
        product[i] = (((sum1 & 0xF) ^ ((sum1 >> 4) & 0xF) ^ ((sum1 >> 8) & 0xF)) << 4) | ((sum2 & 0xF) ^ ((sum2 >> 4) & 0xF) ^ ((sum2 >> 8) & 0xF))
    
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

    expected = np.array([
        51, 51, 51, 52, 51, 51, 51, 67, 51, 51, 51, 51, 35, 51, 52, 51, 
        67, 67, 67, 51, 51, 67, 52, 51, 67, 67, 52, 52, 68, 51, 67, 67
    ], dtype=np.uint16)

    np.testing.assert_array_equal(product, expected, 
                            err_msg="The product does not match the expected values")

    # XOR with original hash
    product ^= np.frombuffer(hash, dtype=np.uint8)

    expected = np.array([
        0x2c, 0xaf, 0x00, 0x27, 0x9d, 0xe5, 0x2e, 0xb2, 0x6d, 0xa4, 0xbe, 0xb7, 0x76, 0x27, 0x9a, 0xbc,
        0x7a, 0xd1, 0x94, 0x9d, 0x86, 0x2f, 0x5a, 0x1d, 0x3a, 0x9c, 0x5f, 0x17, 0x46, 0xbf, 0x8a, 0xc2
    ] , dtype=np.uint8)

    np.testing.assert_array_equal(product, expected, 
                            err_msg="The res does not match the expected values")

def test_todo3():
    return
    block = json.loads(open("tests/test.block.pow.json").read())
    header = block["header"]

    block_hash = calc_block_header_hash(
        hash_merkle_root=header["hash_merkle_root"],
        accepted_id_merkle_root=header["accepted_id_merkle_root"],
        utxo_commitment=header["utxo_commitment"],
        bits=header["bits"],
        timestamp=header["timestamp"],
        nonce=header["nonce"],
        version=header["version"],
        daa_score=header["daa_score"],
        blue_score=header["blue_score"],
        parents_by_level=header["parents_by_level"],
        blue_work=header["blue_work"],
        pruning_point=header["pruning_point"]
    )

    pre_pow_hash = calc_block_header_hash(
        hash_merkle_root=header["hash_merkle_root"],
        accepted_id_merkle_root=header["accepted_id_merkle_root"],
        utxo_commitment=header["utxo_commitment"],
        bits=header["bits"],
        timestamp=0,
        nonce=0,
        version=header["version"],
        daa_score=header["daa_score"],
        blue_score=header["blue_score"],
        parents_by_level=header["parents_by_level"],
        blue_work=header["blue_work"],
        pruning_point=header["pruning_point"]
    )

    nonce = 13789334910599599310

    assert block_hash.hex() == "f47403131919a7018500c0f562df0276cd5f49d6d6e8c1d3a515fc323fa03acc"
    assert pre_pow_hash.hex() == "1aad5f2d82611fbc12f2ead2274191ad3509f4003b3d53ac1b0669366b103026"

    hasher = blake3.blake3()
    hasher.update(pre_pow_hash)
    hasher.update(struct.pack("<Q", header["timestamp"]))
    hasher.update(bytearray(32)) # 32 bytes
    hasher.update(struct.pack("<Q", nonce))
    pow_hash = hasher.digest()

    assert pow_hash.hex() == "25d98097b64293712b2a4dc444c934077ccb2b7799a318ffd7a1e217094c5169"

    mat = Matrix.generate(pre_pow_hash).data
    assert mat[0][0] == 2
    assert mat[0][1] == 13
    assert mat[0][2] == 4
    assert mat[0][-3] == 14
    assert mat[0][-2] == 2
    assert mat[0][-1] == 4

    assert mat[2][0] == 7
    assert mat[2][1] == 11
    assert mat[2][2] == 8
    assert mat[2][-3] == 6
    assert mat[2][-2] == 9
    assert mat[2][-1] == 15

    assert mat[-2][0] == 15
    assert mat[-2][1] == 2
    assert mat[-2][2] == 1
    assert mat[-2][-3] == 14
    assert mat[-2][-2] == 3
    assert mat[-2][-1] == 13

    assert mat[-1][0] == 15
    assert mat[-1][1] == 13
    assert mat[-1][2] == 3
    assert mat[-1][-3] == 7
    assert mat[-1][-2] == 5
    assert mat[-1][-1] == 5

    heavy_hash = calc_heavy_hash(mat, pow_hash)

    assert heavy_hash.hex() == "a982c1731d8f098ea243b2fb6cdb36f04179371c84e9b1475a0909f091366479"

    """
        HASH =  1aad5f2d82611fbc12f2ead2274191ad3509f4003b3d53ac1b0669366b103026
        powHash:  dc5cf45e2677854dcc678afefdb7ed7d789c6801df3e07b070c9350f635e651c
        heavyHash:  eae0752d309aa529ff0c391ccab435d3f0fd7162b554dc7e97e459b07c2856fc
    """

def load_matrix(hash: str):
    matrix_dump_file_path = f"tests/data/{hash}.bin"

    with open(matrix_dump_file_path, "rb") as f:
        raw_data = f.read()

    mat = np.frombuffer(raw_data, dtype=np.uint16)
    return mat.reshape((64, 64))