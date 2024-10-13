import asyncio
import pytest
import json
import struct

import blake3
import utils

from state import *
from matrix import *

"""

Complete flow of mining pow check from block header to check_pow value

"""


def test_mining():
    block = json.loads(open("tests/test.block.mining.json").read())
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
    
    assert block_hash.hex() == "92f4ae326ac67a1f121a8057a6ba94f9e1657271a09839982baca83a0d0fd2a1"

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

    assert pre_pow_hash.hex() == "4135d0e8568ddd2a416946f5cefd6e8d7c20d7f27d15ea289786b9176ba39735"

    hasher = blake3.blake3()
    hasher.update(pre_pow_hash)
    hasher.update(struct.pack("<Q", header["timestamp"]))
    hasher.update(bytearray(32)) # 32 bytes

    mat = Matrix.generate(pre_pow_hash).data

    # repeat: finalize_with_nonce (TODO: Save previous hasher state for re-use)
    hasher.update(struct.pack("<Q", header["nonce"]))
    pow_hash = hasher.digest()

    assert pow_hash.hex() == "b7ad9c0d6a740cbd398cec4589d0891ae7f7861cba3fd1c549e474d01368b891"

    heavy_hash = calc_heavy_hash(mat, pow_hash)

    assert heavy_hash.hex() == "f24009425cc7c74438fdc2f41677e5b532b49f6442aa88619e1ca4b1d91c82ea"

    kheavy_hash = calc_kheavy_hash(heavy_hash)

    assert kheavy_hash.hex() == "fc7098f90a7d78ca7ac14de459803194047d0a2c67420bca4aa8f6d4bb84a2af"

    target = Uint256.from_compact_target_bits(block["header"]["bits"])

    a = Uint256(kheavy_hash)
    b = target

    check_pow = a <= b

    assert check_pow

# @pytest.mark.asyncio
# async def test_submit_block():
#     block = json.loads(open("tests/test.block.mining.json").read())
#     print("block", block)

#     rpc = await utils.connect_node()
#     result = await rpc.submit_block(block, False)

#     print("result", result)