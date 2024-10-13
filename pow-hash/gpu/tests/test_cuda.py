import os
import sys

import numpy as np
import pycuda.driver as drv

import cuda
import cuda_uint256

# Import the miner modules
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.join(root_dir, "miner"))

from uint256 import Uint256

def test_uint256():
    
    a_values = [6292545040062746598, 1403541428189001157, 14763884607633610501, 270302197432715479]
    a = Uint256(a_values)
    
    assert a.hex() == "03c04e6a9bfc08d7cce3dad406dbd705137a617e1ebff5c557539c5fe6160fe6"
    assert all(cuda.uint256_to_uint256_t(a.value) == a_values)

    # TODO: This big numbers doesn't work (we probably don't need to tho)    
    # a_values = [2**255 - 1, 2**256 - 1, 1000, 1403541428189001157]
    # print(a)
    # a = Uint256(a_values)
    
    # assert a.hex() == "10000000000000000937a617e1ebff5c500000000000003e7fffffffffffffffeffffffffffffffff"
    # assert all(cuda.uint256_to_uint256_t(a.value) == a_values)

def test_uint256_compare():
    kernels  = cuda._init_kernels(0)

    # Test values
    value1 = 0x1234567890ABCDEF  # Example 64-bit value
    value2 = 0x1234567890ABCDEF0  # Slightly larger value

    # Convert to arrays
    a = cuda_uint256.uint256_to_array(value1)
    b = cuda_uint256.uint256_to_array(value2)

    assert all(a == [1311768467294899695, 0, 0, 0])
    assert Uint256([1311768467294899695, 0, 0, 0]).value == value1

    assert all(b == [2541551403008843504, 1, 0, 0])
    assert Uint256([2541551403008843504, 1, 0, 0]).value == value2
    
    a, b, result = cuda_uint256.cuda_compare_uint256(kernels, a, b)

    print(f"Uint256 #1: {a:#x}")
    print(f"Uint256 #2: {b:#x}")

    assert result[0] # Uint256 #1 <= Uint256 #2

def test_check_pow():
    kernels  = cuda._init_kernels(0)

    # Invalid PoW (kheavy_hash > target)
    kheavy_hash = 56429235189643032184805928543776564771839988472657994414619247312938887788814
    target = 367414466771395663224089228328317449961234786133490528264076977053955522560

    kheavy_hash = cuda_uint256.uint256_to_array(kheavy_hash)
    target = cuda_uint256.uint256_to_array(target)

    kheavy_hash, target, result = cuda_uint256.cuda_compare_uint256(kernels, kheavy_hash, target)

    print(f"kHeavyHash: {kheavy_hash:#x}")
    print(f"Target: {target:#x}")
    
    assert result[0] == False

    # # Valid PoW (kheavy_hash <= target)
    kheavy_hash = 322776840778782003387486516442409121315005312254145377109239425883631426304
    target = 367414466771395663224089228328317449961234786133490528264076977053955522560

    kheavy_hash = cuda_uint256.uint256_to_array(kheavy_hash)
    target = cuda_uint256.uint256_to_array(target)

    kheavy_hash, target, result = cuda_uint256.cuda_compare_uint256(kernels, kheavy_hash, target)

    print(f"kHeavyHash: {kheavy_hash:#x}")
    print(f"Target: {target:#x}")
    
    assert result[0] == True
