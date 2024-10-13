import os
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

from state import GPUState
import cuda

def test_state():
    vector_size = 0
    kernel = cuda.GPUKernel(vector_size)
    gpu_state = GPUState(kernel)

    assert gpu_state._get_state() == False

    gpu_state._update_state(True)

    assert gpu_state._get_state() == True

    gpu_state.sync(is_synced=False)

    assert gpu_state._get_state() == False
