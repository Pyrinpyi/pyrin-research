#!/usr/bin/env python3

import signal
import sys
import os
import asyncio
import random
import threading

# pkill -9 -f miner.py; python3 miner.py
# TODO: Fix ctrl+c exit

# 20009b47
# 2005cb58

# Import the miner modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(root_dir, "miner"))

import utils
from state import GPUState

import pycuda.driver as drv
from uint256 import Uint256
import cuda

# Initialize CUDA driver
drv.init()

device = drv.Device(0)
context = device.make_context()

rpc = None
state = None
cached_blocks = {}

async def process_template(state: GPUState):
    global rpc, cached_blocks

    address = "pyrin:qzn54t6vpasykvudztupcpwn2gelxf8y9p84szksr73me39mzf69uaalnymtx"
    # address = "pyrintest:qpsqzudxe835dpxk039x528kj4rxhadj2s7xxa33jv63cz82h7usxp6memfly"
    extra_data = b""
    block_template = await rpc.get_block_template(address, extra_data)
    block = block_template.block

    block_hash = block.header.hash

    if block_hash not in cached_blocks:
        print("New block template", block_hash)

        # Push the context for this thread
        context.push()

        try:
            state.sync(block=block, is_synced=block_template.is_synced)
        finally:
            # Always pop the context, even if an exception occurred
            context.pop()

        # Up to 10k blocks in the cache
        if len(cached_blocks.keys()) > 10_000:
            cached_blocks = {}


        print("hey")
        cached_blocks[block_hash] = True

def on_new_block_template_thread(state):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(process_template(state))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        try:
            signal.set_wakeup_fd(-1) # TODO: Add to the rest if needed
        except ValueError:
            # The file descriptor is invalid on this thread
            pass
        loop.close()

    loop.close()

def on_new_block_template(state):
    t = threading.Thread(target=on_new_block_template_thread, args=(state,))
    t.start()

async def start_mining(rpc):
    global state

    print("Staring Pyrin GPU Miner ..")
    cuda_cores, max_threads_per_block, max_shared_memory_per_block, = cuda.get_cuda_info(log_info=True)

    print("Initializing CUDA kernels ..")
    vector_size = 1000000
    vector_size = 0
    kernel = cuda.GPUKernel(vector_size)
    state = GPUState(kernel)

    # Start listening for new block templates
    rpc.on_new_block_template(lambda: on_new_block_template(state))

    # Ask for the current template to start the mining
    on_new_block_template(state)

    # utils.start_logger_thread(state)
    is_synced_i = 0

    nonce = random.getrandbits(64)
    while not utils.exit_gracefully:
        if not state.is_synced:
            await asyncio.sleep(0.1)

            is_synced_i += 1

            if is_synced_i >= 50:
                print("Node is not synced")
                is_synced_i = 0

            continue

        # For debug log
        # state.nonce = nonce

        await asyncio.sleep(1)
        continue

        # Calculate the PoW Hash with the pre made hasher finalize  with nonce
        pow_hash = pow_hasher_finalize(state.pow_hasher.copy(), nonce)

        # heavy_hash = calc_heavy_hash(state.mat, pow_hash)
        heavy_hash = kernel.calc_heavy_hash(state.mat, pow_hash)

        # kheavy_hash = heavy_hash -> blake3
        # kheavy_hash = calc_kheavy_hash(heavy_hash)
        kheavy_hash = kernel.blake3_hash(heavy_hash)

        kheavy_hash = Uint256(kheavy_hash).value

        if kernel.check_pow(kheavy_hash, state.target.value):
            block = state.block
            bh = block.header.hash

            print(f"Found block {bh} with nonce {nonce}")

            block_dict = utils.to_block_dict(block)
            block_dict["header"]["nonce"] = nonce

            # TODO: Check holding the gRPC connection and sending a few with diff nonce
            error = await rpc.submit_block(block_dict, False, "192.168.1.177:16210")
            if not error:
                print(f"Accepted block {bh}")
            else:
                print(f"Failed to submit block with error {error}")

        utils.hashes += 1
        nonce += 1  # TODO: Rotate around a range in parallel in GPU 

async def main():
    def on_exit_gracefully(signum, frame):
        print("on_exit_gracefully", signum, frame)
        try:
            context.pop()
        finally:
            pass 

    utils.init_sig_signals(on_exit_gracefully)

    global rpc
    rpc = await utils.connect_node()

    await start_mining(rpc)

if __name__ == "__main__":
    asyncio.run(main())