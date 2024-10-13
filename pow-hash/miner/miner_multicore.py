import asyncio
import random
import threading
import multiprocessing
import blake3
import struct
from multiprocessing import Manager, Pool, Value, cpu_count

from utils import format_hashrate, from_block_dict, exit_gracefully, init_sig_signals, connect_node, to_block_dict, hashes
from state import State, create_pow_hasher, pow_hasher_finalize, calc_heavy_hash, calc_kheavy_hash
from uint256 import Uint256
from matrix import Matrix

"""
    Multi-processing will require to share the state across threads,
    for this we will use the PickledState to only store pickled data

    We will override 2 functions:

    - sync()        so we can handle power hasher without the inner blake3 (which can't be pickled)
    - check_pow()   so we can use our pow_hasher
"""
class PickledState(State):
    def __init__(self):
        super().__init__()
        self.hashes = 0 # For hashes/s debug log
        self.nonce = 0 # For hashes/s debug log

    def sync(self, block):
        self.block = block

        # Most of the logic moved to start of mine_chunk() which holds the state for the process

    def check_pow(self, nonce: int) -> bool:
        pass

# Each process will have an rpc client
rpc = None
rpc_clients = {}
state = PickledState()
total_hashes = None

# Process-local variable to store RPC connection
process_rpc = None

def initialize_worker(shared_hash_counter):
    global total_hashes, process_rpc
    total_hashes = shared_hash_counter
    
    # Create a new event loop for the process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Initialize RPC connection for this process
    process_rpc = loop.run_until_complete(connect_node())
    print(f"Initialized RPC connection for process {multiprocessing.current_process().pid}")

async def multicore_process_template(state):
    global process_rpc

    address = "pyrin:qzn54t6vpasykvudztupcpwn2gelxf8y9p84szksr73me39mzf69uaalnymtx"
    # address = "pyrintest:qpsqzudxe835dpxk039x528kj4rxhadj2s7xxa33jv63cz82h7usxp6memfly"
    extra_data = b""
    block_template = await process_rpc.get_block_template(address, extra_data)
    block = block_template.block

    state.sync(from_block_dict(block))
    state.is_synced = block_template.is_synced

def on_new_block_template_thread(state):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(multicore_process_template(state))
    loop.close()

def on_new_block_template(state):
    t = threading.Thread(target=on_new_block_template_thread, args=(state,))
    t.start()

async def debug_print(state):
    global total_hashes

    while not exit_gracefully:
        if state.is_synced == None:
            print("Starting miner ..")
        elif not state.is_synced:
            print("Node is not synced")
        else:
            hash_rate = total_hashes.value
            print(f"{format_hashrate(hash_rate)} (nonce {state.nonce})")
        
        await asyncio.sleep(1)

def debug_print_thread(state):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(debug_print(state))
    loop.close()

def start_logger_thread(state):
    t = threading.Thread(target=debug_print_thread, args=(state,))
    t.start()

async def start_mining():
    global state, total_hashes, process_rpc

    process_rpc = await connect_node()
    process_rpc.on_new_block_template(lambda: on_new_block_template(state))
    on_new_block_template(state)

    start_logger_thread(state)

    # num_cores = cpu_count()
    num_cores = 5
    print(f"Using {num_cores} CPU cores for mining")

    total_hashes = Value("i", 0)  # 'i' stands for integer

    with Pool(num_cores, initializer=initialize_worker, initargs=(total_hashes,)) as pool:
        while not exit_gracefully:
            if not state.is_synced:
                await asyncio.sleep(0.1)
                continue

            nonces = [random.getrandbits(64) for _ in range(num_cores)]
            results = pool.map(mine_chunk, [(nonce, state, i) for i, nonce in enumerate(nonces)])


def mine_chunk(args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_mine_chunk(args))
    loop.close()

async def async_mine_chunk(args):
    global process_rpc
    nonce, state, i = args

    pre_pow_hash = state.calc_block_hash(True)
    target = Uint256.from_compact_target_bits(state.block.header.bits)

    # Update the matrix with the new pre_pow_hash
    mat = Matrix.generate(pre_pow_hash).data

    # Save the current PoW Hash state, we will later finalize it with the nonce for each block hash we search
    pow_hasher = create_pow_hasher(pre_pow_hash, state.block.header.timestamp)

    hashes_this_chunk = 0
    for _ in range(1000):  # Process 1000 hashes per chunk
        state.nonce = nonce

        pow_hash = pow_hasher_finalize(pow_hasher.copy(), nonce)
        heavy_hash = calc_heavy_hash(mat, pow_hash)
        kheavy_hash = calc_kheavy_hash(heavy_hash)

        check_pow = Uint256(kheavy_hash) <= target

        hashes_this_chunk += 1

        if check_pow:
            print(f"Found block with nonce {nonce}")
            block_dict = to_block_dict(state.block)
            block_dict["header"]["nonce"] = nonce

            error = await process_rpc.submit_block(block_dict, False)
            if not error:
                print("Block accepted")
            else:
                print(f"Failed to submit block with error {error}")

            with total_hashes.get_lock():
                total_hashes.value += hashes_this_chunk

            return True, nonce, hashes_this_chunk
        
        nonce += 1

    with total_hashes.get_lock():
        total_hashes.value += hashes_this_chunk

    return False, nonce, hashes_this_chunk

async def main():

    init_sig_signals()

    await start_mining()

if __name__ == "__main__":
    asyncio.run(main())