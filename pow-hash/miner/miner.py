import asyncio
import random
import argparse
import time

import utils
from state import State

rpc = None
state = State()

# Config
limit_hashrate = None

async def start_mining(rpc):
    global state, limit_hashrate

    rpc.on_new_block_template(lambda: utils.on_new_block_template(state))

    utils.start_tasks_queue()

    # Ask for the next template to start the mining
    utils.on_new_block_template(state)

    utils.start_logger_thread(state)

    nonce = random.getrandbits(64)
    start_time = time.time()
    hashes_this_second = 0

    while not utils.exit_gracefully:
        if not state.is_synced:
            await asyncio.sleep(0.1)
            continue

        # For debug log
        state.nonce = nonce

        if state.check_pow(nonce):
            block = state.block
            bh = block.header.hash

            print(f"Found block {bh} with nonce {nonce}")
            print("Target:\t", state.target)

            block_dict = utils.to_block_dict(block)
            block_dict["header"]["nonce"] = nonce

            # result = await rpc.submit_block(block_dict, False, "192.168.1.177:16210")
            result = await rpc.submit_block(block_dict, False, "127.0.0.1:17001")
            print("result", result)

        utils.hashes += 1
        hashes_this_second += 1
        nonce += 1  # TODO: Rotate around a range in parallel in GPU 

        # Check if we need to limit the hashrate
        if limit_hashrate:
            utils.hashes = hashes_this_second
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 1:
                if hashes_this_second >= limit_hashrate:
                    sleep_time = 1 - elapsed_time
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

                start_time = time.time()
                hashes_this_second = 0

async def main():
    global limit_hashrate

    parser = argparse.ArgumentParser(description='Concise Miner')
    parser.add_argument("--limit-hashrate", "-lh", type=int, help='Limit hashrate (hashes per second)')
    args = parser.parse_args()

    if args.limit_hashrate:
        limit_hashrate = args.limit_hashrate
        print(f"Hashrate limited to {limit_hashrate} hashes per second")
    else:
        print("No hashrate limit set")

    utils.init_sig_signals()

    global rpc
    rpc = await utils.connect_node()

    await start_mining(rpc)

if __name__ == "__main__":
    asyncio.run(main())