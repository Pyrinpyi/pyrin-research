import asyncio
import queue
import signal
import threading
from typing import Dict
import pyrin

hashes = 0
exit_gracefully = False
testnet = False


async def connect_node():
    global rpc
    # TODO: Handle reconnections (a TODO from the sdk)
    
    rpc = pyrin.RPC()
    # success = await rpc.connect("localhost:17110")
    success = await rpc.connect("192.168.1.177:17110")
    # success = await rpc.connect("localhost:7002", testnet)
    # success = await rpc.connect("192.168.1.177:17000", True) # Testnet
    # success = await rpc.connect("192.168.1.177:17210", True) # Testnet
    # success = await rpc.connect("localhost:14211", True) # Testnet
    # success = await rpc.connect("localhost:18211", True) # Testnet

    if not success:
        print("Failed to connect to node")
    else:
        print("Connected to node")

    return rpc

cached_blocks = {}

async def process_template(state):
    global rpc, cached_blocks

    if not testnet:
        address = "pyrin:qzn54t6vpasykvudztupcpwn2gelxf8y9p84szksr73me39mzf69uaalnymtx"
    else:
        address = "pyrintest:qpsqzudxe835dpxk039x528kj4rxhadj2s7xxa33jv63cz82h7usxp6memfly"

    extra_data = b""
    block_template = await rpc.get_block_template(address, extra_data)
    block = block_template.block

    block_hash = block.header.hash

    if block_hash not in cached_blocks:
        print("New block template", block_hash)

        state.sync(block)
        state.is_synced = block_template.is_synced

        # Up to 10k blocks in the cache
        if len(cached_blocks.keys()) > 10_000:
            cached_blocks = {}

        cached_blocks[block_hash] = True

template_queue = queue.Queue()
STOP_SENTINEL = object()

def start_tasks_queue():
    asyncio.create_task(process_templates())

def on_new_block_template(state):
    template_queue.put(state)

async def process_templates():
    while True:
        state = await asyncio.get_event_loop().run_in_executor(None, template_queue.get)
        if state is STOP_SENTINEL:
            break
        await process_template(state)

def on_new_block_template_thread(state):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(process_template(state))
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        if threading.current_thread() is threading.main_thread():
            loop.close()
        else:
            loop.call_soon_threadsafe(loop.stop)
            loop.run_forever()
            loop.close()

def init_sig_signals(callback=None):
    async def on_exit_gracefully(signum, frame):
        print("Received sig %d, Shutting done miner .." % signum)
        global exit_gracefully
        exit_gracefully = True
        template_queue.put(STOP_SENTINEL)
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        # loop.stop()

        if callback:
            callback(signum, frame)
        
        exit(0)

    signal.signal(signal.SIGINT, on_exit_gracefully)
    signal.signal(signal.SIGTERM, on_exit_gracefully)

def format_hashrate(hash_rate: int) -> str:
    unit = ""
    if hash_rate >= 1_000_000_000:
        hash_rate /= 1_000_000_000
        unit = "G"
    elif hash_rate >= 1_000_000:
        hash_rate /= 1_000_000
        unit = "M"
    elif hash_rate >= 1_000:
        hash_rate /= 1_000
        unit = "K"

    return f"{hash_rate:.2f} {unit}H/s"

async def debug_print(state):
    global hashes
    intervals = 5

    while not exit_gracefully:
        if state.is_synced == None:
            print("Starting miner ..")
        elif not state.is_synced:
            print("Node is not synced")
        else:
            hash_rate = hashes
            hash_rate /= intervals
            bits = f"{state.block.header.bits:x}"
            synced = state.is_synced and "synced" or "not synced"
            print(f"{format_hashrate(hash_rate)} ([{synced}] nonce {state.nonce}, bits {bits})")
        
        hashes = 0
        await asyncio.sleep(intervals)

def debug_print_thread(state):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(debug_print(state))
    loop.close()

def start_logger_thread(state):
    t = threading.Thread(target=debug_print_thread, args=(state,))
    t.start()

def debug_info(state):
    print("Nonce:\t\t", state.nonce)
    print("Target:\t\t", state.target)
    print("Target:\t\t", pow.from_big(state.target))
    print("Pre PoW Hash:\t", state.block_hash.hex())
    print("Matrix:\t", state.mat)


# d = stringify_block_header(block.header)
# with open(os.path.join("blocks", f"{block.header.hash}.json"), "w") as f:
#     f.write(d)

class BlockHeader:
    def __init__(self):
        self.hash = None
        self.version = None
        self.parents_by_level = None
        self.hash_merkle_root = None
        self.accepted_id_merkle_root = None
        self.utxo_commitment = None
        self.timestamp = None
        self.bits = None
        self.nonce = None
        self.daa_score = None
        self.blue_work = None
        self.blue_score = None
        self.pruning_point = None

class Output:
    def __init__(self):
        self.script_public_key = None
        self.value = None

class Input:
    def __init__(self):
        self.previous_outpoint = None
        self.signature_script = None
        self.sequence = None
        self.sig_op_count = None

class Transaction:
    def __init__(self):
        self.gas = None
        self.inputs = []
        self.lock_time = None
        self.mass = None
        self.outputs = []
        self.payload = None
        self.subnetwork_id = None
        self.verbose_data = None
        self.version = None

class Block:
    def __init__(self):
        self.header = BlockHeader()
        self.transactions = []

def from_block_dict(block) -> Block:
    new_block = Block()
    
    if block.header:
        new_block.header = from_block_header_dict(block.header)
    
    if block.transactions:
        for tx_dict in block.transactions:
            tx = Transaction()
            
            for attr in ['gas', 'lock_time', 'mass', 'payload', 'subnetwork_id', 'verbose_data', 'version']:
                if hasattr(tx_dict, attr):
                    setattr(tx, attr, getattr(tx_dict, attr))
            
            # Process inputs
            if tx_dict.inputs:
                for input_dict in tx_dict.inputs:
                    inp = Input()
                    for attr in ['signature_script', 'sequence', 'sig_op_count']:
                        if hasattr(input_dict, attr):
                            setattr(inp, attr, getattr(input_dict, attr))
                    if hasattr(input_dict, 'previous_outpoint'):
                        inp.previous_outpoint = input_dict['previous_outpoint']
                    tx.inputs.append(inp)
            
            # Process outputs
            if tx_dict.outputs:
                for output_dict in tx_dict.outputs:
                    out = Output()
                    for attr in ['script_public_key', 'value']:
                        if hasattr(output_dict, attr):
                            setattr(out, attr, getattr(output_dict, attr))
                    tx.outputs.append(out)
            
            new_block.transactions.append(tx)
    
    return new_block

def to_block_dict(block: Block) -> Dict:
    d = {}
    d["header"] = to_block_header_dict(block.header)
    
    transactions = []

    for t in block.transactions:
        td = {
            "gas": t.gas,
            "lock_time": t.lock_time,
            "mass": t.mass,
            "payload": t.payload,
            "subnetwork_id": t.subnetwork_id,
            "verbose_data": t.verbose_data,
            "version": t.version,
            "outputs": [{
                "script_public_key": output.script_public_key,
                "value": output.value
            } for output in t.outputs],
            "inputs": [{
                "transaction_id": input.previous_outpoint.transaction_id if input.previous_outpoint else None,
                "index": input.previous_outpoint.index if input.previous_outpoint else None,
                "signature_script": input.signature_script,
                "sequence": input.sequence,
                "sig_op_count": input.sig_op_count
            } for input in t.inputs]
        }
        transactions.append(td)

    d["transactions"] = transactions

    return d

def from_block_header_dict(header) -> BlockHeader:
    block_header = BlockHeader()
    
    attributes = [
        "hash", "version", "parents_by_level", "hash_merkle_root", 
        "accepted_id_merkle_root", "utxo_commitment", "timestamp", 
        "bits", "nonce", "daa_score", "blue_work", "blue_score", 
        "pruning_point"
    ]
    
    for attr in attributes:
        if hasattr(header, attr):
            setattr(block_header, attr, getattr(header, attr))
    
    return block_header

def to_block_header_dict(header: BlockHeader) -> Dict:
    return {
        "hash": header.hash,
        "version": header.version,
        "parents_by_level": header.parents_by_level,
        "hash_merkle_root": header.hash_merkle_root,
        "accepted_id_merkle_root": header.accepted_id_merkle_root,
        "utxo_commitment": header.utxo_commitment,
        "timestamp": header.timestamp,
        "bits": header.bits,
        "nonce": header.nonce,
        "daa_score": header.daa_score,
        "blue_work": header.blue_work,
        "blue_score": header.blue_score,
        "pruning_point": header.pruning_point
    }

# def stringify_block(block):
#     return json.dumps(block_dict(block))

# def stringify_block_header(header):
#     return json.dumps(block_header_dict(header))