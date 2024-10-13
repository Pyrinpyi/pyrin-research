import utils
import asyncio
import pytest
import json

# @pytest.mark.asyncio
# async def test_submit_block():
#     block = json.loads(open("tests/test.block.json").read())
#     print("block", block)

#     rpc = await utils.connect_node()
#     result = await rpc.submit_block(block, False)

#     print("result", result)