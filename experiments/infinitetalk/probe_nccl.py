"""Run under an external timeout: NCCL initialization can ignore group timeouts."""
import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
start = time.perf_counter()
print(json.dumps({
    'rank': rank, 'torch': torch.__version__, 'cuda': torch.version.cuda,
    'nccl': torch.cuda.nccl.version(), 'device': torch.cuda.get_device_name(rank),
    'loaded_nccl_libraries': sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libnccl' in line}),
}), flush=True)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=30))
world = dist.get_world_size()
expected = world * (world + 1) / 2
x = torch.empty(1024 * 1024, device=f'cuda:{rank}', dtype=torch.float32)
for _ in range(3):
    x.fill_(rank + 1)
    dist.all_reduce(x)
    torch.cuda.synchronize()
    assert torch.all(x == expected).item(), 'Incorrect all-reduce result'
if rank == 0:
    print(json.dumps({'ranks': world, 'seconds': time.perf_counter()-start, 'status': 'passed', 'correctness_checked': True}), flush=True)
dist.destroy_process_group()
