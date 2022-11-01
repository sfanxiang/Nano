import argparse
import os
import sys
import threading

from lib import batch_args

parser = argparse.ArgumentParser(prog=__package__, description='Gen.')
parser.add_argument('--dev', type=str, required=True)
parser.add_argument('--start-idx', type=int, required=True)
parser.add_argument('--prompts', type=str, required=True, help='Prompts.')
parser.add_argument('--stage', type=int, required=True)
args = parser.parse_args()

dev = args.dev
start_idx = args.start_idx
prompts = batch_args.expand(eval(args.prompts))
stage = args.stage

target = 1

if os.path.exists(f'stage{stage}/gen_log_{target}_{start_idx}.txt'):
    raise Exception(f'stage{stage}/gen_log_{target}_{start_idx}.txt exists')

if stage == 0:
    try:
        os.mkdir(f'stage{stage}')
    except FileExistsError:
        print(f'Directory stage{stage} already exists. Make sure this is what you want!')

def run(device, target, start_idx, ret):
    global stage, prompts
    escaped_prompts = batch_args.compress(prompts)
    escaped_prompts = batch_args.escape(escaped_prompts)
    ret[0] = os.system(f'PYTHONUNBUFFERED=1 python ./scripts/lib/generate.py --device {device} --stage {stage} --target {target} --start-idx {start_idx} --save stage{stage}/results_{target}_{start_idx}.txt {escaped_prompts} | tee stage{stage}/gen_log_{target}_{start_idx}.txt')

ret = []
threads = []

ret.append([1])
threads.append(threading.Thread(target=run, args=(dev, target, start_idx, ret[-1])))
threads[-1].start()

for thread in threads:
    thread.join()

for i, r in enumerate(ret):
    if r[0] != 0:
        raise Exception(f'Thread {i} returned {r[0]}')
