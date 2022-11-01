import argparse
import os
import sys

parser = argparse.ArgumentParser(prog=__package__, description='Train (no arguments).')
args = parser.parse_args()

stages = 0
while True:
    if not os.path.exists(f'stage{stages}'):
        break
    stages += 1

if stages == 0:
    raise Exception('No stages detected')

print(f'Last stage: {stages - 1}')
print(f'Next stage: {stages}')

content = ''
for stage in range(stages):
    with open(f'stage{stage}/labels.csv', 'r') as f:
        cur = f.read()
    if stage > 0:
        cur = '\n'.join(cur.split('\n')[1:]) # Remove CSV header
    if not cur.endswith('\n'):
        cur += '\n'
    content += cur

with open('tmp_labels.csv', 'w') as f:
    f.write(content)

print(f'Training for new stage {stages}...')
os.mkdir(f'stage{stages}')
os._exit(os.system(f'PYTHONUNBUFFERED=1 python ./scripts/lib/run_glue.py \
  --model_name_or_path gpt2-medium \
  --do_train \
  --do_eval \
  --train_file tmp_labels.csv \
  --validation_file tmp_labels.csv \
  --max_seq_length 128 \
  --per_device_train_batch_size 4 \
  --num_train_epochs 5 \
  --save_strategy no \
  --output_dir ./stage{stages}/classifier \
  | tee ./stage{stages}/train_log_classifier.txt'))
