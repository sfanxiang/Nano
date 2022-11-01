import argparse
import csv
import os
import sys

parser = argparse.ArgumentParser(prog=__package__, description='Train LM.')
parser.add_argument('input', type=str)
args = parser.parse_args()

text = args.input

stages = 0
while True:
    if not os.path.exists(f'stage{stages}'):
        break
    stages += 1

stages -= 1

if stages <= 0:
    raise Exception('No stages detected')

print(f'Last stage: {stages - 1}')
print(f'Next stage: {stages}')

if os.path.exists('stage{stages}/train_log_lm.txt'):
    raise Exception('Next stage is already trained. If training for a new stage, train the classifier first.')

stage0_content = ''
content = ''
for stage in range(stages):
    with open(f'stage{stage}/labels.csv', 'r') as f:
        cur = f.read()
    if stage > 0:
        cur = '\n'.join(cur.split('\n')[1:]) # Remove CSV header
    if not cur.endswith('\n'):
        cur += '\n'
    content += cur

    if stage == 0:
        stage0_content = content

with open('tmp_labels.csv', 'w') as f:
    f.write(content)

reader = csv.DictReader(stage0_content.split('\n'))
stage0_len = len([None for _ in reader])
reader = csv.DictReader(stage0_content.split('\n'))
stage0_perfects = sum([(1.0 if int(row['label']) >= 5 else 0.0) for row in reader])
reader = csv.DictReader(content.split('\n'))
positives = sum([(1.0 if int(row['label']) >= 5 else (0.5 if int(row['label']) >= 4 else 0.0)) for row in reader])
del reader

print(f'{stage0_perfects}/{stage0_len} perfect labels in stage 0. {positives} positives in total.')
epochs = 3

text = text.replace('\'', '\'\"\'\"\'')
os._exit(os.system(f'PYTHONUNBUFFERED=1 python ./scripts/lib/run_clm.py \
  --prompts \'{text}\' \
  --model_name_or_path gpt2-medium \
  --do_train \
  --do_eval \
  --train_file tmp_labels.csv \
  --validation_file tmp_labels.csv \
  --per_device_train_batch_size 4 \
  --num_train_epochs {epochs} \
  --save_strategy no \
  --output_dir ./stage{stages}/lm \
  | tee ./stage{stages}/train_log_lm.txt'))
