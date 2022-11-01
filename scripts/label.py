import argparse
import glob
import os
import random
import re
import sys

def input_rating(prompt):
    while True:
        label = input(prompt)
        try:
            label = int(label)
            if label not in [1, 2, 3, 4, 5]:
                raise ValueError('Rating needs to be an integer between 1-5')
            return label
        except ValueError:
            pass

parser = argparse.ArgumentParser(prog=__package__, description='Label.')
parser.add_argument('--continue-from', type=int, default=-1)
parser.add_argument('--stage', type=int, required=True)
args = parser.parse_args()

continue_from = args.continue_from
stage = args.stage

random.seed(stage)
labels = [1]

if continue_from < 0:
    if os.path.exists(f'stage{stage}/labels.csv'):
        raise RuntimeError(f'stage{stage}/labels.csv already exists')
    if os.path.exists(f'stage{stage}/labels-full.csv'):
        raise RuntimeError(f'stage{stage}/labels-full.csv already exists')

gen = []
for label in labels:
    for filename in sorted(glob.glob(f'stage{stage}/results_{label}_*.txt')):
        with open(filename, 'r') as f:
            content = f.read()
        results = content.split('\nc04e780be4d59c675613ca4530db7983\n')[:-1]
        for result in results:
            gen.append((len(gen), result, label))

random.shuffle(gen)

with open(f'stage{stage}/labels.csv', 'a') as f:
    with open(f'stage{stage}/labels-full.csv', 'a') as g:
        if continue_from < 0:
            f.write('sentence,label\n')
            g.write('sentence,label,gen_label\n')
        for i, (ind, x, gen_label) in enumerate(gen):
            if i < continue_from:
                continue

            print(f'{i} of {len(gen)}:')
            print(x.replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t'))

            label = input_rating('\nRating (1-5): ')

            x = '\"' + x.replace('\"', '\"\"') + '\"' # CSV-compliant
            f.write(f'{x},{label}\n')
            g.write(f'{x},{label},{gen_label}\n')

            if stage > 0:
                print('label used for generation: ' + ('positive' if gen_label != 0 else 'negative'))

            if False:
                alt = input(f'Write another input: ')
                alt = alt.strip('\n')
                if alt:
                    alt = '\"' + alt.replace('\"', '\"\"') + '\"'
                    alt_label = input_rating(f'Rate your version (1-5): ')
                    f.write(f'{alt},{alt_label}\n')
                    g.write(f'{alt},{alt_label},-1\n')

            f.flush()
            g.flush()
            print()
            print()
