import argparse
import os
import sys
import torch
import transformers
from transformers import GPT2TokenizerFast

import batch_args
from modeling_gpt2 import GPT2ForSequenceClassification, GPT2LMHeadModel

import generating

parser = argparse.ArgumentParser(prog=__package__, description='Generate.')
parser.add_argument('input', type=str, help='Prompts.')
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--stage', type=int, required=True)
parser.add_argument('--target', type=int, required=True)
parser.add_argument('--start-idx', type=int, required=True)
parser.add_argument('--save', type=str, required=True)
args = parser.parse_args()

prompts = batch_args.expand(eval(args.input))
device = args.device
stage = args.stage
target = args.target
start_idx = args.start_idx
save = args.save

assert(target == 1)

print(f'Arguments: {args}')

if stage == 0:
    classifier = None
    model = GPT2LMHeadModel.from_pretrained(f'gpt2-medium').to(device)
else:
    classifier = GPT2ForSequenceClassification.from_pretrained(f'stage{stage}/classifier').to(device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    if os.path.exists(f'stage{stage}/lm'):
        print('Using custom LM')
        model = GPT2LMHeadModel.from_pretrained(f'stage{stage}/lm').to(device)
    else:
        print('Using pretrained LM')
        model = GPT2LMHeadModel.from_pretrained(f'gpt2-medium').to(device)

model.eval()
for param in model.parameters():
    param.requires_grad = False
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')

batch_size = 16
labels = [1, 0]

for label in labels:
    if label != target:
        continue
    if os.path.exists(save):
        raise Exception(f'{save} exists!')

for label_idx, label in enumerate(labels):
    if label != target:
        continue
    for i in range(start_idx, start_idx + len(prompts), batch_size):
        print('================================================================================')
        print(f'label {label}, starting at index {i}')

        seed = i + stage * 3000000 # Different seeds per stage
        if stage == 0:
            seed = i * len(labels) + label_idx # At stage 0, don't generate the same positives and negatives

        torch.manual_seed(seed)

        bsz = batch_size
        if i - start_idx + bsz > len(prompts):
            bsz = len(prompts) - (i - start_idx)
        current_prompts = prompts[i - start_idx:i - start_idx + bsz]

        results = generating.gen(model, tokenizer, classifier, text=current_prompts, length=50,
                                 target=label, grad_steps_start=30, grad_steps_end=30, step_size=0.02,
                                 top_k=100, loss_temperature=10.0, sample_temperature=1.0) # Try top_k = 100, 0

        print('\n\n================================================================================')
        print('Final results:')
        for j in range(len(results)):
            print(f'Hard score {results[j][0]["score"]}:', tokenizer.decode(results[j][0]['input_ids'].squeeze(0)))

        with open(save, 'a') as f:
            for j in range(len(results)):
                text = tokenizer.decode(results[j][0]['input_ids'].squeeze(0))
                start = '<|endoftext|>'
                assert(text.startswith(start))
                text = text[len(start):]

                f.write(text)
                f.write('\nc04e780be4d59c675613ca4530db7983\n') # Random string
print('================================================================================')
