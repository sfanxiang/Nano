import argparse
import math
import numpy as np
import sys
import torch
from torch import nn
import transformers
from transformers import GPT2TokenizerFast

import batch_args
from modeling_gpt2 import GPT2ForSequenceClassification, GPT2LMHeadModel

SMALL_CONST = 1e-15
BIG_CONST = 1e10

# input_ids: (seq_len,)
def dist_n(input_ids, n):
    distinct = set()
    for i in range(len(input_ids) - n + 1):
        distinct.add(tuple(input_ids[i:i + n].tolist()))
    total = len(input_ids) - n + 1
    return (len(distinct) / total) if total > 0 else 0.0

def get_past_kv_for_index(past_kv, index):
    result = []
    for kv in past_kv:
        result.append([])
        for p in kv:
            result[-1].append(p[index])
    return result

# This function is modified from PPLM source code. Below is a copy of its original license:
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def top_k_filter(logits, k, probs=False):
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k).values
        idx = (slice(None),) * (values.dim() - 1) + (-1,)
        batch_mins = values[idx].unsqueeze(-1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)

def gen_uncontrolled(model, tokenizer, input_ids, input_ids_full, past_kv, length, top_k, sample_temperature):
    new_input_ids = []
    cur_past_kv = past_kv
    while input_ids.size(1) + len(new_input_ids) < length:
        last = new_input_ids[-1] if new_input_ids else input_ids[:, -1:]
        outputs = model(last, past_key_values=cur_past_kv)
        cur_past_kv = outputs.past_key_values

        logits = outputs.logits[:, -1]
        logits = logits / sample_temperature
        logits = top_k_filter(logits, top_k)
        logits[:, tokenizer.eos_token_id] = -BIG_CONST
        sample = torch.multinomial(nn.Softmax(dim=-1)(logits), num_samples=1)
        for b in range(len(input_ids_full)):
            if input_ids.size(1) + len(new_input_ids) < len(input_ids_full[b]):
                sample[b, :] = input_ids_full[b][input_ids.size(1) + len(new_input_ids)]
        new_input_ids.append(sample)

    assert(cur_past_kv[0][0].size(2) == length - 1)

    candidates = [[] for _ in range(len(input_ids))]
    for i in range(len(input_ids)):
        candidates[i].append({
            'score': 0.0,
            'input_ids': torch.cat([input_ids] + new_input_ids, dim=1).detach()[i:i + 1],
        })
    return candidates

def gen_one(model, tokenizer, classifier, input_ids, past_kv, length, target, grad_steps, step_size, top_k, loss_temperature, sample_temperature, gamma):
    if not isinstance(target, torch.Tensor):
        target = torch.tensor([target] * len(input_ids), device=input_ids.device)

    candidates = [[] for _ in range(len(input_ids))]

    for it in range(grad_steps):
        new_input_ids = []
        cur_past_kv = past_kv
        while input_ids.size(1) + len(new_input_ids) < length:
            last = new_input_ids[-1] if new_input_ids else input_ids[:, -1:]
            outputs = model(last, past_key_values=cur_past_kv)
            cur_past_kv = outputs.past_key_values

            logits = outputs.logits[:, -1]
            logits = logits / sample_temperature
            logits = top_k_filter(logits, top_k)
            logits[:, tokenizer.eos_token_id] = -BIG_CONST
            sample = torch.multinomial(nn.Softmax(dim=-1)(logits), num_samples=1)
            new_input_ids.append(sample)

        assert(cur_past_kv[0][0].size(2) == length - 1)

        print('################')
        print(f'Index {input_ids.size(1)} iteration {it}')
        for i in range(len(input_ids)):
            print(f'Sentence {i}:', tokenizer.decode(torch.cat([input_ids] + new_input_ids, dim=1)[i]))

        for kv in past_kv:
            for p in kv:
                p.requires_grad_(True)

        outputs = model(torch.cat([input_ids[:, -1:]] + new_input_ids, dim=1), past_key_values=past_kv)
        assert(outputs.past_key_values[0][0].size(2) == length)

        # Soft embeddings
        inputs_embeds_0 = classifier.transformer.wte(input_ids)
        logits = outputs.logits[:, -len(new_input_ids) - 1:-1]
        logits = logits / loss_temperature
        probs = nn.Softmax(dim=-1)(logits)
        inputs_embeds_1 = torch.matmul(probs, classifier.transformer.wte.weight.data)
        inputs_embeds = torch.cat([inputs_embeds_0, inputs_embeds_1], dim=1)
        assert(inputs_embeds.size(1) == length)

        # Soft loss
        logits = classifier(inputs_embeds=inputs_embeds[:, 1:]).logits # Do not include bos token for the classifier
        logits = logits[:, -2:]
        target = torch.ones_like(logits)
        loss = nn.BCEWithLogitsLoss(reduction='none')(logits.reshape(-1), target.reshape(-1)).reshape(target.size())
        coeff = torch.ones_like(loss)
        coeff[:, 0] = 0.1
        loss = loss * coeff
        loss = torch.sum(loss)
        loss.backward()
        print('Soft loss sum:', loss.item())

        # Hard loss
        logits = classifier(torch.cat([input_ids[:, 1:]] + new_input_ids, dim=1)).logits # Do not include bos token for the classifier
        logits = logits[:, -2:]
        target = torch.ones_like(logits)
        loss = nn.BCEWithLogitsLoss(reduction='none')(logits.reshape(-1), target.reshape(-1)).reshape(target.size()).detach()
        coeff = torch.ones_like(loss)
        coeff[:, 0] = 0.1
        loss = loss * coeff
        loss = torch.sum(loss, dim=-1)
        past_kv_detached = []
        for kv in past_kv:
            past_kv_detached.append([])
            for p in kv:
                past_kv_detached[-1].append(p.detach())

        # Score calculation; lower is better
        for i in range(len(input_ids)):
            dist_scores = [dist_n(torch.cat([input_ids] + new_input_ids, dim=1)[i], n + 1) for n in range(3)]
            dist_scores = sum(dist_scores) / len(dist_scores)

            score = loss[i].item()
            if dist_scores < 0.5:
                score += 10000000.0

            candidates[i].append({
                'score': score,
                'next_input_id': new_input_ids[0].detach()[i:i + 1],
                'input_ids': torch.cat([input_ids] + new_input_ids, dim=1).detach()[i:i + 1],
            })
            print(f'Hard score {i}:', score)

        new_past_kv = []
        for kv in past_kv:
            new_past_kv.append([])
            for p in kv:
                grad_norm = torch.norm(p.grad.reshape(p.grad.size(0), -1), dim=-1) + SMALL_CONST
                grad = -step_size * (p.grad / grad_norm.reshape(p.grad.size(0), *([1] * (p.grad.dim() - 1))) ** gamma)
                new_past_kv[-1].append(p.detach() + grad.detach())
                p.grad.zero_()

        past_kv = new_past_kv

    for i in range(len(input_ids)):
        candidates[i].sort(key=lambda x: x['score'])
    return candidates

# Returns a list (not tensor) of of shape (batch_size, num_candidates).
# Candidates are sorted by score.
# Each candidate is a dict of {'score': float, 'input_ids': tensor of shape (1, seq_len)}.
def gen(model, tokenizer, classifier, text, length, target, grad_steps_start, grad_steps_end, step_size, top_k, loss_temperature, sample_temperature):
    text = [(tokenizer.bos_token + x) for x in text]
    input_ids = []
    for x in text:
        input_ids.append(tokenizer(x).input_ids)
    input_ids_full = input_ids
    min_len = min([len(x) for x in input_ids])
    input_ids = [x[:min_len] for x in input_ids]
    del min_len
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)

    print(input_ids)
    print(f'Input:\n{tokenizer.batch_decode(input_ids)}')

    past_kv = model(input_ids[:, :-1]).past_key_values

    if classifier is None:
        candidates = gen_uncontrolled(model, tokenizer, input_ids=input_ids,
                                      input_ids_full=input_ids_full, past_kv=past_kv,
                                      length=length, top_k=top_k, sample_temperature=sample_temperature)
        for x in candidates:
            for y in x:
                y['input_ids'] = y['input_ids'].cpu()
        return candidates

    all_candidates = [[] for _ in range(len(input_ids))]

    while input_ids.size(1) < length:
        past_kv = model(input_ids[:, :-1]).past_key_values # Reset past_kv
        grad_steps = grad_steps_start + \
            ((grad_steps_end - grad_steps_start) * input_ids.size(1)) // (length - 1)
        assert(grad_steps_start <= grad_steps <= grad_steps_end or grad_steps_start >= grad_steps >= grad_steps_end)

        candidates = gen_one(
            model,
            tokenizer,
            classifier,
            input_ids=input_ids,
            past_kv=past_kv,
            length=length,
            target=target,
            grad_steps=grad_steps,
            step_size=step_size,
            top_k=top_k,
            loss_temperature=loss_temperature,
            sample_temperature=sample_temperature,
            gamma=1.0,
        )
        for i in range(len(input_ids)):
            for x in candidates[i]:
                if len(input_ids_full[i]) <= input_ids.size(1):
                    # Beyond the end of prompt
                    all_candidates[i].append({
                        'score': x['score'],
                        'input_ids': x['input_ids'].cpu(),
                    })
        del candidates
        for i in range(len(input_ids)):
            all_candidates[i].sort(key=lambda x: x['score'])

        best = []
        for i in range(len(input_ids)):
            if all_candidates[i]:
                best.append(all_candidates[i][0]['input_ids'][:, :input_ids.size(1) + 1])
            else:
                best.append(torch.tensor(input_ids_full[i][:input_ids.size(1) + 1], dtype=torch.long)[None, :])
        input_ids = torch.cat(best, dim=0).to(model.device) # Turn into a batch

        print('\nCurrent sentences at this token:')
        for i in range(len(input_ids)):
            print(tokenizer.decode(input_ids[i]))
        print('====\nBest sentences so far:')
        for i in range(len(input_ids)):
            if all_candidates[i]:
                print(f'Hard score {all_candidates[i][0]["score"]}:', tokenizer.decode(all_candidates[i][0]['input_ids'].squeeze(0)))
        print('====')

        break

    for i in range(len(input_ids)):
        all_candidates[i].sort(key=lambda x: x['score'])
    return all_candidates

def main():
    parser = argparse.ArgumentParser(prog=__package__, description='Generation.')
    parser.add_argument('input', type=str)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--grad-steps-start', type=int, required=True)
    parser.add_argument('--grad-steps-end', type=int, required=True)
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--model-classification', type=str, required=True)
    parser.add_argument('--model-language', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--step-size', type=float, required=True)
    parser.add_argument('--target', type=int, required=True)
    parser.add_argument('--loss-temperature', type=float, required=True)
    parser.add_argument('--sample-temperature', type=float, required=True)
    parser.add_argument('--top-k', type=int, required=True)
    args = parser.parse_args()

    print(f'Arguments: {args}')

    prompts = batch_args.expand(eval(args.input))
    device = args.device
    grad_steps_start = args.grad_steps_start
    grad_steps_end = args.grad_steps_end
    length = args.length
    model_classification = args.model_classification
    model_language = args.model_language
    seed = args.seed
    step_size = args.step_size
    target = args.target
    loss_temperature = args.loss_temperature
    sample_temperature = args.sample_temperature
    top_k = args.top_k

    torch.manual_seed(seed)

    if model_classification:
        classifier = GPT2ForSequenceClassification.from_pretrained(model_classification).to(device)
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False
    else:
        classifier = None
    model = GPT2LMHeadModel.from_pretrained(model_language).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = GPT2TokenizerFast.from_pretrained(model_language)

    results = gen(model, tokenizer, classifier, text=prompts, length=length,
                  target=target, grad_steps_start=grad_steps_start, grad_steps_end=grad_steps_end,
                  step_size=step_size, top_k=top_k, loss_temperature=loss_temperature,
                  sample_temperature=sample_temperature)
    print('\n\nFinal results:')
    for i in range(len(results)):
        print(f'Hard score {results[i][0]["score"]}:', tokenizer.decode(results[i][0]['input_ids'].squeeze(0)))

if __name__ == '__main__':
    main()
