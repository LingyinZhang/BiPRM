import os
import json
import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

import deepspeed
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from value_model import AutoModelForCausalLMWithValueHead

def seed_everything(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def instruction_format(s):
    return f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{s}\n\n### Response: Let's think step by step"

def build_answers_from_steps(steps, prm_token):
    joined = f" {prm_token}\n".join(steps) + f" {prm_token}"
    joined_reverse = f" {prm_token}\n".join(steps[::-1]) + f" {prm_token}"
    return joined, joined_reverse

def data_collator_batch(batch_examples, tokenizer, prm_token_id, device):
    inputs = []
    special_ids = []
    orm_ids = []
    inputs_reverse = []
    special_ids_reverse = []
    orm_ids_reverse = []
    idx_list = []
    reward_idx = []
    for d in batch_examples:
        input_ids = tokenizer.encode(d['input_text'], add_special_tokens=False)
        inputs.append(torch.tensor(input_ids, dtype=torch.long))

        cur_special_ids = [i for i, tok in enumerate(input_ids) if tok == prm_token_id]
        special_ids.append(torch.tensor(cur_special_ids, dtype=torch.long) if cur_special_ids else torch.tensor([-100], dtype=torch.long))
        orm_ids.append(cur_special_ids[-1] if cur_special_ids else -100)
        idx_list.append(d['_idx'])
        reward_idx.append(d['_idx'])

        input_ids_reverse = tokenizer.encode(d['input_text_reverse'], add_special_tokens=False)
        inputs_reverse.append(torch.tensor(input_ids_reverse, dtype=torch.long))
        cur_special_ids_r = [i for i, tok in enumerate(input_ids_reverse) if tok == prm_token_id]
        special_ids_reverse.append(torch.tensor(cur_special_ids_r, dtype=torch.long) if cur_special_ids_r else torch.tensor([-100], dtype=torch.long))
        orm_ids_reverse.append(cur_special_ids_r[-1] if cur_special_ids_r else -100)

    inputs = pad_sequence(inputs, padding_value=tokenizer.pad_token_id, batch_first=True)
    attention_mask = (inputs != tokenizer.pad_token_id).long()
    special_ids = pad_sequence(special_ids, padding_value=-100, batch_first=True)
    inputs_reverse = pad_sequence(inputs_reverse, padding_value=tokenizer.pad_token_id, batch_first=True)
    attention_mask_reverse = (inputs_reverse != tokenizer.pad_token_id).long()
    special_ids_reverse = pad_sequence(special_ids_reverse, padding_value=-100, batch_first=True)

    return {
        'input_ids': inputs.to(device),
        'attention_mask': attention_mask.to(device),
        'special_tokens': special_ids.to(device),
        'orm_tokens': torch.tensor(orm_ids, device=device),
        'input_ids_reverse': inputs_reverse.to(device),
        'attention_mask_reverse': attention_mask_reverse.to(device),
        'special_tokens_reverse': special_ids_reverse.to(device),
        'orm_tokens_reverse': torch.tensor(orm_ids_reverse, device=device),
        'idx': torch.tensor(idx_list, device=device),
        'reward_idx': torch.tensor(reward_idx, device=device),
    }

def find_first_error_from_step_rewards(step_rewards, threshold):
    for i, r in enumerate(step_rewards):
        if r < threshold:
            return i
    return -1
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--backbone-path", type=str, required=True,
                        help="backbone model path (e.g. Qwen/Qwen2.5-Math-1.5B)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="trained model checkpoint folder containing pytorch_model.bin")
    parser.add_argument("--subset", type=str, default="gsm8k", help="ProcessBench subset to load")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold above which step considered correct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=2, help="tensor parallel size for deepspeed.init_inference")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator()
    seed_everything(0)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    prm_token = "[PRM]"
    tokenizer.add_special_tokens({'additional_special_tokens': [prm_token]})
    prm_token_id = tokenizer.encode(prm_token, add_special_tokens=False)[-1]

    backbone = AutoModelForCausalLM.from_pretrained(args.backbone_path, torch_dtype=torch.bfloat16, local_files_only=False)
    backbone.resize_token_embeddings(len(tokenizer))
    model = AutoModelForCausalLMWithValueHead(backbone)

    # load sigma mlp
    hidden_size = model.pretrained_model.config.hidden_size
    model.sigma_mlp = nn.Sequential(
        nn.Linear(hidden_size * 2, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)
    )

    state_dict = os.path.join(args.model_path, "pytorch_model.bin")
    state_dict = torch.load(state_dict, map_location='cpu')
        
    model.load_state_dict(state_dict)

    ds_engine = deepspeed.init_inference(model, tensor_parallel={"tp_size": args.tp_size}, dtype=torch.bfloat16)
    model = ds_engine.module
    model.eval()
    model.to(device)

    # load ProcessBench subset
    ds = load_dataset("Qwen/ProcessBench")[args.subset]
    hf_dataset = ds

    # normalize using your specified format: problem, label, steps
    normalized = []
    for i, ex in enumerate(hf_dataset):
        if not ('problem' in ex and 'steps' in ex and 'label' in ex):
            # skip entries that do not match expected format
            continue
        problem = ex['problem']
        steps = ex['steps']
        label = int(ex['label'])
        normalized.append({
            '_idx': i,
            'raw': ex,
            'problem': problem,
            'steps': steps,
            'label': label
        })

    examples = []
    for e in normalized:
        steps = e['steps']
        answer, answer_reverse = build_answers_from_steps(steps, prm_token)
        query = instruction_format(e['problem'])
        input_text = f"{query}\n{answer}"
        input_text_reverse = f"{query}\n{answer_reverse}"
        examples.append({
            '_idx': e['_idx'],
            'input_text': input_text,
            'input_text_reverse': input_text_reverse,
            'steps_len': len(steps),
            'label': e['label'],
            'raw': e['raw'],
            'problem': e['problem'],
        })

    batch_size = args.batch_size
    res_data = []
    predictions = {e['_idx']: None for e in examples}

    dataloader = DataLoader(examples, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda batch: data_collator_batch(batch, tokenizer, prm_token_id, device))

    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Scoring"):
            out = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            if len(out) >= 3:
                _, _, rewards_positive = out
            else:
                raise RuntimeError("Model forward did not return expected outputs (lm_logits, ..., rewards).")

            out_rev = model(input_ids=inputs['input_ids_reverse'], attention_mask=inputs['attention_mask_reverse'])
            if len(out_rev) >= 3:
                _, _, rewards_reverse = out_rev
            else:
                raise RuntimeError("Model forward for reverse did not return expected outputs.")

            cur_index = torch.where(inputs['special_tokens'] == -100, 0, inputs['special_tokens'])
            cur_index_reverse = torch.where(inputs['special_tokens_reverse'] == -100, 0, inputs['special_tokens_reverse'])

            if rewards_positive.dim() == 3 and rewards_positive.size(-1) == 1:
                rewards_positive = rewards_positive.squeeze(-1)
            if rewards_reverse.dim() == 3 and rewards_reverse.size(-1) == 1:
                rewards_reverse = rewards_reverse.squeeze(-1)

            rewards_pos_gathered = rewards_positive.gather(dim=-1, index=cur_index)
            rewards_rev_gathered = rewards_reverse.gather(dim=-1, index=cur_index_reverse)

            special_tokens_mask = (inputs['special_tokens'] != -100).long()
            special_tokens_mask_rev = (inputs['special_tokens_reverse'] != -100).long()

            a = rewards_pos_gathered * special_tokens_mask
            b = rewards_rev_gathered * special_tokens_mask_rev

            c = torch.zeros_like(b)
            for i in range(b.size(0)):
                non_zero = b[i, b[i] != 0]
                if non_zero.numel() > 0:
                    reversed_non_zero = torch.flip(non_zero, [0])
                    c[i, :len(reversed_non_zero)] = reversed_non_zero


            with torch.no_grad():
                base_model = model.pretrained_model
                out_f = base_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
                out_r = base_model(input_ids=inputs["input_ids_reverse"], attention_mask=inputs["attention_mask_reverse"], output_hidden_states=True)
                hidden_fwd = out_f.hidden_states[-1]
                hidden_rev = out_r.hidden_states[-1]
            # gather hidden states at PRM positions
            hidden_fwd_selected = hidden_fwd.gather(1, cur_index.unsqueeze(-1).expand(-1, -1, hidden_fwd.size(-1)))
            hidden_rev_selected = hidden_rev.gather(1, cur_index_reverse.unsqueeze(-1).expand(-1, -1, hidden_rev.size(-1)))
            # align reversed hidden states
            hidden_rev_flip = torch.zeros_like(hidden_fwd_selected)
            mask_rev = special_tokens_mask_rev
            for i in range(hidden_rev_selected.size(0)):
                non_zero_hidden = hidden_rev_selected[i][mask_rev[i]]
                reversed_non_zero_hidden = torch.flip(non_zero_hidden, [0])
                hidden_rev_flip[i, :len(reversed_non_zero_hidden)] = reversed_non_zero_hidden
            # compute sigma weights
            concat_hidden = torch.cat([hidden_fwd_selected, hidden_rev_flip], dim=-1)
            with torch.no_grad():
                sigma = model.sigma_mlp(concat_hidden).squeeze(-1)
            sigma = torch.sigmoid(sigma)
            # merge
            rewards_combined = sigma * a + (1 - sigma) * c
            rewards_combined = torch.sigmoid(rewards_combined)
            for i in range(rewards_combined.size(0)):
                row_mask = (inputs['special_tokens'][i] != -100)
                step_rewards = rewards_combined[i, :].cpu().tolist()
                step_rewards_filtered = [r for (r, m) in zip(step_rewards, row_mask.cpu().tolist()) if m]
                pred = find_first_error_from_step_rewards(step_rewards_filtered, args.threshold)
                orig_idx = int(inputs['reward_idx'][i].item())
                # find label and raw by matching examples list
                label = -1
                raw = {}
                for ex in examples:
                    if ex['_idx'] == orig_idx:
                        label = ex['label']
                        raw = ex['raw']
                        break
                match = (pred == label)
                out_entry = deepcopy(raw)
                out_entry['prediction'] = pred
                out_entry['match'] = match
                out_entry['step_rewards'] = step_rewards_filtered
                out_entry['label'] = label
                res_data.append(out_entry)
                predictions[orig_idx] = pred

    data1 = [e for e in res_data if e.get('label', -1) != -1]
    data2 = [e for e in res_data if e.get('label', -1) == -1]

    config = args.subset
    err_path = os.path.join(args.output_dir, f'{config}_error.jsonl')
    cor_path = os.path.join(args.output_dir, f'{config}_correct.jsonl')
    with open(err_path, 'w') as f:
        for e in data1:
            f.write(json.dumps(e) + '\n')
    with open(cor_path, 'w') as f:
        for e in data2:
            f.write(json.dumps(e) + '\n')

    acc1 = float(np.mean([1.0 if e.get('match', False) else 0.0 for e in data1]) * 100) if len(data1) > 0 else 0.0
    acc2 = float(np.mean([1.0 if e.get('match', False) else 0.0 for e in data2]) * 100) if len(data2) > 0 else 0.0
    if acc1 + acc2 == 0:
        f1 = 0.0
    else:
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)

    if accelerator.is_local_main_process:
        print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')
        print("Saved:", err_path, cor_path)

if __name__ == '__main__':
    main()