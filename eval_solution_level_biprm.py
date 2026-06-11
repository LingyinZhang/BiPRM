import os
import re
import json
import random
from copy import deepcopy
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

import deepspeed
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from value_model import AutoModelForCausalLMWithValueHead
from eval_solution_level_utils import eval_gsm8k, eval_math_prm

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")



def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def instruction_format(s):
    return f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{s}\n\n### Response: Let's think step by step"

def best_of_n(splitted_completions):
    selected_completions = []
    for n_completions_per_query in splitted_completions:
        n_completions_per_query = sorted(n_completions_per_query, key=lambda x: x["reward"], reverse=True)
        assert all([n_completions_per_query[0]["reward"] >= completion["reward"] for completion in n_completions_per_query])
        selected_completions.append(n_completions_per_query[0])
    return selected_completions

def the_first_n(completions, n, N=128): 
    splitted_completions = []
    for idx in range(int(len(completions) / N)):
        samples = [sample for sample in completions if sample["idx"] == idx]
        splitted_completions.append(samples[:n])
    return splitted_completions

def compute_metrics(dataset_name, scored_results):
    metrics = {}
    sample_nums = [1, 8, 16, 32, 64, 128]

    if dataset_name == 'gsm8k':
        original_dataset = load_dataset('qintongli/GSM-Plus')['testmini']
    else:
        original_dataset = load_dataset('HuggingFaceH4/MATH-500')['test']

    for n in sample_nums:
        splitted_completions = the_first_n(scored_results, n, max(sample_nums))
        if not args.baseline and not args.combine:
            selected_completions = best_of_n(splitted_completions)   
                         
            assert len(original_dataset) == len(selected_completions)
            if dataset_name == 'math':
                acc, _, _ = eval_math_prm([{'response': query['response'],'idx': query['idx']} for query in selected_completions],
                                          all_problems=[{'solution': data['solution'], 'question': data['problem'], 'answer': data['answer']} for
                                                        data in original_dataset], is_extract=False)
            else:
                acc, acc_list, _ = eval_gsm8k([{'response': query['response'],'idx': query['idx']} for query in selected_completions],
                                        answers=[data['answer'] for data in original_dataset],is_extract=True)
            metrics[n] = acc
            if accelerator.is_local_main_process:
                print(f"BON@ {n:<3} Accuracy: {acc:.2f}%")

        else:
            selected_completions = []
            for comps in splitted_completions:
                selected_completions += comps
            if dataset_name == 'math':
                acc, acc_list, output_list = eval_math_prm([{'response': query['response'],'idx': query['idx']} for query in selected_completions],
                                          all_problems=[{'solution': data['solution'], 'question': data['problem']} for
                                                        data in original_dataset for _ in range(n)], is_extract=False)
            else:
                acc, acc_list, output_list = eval_gsm8k([{'response': query['response'],'idx': query['idx']} for query in selected_completions],
                                       answers=[data['answer'] for data in original_dataset for _ in range(n)],is_extract=True)
            total_index = int(len(acc_list) / n)
            if args.baseline:
                pass_k = sum([1 for ii in range(total_index) if True in acc_list[ii*n:(ii+1)*n]])/total_index
                consistent_outputs = [Counter(output_list[ii*n:(ii+1)*n]).most_common(1)[0][0] for ii in range(total_index)] 
                position_of_consistent_outputs = [output_list[ii*n:(ii+1)*n].index(consistent_outputs[ii]) for ii in range(total_index)]
                acc_of_consistency = [acc_list[ii*n:(ii+1)*n][idx_of_split] for ii, idx_of_split in enumerate(position_of_consistent_outputs)]
                sc = sum(acc_of_consistency)/total_index
                if accelerator.is_local_main_process:
                    print('*********')
                    print(n,pass_k,sc)
                    print('*********')
            else:
                correct,sumv = 0,0
                for ii in range(total_index):
                    answer_dict = {k:0 for k in set(output_list[ii*n:(ii+1)*n])}
                    reward_list = [ele['reward'] for ele in selected_completions[ii*n:(ii+1)*n]]
                    for ele,reward in zip(output_list[ii*n:(ii+1)*n],reward_list):
                        answer_dict[ele]+=torch.sigmoid(torch.tensor(reward)).item()
                    select_answer = sorted(answer_dict.items(),key=lambda x:x[1],reverse=True)[0][0]
                    correct += acc_list[ii*n:(ii+1)*n][output_list[ii * n:(ii + 1) * n].index(select_answer)]
                    sumv+=1
                if accelerator.is_local_main_process:
                    print('*********')
                    print(n,correct/sumv)
                    print('*********')
    return metrics



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--baseline", type=int, default=0)
    parser.add_argument("--combine", type=int, default=0)
    parser.add_argument("--orm", type=int, default=0)
    parser.add_argument("--backbone-path", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--model-path", type=str, default="checkpoint")
    parser.add_argument("--data-name", type=str, choices=['math','gsm8k'], default='math')
    parser.add_argument("--data-file", type=str, required=True, default="./test_data/math-muggle-128.json")
    parser.add_argument("--save-file", type=str, default="./bon_eval_results.json")
    parser.add_argument("--load-file", type=str, default="")    

    args = parser.parse_args()

    seed_everything(0)
    accelerator = Accelerator()
    if not args.baseline:
        prm_token = '[PRM]'
        model_path = args.backbone_path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.bfloat16)
        tokenizer.add_special_tokens({'additional_special_tokens':[prm_token]})
        prm_token_id = tokenizer.encode(prm_token, add_special_tokens=False)[-1]
        model.resize_token_embeddings(len(tokenizer))
        model = AutoModelForCausalLMWithValueHead(model)

        # load sigma mlp
        hidden_size = model.pretrained_model.config.hidden_size
        model.sigma_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
            
        state_dict = torch.load(args.model_path + "/pytorch_model.bin", map_location=torch.device('cuda'))
        model.load_state_dict(state_dict)

        ds_engine = deepspeed.init_inference(model,
                                             tensor_parallel={"tp_size": 1},
                                             dtype=torch.bfloat16)

        model = ds_engine.module
        model.eval()

        def data_collator(example, tokenizer=tokenizer):
            inputs = []
            special_ids = []
            orm_ids = []
            inputs_reverse = []
            special_ids_reverse = []
            orm_ids_reverse = []
            idx,reward_idx = [],[]
            template = '{query}\n{answer}'
            for d in example:
                input_ids = tokenizer.encode(template.format(query=d['query'],answer=d['answer']),
                                              add_special_tokens=False)
                inputs.append(torch.tensor(input_ids))

                cur_special_ids = []
                for ii,id in enumerate(input_ids):
                    if id==prm_token_id:
                        cur_special_ids.append(ii)
                special_ids.append(torch.tensor(cur_special_ids))
                orm_ids.append(cur_special_ids[-1])
                idx.append(d['idx'])
                reward_idx.append(d['reward_idx'])

                input_ids_reverse = tokenizer.encode(template.format(query=d['query'],answer=d['answer_reverse']),
                                              add_special_tokens=False)
                inputs_reverse.append(torch.tensor(input_ids_reverse))

                cur_special_ids = []
                for ii,id in enumerate(input_ids_reverse):
                    if id==prm_token_id:
                        cur_special_ids.append(ii)
                special_ids_reverse.append(torch.tensor(cur_special_ids))
                orm_ids_reverse.append(cur_special_ids[-1])

            inputs = pad_sequence(inputs, padding_value=tokenizer.pad_token_id, batch_first=True)
            attention_mask = (inputs!=tokenizer.pad_token_id)
            special_ids = pad_sequence(special_ids, padding_value=-100, batch_first=True)
            inputs_reverse = pad_sequence(inputs_reverse, padding_value=tokenizer.pad_token_id, batch_first=True)
            attention_mask_reverse = (inputs_reverse!=tokenizer.pad_token_id)
            special_ids_reverse = pad_sequence(special_ids_reverse, padding_value=-100, batch_first=True)

            return {
                'input_ids': inputs.int().to(accelerator.device),
                'attention_mask': attention_mask.int().to(accelerator.device),
                'special_tokens':special_ids.to(accelerator.device),
                'orm_tokens': torch.tensor(orm_ids).to(accelerator.device),
                'input_ids_reverse': inputs_reverse.int().to(accelerator.device),
                'attention_mask_reverse': attention_mask_reverse.int().to(accelerator.device),
                'special_tokens_reverse':special_ids_reverse.to(accelerator.device),
                'orm_tokens_reverse': torch.tensor(orm_ids_reverse).to(accelerator.device),
                'idx':torch.tensor(idx).to(accelerator.device),
                'reward_idx':torch.tensor(reward_idx).to(accelerator.device)
            }
    data_name = args.data_name

    if args.load_file == "":
        if data_name == 'gsm8k':
            file_list = [
                args.data_file,
            ]
            queries = []
            cur_queries = []
            origin_dataset = load_dataset('qintongli/GSM-Plus')['testmini']
            for file_name in file_list:
                cur_data = json.load(open(file_name))
                if len(cur_queries) == len(cur_data):
                    for cur_q, cur_d in zip(cur_queries, cur_data):
                        cur_q['responses'].extend(cur_d['responses'])
                else:
                    cur_queries = deepcopy(cur_data)

            assert len(origin_dataset) == len(cur_queries), (len(origin_dataset), len(queries))
            for idx, (data, ori) in enumerate(zip(cur_queries, origin_dataset)):
                assert data['question'] == ori['question']
                assert len(data['responses']) == 128
                for response_dict in data['responses']:
                    queries.append({
                        'idx': idx,
                        'prompt': data['question'],
                        'response': response_dict['text'],
                        'solution': ori['answer'],
                        'logprobs': 0,
                    })
        elif data_name == 'math':
            file_list = [
                args.data_file,
            ]
            queries = []
            cur_queries = []
            origin_dataset = load_dataset('HuggingFaceH4/MATH-500')['test']
            for file_name in file_list:
                cur_data = json.load(open(file_name))
                if len(cur_queries) == len(cur_data):
                    for cur_q, cur_d in zip(cur_queries, cur_data):
                        cur_q['responses'].extend(cur_d['responses'])
                else:
                    cur_queries = deepcopy(cur_data)

            assert len(origin_dataset) == len(cur_queries), (len(origin_dataset), len(queries))
            for idx, (data, ori) in enumerate(zip(cur_queries, origin_dataset)):
                assert data['question'] == ori['problem']
                assert len(data['responses']) % 128==0
                for response_dict in data['responses']:
                    queries.append({
                        'idx': idx,
                        'prompt': data['question'],
                        'response': response_dict['text'],
                        'solution': ori['answer'],
                        'logprobs': 0,
                    })

        if not args.baseline:
            i=0
            for idx, data in enumerate(queries):
                data['reward_idx'] = idx
                data["query"] = instruction_format(data["prompt"])
                steps = re.split('Step \d+:', data['response'])
                steps = [f'Step {id + 1}: ' + step.strip() for id, step in enumerate(steps) if step.strip()!='']
                steps_reverse = steps[::-1]
                data["answer"] = f" {prm_token}\n".join(steps) + f" {prm_token}"
                data["answer_reverse"] = f" {prm_token}\n".join(steps_reverse) + f" {prm_token}"
            
            dataset = Dataset.from_pandas(pd.DataFrame.from_records(queries))
            dataloader = DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=data_collator)

            for inputs in tqdm(dataloader):
                with torch.no_grad():
                    _, _, rewards_positive = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                    _, _, rewards_reverse = model(input_ids=inputs['input_ids_reverse'], attention_mask=inputs['attention_mask_reverse'])
                cur_index = torch.where(inputs['special_tokens']==-100,0,inputs['special_tokens'])
                cur_index_reverse = torch.where(inputs['special_tokens_reverse']==-100,0,inputs['special_tokens_reverse'])
                if not args.orm:
                    special_tokens = inputs.get('special_tokens')
                    special_tokens_reverse = inputs.get('special_tokens_reverse')
                    rewards_positive = rewards_positive.gather(dim=-1, index=cur_index)
                    rewards_reverse = rewards_reverse.gather(dim=-1, index=cur_index_reverse)
                    a = rewards_positive * (special_tokens != -100)
                    b = rewards_reverse * (special_tokens_reverse != -100)
                    c = torch.zeros_like(b)
                    for i in range(b.size(0)):
                        non_zero_b = b[i, b[i] != 0]
                        reversed_non_zero_b = torch.flip(non_zero_b, [0])
                        c[i, :len(reversed_non_zero_b)] = reversed_non_zero_b

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
                    mask_rev = (special_tokens_reverse != -100)
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
                    rewards = sigma * a + (1 - sigma) * c
                    final_rewards = torch.where(special_tokens==-100,1e5,rewards).min(-1).values # 获取最后一个维的最小值
                else:
                    rewards = rewards.gather(dim=-1, index=inputs['orm_tokens'][...,None])
                    final_rewards = rewards.squeeze(1)
                for step_reward,final_reward,reward_idx in zip(rewards.tolist(),final_rewards.tolist(),inputs['reward_idx'].tolist()):
                    queries[int(reward_idx)]['reward'] = final_reward
                    queries[int(reward_idx)]['step_reward'] = [r for r in step_reward if r!=1e5]                    

        os.makedirs(os.path.dirname(args.save_file) or '.', exist_ok=True)
        with open(args.save_file,'w') as f:
            json.dump(queries,f)
    else:
        queries = json.load(open(args.load_file))
    
    metrics = compute_metrics(data_name, queries)
    