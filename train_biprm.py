import os
import sys
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

import datasets
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    set_seed,
)

from value_model import AutoModelForCausalLMWithValueHead
from utils import generate_dataset_double
from training_arguments import PR2MTrainingArguments

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

class PRMTrainer(Trainer):
    def __init__(self, model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None):
        super().__init__(model=model,
                         args=args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         tokenizer=tokenizer,
                         model_init=model_init,
                         compute_metrics=compute_metrics,
                         callbacks=callbacks,
                         optimizers=optimizers,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        self.loss_type = args.loss_type
        self.batch_size = args.per_device_train_batch_size
        if self.loss_type == 'bce':
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            _, _, rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            _, _, rewards_reverse = model(input_ids=inputs["input_ids_reverse"], attention_mask=inputs["attention_mask_reverse"])
        labels = inputs.get('step_labels')
        rewards = rewards.gather(dim=-1, index=inputs['prm_idxs'])
        rewards_reverse = rewards_reverse.gather(dim=-1, index=inputs['prm_idxs_reverse'])
        rewards = rewards * (inputs['step_labels'] != -100)
        rewards_reverse = rewards_reverse * (inputs['step_labels_reverse'] != -100)
        # compute reversed alignment
        c = torch.zeros_like(rewards_reverse)
        for i in range(rewards_reverse.size(0)):
            non_zero_b = rewards_reverse[i, rewards_reverse[i] != 0]
            reversed_non_zero_b = torch.flip(non_zero_b, [0])
            c[i, :len(reversed_non_zero_b)] = reversed_non_zero_b
        # dynamic merge forward and reverse using learned weights
        with torch.no_grad():
            base_model = model.pretrained_model
            out_f = base_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
            out_r = base_model(input_ids=inputs["input_ids_reverse"], attention_mask=inputs["attention_mask_reverse"], output_hidden_states=True)
            hidden_fwd = out_f.hidden_states[-1]
            hidden_rev = out_r.hidden_states[-1]
        # gather hidden states at PRM positions
        hidden_fwd_selected = hidden_fwd.gather(1, inputs['prm_idxs'].unsqueeze(-1).expand(-1, -1, hidden_fwd.size(-1)))
        hidden_rev_selected = hidden_rev.gather(1, inputs['prm_idxs_reverse'].unsqueeze(-1).expand(-1, -1, hidden_rev.size(-1)))
        # align reversed hidden states
        hidden_rev_flip = torch.zeros_like(hidden_fwd_selected)
        mask_rev = (inputs['step_labels_reverse'] != -100)
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
        rewards = sigma * rewards + (1 - sigma) * c

        if self.loss_type == 'mse':
            rewards = rewards.sigmoid()
            loss = (self.loss_fn(rewards,
                                 torch.where(inputs['step_labels'] != -100, inputs['step_labels'], 0).bfloat16()) *
                    (inputs['step_labels'] != -100)).sum() / (inputs['step_labels'] != -100).sum()
        elif self.loss_type == 'bce':
            rewards = rewards.sigmoid().flatten()
            rewards = torch.stack([1 - rewards, rewards])
            loss = self.loss_fn(rewards.T, inputs['step_labels'].flatten())
        elif self.loss_type == 'rank':
            loss = self.ranking_loss(rewards, inputs['step_labels'], inputs['has_neg'])

        return (loss, rewards, labels)

    def ranking_loss(self, rewards, labels, has_neg):
        pos_rewards_exp = torch.where(labels == 1, (rewards).exp(), 0)
        neg_rewards_exp = torch.where(labels == 0, (rewards + 4).exp(), 0).flip(dims=[-1])
        neg_reward_sum = neg_rewards_exp.sum(-1)
        pos_rewards_cumsum = torch.cat(
            [torch.zeros(rewards.shape[0], 1, device=rewards.device).exp(), pos_rewards_exp],
            dim=1).cumsum(-1)[:, :-1]
        pos_rewards_cumsum = torch.cat(
            [torch.zeros(rewards.shape[0], 1, device=rewards.device), pos_rewards_cumsum],
            dim=-1)
        reward_exp_cur = torch.where(labels == 1, pos_rewards_exp, 1)
        reward_exp_cur = torch.cat(
            [torch.zeros(rewards.shape[0], 1, device=rewards.device).exp(), reward_exp_cur], dim=-1)
        loss = -torch.log(
            reward_exp_cur / (reward_exp_cur + pos_rewards_cumsum + neg_reward_sum[..., None] + 1e-5))
        labels = torch.cat([has_neg[..., None], labels], dim=-1)
        loss = (torch.where(labels == 1, loss, 0).sum(-1) /
                torch.where(labels == 1, 1, 0).sum(-1)).mean()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        _, _, rewards = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        _, _, rewards_reverse = model(input_ids=inputs['input_ids_reverse'], attention_mask=inputs['attention_mask_reverse'])
        rewards = rewards.gather(dim=-1, index=inputs['prm_idxs'])
        rewards_reverse = rewards_reverse.gather(dim=-1, index=inputs['prm_idxs_reverse'])
        rewards = rewards * (inputs['step_labels'] != -100)
        rewards_reverse = rewards_reverse * (inputs['step_labels_reverse'] != -100)
        c = torch.zeros_like(rewards_reverse)
        for i in range(rewards_reverse.size(0)):
            non_zero_b = rewards_reverse[i, rewards_reverse[i] != 0]
            reversed_non_zero_b = torch.flip(non_zero_b, [0])
            c[i, :len(reversed_non_zero_b)] = reversed_non_zero_b
        # dynamic merge
        with torch.no_grad():
            base_model = model.pretrained_model
            out_f = base_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
            out_r = base_model(input_ids=inputs['input_ids_reverse'], attention_mask=inputs['attention_mask_reverse'], output_hidden_states=True)
            hidden_fwd = out_f.hidden_states[-1]
            hidden_rev = out_r.hidden_states[-1]
        hidden_fwd_selected = hidden_fwd.gather(1, inputs['prm_idxs'].unsqueeze(-1).expand(-1, -1, hidden_fwd.size(-1)))
        hidden_rev_selected = hidden_rev.gather(1, inputs['prm_idxs_reverse'].unsqueeze(-1).expand(-1, -1, hidden_rev.size(-1)))
        hidden_rev_flip = torch.zeros_like(hidden_fwd_selected)
        mask_rev = (inputs['step_labels_reverse'] != -100)
        for i in range(hidden_rev_selected.size(0)):
            non_zero_hidden = hidden_rev_selected[i][mask_rev[i]]
            reversed_non_zero_hidden = torch.flip(non_zero_hidden, [0])
            hidden_rev_flip[i, :len(reversed_non_zero_hidden)] = reversed_non_zero_hidden
        concat_hidden = torch.cat([hidden_fwd_selected, hidden_rev_flip], dim=-1)
        sigma = model.sigma_mlp(concat_hidden).squeeze(-1)
        sigma = torch.sigmoid(sigma)
        rewards = sigma * rewards + (1 - sigma) * c

        if self.loss_type == 'mse':
            rewards = rewards.sigmoid()
            loss = (self.loss_fn(rewards,
                                 torch.where(inputs['step_labels'] != -100, inputs['step_labels'], 0).bfloat16()) *
                    (inputs['step_labels'] != -100)).sum() / (inputs['step_labels'] != -100).sum()
        elif self.loss_type == 'bce':
            rewards = rewards.sigmoid().flatten()
            rewards = torch.stack([1 - rewards, rewards])
            loss = self.loss_fn(rewards.T, inputs['step_labels'].flatten())
        elif self.loss_type == 'rank':
            loss = self.ranking_loss(rewards, inputs['step_labels'], inputs['has_neg'])
        return loss

def collect_fn(inputs):
    inputs = {key: [example[key] for example in inputs] for key in inputs[0].keys()}
    inputs['input_ids'] = pad_sequence(inputs['input_ids'], padding_value=tokenizer.pad_token_id, batch_first=True)
    inputs['attention_mask'] = pad_sequence(inputs['attention_mask'], padding_value=0, batch_first=True)
    inputs['prm_idxs'] = pad_sequence(inputs['prm_idxs'], padding_value=0, batch_first=True)
    inputs['step_labels'] = pad_sequence(inputs['step_labels'], padding_value=-100, batch_first=True)
    inputs['has_neg'] = torch.stack(inputs['has_neg'])
    inputs['orm_tokens'] = torch.stack(inputs['orm_tokens'])
    inputs['input_ids_reverse'] = pad_sequence(inputs['input_ids_reverse'], padding_value=tokenizer.pad_token_id, batch_first=True)
    inputs['attention_mask_reverse'] = pad_sequence(inputs['attention_mask_reverse'], padding_value=0, batch_first=True)
    inputs['prm_idxs_reverse'] = pad_sequence(inputs['prm_idxs_reverse'], padding_value=0, batch_first=True)
    inputs['step_labels_reverse'] = pad_sequence(inputs['step_labels_reverse'], padding_value=-100, batch_first=True)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, dataset, tokenizer, prm_token_id):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.prm_token_id = prm_token_id
        self.processed_data = []
        rank = int(os.environ.get('RANK', 0))
        is_main_process = rank == 0
        pbar = tqdm(dataset, desc="Preprocessing", disable=not is_main_process)
        for example in pbar:
            processed = self.preprocess(example)
            if processed:
                self.processed_data.append(processed)
        pbar.close()

    def preprocess(self, example):
        template = '{query}\\n{answer}'
        inputs = self.tokenizer(template.format(query=example['query'], answer=example['answer']), return_tensors='pt', add_special_tokens=False)
        inputs_reverse = self.tokenizer(template.format(query=example['query'], answer=example['answer_reverse']), return_tensors='pt', add_special_tokens=False)
        input_ids = inputs['input_ids'].flatten()
        input_ids_reverse = inputs_reverse['input_ids'].flatten()
        prm_idxs = (input_ids == self.prm_token_id).nonzero().flatten()
        prm_idxs_reverse = (input_ids_reverse == self.prm_token_id).nonzero().flatten()

        if len(prm_idxs) != len(example['labels']):
            return None
        step_labels = torch.tensor(example['labels'])
        step_labels_reverse = torch.tensor(example['labels_reverse'])
        orm_labels = prm_idxs[-1]
        has_neg = 0 in example['labels']
        return {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'].flatten(),
            'prm_idxs': prm_idxs,
            'step_labels': step_labels.flatten(),
            'orm_tokens': orm_labels,
            'input_ids_reverse': input_ids_reverse,
            'attention_mask_reverse': inputs_reverse['attention_mask'].flatten(),
            'prm_idxs_reverse': prm_idxs_reverse,
            'step_labels_reverse': step_labels_reverse.flatten(),
            'has_neg': torch.tensor(has_neg)
        }

    def __getitem__(self, index):
        return self.processed_data[index]

    def __len__(self):
        return len(self.processed_data)

if __name__=='__main__':
    parser = HfArgumentParser(PR2MTrainingArguments)
    json_file = None
    for arg in sys.argv[1:]:
        if arg.endswith(".json"):
            json_file = arg
            break
    if json_file:
        (training_args,) = parser.parse_json_file(json_file=os.path.abspath(json_file))
    else:
        (training_args,) = parser.parse_args_into_dataclasses()
    cmd_args_str = [arg for arg in sys.argv[1:] if not arg.endswith(".json")]
    cmd_args = parser.parse_args_into_dataclasses(args=cmd_args_str)[0]
    cmd_args_keys = [arg.split('=')[0].lstrip('-') for arg in cmd_args_str if '=' in arg]
    for key in cmd_args_keys:
        if hasattr(cmd_args, key):
            value = getattr(cmd_args, key)
            if value is not None:
                setattr(training_args, key, value)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    prm_token = '[PRM]'
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name_or_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(training_args.model_name_or_path,
                                                 torch_dtype=torch.bfloat16, local_files_only=False, trust_remote_code=True)
    tokenizer.add_special_tokens({'additional_special_tokens': [prm_token]})
    prm_token_id = tokenizer.encode(prm_token, add_special_tokens=False)[-1]

    model.resize_token_embeddings(len(tokenizer))
    reward_model = AutoModelForCausalLMWithValueHead(model)

    # initialize MLP to generate dynamic weights
    hidden_size = reward_model.pretrained_model.config.hidden_size
    reward_model.sigma_mlp = nn.Sequential(
        nn.Linear(hidden_size * 2, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)
    )
    # ensure MLP is trainable
    for param in reward_model.sigma_mlp.parameters():
        param.requires_grad = True
        
    # move model (including MLP) to correct device
    reward_model.to(training_args.device)

    # BiPRM requires bidirectional data (forward + reversed answers)
    split_dataset = generate_dataset_double(training_args, prm_token)
    train_dataset = TrainDataset(split_dataset["train"], tokenizer, prm_token_id)
    eval_dataset = TrainDataset(split_dataset["test"], tokenizer, prm_token_id)

    trainer = PRMTrainer(
        reward_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collect_fn,
    )
    trainer.train()