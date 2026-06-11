import torch
import random
import numpy as np
from datasets import load_dataset, Dataset


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_dataset_double(training_args, prm_token):
    """
    Build the bidirectional training dataset for BiPRM.
    For each sample, we construct:
      - the forward answer (steps in original order) and its labels;
      - the reverse answer (steps reversed) and its reversed labels.
    """
    def instruction_format(s):
        return f'[INST] {s} [/INST]'

    ds = load_dataset(training_args.train_file)['train']
    queries = []
    for d in ds:
        question = d['prompt']
        steps = [s for s in d['completions'] if s.strip() != '']
        if len(steps) <= 1:
            continue

        steps = [step.strip().replace('ки', '').strip() for step in steps if step.strip() != '']
        step_labels = [1 if l else 0 for l in d['labels']]
        reversed_steps = steps[::-1]
        reversed_step_labels = step_labels[::-1]

        queries.append({
            "query": instruction_format(question),
            "answer": f" {prm_token}\n".join(steps) + f" {prm_token}",
            "labels": step_labels,
            "answer_reverse": f" {prm_token}\n".join(reversed_steps) + f" {prm_token}",
            "labels_reverse": reversed_step_labels,
        })

    full_dataset = Dataset.from_list(queries)
    split_dataset = full_dataset.train_test_split(test_size=0.05, seed=training_args.seed)
    return split_dataset
