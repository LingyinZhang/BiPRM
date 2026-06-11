"""
Evaluation utilities used by the solution-level (Best-of-N) evaluator.
Provides:
    - eval_gsm8k: extracts numeric answers from GSM-Plus / GSM8K-style completions.
    - eval_math_prm: evaluates MATH / MATH-500 style completions using math_equal.
"""

import os
import re
import math
import random

import pandas as pd
from datasets import load_from_disk

from grader import math_equal
from normalizer import extract_math_answer_new

random.seed(2)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except Exception:
        return None


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1: right_brace_idx].strip()


def eval_gsm8k(scored_results, print_acc=False, answers=None, is_extract=False):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"

    def extract_answer_hf(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip().replace(",", "")
            return eval(match_str)
        return INVALID_ANS

    def extract_answer(completion):
        try:
            last_number = re.findall(r'\d+\.\d+|\d+', completion)[-1]
            return eval(last_number)
        except Exception:
            return INVALID_ANS

    def is_correct(completion, answer, is_extract):
        if is_extract:
            try:
                gold = eval(answer)
            except Exception:
                gold = answer
        else:
            gold = extract_answer_hf(answer)
        assert gold != INVALID_ANS, f"No ground truth answer found in the document:{answer}"
        return extract_answer(completion) == gold

    completions = [result["response"] for result in scored_results]

    if answers is None:
        # Fallback: load a local GSM8K test split if explicitly available.
        test = load_from_disk(os.path.join('./eval_data/gsm8k', "test"))
        answers = [d['solution'] for d in test]

    acc_list = [is_correct(c, a, is_extract) for c, a in zip(completions, answers)]
    acc = 100 * sum(acc_list) / len(acc_list)
    if print_acc:
        print("Accuracy:", acc)
    return acc, acc_list, [extract_answer(c) for c in completions]


def eval_math_prm(scored_results, print_acc=False, all_problems=None, is_extract=False):
    """
    Evaluate MATH-style problems. `all_problems` is a list of dicts with keys
    such as 'question', 'solution' (and optionally 'answer', 'level', 'type').
    """

    def match_answer(response):
        is_matched = False
        for ans_marker in ('answer:\n', 'answer:', 'the answer is: ', 'the final answer is '):
            ans_idx = response.lower().rfind(ans_marker)
            if ans_idx != -1:
                is_matched = True
                response = response[ans_idx + len(ans_marker):].strip()
                response = response.replace('I hope it is correct.', '').strip()
                if response.endswith("\n"):
                    response = response[:-2]
                if response.endswith("."):
                    response = response[:-1]

        # Find boxed
        ans_boxed = _last_boxed_only_string(response)
        if ans_boxed:
            is_matched = True
            response = ans_boxed
        return is_matched, response

    if not all_problems:
        raise ValueError("eval_math_prm requires `all_problems` to be provided.")

    completions = [result["response"] for result in scored_results]
    assert len(all_problems) == len(completions), f"{len(all_problems)} vs {len(completions)}"

    outputs = []
    correct = []
    for problem_data, model_output in zip(all_problems, completions):
        try:
            answer = extract_math_answer_new(problem_data['question'], problem_data["solution"], is_extract)
            _, model_output = match_answer(model_output)
        except Exception:
            model_output = None
            answer = None

        outputs.append(model_output)
        try:
            equiv = math_equal(model_output, answer, timeout=True)
        except Exception:
            equiv = False
        correct.append(equiv)

    acc = math.fsum(correct) / len(all_problems) * 100
    if print_acc:
        print("Overall Accuracy = {}/{} = {:.4f}".format(math.fsum(correct), len(all_problems), acc))
    return acc, correct, outputs
