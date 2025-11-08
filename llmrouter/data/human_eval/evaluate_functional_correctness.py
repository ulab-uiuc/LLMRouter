import fire
import sys

from human_eval.data import HUMAN_EVAL
# from human_eval.evaluation import evaluate_functional_correctness
from human_eval.evaluation import evaluate_functional_correctness_item
import json

# def entry_point(
#     sample_file: str,
#     k: str = "1,10,100",
#     n_workers: int = 4,
#     timeout: float = 3.0,
#     problem_file: str = HUMAN_EVAL,
# ):
#     """
#     Evaluates the functional correctness of generated samples, and writes
#     results to f"{sample_file}_results.jsonl.gz"
#     """
#     k = list(map(int, k.split(",")))
#     results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
#     print(results)


# def entry_point_item(
#     sample_path: str,
#     problem_file: str = HUMAN_EVAL,
# ):
#     """
#     Evaluates the functional correctness of a single sample.
#     `sample_path` should be a path to a JSON file containing one sample.
#     """
#     # Load the sample (expects a JSON file with a dict like {"task_id": ..., "completion": ...})
#     with open(sample_path, "r") as f:
#         sample = json.load(f)

#     result = evaluate_functional_correctness_item(sample, problem_file)
#     print(result)

def entry_point_item(sample_input, problem_file=HUMAN_EVAL):
    """
    Evaluates the functional correctness of a single sample.
    
    Args:
        sample_input: Either a path to a JSON file or a dictionary with task_id and completion
        problem_file: Path to the HumanEval problems file
    """
    # Check if sample_input is a string (path) or dict
    if isinstance(sample_input, str):
        # Load the sample from file
        with open(sample_input, 'r') as f:
            sample = json.load(f)
    else:
        # Already a dictionary
        sample = sample_input
    
    result = evaluate_functional_correctness_item(sample, problem_file)
    return result


def main():
    fire.Fire(entry_point_item)


# sys.exit(main())