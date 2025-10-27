import json
import os
import re
import tempfile
import contextlib
import io
import signal
import multiprocessing
from typing import Dict, List, Optional

# Similar timeout and safety utilities as in execution.py
class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

@contextlib.contextmanager
def swallow_io():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            yield

def check_mbpp_correctness(
    problem: Dict, completion: str, timeout: float = 3.0, completion_id: Optional[int] = None
) -> Dict:
    """
    Evaluates the functional correctness of an MBPP completion by running the test
    assertions provided in the problem.
    """
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(
        target=unsafe_execute_mbpp, 
        args=(problem, completion, timeout, result)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=problem["task_id"],
        passed=result[0] == "passed",
        result=result[0],
        completion_id=completion_id,
    )

def unsafe_execute_mbpp(problem: Dict, completion: str, timeout: float, result):
    """Executes the MBPP code with safety precautions."""
    # Get the test cases from the problem
    test_list = problem.get("test_list", [])
    
    try:
        exec_globals = {}
        with swallow_io():
            with time_limit(timeout):
                # Execute the completion (full function definition)
                exec(completion, exec_globals)
                
                # Then execute each test assertion
                for test in test_list:
                    exec(test, exec_globals)
                
        # If we got here without exceptions, all tests passed
        result.append("passed")
    except TimeoutException:
        result.append("timed out")
    except BaseException as e:
        result.append(f"failed: {e}")

def evaluate_functional_correctness_item_mbpp(sample, problem_file):
    """
    Evaluates the functional correctness of an MBPP sample against the problem tests.
    """
    # Load the problems
    problems = {}
    with open(problem_file, 'r') as f:
        for line in f:
            if line.strip():
                problem = json.loads(line)
                problems[problem["task_id"]] = problem
    
    task_id = sample["task_id"]
    completion = sample["completion"]
    
    if task_id not in problems:
        return {"pass@1": 0.0}
    
    # Check if the completion passes all tests
    result = check_mbpp_correctness(problems[task_id], completion)
    
    # For a single sample, pass@1 is just 1.0 if passed, 0.0 if failed
    pass_at_k = {"pass@1": 1.0 if result["passed"] else 0.0}
    
    return pass_at_k

# def entry_point_item_mbpp(sample_path, problem_file):
#     """Entry point for evaluating a single MBPP sample."""
#     # Load the sample
#     with open(sample_path, 'r') as f:
#         sample = json.load(f)
    
#     # Evaluate it
#     result = evaluate_functional_correctness_item_mbpp(sample, problem_file)
    
#     return result

def entry_point_item_mbpp(sample_input, problem_file):
    """
    Entry point for evaluating a single MBPP sample.
    
    Args:
        sample_input: Either a path to a JSON file or a dictionary with task_id and completion
        problem_file: Path to the MBPP problems file
    """
    # Check if sample_input is a string (path) or dict
    if isinstance(sample_input, str):
        # Load the sample from file
        with open(sample_input, 'r') as f:
            sample = json.load(f)
    else:
        # Already a dictionary
        sample = sample_input
    
    # Evaluate it
    result = evaluate_functional_correctness_item_mbpp(sample, problem_file)
    
    return result