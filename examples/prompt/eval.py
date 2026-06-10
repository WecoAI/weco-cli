# weco-cli/examples/prompt/eval.py
"""
eval.py  (parallel with progress logs)

Downloads a slice of AIME 2024, calls optimize.solve in parallel,
prints progress every N samples, and finally prints accuracy
in the format that Weco expects.
The LLM model to use is defined in this file.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import concurrent.futures

from datasets import load_dataset
import optimize  # the file Weco mutates

# ---------------------------------------------------------------------
# Configuration
TOTAL_SAMPLES = 30  # how many problems to load
NUM_WORKERS = 30  # concurrent LLM calls
LOG_EVERY = 5  # print progress after this many
MODEL_TO_USE = "gpt-4.1"  # Define the model to use HERE
TASK_TIMEOUT = 300  # seconds per LLM call
# ---------------------------------------------------------------------

print(f"[setup] loading {TOTAL_SAMPLES} problems from AIME 2024 …", flush=True)
DATA = load_dataset("Maxwell-Jia/AIME_2024", split=f"train[:{TOTAL_SAMPLES}]", cache_dir=".cache")


def extract_final_answer(text: str) -> str:
    """
    Extracts the final AIME answer (000-999) from the LLM response.
    Prioritizes answers within \boxed{}, then looks for patterns,
    and falls back to finding the last 3-digit number.
    """
    # 1. Check for \boxed{...}
    boxed_match = re.search(r"\\boxed\{(\d{1,3})\}", text)
    if boxed_match:
        return boxed_match.group(1).zfill(3)  # Pad with leading zeros if needed

    # 2. Check for "final answer is ..." patterns (case-insensitive)
    # Make sure pattern captures potential variations like "is: 123", "is 123."
    answer_pattern = r"(?:final|answer is|result is)[:\s]*(\d{1,3})\b"
    answer_match = re.search(answer_pattern, text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).zfill(3)

    # 3. Fallback: Find the last occurrence of a 1-3 digit number in the text
    #    This is less reliable but can be a fallback.
    #    Let's refine the fallback regex to be slightly more specific
    #    Look for isolated 1-3 digit numbers, possibly at the end or after keywords.
    fallback_matches = re.findall(r"\b(\d{1,3})\b", text)
    if fallback_matches:
        # Return the last found number, assuming it's the most likely answer candidate
        return fallback_matches[-1].zfill(3)

    return ""  # Return empty if no answer found


def grade_answer(llm_output: str, ground_truth_answer: str) -> bool:
    """Compares the extracted LLM answer to the ground truth."""
    extracted_guess = extract_final_answer(llm_output)
    # Ground truth answers in AIME are typically strings "000" to "999"
    # Ensure comparison is consistent (e.g., both as strings, potentially padded)
    # The ground truth from the dataset seems to be string integers already.
    # Let's ensure the extracted guess is also treated as a simple integer string for comparison.
    # The ground truth might not be zero-padded in the dataset, so compare integers.
    try:
        # Check if both can be converted to integers for comparison
        return int(extracted_guess) == int(ground_truth_answer)
    except ValueError:
        # If conversion fails (e.g., empty string), they don't match
        return False


def run_evaluation() -> float:
    """Runs the evaluation on the dataset and returns the accuracy."""
    correct = 0
    start = time.time()
    results = []  # Store results for potential later analysis if needed

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        # Submit all tasks, passing the MODEL_TO_USE
        futures = {
            pool.submit(optimize.solve, row["Problem"], MODEL_TO_USE): row["Answer"] for row in DATA
        }  # Pass MODEL_TO_USE here

        try:
            # Process completed tasks
            for idx, future in enumerate(as_completed(futures), 1):
                problem_answer = futures[future]  # Get the corresponding ground truth answer
                try:
                    # Wait up to TASK_TIMEOUT seconds for each LLM call
                    llm_raw_output = future.result(timeout=TASK_TIMEOUT)
                    is_correct = grade_answer(llm_raw_output, str(problem_answer))
                    if is_correct:
                        correct += 1
                    results.append({"raw_output": llm_raw_output, "correct_answer": problem_answer, "is_correct": is_correct})

                except Exception as exc:
                    print(f"[error] Generated an exception: {exc}")
                    results.append({"raw_output": f"Error: {exc}", "correct_answer": problem_answer, "is_correct": False})

                if idx % LOG_EVERY == 0 or idx == TOTAL_SAMPLES:
                    elapsed = time.time() - start
                    current_accuracy = correct / idx if idx > 0 else 0
                    print(
                        f"[progress] {idx}/{TOTAL_SAMPLES} completed, accuracy: {current_accuracy:.4f}, elapsed {elapsed:.1f} s",
                        flush=True,
                    )
        except concurrent.futures.TimeoutError:
            # Abort any stuck LLM calls
            print(f"[error] LLM call timed out after {TASK_TIMEOUT}s", flush=True)
            # Cancel all pending futures and exit
            for f in futures:
                f.cancel()
            print("Exiting due to timeout", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user", file=sys.stderr)
            sys.exit(1)

    # Final accuracy calculation
    total_evaluated = len(results)
    final_accuracy = correct / total_evaluated if total_evaluated > 0 else 0
    return final_accuracy


if __name__ == "__main__":
    acc = run_evaluation()
    # Weco parses this exact line format
    print(f"accuracy: {acc:.4f}")
