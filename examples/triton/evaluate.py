import sys
import pathlib
import importlib
import importlib.util
import traceback
import torch
import torch.nn as nn
from triton.testing import do_bench


########################################################
# Baseline
########################################################
class Model(nn.Module):
    """
    Simple model that performs a Swish activation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        return x * torch.sigmoid(x)


########################################################
# Benchmark
########################################################
def load_module_from_path(module_path: str, add_to_sys_modules: bool = False):
    # Clean out all old compiled extensions to prevent namespace collisions during build
    module_path = pathlib.Path(module_path)
    name = module_path.stem
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    if add_to_sys_modules:
        sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def get_inputs(batch_size, dim, device):
    return torch.randn(batch_size, dim, device=device, dtype=torch.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    # benchmarking parameters
    n_correctness_trials = 10
    correctness_tolerance = 1e-5
    warmup_ms = 100
    rep_ms = 500

    # input parameters
    batch_size = 2000
    dim = 16384

    # load solution module
    try:
        torch.manual_seed(0)
        solution_module = load_module_from_path(args.path, add_to_sys_modules=False)
        solution_model = solution_module.Model().to("cuda")
        assert isinstance(solution_model, nn.Module)
    except Exception:
        print(f"Candidate module initialization failed: {traceback.format_exc()}")
        exit(1)

    torch.manual_seed(0)
    baseline_model = Model().to("cuda")

    # measure correctness
    max_diff_avg = 0
    for _ in range(n_correctness_trials):
        inputs = get_inputs(batch_size=batch_size, dim=dim, device="cuda")
        with torch.no_grad():
            optimized_output = solution_model(inputs)
            if torch.isnan(optimized_output).any():
                print("Incorrect solution: NaN detected in optimized model output")
            if torch.isinf(optimized_output).any():
                print("Incorrect solution: Inf detected in optimized model output")
            baseline_output = baseline_model(inputs)
            max_diff_avg += torch.max(torch.abs(optimized_output - baseline_output))
    max_diff_avg /= n_correctness_trials
    print(f"max float diff between values of baseline and optimized model: {max_diff_avg}")
    if max_diff_avg > correctness_tolerance:
        print("Incorrect solution: max float diff is too high")

    # measure performance
    inputs = get_inputs(batch_size=batch_size, dim=dim, device="cuda")
    t_avg_baseline = do_bench(lambda: baseline_model(inputs), warmup=warmup_ms, rep=rep_ms)
    print(f"baseline time: {t_avg_baseline:.2f}ms")
    t_avg_optimized = do_bench(lambda: solution_model(inputs), warmup=warmup_ms, rep=rep_ms)
    print(f"optimized time: {t_avg_optimized:.2f}ms")
    print(f"speedup: {t_avg_baseline / t_avg_optimized:.2f}x")
