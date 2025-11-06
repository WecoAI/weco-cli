import sys
import os
import shutil
import pathlib
import importlib
import importlib.util
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from triton.testing import do_bench


########################################################
# Baseline
########################################################
class Model(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


########################################################
# Benchmark
########################################################
def load_module_from_path(module_path: str, add_to_sys_modules: bool = False):
    module_path = pathlib.Path(module_path)
    name = module_path.stem
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    if add_to_sys_modules:
        sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def get_inputs(batch_size, seq_len, n_embd, device):
    return torch.randn(batch_size, seq_len, n_embd, device=device, dtype=torch.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    # setup local cache for PyTorch extensions
    cache_dir = pathlib.Path.cwd() / ".weco-temp/torch_extensions"
    shutil.rmtree(cache_dir.parent, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache_dir)

    # benchmarking parameters
    n_correctness_trials = 10
    correctness_tolerance = 1e-5
    warmup_ms = 1e3
    rep_ms = 5 * 1e3

    # init parameters
    max_seqlen = 512
    seq_len = 256
    n_embd = 768
    n_head = 8
    # turn off dropout to measure correctness
    attn_pdrop = 0.0
    resid_pdrop = 0.0

    # input parameters
    batch_size = 32

    # load solution module
    try:
        torch.manual_seed(0)
        solution_module = load_module_from_path(args.path, add_to_sys_modules=False)
        solution_model = solution_module.Model(
            n_embd=n_embd, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, max_seqlen=max_seqlen
        ).to("cuda")
        assert isinstance(solution_model, nn.Module)
    except Exception:
        print(f"Candidate module initialization failed: {traceback.format_exc()}")
        exit(1)

    torch.manual_seed(0)
    baseline_model = Model(
        n_embd=n_embd, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, max_seqlen=max_seqlen
    ).to("cuda")

    # measure correctness
    max_diff_avg = 0
    for _ in range(n_correctness_trials):
        inputs = get_inputs(batch_size=batch_size, seq_len=seq_len, n_embd=n_embd, device="cuda")
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
    inputs = get_inputs(batch_size=batch_size, seq_len=seq_len, n_embd=n_embd, device="cuda")
    t_avg_baseline = do_bench(lambda: baseline_model(inputs), warmup=warmup_ms, rep=rep_ms)
    print(f"baseline time: {t_avg_baseline:.2f}ms")
    t_avg_optimized = do_bench(lambda: solution_model(inputs), warmup=warmup_ms, rep=rep_ms)
    print(f"optimized time: {t_avg_optimized:.2f}ms")
    print(f"speedup: {t_avg_baseline / t_avg_optimized:.2f}x")

    # clean up
    shutil.rmtree(cache_dir.parent, ignore_errors=True)
