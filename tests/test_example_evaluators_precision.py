"""The CUDA/Triton example evaluators must compute the baseline in pinned FP32,
even when the in-process candidate enables TF32 globally."""

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"weco_example_{name}_evaluate", EXAMPLES / name / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _candidate_enables_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


NEW_API = hasattr(torch.backends.cuda.matmul, "fp32_precision")  # PyTorch >= 2.9


@pytest.fixture(autouse=True)
def _restore_flags():
    if NEW_API:
        backends = (torch.backends.cuda.matmul, torch.backends.cudnn.conv, torch.backends.cudnn.rnn)
        saved = [backend.fp32_precision for backend in backends]
        yield
        for backend, precision in zip(backends, saved):
            backend.fp32_precision = precision
        return
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[0], saved[1]
    torch.set_float32_matmul_precision(saved[2])


@pytest.mark.parametrize("name", ["cuda", "triton"])
def test_tf32_candidate_does_not_change_baseline_output(name):
    ev = _load(name)
    torch.manual_seed(0)
    a = torch.randn(512, 1024, device="cuda")
    b = torch.randn(1024, 512, device="cuda")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    reference = a @ b

    _candidate_enables_tf32()
    with ev.fp32_reference():
        pinned = a @ b
    assert torch.equal(pinned, reference)
    # the candidate's own setting is restored after the baseline call
    assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "high"


@pytest.mark.skipif(not NEW_API, reason="fp32_precision API needs PyTorch >= 2.9")
@pytest.mark.parametrize("name", ["cuda", "triton"])
def test_new_api_tf32_candidate_does_not_change_baseline_output(name):
    ev = _load(name)
    torch.manual_seed(0)
    a = torch.randn(512, 1024, device="cuda")
    b = torch.randn(1024, 512, device="cuda")

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    reference = a @ b

    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    with ev.fp32_reference():
        pinned = a @ b
    assert torch.equal(pinned, reference)
    # the candidate's own setting is restored after the baseline call
    assert torch.backends.cuda.matmul.fp32_precision == "tf32"
    assert torch.backends.cudnn.conv.fp32_precision == "tf32"


def test_cuda_example_baseline_model_is_pinned():
    ev = _load("cuda")
    torch.manual_seed(0)
    model = ev.Model(n_embd=768, n_head=8, attn_pdrop=0.0, resid_pdrop=0.0, max_seqlen=512).to("cuda").eval()
    x = torch.randn(4, 256, 768, device="cuda")
    with torch.no_grad():
        with ev.fp32_reference():
            reference = model(x)
        _candidate_enables_tf32()
        leaked = model(x)
        with ev.fp32_reference():
            pinned = model(x)
    assert torch.equal(pinned, reference)
    # positive control: on TF32-capable GPUs the unpinned call really differs
    if torch.cuda.get_device_capability()[0] >= 8:
        assert not torch.equal(leaked, reference)
