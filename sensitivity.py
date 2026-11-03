"""
sensitivity.py
--------------
Per-layer sensitivity analysis for mixed precision quantization, following
the HAWQ family of methods (Dong et al., "HAWQ: Hessian AWare Quantization"):
quantization error hurts most in layers whose loss landscape is most
curved, so we estimate the trace of the loss Hessian restricted to each
layer's weights (its "average curvature") and treat that as the layer's
sensitivity score.

The full Hessian is far too large to form explicitly, so we use
Hutchinson's stochastic trace estimator with Pearlmutter's Hessian-vector
product trick (both are standard, exact-in-expectation techniques -- no
approximation of the Hessian itself is needed, only of its trace):

    tr(H) ≈ (1/m) * sum_{i=1}^{m} v_i^T H v_i ,   v_i ~ Rademacher(+-1)

and H v is obtained without ever materializing H, via double
backpropagation:  H v = d/dw ( (dL/dw)^T v ).
"""
import torch

# torch.cuda.OutOfMemoryError only exists from torch>=2.0; older versions
# raise a plain RuntimeError whose message contains "out of memory".
# Catching both keeps the OOM backoff below working across torch versions.
_OOM_EXCEPTIONS = (RuntimeError,)
if hasattr(torch.cuda, "OutOfMemoryError"):
    _OOM_EXCEPTIONS = (torch.cuda.OutOfMemoryError, RuntimeError)


def _is_oom(e: Exception) -> bool:
    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


@torch.no_grad()
def _rademacher_like(p):
    return torch.randint(0, 2, p.shape, device=p.device, dtype=p.dtype) * 2 - 1


def _one_batch_traces(model, loss_fn, x, y, params, param_names, num_samples):
    """Runs the double-backward Hessian-trace estimate for a single batch.
    Split out so it can be retried on a smaller sub-batch if it OOMs."""
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model, x, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    batch_traces = {name: [] for name in param_names}
    for _ in range(num_samples):
        vs = [_rademacher_like(p) for p in params]
        gv = sum((g * v).sum() for g, v in zip(grads, vs))
        hvs = torch.autograd.grad(gv, params, retain_graph=True)
        for name, v, hv in zip(param_names, vs, hvs):
            batch_traces[name].append((hv * v).sum().item())
        del vs, gv, hvs

    # Explicitly drop every tensor holding onto this batch's (expensive,
    # double-backward) autograd graph before returning, and release
    # PyTorch's cached-but-unallocated CUDA blocks back to the allocator.
    # create_graph=True graphs are ~2-3x the memory of a normal backward
    # pass (they retain everything needed to differentiate through the
    # backward pass itself); on a small/shared GPU this is the single
    # most memory-hungry step in the whole pipeline, so being prompt
    # about freeing it matters far more here than anywhere else in the
    # codebase.
    del loss, grads
    if x.is_cuda:
        torch.cuda.empty_cache()
    return batch_traces


def hutchinson_layer_trace(model, loss_fn, data_iter, layers: dict, num_samples: int = 10,
                            num_batches: int = 3, min_batch_size: int = 4):
    """
    Estimate the Hessian trace of `loss_fn` restricted to each layer's
    weight tensor.

    Args:
        model: the network (must be in train() mode with grads enabled).
        loss_fn: callable(model, x, y) -> scalar loss.
        data_iter: iterable yielding (x, y) batches.
        layers: dict {layer_name: module}, module must have `.weight`.
        num_samples: number of Rademacher probe vectors per batch.
        num_batches: number of data batches to average the estimate over.
        min_batch_size: if a batch OOMs, it's automatically retried on
            progressively smaller halves (this is the memory-heaviest step
            in the pipeline -- double-backward graphs run ~2-3x a normal
            backward pass). Gives up on a given batch (skips it, moves to
            the next) once a half-split would go below this size, rather
            than looping forever.

    Returns:
        dict {layer_name: sensitivity_score} (mean Hessian trace estimate).
    """
    device = next(model.parameters()).device
    weights = {name: m.weight for name, m in layers.items()}
    params = list(weights.values())
    param_names = list(weights.keys())
    traces = {name: [] for name in layers}

    def run_with_oom_backoff(x, y):
        """Try the full sub-batch; on CUDA OOM, clear the cache and retry
        on progressively smaller halves instead of crashing the whole
        sensitivity pass."""
        try:
            return _one_batch_traces(model, loss_fn, x, y, params, param_names, num_samples)
        except _OOM_EXCEPTIONS as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            n = x.shape[0]
            if n <= min_batch_size:
                print(f"    [sensitivity] OOM even at batch size {n} -- skipping this micro-batch "
                      f"(consider lowering --sensitivity-samples or --batch-size).")
                return {name: [] for name in param_names}
            print(f"    [sensitivity] OOM at batch size {n}, retrying at {n // 2} "
                  f"(set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation further)")
            half = n // 2
            merged = {name: [] for name in param_names}
            for xs, ys in [(x[:half], y[:half]), (x[half:], y[half:])]:
                sub = run_with_oom_backoff(xs, ys)
                for name in param_names:
                    merged[name].extend(sub[name])
            return merged

    batch_count = 0
    for x, y in data_iter:
        if batch_count >= num_batches:
            break
        x, y = x.to(device), y.to(device)

        batch_traces = run_with_oom_backoff(x, y)
        for name in param_names:
            traces[name].extend(batch_traces[name])

        batch_count += 1

    return {name: (sum(vals) / len(vals) if vals else 0.0) for name, vals in traces.items()}