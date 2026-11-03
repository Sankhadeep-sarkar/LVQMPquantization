"""
pipeline.py
-----------
End-to-end driver, generalized to run on any of the three architectures
in this project (resnet20, mobilenetv2, inceptionv3), all sharing the
identical mixed-precision + LVQ recipe:

  1. Train the chosen architecture briefly in full precision.
  2. Measure per-layer sensitivity via Hutchinson-estimated Hessian trace.
  3. Cosine-distance + elbow-method mixed precision bitwidth search.
  4. Apply those bitwidths, switch every layer's weight quantizer to LVQ,
     recalibrate BatchNorm against the now-quantized weights.
  5. Fine-tune (backprop + explicit LVQ1 competitive update) to recover
     accuracy.

Every stage's numbers -- per-epoch training curves, the sensitivity /
bitwidth-search trace, model size before & after -- are written to
results/<model_name>.json via results_logger.py, so plotting.py can
later assemble the accuracy tables, training curves, and model-size
comparison graphs across however many of the three architectures you've
actually run (including re-running resnet20 to refresh its numbers).

Run with:  python pipeline.py --model resnet20
       or: python pipeline.py --model mobilenetv2
       or: python pipeline.py --model inceptionv3

Needs: torch, torchvision. Falls back to synthetic random data if CIFAR10
can't be downloaded (no internet), purely so the *pipeline mechanics* can
still be exercised end to end -- swap in real CIFAR10 for real results.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from resnet20 import resnet20
from mobilenetv2 import mobilenetv2
from inceptionv3 import inceptionv3
from sensitivity import hutchinson_layer_trace
from bitwidth_search import search_mixed_precision
from model_size import compute_model_size, print_size_report
from results_logger import RunLog

def save_model(model, model_name, epoch, accuracy, suffix=""):
    """Save model weights to a file."""
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{model_name}_epoch{epoch}_acc{accuracy:.4f}{suffix}.pt"
    filepath = os.path.join(save_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'model_name': model_name,
    }, filepath)
    print(f"    [saved] {filepath}")
    return filepath

MODEL_REGISTRY = {
    "resnet20": resnet20,
    "mobilenetv2": mobilenetv2,
    "inceptionv3": inceptionv3,
}


def get_dataloaders(batch_size=128, data_root="./data"):
    import torchvision
    import torchvision.transforms as T

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    try:
        train_set = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
    except Exception as e:
        print(f"[warn] Could not download CIFAR10 ({e}); falling back to synthetic FakeData "
              f"so the pipeline can still be run end-to-end. Use real CIFAR10 for real results.")
        train_set = torchvision.datasets.FakeData(2000, (3, 32, 32), 10, transform=train_tf)
        test_set = torchvision.datasets.FakeData(500, (3, 32, 32), 10, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, train_set, test_set


def make_loader(dataset, batch_size, shuffle=True, drop_last=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, drop_last=drop_last)


def loss_fn(model, x, y):
    return nn.functional.cross_entropy(model(x), y)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_epochs(model, train_loader, test_loader, device, epochs, lr,
                  lvq_layers=None, weight_decay=5e-4, log_fn=None):
    """Generic training loop used both for the float baseline and for
    fine-tuning after mixed precision + LVQ quantization are applied.
    If `lvq_layers` is given, also runs the explicit LVQ1 competitive
    codebook update once per step. If `log_fn` is given, it's called
    once per epoch as log_fn(epoch, train_loss, test_acc) so the caller
    can persist a training curve for plotting."""
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    for epoch in range(epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, x, y)
            loss.backward()
            opt.step()

            if lvq_layers:
                for m in lvq_layers:
                    m.lvq_w.lvq1_update(m.weight)

            running_loss += loss.item()
            n_batches += 1
        sched.step()
        train_loss = running_loss / max(n_batches, 1)

        test_acc = None
        if log_fn is not None:
            test_acc = evaluate(model, test_loader, device)
            log_fn(epoch, train_loss, test_acc)

        acc_str = f"  test_acc={test_acc:.4f}" if test_acc is not None else ""
        print(f"    epoch {epoch + 1}/{epochs}  loss={train_loss:.4f}{acc_str}")


def recalibrate_batchnorm(model, loader, device, num_batches=50):
    """Re-estimates every BatchNorm layer's running mean/var under the
    *current* (now quantized) weights -- see pipeline.py history / README
    for why this matters: BN's running stats were fit to the float
    activation distribution, and quantizing weights shifts that
    distribution, causing a large accuracy drop right at the point
    quantization is applied unless BN is re-estimated for it.

    Uses momentum=None (cumulative moving average) during the pass, the
    standard trick for a clean one-shot re-estimation.
    """
    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    saved_momentum = [m.momentum for m in bn_layers]
    saved_running_mean = [m.running_mean.clone() for m in bn_layers]
    saved_running_var = [m.running_var.clone() for m in bn_layers]
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None

    model.train()
    seen = 0
    with torch.no_grad():
        for x, _ in loader:
            if seen >= num_batches:
                break
            model(x.to(device))
            seen += 1

    if seen == 0:
        for m, mean, var, mom in zip(bn_layers, saved_running_mean, saved_running_var, saved_momentum):
            m.running_mean.copy_(mean)
            m.running_var.copy_(var)
            m.momentum = mom
    else:
        for m, mom in zip(bn_layers, saved_momentum):
            m.momentum = mom


def apply_mixed_precision(model, chosen_bits: dict, use_lvq: bool = True):
    layers = model.quantizable_layers()
    for name, bits in chosen_bits.items():
        layers[name].set_weight_bits(bits, use_lvq=use_lvq)
    # Propagate each conv's searched weight-bitwidth to the activation
    # quantizer that follows it. activation_pairs() makes this
    # architecture-agnostic -- works identically for resnet20,
    # mobilenetv2, and inceptionv3.
    for conv, act in model.activation_pairs():
        act.set_bits(conv.weight_bits)
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="resnet20")
    parser.add_argument("--baseline-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bit-choices", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--sensitivity-samples", type=int, default=5)
    parser.add_argument("--sensitivity-batches", type=int, default=2)
    parser.add_argument("--sensitivity-batch-size", type=int, default=32,
                         help="Batch size used only for the Hessian-trace sensitivity pass. "
                              "Kept separate (and smaller by default) from --batch-size because "
                              "double-backward (create_graph=True) uses roughly 2-3x the memory "
                              "of a normal backward pass -- this is the single most memory-hungry "
                              "step in the pipeline, so it gets its own smaller batch size rather "
                              "than forcing the whole run to use a small batch size everywhere.")
    parser.add_argument("--bn-recalib-batches", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model: {args.model}   device: {device}")
    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"    GPU memory free: {free_b/1024**3:.2f} GiB / {total_b/1024**3:.2f} GiB total")
        if free_b / total_b < 0.5:
            print("    [note] less than half the GPU's memory is free (likely shared with other "
                  "processes). If you hit CUDA OOM, try: lower --batch-size and/or "
                  "--sensitivity-batch-size, lower --sensitivity-samples, or set the env var "
                  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before running.")

    log = RunLog(model_name=args.model)
    train_loader, test_loader, train_set, _ = get_dataloaders(batch_size=args.batch_size)
    sensitivity_loader = make_loader(train_set, batch_size=args.sensitivity_batch_size)

    # ---- 1. Full-precision baseline -------------------------------------
    model = MODEL_REGISTRY[args.model]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[1/5] training full-precision baseline ({n_params:,} params) ...")
    train_epochs(model, train_loader, test_loader, device, epochs=args.baseline_epochs, lr=0.1,
                 log_fn=lambda e, l, a: log.log_epoch("baseline", e, l, a))
    acc_fp32 = evaluate(model, test_loader, device)
    print(f"    baseline fp32 accuracy: {acc_fp32:.4f}")
    log.acc_fp32 = acc_fp32

    # ---- 2. Per-layer sensitivity (Hessian trace) ------------------------
    print("[2/5] estimating per-layer sensitivity (Hutchinson Hessian trace) ...")
    if device.type == "cuda":
        torch.cuda.empty_cache()  # release baseline-training's cached blocks before the
                                   # double-backward pass, which is the most memory-hungry step
    layers = model.quantizable_layers()
    sensitivity = hutchinson_layer_trace(
        model, loss_fn, iter(sensitivity_loader), layers,
        num_samples=args.sensitivity_samples, num_batches=args.sensitivity_batches,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log.sensitivity = sensitivity
    for name, s in sorted(sensitivity.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {name:30s} sensitivity={s:.4e}")
    print(f"    ... ({len(sensitivity)} layers total)")

    # ---- 3. Cosine-distance + elbow mixed-precision bitwidth search ------
    print("[3/5] searching mixed-precision bitwidths (cosine distance + elbow) ...")
    chosen_bits, trace = search_mixed_precision(sensitivity, bit_choices=tuple(args.bit_choices))
    log.bitwidth_search_trace = trace
    log.chosen_bits = chosen_bits
    print(f"    elbow point: avg_bits={trace['chosen_avg_bits']:.2f}  "
          f"cos_dist={trace['chosen_cos_dist']:.4f}  (index {trace['elbow_index']} of {len(trace['avg_bits'])})")

    # ---- 4. Apply mixed precision + LVQ weight quantization --------------
    print("[4/5] applying mixed-precision bitwidths with LVQ weight quantization ...")
    quant_layers = apply_mixed_precision(model, chosen_bits, use_lvq=True)

    acc_before_finetune_raw = evaluate(model, test_loader, device)
    print(f"    accuracy right after quantization, BEFORE BN recalibration: {acc_before_finetune_raw:.4f}")
    log.acc_post_quant_pre_bn_recal = acc_before_finetune_raw

    print("    recalibrating BatchNorm statistics against quantized weights ...")
    recalibrate_batchnorm(model, train_loader, device, num_batches=args.bn_recalib_batches)
    acc_before_finetune = evaluate(model, test_loader, device)
    print(f"    accuracy after quantization, AFTER BN recalibration (pre-finetune): {acc_before_finetune:.4f}")
    log.acc_post_quant_post_bn_recal = acc_before_finetune

    size_report = compute_model_size(model, use_lvq=True)
    print_size_report(args.model, size_report)
    log.size_fp32_mb = size_report.fp32_equivalent_mb
    log.size_quant_mb = size_report.total_mb
    log.compression_ratio = size_report.compression_ratio
    log.avg_weight_bits = size_report.avg_weight_bits

    # ---- 5. Fine-tune the quantized model (backprop + explicit LVQ1) -----
    print("[5/5] fine-tuning mixed-precision LVQ model ...")
    lvq_modules = [m for m in quant_layers.values() if m.use_lvq]
    train_epochs(model, train_loader, test_loader, device, epochs=args.finetune_epochs, lr=0.01,
                 lvq_layers=lvq_modules,
                 log_fn=lambda e, l, a: log.log_epoch("finetune", e, l, a))
    acc_final = evaluate(model, test_loader, device)
    log.acc_final = acc_final

    print("\n===== summary =====")
    print(f"model:                              {args.model}")
    print(f"fp32 baseline accuracy:             {acc_fp32:.4f}")
    print(f"post-quantization (pre-finetune):   {acc_before_finetune:.4f}")
    print(f"post-finetune (mixed-prec + LVQ):   {acc_final:.4f}")
    print(f"fp32 model size:                    {size_report.fp32_equivalent_mb:.3f} MB")
    print(f"mixed-precision + LVQ model size:   {size_report.total_mb:.3f} MB")
    print(f"compression ratio:                  {size_report.compression_ratio:.2f}x")

    path = log.save()
    print(f"\nresults saved to {path} -- run plotting.py to generate comparison graphs/tables.")
        # Save the final model
    save_model(model, args.model, args.finetune_epochs, acc_final, "_mixed_precision_lvq")


if __name__ == "__main__":
    main()