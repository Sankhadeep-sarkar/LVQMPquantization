"""
results_logger.py
------------------
Persists per-epoch training curves and per-run summary metrics (final
accuracy at each pipeline stage, model size, chosen bitwidths, the
cosine-distance/elbow search trace) to a JSON file per model, under
`results/<model_name>.json`.

Running the pipeline again for the same model overwrites its file (that's
the "recalculate" case: rerun resnet20, its results.json gets replaced
with the fresh numbers). Running it for a different model adds a new
file. plotting.py then reads every file present in `results/` and
aggregates whatever has actually been run -- it never invents numbers for
a model that hasn't been run yet.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

RESULTS_DIR = "results"


@dataclass
class RunLog:
    model_name: str
    baseline_epoch_log: List[dict] = field(default_factory=list)   # [{epoch, train_loss, test_acc}, ...]
    finetune_epoch_log: List[dict] = field(default_factory=list)
    sensitivity: Dict[str, float] = field(default_factory=dict)
    bitwidth_search_trace: dict = field(default_factory=dict)      # avg_bits[], cos_dist[], elbow_index
    chosen_bits: Dict[str, int] = field(default_factory=dict)
    acc_fp32: float = None
    acc_post_quant_pre_bn_recal: float = None
    acc_post_quant_post_bn_recal: float = None
    acc_final: float = None
    size_fp32_mb: float = None
    size_quant_mb: float = None
    compression_ratio: float = None
    avg_weight_bits: float = None

    def log_epoch(self, phase: str, epoch: int, train_loss: float, test_acc: float):
        entry = {"epoch": epoch, "train_loss": train_loss, "test_acc": test_acc}
        if phase == "baseline":
            self.baseline_epoch_log.append(entry)
        elif phase == "finetune":
            self.finetune_epoch_log.append(entry)
        else:
            raise ValueError(f"unknown phase: {phase}")

    def save(self, results_dir: str = RESULTS_DIR):
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"{self.model_name}.json")
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
        print(f"    [results_logger] saved {path}")
        return path

    @staticmethod
    def load(model_name: str, results_dir: str = RESULTS_DIR) -> "RunLog":
        path = os.path.join(results_dir, f"{model_name}.json")
        with open(path) as f:
            data = json.load(f)
        return RunLog(**data)


def list_available_results(results_dir: str = RESULTS_DIR) -> List[str]:
    if not os.path.isdir(results_dir):
        return []
    return sorted(
        fname[:-5] for fname in os.listdir(results_dir)
        if fname.endswith(".json")
    )
