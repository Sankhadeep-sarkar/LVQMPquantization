"""
bitwidth_search.py
-------------------
Turns a per-layer Hessian-sensitivity vector into a per-layer bitwidth
assignment, using cosine distance + the elbow method, as requested:

  1. Sensitivity vector s = [s_1, ..., s_L] comes from sensitivity.py
     (Hutchinson-estimated Hessian trace per layer). Higher s_l means
     layer l's loss is more curved w.r.t. its weights -> that layer should
     get MORE bits (it is more sensitive to quantization error).

  2. For a budget sweep (a range of target average bitwidths, e.g.
     2 -> 8), build a *candidate bitwidth vector* b(budget) by allocating
     bits proportionally to normalized rank/magnitude of sensitivity,
     rounded to the nearest allowed bit choice (e.g. {2,3,4,5,6,7,8}).
     This gives one candidate bitwidth vector per budget point.

  3. For every candidate, compute the cosine distance between the
     candidate bitwidth vector and the sensitivity vector:

        cos_dist(b, s) = 1 - (b . s) / (||b|| ||s||)

     A small cosine distance means the *shape* of the bitwidth
     allocation tracks the *shape* of the sensitivity profile well
     (proportionally more bits where the Hessian says it matters more).

  4. Plot cosine distance (fidelity to sensitivity) against average
     bitwidth (compression cost) across the budget sweep, and apply the
     elbow method (max-distance-from-chord / "kneedle" heuristic) to find
     the knee: the point past which spending more bits buys rapidly
     diminishing improvement in how well the allocation matches
     sensitivity. That knee's bitwidth vector is the chosen mixed
     precision configuration.
"""
import math
from typing import Dict, List, Sequence


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    cos_sim = dot / (na * nb)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


def _normalize01(vals: Sequence[float]) -> List[float]:
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [0.5 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def _nearest_allowed(x: float, choices: Sequence[int]) -> int:
    return min(choices, key=lambda c: abs(c - x))


def build_candidate(sensitivity: Dict[str, float], target_avg_bits: float,
                     bit_choices: Sequence[int]) -> Dict[str, int]:
    """Allocate bits proportionally to normalized sensitivity, so that the
    *average* bitwidth across layers is close to `target_avg_bits`, then
    snap every value to the nearest allowed bit choice."""
    names = list(sensitivity.keys())
    raw = [sensitivity[n] for n in names]
    norm = _normalize01(raw)  # in [0, 1], 0 = least sensitive, 1 = most

    lo, hi = min(bit_choices), max(bit_choices)
    # Center the allocation on target_avg_bits: spread +/- around it in
    # proportion to normalized sensitivity, then clip into [lo, hi].
    half_range = min(target_avg_bits - lo, hi - target_avg_bits)
    half_range = max(half_range, 0.0)
    raw_bits = {
        n: target_avg_bits + (norm[i] - 0.5) * 2 * half_range
        for i, n in enumerate(names)
    }
    return {n: _nearest_allowed(v, bit_choices) for n, v in raw_bits.items()}


def elbow_index(xs: Sequence[float], ys: Sequence[float]) -> int:
    """Classic 'kneedle' elbow detector: normalize both axes to [0,1],
    connect the first and last point with a straight line (the chord),
    and return the index of the point with maximum perpendicular distance
    from that chord. Works for a monotonically decreasing ys(xs) curve,
    which is what cosine distance vs. average bitwidth looks like here
    (more bits -> allocation can track sensitivity more faithfully ->
    lower cosine distance, with diminishing returns)."""
    n = len(xs)
    if n < 3:
        return 0
    nx = _normalize01(xs)
    ny = _normalize01(ys)
    x1, y1 = nx[0], ny[0]
    x2, y2 = nx[-1], ny[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = math.sqrt(dx * dx + dy * dy) or 1e-12

    best_i, best_d = 0, -1.0
    for i in range(n):
        # perpendicular distance from point i to the chord (x1,y1)-(x2,y2)
        d = abs(dy * nx[i] - dx * ny[i] + x2 * y1 - y2 * x1) / norm
        if d > best_d:
            best_d = d
            best_i = i
    return best_i


def search_mixed_precision(sensitivity: Dict[str, float],
                            bit_choices: Sequence[int] = (2, 3, 4, 5, 6, 8),
                            num_budget_points: int = 13):
    """Full search: sweep target average bitwidth, score each candidate by
    cosine distance to the sensitivity vector, elbow-pick the budget.

    Returns:
        chosen_bits: dict {layer_name: bitwidth}
        trace: dict with 'avg_bits', 'cos_dist', 'elbow_index' for
               plotting/diagnostics.
    """
    names = list(sensitivity.keys())
    sens_vec = [sensitivity[n] for n in names]
    sens_norm = _normalize01(sens_vec)  # compare shapes on a common scale

    lo, hi = min(bit_choices), max(bit_choices)
    budgets = [lo + (hi - lo) * i / (num_budget_points - 1) for i in range(num_budget_points)]

    candidates = []
    avg_bits_list = []
    cos_dist_list = []
    for budget in budgets:
        cand = build_candidate(sensitivity, budget, bit_choices)
        bits_vec = [cand[n] for n in names]
        bits_norm = _normalize01(bits_vec) if max(bits_vec) != min(bits_vec) else [0.5] * len(bits_vec)
        d = cosine_distance(bits_norm, sens_norm)
        avg = sum(bits_vec) / len(bits_vec)
        candidates.append(cand)
        avg_bits_list.append(avg)
        cos_dist_list.append(d)

    # Sort by avg_bits so the curve is monotonic for the elbow detector.
    order = sorted(range(len(avg_bits_list)), key=lambda i: avg_bits_list[i])
    xs = [avg_bits_list[i] for i in order]
    ys = [cos_dist_list[i] for i in order]
    ordered_candidates = [candidates[i] for i in order]

    idx = elbow_index(xs, ys)
    chosen_bits = ordered_candidates[idx]

    trace = {
        "avg_bits": xs,
        "cos_dist": ys,
        "elbow_index": idx,
        "chosen_avg_bits": xs[idx],
        "chosen_cos_dist": ys[idx],
    }
    return chosen_bits, trace
