"""
lvq.py
------
Learning Vector Quantization (LVQ) applied to *weight quantization*.

Standard LVQ (Kohonen) is a supervised, prototype-based classifier: a set
of codebook (prototype) vectors live in input space, each input pulls its
nearest prototype towards it (and, in supervised variants LVQ2/LVQ3, pushes
away the nearest wrong-class prototype). There are no class labels for
convolution weights, so we use the unsupervised competitive-learning core
of LVQ1 as a *trainable vector-quantization codebook*: it is exactly LVQ's
winner-take-all update rule, just without the "attract same class / repel
different class" split, since weights don't carry class labels. This
gives a much better weight quantizer than plain uniform rounding, because
the codebook can move its prototypes to sit wherever weight mass actually
concentrates (a non-uniform quantization grid, learned from data).

How it's used in this project:
  - Once the mixed-precision search (bitwidth_search.py) has picked a
    bitwidth b_l for layer l, we set the LVQ codebook size for that layer
    to 2**b_l prototypes -- so the *number of quantization levels* is what
    mixed precision actually controls, and LVQ decides *where* those
    levels sit.
  - Weights are reshaped into small groups of size `vec_dim` (real vector
    quantization, not scalar), each group is replaced by its nearest
    codebook prototype (forward pass, straight-through gradients for the
    backward pass so the underlying float weights keep training), and the
    codebook itself is refined either by ordinary backprop or by an
    explicit LVQ1 competitive update (see `lvq1_update`), which can be
    called once per training step for a more "textbook" LVQ behaviour.
"""
import torch
import torch.nn as nn


class LVQVectorQuantizer(nn.Module):
    """Vector-quantizes a weight tensor with a learnable LVQ codebook.

    Args:
        num_levels: number of prototypes in the codebook (== 2**bitwidth).
        vec_dim: how many scalar weights form one vector that gets jointly
            quantized. vec_dim=1 reduces to scalar (per-weight) VQ.
        lvq_lr: learning rate used by the explicit LVQ1 competitive update.
    """

    def __init__(self, num_levels: int = 16, vec_dim: int = 4, lvq_lr: float = 0.05):
        super().__init__()
        self.num_levels = num_levels
        self.vec_dim = vec_dim
        self.lvq_lr = lvq_lr
        self.codebook = nn.Parameter(torch.randn(num_levels, vec_dim) * 0.05)
        self._initialized = False

    def set_num_levels(self, num_levels: int):
        """Re-sizes the codebook (used after the bitwidth search updates
        this layer's assigned bitwidth). Re-initializes from scratch since
        the number of prototypes changed."""
        if num_levels == self.num_levels:
            return
        self.num_levels = num_levels
        device = self.codebook.device
        self.codebook = nn.Parameter(torch.randn(num_levels, self.vec_dim, device=device) * 0.05)
        self._initialized = False

    @torch.no_grad()
    def _init_codebook(self, vectors, lloyd_iters: int = 25):
        # Farthest-point sampling is used ONLY as a seed here (it's good
        # at guaranteeing k diverse, non-degenerate starting points and
        # avoiding duplicate/collapsed prototypes). On its own, though, it
        # is a bad *final* init for quantization: it explicitly maximizes
        # spread, i.e. it prefers outliers -- for a roughly bell-shaped
        # weight distribution that starves the codebook of levels near
        # zero, where most of the weight mass actually lives, causing
        # large initial quantization error (very visible right after
        # switching a layer into quantized mode, before any fine-tuning
        # has had a chance to fix it).
        #
        # So: seed with farthest-point, then refine with a few iterations
        # of standard Lloyd's k-means (nearest-assignment + centroid
        # update). This pulls every prototype to the *mean of the data
        # points that actually pick it*, i.e. density-weighted placement
        # -- the thing that minimizes quantization MSE for a fixed number
        # of levels.
        n = vectors.shape[0]
        k = min(self.num_levels, n)
        idx = [torch.randint(0, n, (1,)).item()]
        chosen = vectors[idx[0]].unsqueeze(0)
        for _ in range(k - 1):
            d = torch.cdist(vectors, chosen).min(dim=1).values
            next_idx = torch.argmax(d).item()
            idx.append(next_idx)
            chosen = vectors[idx]
        centroids = vectors[idx].clone()
        if k < self.num_levels:  # pad with jittered copies if too few unique points
            pad = centroids[torch.randint(0, k, (self.num_levels - k,))]
            pad = pad + 0.01 * torch.randn_like(pad)
            centroids = torch.cat([centroids, pad], dim=0)

        # Lloyd's algorithm refinement.
        for _ in range(lloyd_iters):
            dists = torch.cdist(vectors, centroids)
            assign = torch.argmin(dists, dim=1)
            new_centroids = centroids.clone()
            moved = False
            for level in range(self.num_levels):
                mask = assign == level
                if mask.any():
                    new_centroids[level] = vectors[mask].mean(dim=0)
                else:
                    # Empty cluster: re-seed it at the point currently
                    # farthest from its own nearest centroid, so no
                    # codebook level goes permanently unused (a dead
                    # prototype wastes a bit of precision the mixed
                    # precision search paid for).
                    d = torch.cdist(vectors, new_centroids).min(dim=1).values
                    new_centroids[level] = vectors[torch.argmax(d)]
                    moved = True
            if not moved and torch.allclose(new_centroids, centroids, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids

        self.codebook.data.copy_(centroids)
        self._initialized = True

    def _reshape_to_vectors(self, w):
        flat = w.reshape(-1)
        pad = (-flat.numel()) % self.vec_dim
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        return flat.view(-1, self.vec_dim), pad

    def quantize(self, w: torch.Tensor):
        """Returns (quantized_weight_same_shape_as_w, assignment_indices)."""
        orig_shape = w.shape
        vectors, pad = self._reshape_to_vectors(w)

        if not self._initialized:
            self._init_codebook(vectors.detach())

        # Nearest-prototype assignment (competitive "winner" selection).
        dists = torch.cdist(vectors.detach(), self.codebook.detach())  # [N, num_levels]
        assign = torch.argmin(dists, dim=1)  # [N]

        selected = self.codebook[assign]  # [N, vec_dim], differentiable wrt codebook

        # Straight-through estimator: forward value == selected; gradient
        # wrt the underlying float weight passes straight through.
        quant_vectors = vectors + (selected - vectors).detach()
        # Also let gradient flow into the codebook entries that were
        # actually used, via a zero-valued term:
        quant_vectors = quant_vectors + (selected - selected.detach())

        flat = quant_vectors.reshape(-1)
        if pad:
            flat = flat[: flat.numel() - pad]
        return flat.view(orig_shape), assign

    @torch.no_grad()
    def lvq1_update(self, w: torch.Tensor):
        """Explicit LVQ1-style competitive update (optional -- call once
        per training step from the training loop for 'classic' LVQ
        dynamics in addition to / instead of backprop through the
        codebook). Moves each winning prototype toward the mean of the
        vectors that picked it as their nearest neighbour."""
        vectors, _ = self._reshape_to_vectors(w.detach())
        dists = torch.cdist(vectors, self.codebook)
        assign = torch.argmin(dists, dim=1)
        for level in range(self.num_levels):
            mask = assign == level
            if mask.any():
                target = vectors[mask].mean(dim=0)
                self.codebook.data[level] += self.lvq_lr * (target - self.codebook.data[level])

    def forward(self, w: torch.Tensor):
        q, _ = self.quantize(w)
        return q