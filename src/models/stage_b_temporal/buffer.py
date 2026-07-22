"""
Reference buffer for Stage B's recursive temporal deformation.

Semantics (design_doc_v2.md Sec 2.1, confirmed-recursive, NOT re-anchored):

    buffer <- G_0                                   # after Stage A, once
    for t = 1 .. T:
        G_t <- deform(buffer, Camera_t, LiDAR_t)     # Stage B
        O_t <- splat(G_t)                            # Stage C
        buffer <- G_t                                 # write-back, recursive,
                                                        # never reset to G_0

No periodic re-anchoring, no keyframe reset -- drift is handled at the
training level via truncated BPTT (see design_doc_v2.md Sec 2.6), not by
changing this module's behavior.
"""

from dataclasses import dataclass

import torch


@dataclass
class GaussianState:
    """
    A full set of Gaussian primitives at a single timestep.

    means:      (N, 3)              mu_i
    rotations:  (N, 4)               r_i, quaternion (w, x, y, z), assumed
                                     unit-norm on entry/exit of every op
    scales:     (N, 3)              s_i
    opacities:  (N, 1)               alpha_i
    semantics:  (N, semantic_dim)     c_i  (logits, per design_doc_v2.md)

    scales/opacities/semantics are time-invariant under Stage B's update rule
    (design_doc_v2.md Sec 2.6) -- only means and rotations are deformed.
    """

    means: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    semantics: torch.Tensor

    def __post_init__(self):
        n = self.means.shape[0]
        for name, t, last_dim in [
            ("means", self.means, 3),
            ("rotations", self.rotations, 4),
            ("scales", self.scales, 3),
            ("opacities", self.opacities, 1),
            ("semantics", self.semantics, None),
        ]:
            if t.shape[0] != n:
                raise ValueError(
                    f"GaussianState field '{name}' has {t.shape[0]} rows, "
                    f"expected {n} (matching 'means')."
                )
            if last_dim is not None and t.shape[-1] != last_dim:
                raise ValueError(
                    f"GaussianState field '{name}' has last dim {t.shape[-1]}, "
                    f"expected {last_dim}."
                )

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def clone(self) -> "GaussianState":
        """Deep-copy every tensor field. Used on buffer init/write so the
        buffer never holds an aliased reference to a caller's tensors --
        this is what makes the 'is it provably G_t, not G_0' test meaningful
        rather than a tensor-identity coincidence."""
        return GaussianState(
            means=self.means.clone(),
            rotations=self.rotations.clone(),
            scales=self.scales.clone(),
            opacities=self.opacities.clone(),
            semantics=self.semantics.clone(),
        )


class ReferenceBuffer:
    """
    Holds exactly one GaussianState: the most recently written G_t (or G_0,
    before any writes). Recursive by construction -- there is no method that
    lets a caller get back to G_0 after the first write() except by having
    kept their own separate reference to it.
    """

    def __init__(self, g0: GaussianState):
        self._state = g0.clone()
        self._write_count = 0  # 0 => still holding G_0, unmodified

    def read(self) -> GaussianState:
        """Returns the buffer's current state. Does NOT clone -- callers
        must not mutate the returned object in place; use clone() first."""
        return self._state

    def write(self, g_t: GaussianState) -> None:
        if g_t.num_gaussians != self._state.num_gaussians:
            raise ValueError(
                f"Cannot write GaussianState with {g_t.num_gaussians} "
                f"Gaussians into a buffer holding {self._state.num_gaussians} "
                f"-- N_g must stay fixed across the recursion (no "
                f"re-anchoring/resampling mid-sequence)."
            )
        self._state = g_t.clone()
        self._write_count += 1

    @property
    def write_count(self) -> int:
        """0 immediately after construction (holding G_0, never written to);
        increments by 1 on every write(). Useful in tests/asserts to confirm
        recursion actually advanced rather than silently re-reading G_0."""
        return self._write_count
