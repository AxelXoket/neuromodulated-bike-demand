"""Four small networks that differ only in how the state of the day reaches them.

- no-state: the state is not given at all (how much does it matter?).
- concat: the state is appended to the input, the usual way.
- shift: the state adds a learned offset to every hidden unit of every layer, h = relu(W x + b + B s).
- film: the state also scales every hidden unit (FiLM, Perez et al. 2018):
  h = relu((W x + b) * gamma(s) + beta(s)), gamma(s) = 1 + A s, beta(s) = B s.
A and B start at zero, so shift and film start as the no-state network and learn how far the state should bend it.
shift separates the two parts of FiLM: anything film gains over shift comes from the scaling.
"""
from __future__ import annotations

import torch
from torch import nn

KINDS = ("no-state", "concat", "shift", "film")
DEPTH = 2
REFERENCE_WIDTH = 128  # the concat network's width; the others get the width that matches its parameter count


class Modulation(nn.Module):
    def __init__(self, state_dim: int, width: int, scale: bool):
        super().__init__()
        self.gamma = nn.Linear(state_dim, width, bias=False) if scale else None
        self.beta = nn.Linear(state_dim, width, bias=False)
        for layer in (self.gamma, self.beta):
            if layer is not None:
                nn.init.zeros_(layer.weight)

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return (h if self.gamma is None else h * (1 + self.gamma(s))) + self.beta(s)


class Net(nn.Module):
    def __init__(self, kind: str, input_dim: int, state_dim: int, width: int):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
        self.kind = kind
        first = input_dim + state_dim if kind == "concat" else input_dim
        self.hidden = nn.ModuleList([nn.Linear(first if i == 0 else width, width) for i in range(DEPTH)])
        modulated = kind in ("shift", "film")
        self.modulation = nn.ModuleList([Modulation(state_dim, width, scale=kind == "film") for _ in range(DEPTH)]) \
            if modulated else None
        self.out = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, s], dim=1) if self.kind == "concat" else x
        for i, layer in enumerate(self.hidden):
            h = layer(h)
            if self.modulation is not None:
                h = self.modulation[i](h, s)
            h = torch.relu(h)
        return self.out(h).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def matched_width(kind: str, input_dim: int, state_dim: int, target: int) -> int:
    """The width whose parameter count is closest to target."""
    def count(width: int) -> int:
        return parameter_count(Net(kind, input_dim, state_dim, width))
    return min(range(8, 2 * REFERENCE_WIDTH + 1), key=lambda width: abs(count(width) - target))


def widths(input_dim: int, state_dim: int) -> dict[str, int]:
    target = parameter_count(Net("concat", input_dim, state_dim, REFERENCE_WIDTH))
    return {kind: REFERENCE_WIDTH if kind == "concat" else matched_width(kind, input_dim, state_dim, target)
            for kind in KINDS}
