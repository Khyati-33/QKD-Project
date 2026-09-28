"""Reusable CPU inference utilities for deployment benchmarks."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from models import stack_observations


def configure_cpu(*, threads: int = 8, interop_threads: int = 2) -> None:
    """Configure bounded CPU parallelism before model execution starts."""
    torch.set_num_threads(max(1, int(threads)))
    try:
        torch.set_num_interop_threads(max(1, int(interop_threads)))
    except RuntimeError:
        # PyTorch permits changing inter-op threads only before parallel work.
        pass


def quantize_dynamic_int8(model: torch.nn.Module) -> torch.nn.Module:
    """Quantize supported linear layers while retaining graph operations in FP32."""
    model = model.cpu().eval()
    return torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8).eval()


@torch.inference_mode()
def act(model: torch.nn.Module, observation: Mapping[str, Any]) -> int:
    """Run one deterministic action with autograd fully disabled."""
    action, _, _, _ = model.act(observation, deterministic=True)
    return int(action)


@torch.inference_mode()
def batched_actions(model: torch.nn.Module,
                    observations: Sequence[Mapping[str, Any]]) -> list[int]:
    """Run independent routing decisions together for higher throughput."""
    batch = stack_observations(observations)
    logits, _ = model(batch)
    return logits.argmax(dim=-1).to(device="cpu").tolist()
