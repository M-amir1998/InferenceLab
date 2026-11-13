"""InferenceLab: reproducible LLM inference engineering platform.

InferenceLab is the control/data-plane layer *around* production inference engines
(vLLM, SGLang, TensorRT-LLM). It does not reimplement those engines; it turns models,
deployments, traffic, routing, SLOs and cost into reproducible engineering experiments.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0.dev0"
