"""TQP frontends that compile SQL into internal plans."""

from tpch_torch.frontend.auto import compile_auto_plan
from tpch_torch.frontend.sirius import compile_sirius_plan
from tpch_torch.frontend.substrait import compile_substrait_plan

__all__ = ["compile_auto_plan", "compile_sirius_plan", "compile_substrait_plan"]
