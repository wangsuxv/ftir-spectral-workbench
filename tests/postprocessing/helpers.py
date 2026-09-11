"""Create known, confirmed baseline parents through the existing public workflow."""

from __future__ import annotations

import numpy as np

from ftir_baseline.config import CoarseBaselineConfig
from ftir_workbench.batch.importing import import_sources
from ftir_workbench.batch.models import BatchWorkspace
from ftir_workbench.batch.service import preview_coarse
from ftir_workbench.batch.state import confirm_coarse, set_coarse_draft, skip_fine
from tests.batch.test_importing import table_bytes


def confirmed_workspace(
    x: np.ndarray | None = None, y: np.ndarray | None = None,
    *, workspace: BatchWorkspace | None = None, name: str = "synthetic.csv",
    skip: bool = True,
) -> tuple[BatchWorkspace, str]:
    """The analytical fixture uses no baseline subtraction, then explicit skip."""
    x = np.linspace(1800.0, 900.0, 181) if x is None else x
    y = (0.2 + 0.8 * np.exp(-((x - 1450.0) / 30.0) ** 2)
         + 0.03 * np.sin(x / 5)) if y is None else y
    workspace = BatchWorkspace() if workspace is None else workspace
    sid = import_sources(workspace, [(name, table_bytes(x, y, labels=("sample",)))])[0]
    set_coarse_draft(workspace, sid, CoarseBaselineConfig(method="none"))
    preview_coarse(workspace, sid)
    confirm_coarse(workspace, sid)
    if skip:
        skip_fine(workspace, sid)
    return workspace, sid
