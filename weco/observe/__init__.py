"""Weco Observe — observability SDK for external optimization loops.

Usage:
    from weco.observe import WecoObserver

    obs = WecoObserver()
    run = obs.create_run(
        name="val_bpb sweep v3",
        source_code={"train.py": open("train.py").read()},
        primary_metric="val_bpb",
        maximize=False,
    )

    run.log_step(
        step=i,
        status="completed",
        description="Added RMSNorm",
        metrics={"val_bpb": 1.03, "memory_gb": 34.5},
        code={"train.py": open("train.py").read()},
    )

"""

from .observer import WecoObserver, ObserveRun

__all__ = ["WecoObserver", "ObserveRun"]
