"""CPU placement for the evaluation process and its real-time runner."""

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _available_cpus():
    # Both placements must honor the mask inherited from the job scheduler.
    return sorted(os.sched_getaffinity(0))


def bind_evaluation_cpus(start: int, count: int):
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        cpus = _available_cpus()
        os.sched_setaffinity(0, cpus[start:start + count] or cpus[-1:])
