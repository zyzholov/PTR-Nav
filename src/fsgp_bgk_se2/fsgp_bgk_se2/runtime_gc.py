"""Bound periodic GC work in the dedicated terrain executable.

Imported Torch/CuPy/ROS modules and compiled geometry live for the process.
Repeated full generation scans of that graph can pause every Python callback.
Freeze the initial imports, then once after the first complete map has warmed
lazy imports/geometry. Keep collection of subsequently allocated maps, workers
and cycles enabled, with unchanged thresholds. Refcount reclamation continues
even for frozen objects; unfreeze the complete owned graph on shutdown.

This context belongs to the standalone entry point, never to a library import
or Node constructor. Existing external freeze ownership is left untouched.
"""
from contextlib import contextmanager
import gc


@contextmanager
def initialized_module_gc():
    owned=gc.get_freeze_count()==0
    if owned:
        gc.collect()
        gc.freeze()
    warmed=False
    def after_first_map():
        nonlocal warmed
        if owned and not warmed:
            gc.freeze()
            warmed=True
    try:
        yield after_first_map
    finally:
        if owned:gc.unfreeze()
