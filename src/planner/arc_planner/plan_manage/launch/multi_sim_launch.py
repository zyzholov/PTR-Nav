"""Compatibility wrapper.

Keep `multi_sim_launch.py` as user-facing entrypoint, but delegate to the
validated implementation in `multi_sim.launch.py` to avoid config drift.
"""

import importlib.util
import os


def generate_launch_description():
  launch_dir = os.path.dirname(__file__)
  target_file = os.path.join(launch_dir, "multi_sim.launch.py")

  spec = importlib.util.spec_from_file_location("plan_manage_multi_sim_launch", target_file)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load delegated launch file: {target_file}")

  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)

  if not hasattr(module, "generate_launch_description"):
    raise RuntimeError(f"Delegated launch file has no generate_launch_description(): {target_file}")

  return module.generate_launch_description()

