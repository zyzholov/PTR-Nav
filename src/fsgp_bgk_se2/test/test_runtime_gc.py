"""Use child interpreters: GC policy in the pytest host must remain untouched."""
import subprocess
import sys


def check(code):
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True,timeout=15)


def test_import_graph_freeze_keeps_new_cycles_collectable_and_restores_on_failure():
    check('''
import gc,weakref
from fsgp_bgk_se2.runtime_gc import initialized_module_gc
gc.unfreeze();thresholds=gc.get_threshold();enabled=gc.isenabled()
class Cycle:pass
try:
 with initialized_module_gc() as warmed:
  assert gc.get_freeze_count()>0
  assert gc.get_threshold()==thresholds and gc.isenabled()==enabled
  lazy=Cycle();lazy.self=lazy
  warmed();frozen=gc.get_freeze_count()
  value=Cycle();value.self=value
  warmed();assert gc.get_freeze_count()==frozen
  del value;del lazy
  value=Cycle();value.self=value;ref=weakref.ref(value);del value
  gc.collect();assert ref() is None
  raise ValueError('shutdown')
except ValueError:pass
assert gc.get_freeze_count()==0
assert gc.get_threshold()==thresholds and gc.isenabled()==enabled
''')


def test_existing_external_freeze_is_not_released_or_extended_by_nested_users():
    check('''
import gc
from fsgp_bgk_se2.runtime_gc import initialized_module_gc
gc.collect();gc.freeze();original=gc.get_freeze_count()
try:
 with initialized_module_gc() as warmed:
  assert gc.get_freeze_count()==original
  warmed();assert gc.get_freeze_count()==original
  with initialized_module_gc():assert gc.get_freeze_count()==original
 assert gc.get_freeze_count()==original
finally:gc.unfreeze()
''')
