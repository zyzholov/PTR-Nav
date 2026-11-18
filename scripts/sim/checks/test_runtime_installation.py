import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('runtime_installation',
    Path(__file__).resolve().parents[2] / 'tools' / 'check_runtime_installation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_installed_copy_must_match_source_even_when_it_exists(tmp_path):
    source, runtime = tmp_path / 'source.py', tmp_path / 'runtime.py'
    source.write_text('version = 2\n')
    runtime.write_text('version = 1\n')
    assert not module.compare_files(source, runtime)['matches']
    runtime.write_bytes(source.read_bytes())
    assert module.compare_files(source, runtime)['matches']
    runtime.unlink()
    assert not module.compare_files(source, runtime)['matches']
