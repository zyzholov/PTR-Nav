#!/usr/bin/env python3
"""Verify the Python modules and launch resources ROS will actually load."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def compare_files(source, runtime):
    source = Path(source)
    runtime = Path(runtime)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    runtime_hash = hashlib.sha256(runtime.read_bytes()).hexdigest() if runtime.is_file() else None
    return dict(source=str(source), runtime=str(runtime), source_sha256=source_hash,
                runtime_sha256=runtime_hash, matches=source_hash == runtime_hash)


def audit():
    from ament_index_python.packages import get_package_share_directory
    package = 'fsgp_bgk_se2'
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise RuntimeError('fsgp_bgk_se2 is not importable from the current ROS environment')
    source_root = ROOT / 'src' / package
    runtime_root = Path(spec.origin).parent
    share = Path(get_package_share_directory(package))
    files = [compare_files(path, runtime_root / path.relative_to(source_root / package))
             for path in sorted((source_root / package).rglob('*.py'))]
    for directory, pattern in (('launch', '*.launch.py'), ('config', '*.yaml'), ('rviz', '*.rviz')):
        files.extend(compare_files(path, share / directory / path.name)
                     for path in sorted((source_root / directory).glob(pattern)))
    return dict(passed=bool(files) and all(row['matches'] for row in files),
                scope='Python import resolution and installed launch/config/RViz content',
                package=package, import_root=str(runtime_root), share_root=str(share), files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = audit()
    except (OSError, RuntimeError, LookupError) as exc:
        result = dict(passed=False, error=str(exc))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    mismatches = [row for row in result.get('files', []) if not row['matches']]
    print(json.dumps(dict(passed=result['passed'], checked=len(result.get('files', [])),
                         mismatches=mismatches, error=result.get('error'))), flush=True)
    if not result['passed']:
        print('ROS runtime differs from source. Rebuild: colcon build --packages-select fsgp_bgk_se2')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
