import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import navigation_sandbox as sandbox


def test_native_source_fingerprint_rejects_unrebuilt_code(monkeypatch,tmp_path):
    source=tmp_path/"component.cpp";source.write_text("current source")
    monkeypatch.setattr(sandbox,"ROOT",tmp_path)
    compiled={source.name:sandbox.digest(source)}
    monkeypatch.setattr(sandbox.subprocess,"check_output",lambda *a,**k:json.dumps(compiled))
    assert sandbox.verify_native_build(Path("unused"))==compiled
    source.write_text("new source without a rebuild")
    with pytest.raises(ValueError,match="stale"):
        sandbox.verify_native_build(Path("unused"))
