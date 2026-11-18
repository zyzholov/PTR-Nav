import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from select_regression_catalog import select


def test_regression_selection_balances_sources_without_reading_outcomes():
    items=[{"id":str(i),"cluster_id":"a"} for i in range(100)]
    items += [{"id":"b","cluster_id":"b"},{"id":"c","cluster_id":"c"}]
    selected=select(items,5,17)
    assert {d["cluster_id"] for d in selected}=={"a","b","c"}
    assert len({d["id"] for d in selected})==5 and selected==select(items,5,17)
