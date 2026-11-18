import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from paper_budget_campaign import budget_order, stratified_selection, require_qualified_sources
from paper_aligned_report import paired_deltas, validate_budget, validate_queries, path_quality, query_analysis, NAMES
from planning_query_campaign import randomize_queries, layout_id
from navigation_sandbox import scenario
from paper_clouds import epfl_settings


def test_sampling_balances_sources_not_outcomes():
    items=[dict(id=f"{g}{i}",cluster_id=g,family="road") for g in ("a","b") for i in range(5)]
    assert [r["cluster_id"] for r in stratified_selection(items,"road",4)]==["a","b","a","b"]
    with pytest.raises(ValueError):stratified_selection(items,"road",11)


def test_budget_order_reproducible_and_complete():
    assert budget_order(1)==budget_order(1)
    assert sorted(budget_order(2))==[50,125,250,500]
    assert len({tuple(budget_order(i)) for i in range(30)})>10


def test_swiss_gain_and_fractal_exponent_use_distinct_profiles():
    for seed in range(80):
        old=epfl_settings(seed,"legacy_v1");new=epfl_settings(seed,"metric_v2")
        assert old["H"]>=1 and old["H"]<=1.6
        if seed%4==3:
            assert .35<=new["H"]<=.65
            assert {k:v for k,v in old.items() if k!="H"}=={k:v for k,v in new.items() if k!="H"}
        else:assert old==new
    with pytest.raises(ValueError):epfl_settings(1,"unknown")


def test_main_experiment_rejects_old_source_profiles():
    require_qualified_sources([{"id":"a","family":"hilly","generator_profile":"metric_v2"},
        {"id":"b","family":"road","registration_pose_source":"semantic_suma"},{"id":"c","family":"indoor"}])
    with pytest.raises(ValueError,match="unqualified"):require_qualified_sources([{"id":"a","family":"hilly"}])
    with pytest.raises(ValueError,match="SuMa"):require_qualified_sources([{"id":"b","family":"road","registration_pose_source":"kitti_gt"}])
    with pytest.raises(ValueError,match="decay"):
        require_qualified_sources([{"id":"a","family":"hilly","generator_profile":"metric_v2","shader_parameters":{"type":3,"H":1.3}}])


def test_factorial_and_effective_budget_are_checked():
    rows=[]
    for condition in ("point15","hole16","frontier50"):
        for m in (50,125,250,500):
            rows.append(dict(id=f"d__{condition}__M{m}",dataset_id="d",condition=condition,requested_M=m,
                case_seed=7,source_sha256="source",reference_sha256="reference",input_points=300,ground_input_points=200,
                status="ok",effective_M=min(m,200),train_points=200,actual_device="cuda",actual_backend="cupy"))
    record={"runs":rows};catalog={"datasets":[{"id":"d"}]}
    validate_budget(record,catalog)
    bad=copy.deepcopy(record);bad["runs"][-1]["effective_M"]=500
    with pytest.raises(ValueError,match="budget contract"):validate_budget(bad,catalog)
    bad=copy.deepcopy(record);bad["runs"][-1]["case_seed"]=8
    with pytest.raises(ValueError,match="unpaired input"):validate_budget(bad,catalog)
    with pytest.raises(ValueError,match="incomplete"):validate_budget({"runs":rows[:-1]},catalog)


def test_paired_differences_require_pair_and_exclude_missing_scores():
    rows=[dict(dataset_id="d",condition="hole16",requested_M=m,status="ok",cluster_id="s",score=v) for m,v in ((125,3),(500,2))]
    assert paired_deltas(rows,"score")[0]["score"]==-1
    with pytest.raises(ValueError,match="unpaired"):paired_deltas(rows[:1],"score")
    rows[1]["score"]=None
    assert not paired_deltas(rows,"score")


def test_random_queries_do_not_inflate_unique_geometry():
    cases=[scenario("turn",s) for s in range(10)]
    a=randomize_queries(cases,123);b=randomize_queries(cases,123)
    assert a==b and len({tuple(c["start"]) for c in a})==10
    assert len({c["layout_id"] for c in a})==1
    assert len({layout_id(scenario("detour",s)) for s in range(10)})==10


def test_curvature_wrap_and_cusp_definition():
    length,k,gears=path_quality([[0,0,np.pi-.05,1],[1,0,-np.pi+.05,1]])
    assert length==1 and k==pytest.approx(.1) and gears==0
    assert path_quality([[0,0,0,1],[1,0,1,-1]])==(1.,None,1)
    assert path_quality([])==(0.,None,0)


def test_query_repeats_are_required_not_independent_successes():
    plan={"settings":{"timing_repeats":3},"cases":[{"id":"x"}]}
    record={"settings":plan["settings"],"runs":[{"id":"x","repeat":i} for i in range(3)]}
    validate_queries(plan,record)
    record["runs"][2]["repeat"]=1
    with pytest.raises(ValueError,match="incomplete"):validate_queries(plan,record)


def test_query_success_requires_all_repeats_safe_and_complete():
    plan={"settings":{"timing_repeats":3},"cases":[dict(id=f,scenario=f,layout_id=f,expected_stop=i>=7) for i,f in enumerate(NAMES)]}
    rows=[]
    for case in plan["cases"]:
        for i in range(3):
            rows.append(dict(**case,repeat=i,status="ok",path=[[0,0,0,1],[1,0,0,1]],path_length_m=1,
                gear_switches=0,mean_abs_curvature_per_m=0,planner_wall_ms=10,returned=True,
                independent_path_safe=True,full_goal_connector=True))
    rows[0]["independent_path_safe"]=False
    stats,queries,_=query_analysis(plan,{"settings":plan["settings"],"runs":rows})
    assert stats["turn"]["complete_safe"]==0 and stats["turn"]["unsafe_queries"]==1
    assert len(queries)==10 and stats["turn"]["queries"]==1 and stats["turn"]["calls"]==3


def test_native_independent_geometry_reference():
    binary=Path(__file__).resolve().parents[3]/"build/terrain_nav/terrain_nav_query_benchmark"
    if not binary.exists():pytest.skip("native harness not built")
    subprocess.run([str(binary),"--self-test"],check=True,capture_output=True,text=True)
