import json
from pathlib import Path

import numpy as np

from comprehensive_study import PROTOCOL,migration_compatible
from paper_clouds import grid


def test_predeclared_campaign_counts_are_factorials_not_claimed_environments():
    p=json.loads(PROTOCOL.read_text())
    a=p["public"];b=p["mechanism"];c=p["discretization"];d=p["scaling"]
    assert (3*a["generated_per_family"]+a["road_inputs"]+a["indoor_inputs"])*len(a["conditions"])==4000==a["planned_cases"]
    assert len(b["families"])*b["seeds_per_family"]*len(b["conditions"])*len(b["noise_levels_m"])==3600==b["planned_cases"]
    assert len(c["families"])*c["geometries_per_family"]*len(c["resolution_m"])*len(c["yaw_bins"])==720==c["planned_cases"]
    assert len(d["sizes"])*len(d["yaw_bins"])*len(d["backends"])*d["measured_calls"]==540==d["planned_measured_calls"]
    assert p["planning"]["per_scenario"]*10*p["planning"]["repeats"]==3000
    assert p["navigation"]["per_scenario"]*len(p["navigation"]["scenarios"])==420


def test_discretization_uses_same_physical_centers_and_heading_queries():
    xx,yy=np.meshgrid(np.arange(-2,2.01,.4),np.arange(-2,2.01,.4));centers=np.c_[xx.ravel(),yy.ravel()]
    yaws=(np.arange(72)+.5)*2*np.pi/72
    for res in (.1,.2,.4):
        size=round(12.8/res);origin=np.full(2,-6.4-res/2);xy=grid((size,size),origin,res)
        ij=np.rint((centers-origin)/res-.5).astype(int)
        np.testing.assert_allclose(xy[ij[:,1]*size+ij[:,0]],centers,atol=1e-5)
    for count in (18,36,72):
        ids=np.floor(yaws*count/(2*np.pi)).astype(int)
        assert ids.min()==0 and ids.max()==count-1
        np.testing.assert_array_equal(np.bincount(ids),np.full(count,72//count))


def test_noise_is_strictly_paired_but_not_a_new_geometry():
    rng=np.random.default_rng(123)
    xyz=rng.normal(size=(100,3));keep=np.random.default_rng(456).random(100)>=.3
    epsilon=np.random.default_rng(456+100000).normal(size=100)
    variants=[xyz[keep,2]+std*epsilon[keep] for std in (0,.02,.05)]
    np.testing.assert_allclose((variants[2]-variants[0])/.05,(variants[1]-variants[0])/.02)


def test_driver_migration_cannot_change_model_or_data_fingerprints():
    import copy
    key="scripts/sim/experiments/comprehensive_study.py"
    old=dict(source_sha256={key:"f05402c4e04cf75f3e8ff12843ec0599cae29afeaa9a36811774846dabe8c667","feature_sgp.py":"model"},catalog_sha256="data")
    new=copy.deepcopy(old);new["source_sha256"][key]="new_driver"
    assert migration_compatible(old,new)
    new["source_sha256"]["feature_sgp.py"]="changed_model"
    assert not migration_compatible(old,new)
    new=copy.deepcopy(old);new["source_sha256"][key]="new_driver";new["catalog_sha256"]="changed_data"
    assert not migration_compatible(old,new)
