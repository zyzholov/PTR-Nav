"""Cross-module specification and independent CAD tyre-sweep containment."""
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/isaac_sim_hunter_livo2/scripts'))
from vehicle_geometry import geometry,load_spec,swept_bounds,fit_neos_urdf


def test_turn_radius_is_chassis_centre_not_bicycle_rear_axle_or_inner_wheel():
    v=load_spec();g=geometry();rear=g['wheelbase']/math.tan(g['max_steer'])
    assert math.hypot(rear,g['wheelbase']/2)==pytest.approx(v['center_turn_radius_m'])
    assert rear<v['center_turn_radius_m']
    assert g['max_steer']<.40 # former clamp was already slightly tighter


def test_physical_collision_envelope_contains_every_steered_cad_tyre():
    # Read transforms and tyre dimensions independently from the actual xacro.
    root=ET.fromstring(fit_neos_urdf((ROOT/'src/simulator/fake_ugv/hunter_se/hunter_se_gazebo/xacro/hunter_se_gazebo.xacro').read_text()))
    origins={e.attrib['name']:list(map(float,e.find('origin').attrib['xyz'].split()))
        for e in root.iter('joint') if e.find('origin') is not None and e.attrib['name'].startswith(('fr_susp','re_susp'))}
    cfg=yaml.safe_load((ROOT/'src/planner/terrain_nav/config/hunter_gazebo.yaml').read_text())['terrain_nav_node']['ros__parameters']
    front,back,width=(cfg[k]for k in ('body_collision_front','body_collision_back','body_collision_half_width'))
    g=geometry()
    for steer in np.linspace(-g['max_steer'],g['max_steer'],1601):
        k=math.tan(steer)/g['wheelbase']
        for side,offset in [('left',.0214),('right',-.0210)]:
            x,y,_=origins['fr_susp_'+side+'_joint']
            angle=math.atan2(k*(x+.215),1-k*y);c,s=math.cos(angle),math.sin(angle)
            for u in (-.1348,.1348):
                for v in (-.078/2,.078/2):
                    px=x-offset*s+u*c-v*s;py=y+offset*c+u*s+v*c
                    assert -back<=px<=front
                    assert abs(py)<=width
    assert width>.626/2 # straight datasheet width cannot bound turning tyres
    assert front+back<.86 # old artificial 1.06m bumpers are absent


def test_planner_and_probability_map_share_asymmetric_frame_and_real_limits():
    p=yaml.safe_load((ROOT/'src/planner/terrain_nav/config/hunter_gazebo.yaml').read_text())['terrain_nav_node']['ros__parameters']
    f=yaml.safe_load((ROOT/'src/fsgp_bgk_se2/config/hunter_fastlio.yaml').read_text())['fsgp_bgk_se2_node']['ros__parameters']
    g=geometry();b=swept_bounds();v=load_spec()
    for k in ('front','back','half_width'):
        assert p['footprint_'+k]==pytest.approx(g[k])
        assert f['footprint_'+k]==pytest.approx(g[k])
        assert p['body_collision_'+k]==pytest.approx(b[k])
        assert f['collision_body_'+k+'_m']==pytest.approx(b[k])
    assert p['max_steer']==pytest.approx(g['max_steer'])
    assert p['chance_max_pitch_rad']==pytest.approx(math.radians(v['max_climb_deg']))
    assert f['max_pitch_rad']==pytest.approx(p['chance_max_pitch_rad'])
    assert f['max_step_m']==p['chance_max_step_m']==v['vertical_obstacle_m']
    assert f['vehicle_ground_clearance_m']==p['vehicle_ground_clearance_m']==v['ground_clearance_m']
    assert f['underbody_clearance_margin_m']==p['underbody_clearance_margin_m']>0
    assert 0<f['probability_roughness_limit_m']<=v['ground_clearance_m']-p['underbody_clearance_margin_m']
    assert p['footprint_margin']==f['footprint_margin']==v['collision_margin_m']
    assert p['max_speed']<=v['max_speed_kmh']/3.6
