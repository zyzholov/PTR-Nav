#!/usr/bin/env python3
"""Compare two geometric proofs on the same frozen posterior, not arrival rates.

Inputs are the map/conditioning scan captured by probability_contract_probe.
The CLI reconstructs archived model settings and measured initial-plane source
envelopes before comparing proofs. The separately selected mixed-source field
is not a physical navigation replay. No hidden simulator heights are used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from dataclasses import asdict

import numpy as np
import torch
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from se2_grid_msgs.msg import SE2Grid

from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices,grid_buffer_to_world,layer_refs
from fsgp_bgk_se2.shared_terrain import (SharedTerrainParams,shared_terrain_prediction,
                                       shared_geometric_probability)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def frozen_prediction(path,device,*,replay_initial=False,detail_override=None,halo_override=None,
                      condition_residual_override=None,full_residual_override=None,step_model_override=None,flat_override=None,
                      complete_halo_override=None,column_seams_override=None,measured_seams_override=None,
                      column_ground_override=None,verify_posterior=True,measured_floor_labels_override=False,
                      residual_sources_override=None,residual_lengthscale_override=None,
                      pending_column_separation_override=False,column_shadow_reference_override=None,
                      posterior_radius_override=None,basis_spacing_override=None):
    archive=(path/'map.cdr').read_bytes();scan=(path/'completion_cloud.cdr').read_bytes()
    diagnostic=json.loads((path/'diagnostic.json').read_text())
    if digest(archive)!=diagnostic['map_sha256'] or not diagnostic['completion_cloud'] or \
            digest(scan)!=diagnostic['completion_cloud']['sha256']:
        raise ValueError('frozen map/conditioning-scan hash mismatch')
    msg=deserialize_message(archive,SE2Grid);cloud=deserialize_message(scan,PointCloud2)
    age=diagnostic['completion_cloud']['source_age_s']
    stamp=lambda s:s.sec*10**9+s.nanosec
    if (msg.info.header.frame_id!=cloud.header.frame_id or
            abs(stamp(msg.info.header.stamp)-stamp(cloud.header.stamp)-round(age*1e9))>5):
        raise ValueError('conditioning scan is not timestamp/frame matched')
    def layer(name):
        values=get_layer_slices(msg,name)[0]
        if not values:raise ValueError(f'missing frozen layer: {name}')
        return grid_buffer_to_world(values[0],msg)
    def scalar(name):
        values=layer(name);value=float(values.flat[0])
        if not math.isfinite(value) or not np.all(values==value):
            raise ValueError(f'nonconstant probability metadata: {name}')
        return value
    elevation=layer('gp_elevation_observed');shape=elevation.shape;resolution=msg.info.pos_resolution
    x,y=np.meshgrid(msg.info.pose.position.x-msg.info.length_x/2+(np.arange(shape[1])+.5)*resolution,
                    msg.info.pose.position.y-msg.info.length_y/2+(np.arange(shape[0])+.5)*resolution)
    rows=read_points(cloud,field_names=('x','y','z'))
    points=np.column_stack([rows[name] for name in ('x','y','z')])
    collision=layer('collision_risk')>=.5
    pending_name='ground_pending_effective' if 'ground_pending_effective' in layer_refs(msg) else 'ground_pending'
    pending=layer(pending_name)>=.5 if pending_name in layer_refs(msg) else np.zeros(shape,bool)
    metadata=json.loads((path.parent/'run_metadata.json').read_text())
    if metadata['arguments']['probability_model']!='shared_gp':
        raise ValueError('this replay requires an explicitly recorded shared-GP run')
    params=ProbabilityParams(resolution=resolution,yaw_bins=round(2*math.pi/msg.info.yaw_resolution),
        front=scalar('probability_front'),back=scalar('probability_back'),
        half_width=scalar('probability_half_width'),margin=0,
        max_pitch=scalar('probability_max_pitch'),max_roll=scalar('probability_max_roll'),
        max_step=scalar('probability_max_step'),height_std_floor=scalar('probability_height_std_floor'),
        variance_scale=scalar('probability_variance_scale'),internal_steps=True,
        exact_footprint=scalar('probability_footprint_geometry')==2 if 'probability_footprint_geometry' in layer_refs(msg) else False,
        internal_step_model='height_jump' if scalar('probability_contract') in (6,7,8) else 'second_difference',
        max_roughness=scalar('probability_max_roughness') if scalar('probability_contract') in (7,8) else None,
        independent_model_floor=scalar('probability_contract')==8)
    if params.independent_model_floor and scalar('probability_floor_covariance_model')!=1:
        raise ValueError('independent-floor covariance contract mismatch')
    if step_model_override is not None:
        from dataclasses import replace
        params=replace(params,internal_step_model=step_model_override)
    options={}
    gp=SharedTerrainParams(max_distance_m=metadata['arguments']['gp_support_distance'])
    if replay_initial:
        exact_initial='gp_model_initial_support' in layer_refs(msg)
        prior=layer('gp_model_initial_support' if exact_initial else 'startup_prior_used')>.5
        if prior.any():options=dict(initial_support=prior,initial_height=layer(
            'gp_model_initial_elevation' if exact_initial else 'initial_assumption_elevation'))
        gp=SharedTerrainParams(radius_m=scalar('probability_shared_posterior_radius_m') if 'probability_shared_posterior_radius_m' in layer_refs(msg) else 3.,
            basis_spacing_m=scalar('probability_shared_basis_spacing_m') if 'probability_shared_basis_spacing_m' in layer_refs(msg) else 1.,
            max_distance_m=metadata['arguments']['gp_support_distance'],
            max_extrapolation_m=metadata['arguments']['gp_extrapolation_distance'],
            flat_initial_basis=bool(scalar('probability_initial_affine_basis')),
            condition_local_residual=bool(scalar('probability_conditioned_local_residual')) if 'probability_conditioned_local_residual' in layer_refs(msg) else False,
            condition_full_residual=bool(scalar('probability_conditioned_full_residual')) if 'probability_conditioned_full_residual' in layer_refs(msg) else False,
            residual_inducing_points=int(scalar('probability_residual_inducing_points')) if 'probability_residual_inducing_points' in layer_refs(msg) else 0,
            residual_lengthscale_m=scalar('probability_residual_lengthscale_m') if 'probability_residual_lengthscale_m' in layer_refs(msg) else .35,
            bounded_queries=bool(scalar('probability_bounded_posterior_queries')) if 'probability_bounded_posterior_queries' in layer_refs(msg) else False,
            conditioned_measurement_std_m=scalar('probability_conditioned_measurement_std_m') if 'probability_conditioned_measurement_std_m' in layer_refs(msg) else .01,
            max_detail_sources=int(scalar('probability_detail_max_sources')) if 'probability_detail_max_sources' in layer_refs(msg) else 0,
            complete_footprint_halo=scalar('probability_completion_footprint_halo')==1 if 'probability_completion_footprint_halo' in layer_refs(msg) else False,
            current_floor_column_seams=scalar('probability_current_floor_column_seams')==1 if 'probability_current_floor_column_seams' in layer_refs(msg) else False,
            condition_measured_column_seams=scalar('probability_condition_measured_column_seams')==1 if 'probability_condition_measured_column_seams' in layer_refs(msg) else False,
            predict_ground_under_columns=scalar('probability_predict_ground_under_columns')==1 if 'probability_predict_ground_under_columns' in layer_refs(msg) else False,
            column_shadow_floor_reference=scalar('probability_column_shadow_floor_reference')==1 if 'probability_column_shadow_floor_reference' in layer_refs(msg) else False,
            observation_halo_m=scalar('probability_observation_halo_m') if 'probability_observation_halo_m' in layer_refs(msg) else 0.)
        if ('probability_initial_plane_source_bound' in layer_refs(msg) and scalar('probability_initial_plane_source_bound')==1):
            from fsgp_bgk_se2.shared_terrain import InitialPlaneContext
            options['plane_context']=InitialPlaneContext(np.array([scalar(f'gp_model_initial_anchor_{i}') for i in range(2)]),
                np.array([scalar(f'gp_model_initial_coefficient_{i}') for i in range(3)]),
                np.array([scalar(f'gp_model_initial_std_bound_{i}') for i in range(3)]),layer('gp_model_initial_area')>.5,
                np.array([[scalar(f'gp_model_initial_covariance_{i}_{j}') for j in range(3)] for i in range(3)])
                    if 'gp_model_initial_covariance_0_0' in layer_refs(msg) else None)
    if posterior_radius_override is not None:
        if verify_posterior:raise ValueError('changed posterior tile is a counterfactual, not an unchanged archive replay')
        from dataclasses import replace
        gp=replace(gp,radius_m=posterior_radius_override)
    if basis_spacing_override is not None:
        if verify_posterior:raise ValueError('changed global basis is a counterfactual, not an unchanged archive replay')
        from dataclasses import replace
        gp=replace(gp,basis_spacing_m=basis_spacing_override)
    if detail_override is not None:
        from dataclasses import replace
        gp=replace(gp,max_detail_sources=detail_override)
    if halo_override is not None:
        from dataclasses import replace
        gp=replace(gp,observation_halo_m=halo_override)
    if condition_residual_override is not None:
        from dataclasses import replace
        gp=replace(gp,condition_local_residual=condition_residual_override)
    if flat_override is not None:
        from dataclasses import replace
        gp=replace(gp,flat_initial_basis=flat_override)
    if full_residual_override is not None:
        from dataclasses import replace
        gp=replace(gp,condition_full_residual=full_residual_override)
    if residual_sources_override is not None or residual_lengthscale_override is not None:
        from dataclasses import replace
        gp=replace(gp,residual_inducing_points=gp.residual_inducing_points if residual_sources_override is None else residual_sources_override,
            residual_lengthscale_m=gp.residual_lengthscale_m if residual_lengthscale_override is None else residual_lengthscale_override)
    if complete_halo_override is not None or column_seams_override is not None:
        from dataclasses import replace
        gp=replace(gp,complete_footprint_halo=(gp.complete_footprint_halo if complete_halo_override is None else complete_halo_override),
            current_floor_column_seams=(gp.current_floor_column_seams if column_seams_override is None else column_seams_override))
    if measured_seams_override is not None:
        from dataclasses import replace
        gp=replace(gp,condition_measured_column_seams=measured_seams_override)
    if column_ground_override is not None:
        from dataclasses import replace
        gp=replace(gp,predict_ground_under_columns=column_ground_override)
    if column_shadow_reference_override is not None:
        if verify_posterior:raise ValueError('changed discontinuity admission is a counterfactual, not an unchanged archive replay')
        from dataclasses import replace
        gp=replace(gp,column_shadow_floor_reference=column_shadow_reference_override)
    ground_observed=layer('gp_model_ground_observed')>.5 if 'gp_model_ground_observed' in layer_refs(msg) else layer('terrain_observed')>.5
    columns=layer('gp_model_positive_columns')>.5 if 'gp_model_positive_columns' in layer_refs(msg) else collision|(
        layer('historical_obstacle_advisory')>.5 if 'historical_obstacle_advisory' in layer_refs(msg) else False)
    if pending_column_separation_override:
        if verify_posterior:raise ValueError('pending-label counterfactual cannot verify an unchanged archived admission mask')
        from fsgp_bgk_se2.obstacle_columns import separate_pending_column_heights
        current_columns=(layer('gp_model_current_positive_columns')>.5 if 'gp_model_current_positive_columns' in layer_refs(msg)
            else collision&(layer('gp_column_label_candidates')>.5) if 'gp_column_label_candidates' in layer_refs(msg) else columns&collision)
        pending,_=separate_pending_column_heights(pending,layer('ground_pending_elevation'),
            layer('ground_pending_variance'),current_columns,elevation,layer('gp_variance_observed'),
            ground_observed&~columns,resolution,max_step_m=params.max_step,
            max_ground_slope_rad=max(params.max_pitch,params.max_roll),ground_exclusions=columns)
    preserved_floor=np.zeros(shape,bool)
    if measured_floor_labels_override:
        if verify_posterior:raise ValueError('changed floor labels are a counterfactual, not an archived posterior replay')
        from fsgp_bgk_se2.obstacle_columns import separate_measured_floor_labels
        ground_observed=(layer('terrain_observed')>.5)&~pending
        if 'gp_column_label_candidates' in layer_refs(msg):columns=layer('gp_column_label_candidates')>.5
        columns,preserved_floor=separate_measured_floor_labels(columns,elevation,layer('gp_variance_observed'),
            ground_observed,resolution,max_step_m=params.max_step,
            max_ground_slope_rad=max(params.max_pitch,params.max_roll))
    if gp.predict_ground_under_columns:ground_observed&=~columns
    prediction=shared_terrain_prediction(elevation,layer('gp_variance_observed'),
        ground_observed,x,y,points,age,resolution,
        (np.zeros_like(collision) if gp.predict_ground_under_columns else collision)|pending|(layer('probability_raw_veto')>.5),device=device,
        params=gp,max_step=params.max_step,positive_veto_height=None,
        columns=columns if gp.predict_ground_under_columns else None,**options)
    if prediction is None:raise ValueError('frozen observations do not support a shared posterior')
    maximum=float(np.nanmax(abs(prediction.mean-layer('gp_completion_mean'))))
    if verify_posterior and maximum>2e-5:raise ValueError(f'posterior differs from archived mean: {maximum} m')
    variance_maximum=float(np.nanmax(abs(prediction.variance-layer('gp_completion_variance'))))
    if verify_posterior and variance_maximum>2e-6:raise ValueError(f'covariance differs from archive: {variance_maximum} m2')
    support_verified=None
    if 'gp_completion_valid' in layer_refs(msg):
        support_verified=bool(np.array_equal(prediction.valid,layer('gp_completion_valid')>.5))
        if verify_posterior and not support_verified:raise ValueError('completion admission differs from archived mask')
    return prediction,params,collision,dict(map_sha256=digest(archive),cloud_sha256=digest(scan),
        posterior_mean_max_replay_difference_m=maximum,geometry=vars(params),
        covariance_max_replay_difference_m2=variance_maximum,support_replay_verified=support_verified,
        queries_use_initial_stance=replay_initial,model=asdict(gp),query_pose=diagnostic['pose'],
        selected_initial_source_preserved=bool(scalar('probability_initial_source_preserved')) if 'probability_initial_source_preserved' in layer_refs(msg) else False,
        replay_scope=('source-aware residual field with covariance envelope; not physical navigation' if prediction.initial_plane_used else
            'raw shared-GP posterior; does not reconstruct the separately selected fixed-plane source or mixed-source bounds'),
        map_min=[msg.info.pose.position.x-msg.info.length_x/2,msg.info.pose.position.y-msg.info.length_y/2],
        modified_model=not verify_posterior,measured_floor_preserved_cells=int(preserved_floor.sum()))


def compare(path,device,repeats,output):
    prediction,params,collision,provenance=frozen_prediction(path,device,replay_initial=True)
    shape=(params.yaw_bins,*prediction.mean.shape)
    def evaluate(tight):
        begin=time.perf_counter()
        result=shared_geometric_probability(prediction,params,collision,tighten_span=tight)
        return result,(time.perf_counter()-begin)*1000
    before,cold_before=evaluate(False);after,cold_after=evaluate(True)
    timings={key:[] for key in ('half_span_ms','direct_span_ms')}
    # Alternate order; these small repetitions qualify implementation timing,
    # not a formal hardware benchmark or independent terrain experiments.
    for index in range(repeats):
        for tight in ((False,True) if index%2==0 else (True,False)):
            _,elapsed=evaluate(tight)
            timings['direct_span_ms' if tight else 'half_span_ms'].append(elapsed)
    old=np.full(shape,np.nan,np.float32);new=old.copy()
    for k,(a,b) in enumerate(zip(before,after)):
        if not np.array_equal(a[0],b[0]) or not np.array_equal(a[1],b[1]):
            raise ValueError('tighter proof changed the terrain support domain')
        old[k,a[0],a[1]]=a[2];new[k,b[0],b[1]]=b[2]
    finite=np.isfinite(old)
    if not finite.any():raise ValueError('no eligible whole-pose queries')
    increase=float(np.max(new[finite]-old[finite]))
    if increase>2e-6:raise ValueError('direct-span minimum increased the original bound')
    artifact=output/(path.parent.name+'_'+path.name+'.npz')
    np.savez_compressed(artifact,mean=prediction.mean,variance=prediction.variance,
        support=prediction.valid,collision=collision,half_span_upper=old,direct_span_upper=new)
    return dict(input=str(path.resolve()),provenance=provenance,
        eligible_poses=int(finite.sum()),half_span_state_limit_005_count=int((old<=.05).sum()),
        direct_span_state_limit_005_count=int((new<=.05).sum()),
        maximum_bound_increase=increase,maximum_bound_reduction=float(np.max(old[finite]-new[finite])),
        timing_scope='geometric-certificate wall time including GPU synchronization; GP fit excluded',
        first_call_ms=dict(half_span=cold_before,direct_span=cold_after),timings=timings,
        arrays=str(artifact.resolve()),arrays_sha256=digest(artifact.read_bytes()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.repeats<=10:parser.error('qualification repeats must be within 1..10')
    if len({p.resolve() for p in args.inputs})!=len(args.inputs):parser.error('duplicate frozen input')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    results=[compare(p,args.device,args.repeats,args.output) for p in args.inputs]
    root=Path(__file__).resolve().parents[3]
    names=['scripts/sim/checks/shared_certificate_qualification.py',
           'src/fsgp_bgk_se2/fsgp_bgk_se2/shared_terrain.py',
           'src/fsgp_bgk_se2/fsgp_bgk_se2/shared_terrain_cuda.py']
    record=dict(passed=True,scope=__doc__,device=args.device,
        source_sha256={n:digest((root/n).read_bytes()) for n in names},results=results)
    (args.output/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,indent=2,allow_nan=False))


if __name__=='__main__':main()
