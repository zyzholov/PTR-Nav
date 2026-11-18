from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict,replace
import math
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped,Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor,SingleThreadedExecutor
from rclpy.node import Node
from .grid_cdr import serialize_grid,deserialize_grid
from .runtime_gc import initialized_module_gc
from rclpy.qos import QoSProfile, qos_profile_sensor_data, ReliabilityPolicy,DurabilityPolicy
from se2_grid_msgs.msg import SE2Grid
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, Header

from .bgk_seb import BgkSebEstimator, BgkSebParams, BgkSebResult
from .obstacle_columns import preserve_collision_risk, vertical_columns, collision_subcells, collision_pose_masks, occupied_column_extents
from .overhead_clearance import measured_ceiling, clip_overhead, floor_witness_points,FloorReferenceHistory
from .supported_completion import supported_completion
from .prediction_worker import FeaturePredictionWorker,shift_prediction
from .probabilistic_risk import ProbabilityParams, geometric_probability, height_support_radius
from .startup_support import StartupSupport,Stance
from .refinement_region import refinement_query_mask
from .processing_loop import ProcessingLoop, SourceBudget
from .bounded_egress import BoundedEgress
from .se2_grid_tools import (
    first_data_template,
    serialized_header,
    get_layer_2d,
    get_layer_slices,
    grid_buffer_to_world,
    layer_refs,
    matrix_shape,
    matrix_to_multiarray_like,
    multiarray_to_matrix,
    repeat_or_trim_slices,
    yaw_slices,
    world_to_grid_buffer,
)


SE2_GRID_LENGTH_YAW = 2.0 * math.pi


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def map_deadline(now,deadline,period):
    if period<=0:return now
    if now<deadline:return None
    # Keep the phase when input is 10 Hz and output is 8 Hz. Restart after a
    # stall instead of processing a backlog or reducing the output to 5 Hz.
    return now+period if now>deadline+period else deadline+period


class FsgpBgkSe2Node(Node):
    def __init__(self) -> None:
        super().__init__("fsgp_bgk_se2_node")
        self._declare_parameters()
        self._load_parameters()
        self._runtime_gc_warmup=None  # Set only by the dedicated executable.

        self.estimator = BgkSebEstimator(self.params)
        self.predictor = None
        self.completion_predictor = None
        self._gp_fit_wall = 0.0
        self.prediction_worker=None
        self._gp_prediction_fallbacks=0
        directional=_as_bool(self.get_parameter('startup_support_directional').value)
        if directional and (not self.get_parameter('use_sim_time').value or
                            self.get_parameter('startup_support_mode').value!='stable_stance'):
            raise ValueError('directional startup is an explicit simulation stable-stance hypothesis only')
        self.startup_mode=str(self.get_parameter("startup_support_mode").value)
        self.startup_support=StartupSupport(allow_one_sided=directional,require_goal=self.startup_mode=='stable_stance',
            require_geometry=self.startup_mode=='stable_stance',
            retire_after_departure=bool(self.get_parameter('use_sim_time').value and
                self.startup_mode=='stable_stance' and self.get_parameter('probability_model').value=='shared_gp'))
        self._pending_startup_goal=None
        self.ground_history=None
        self.probability_model=str(self.get_parameter('probability_model').value)
        self.shared_veto_history=None
        self.clearance_floor_history=None
        self.clearance_gpu_failed=False
        if self.probability_model not in ('marginal','shared_gp'):
            raise ValueError('invalid probability terrain model')
        self.shared_posterior_radius=float(self.get_parameter('shared_gp_posterior_radius_m').value)
        if not math.isfinite(self.shared_posterior_radius) or not 1.<=self.shared_posterior_radius<=5.:
            raise ValueError('shared-GP posterior radius must be 1..5m; support-distance gates are separate')
        self.shared_basis_spacing=float(self.get_parameter('shared_gp_basis_spacing_m').value)
        if not math.isfinite(self.shared_basis_spacing) or not .5<=self.shared_basis_spacing<=2.:
            raise ValueError('shared-GP global basis spacing must be .5..2m')
        self.shared_refinement_radius=float(self.get_parameter('shared_gp_refinement_radius_m').value)
        if (not math.isfinite(self.shared_refinement_radius) or
                self.shared_refinement_radius!=0 and not 2.5<=self.shared_refinement_radius<=5. or
                self.shared_refinement_radius>0 and self.probability_model!='shared_gp'):
            raise ValueError('optional shared-GP refinement radius must be 0 or 2.5..5m')
        self.condition_full_residual=_as_bool(self.get_parameter('shared_gp_condition_full_residual').value)
        self.independent_model_floor=_as_bool(self.get_parameter('shared_gp_independent_model_floor').value)
        self.bounded_pending=_as_bool(self.get_parameter('shared_gp_bounded_pending').value)
        if self.bounded_pending and (not self.independent_model_floor or not self.condition_full_residual):
            raise ValueError('bounded pending family requires explicit simulation full-residual independent-floor model')
        if self.independent_model_floor and (self.probability_model!='shared_gp' or
                not self.get_parameter('use_sim_time').value):
            raise ValueError('independent model-floor prior is explicit simulation shared-GP qualification only')
        if int(self.get_parameter('shared_gp_residual_inducing_points').value)>0 and not self.condition_full_residual:
            raise ValueError('correlated residual requires the explicit full conditional shared-GP model')
        if self.condition_full_residual and self.probability_model!='shared_gp':
            raise ValueError('conditioned full residual requires explicit simulation shared_gp')
        self.completion_distance=float(self.get_parameter('probability_gp_max_distance_m').value)
        extrapolation=float(self.get_parameter('probability_gp_max_extrapolation_m').value)
        if (not math.isfinite(extrapolation) or not 0<=extrapolation<=min(.75,self.completion_distance) or
                extrapolation>0 and self.probability_model!='shared_gp'):
            raise ValueError('bounded extrapolation requires the explicit simulation shared-GP model')
        if self.probability_model=='shared_gp' and (not self.get_parameter('use_sim_time').value or
                not self.probability_use_gp_completion or self.prediction_backend!='feature_sgp'):
            raise ValueError('shared GP is an explicit simulation completion qualification model')
        if self.probability_model=='shared_gp':
            from .ground_evidence import GroundEvidenceHistory
            # Reuse only its spatial veto state; never insert positive heights.
            self.shared_veto_history=GroundEvidenceHistory(1.)
            # Explicit shared-GP simulation qualification only. This independent
            # cache feeds clearance references, never GP training or support.
            self.clearance_floor_history=FloorReferenceHistory()
        if not math.isfinite(self.completion_distance) or not 0<self.completion_distance<=1.25:
            raise ValueError('probability GP support distance must be within (0,1.25] metres')
        if self.completion_distance>.75 and self.probability_model!='shared_gp':
            raise ValueError('support beyond .75m requires the shared covariance model')
        if self.completion_distance>.35 and not self.get_parameter('use_sim_time').value:
            raise ValueError('extended GP support is an opt-in simulation qualification feature')
        history_s=float(self.get_parameter('probability_ground_history_s').value)
        if history_s:
            if self.probability_model=='shared_gp':
                raise ValueError('shared GP does not support the separate raw-return history option')
            if not self.get_parameter('use_sim_time').value or not self.probability_use_gp_completion:
                raise ValueError('ground history is an opt-in simulation GP qualification feature')
            from .ground_evidence import GroundEvidenceHistory
            self.ground_history=GroundEvidenceHistory(history_s)
        if self.startup_mode not in ("unknown","stable_stance"):
            raise ValueError("invalid startup support mode")
        if self.startup_mode!="unknown" and not self.get_parameter("use_sim_time").value:
            raise ValueError("stable-stance assumption is permitted only with simulation time")
        if self.prediction_backend == "feature_sgp":
            from .feature_sgp import FeatureSgpParams, FeatureSgpPredictor
            self.predictor = FeatureSgpPredictor(FeatureSgpParams(
                device=str(self.get_parameter("gp_device").value),
                training_steps=int(self.get_parameter("gp_training_steps").value),
                query_chunk=int(self.get_parameter("gp_query_chunk").value),
                support_radius_m=float(self.get_parameter("gp_support_radius_m").value),
                accept_std_m=float(self.get_parameter("gp_accept_std_m").value),
                local_conditioning=_as_bool(self.get_parameter("gp_local_conditioning").value)))
            if self.probability_use_gp_completion and self.probability_model=='marginal':
                self.completion_predictor = FeatureSgpPredictor(FeatureSgpParams(
                    device=str(self.predictor.device), training_steps=0,
                    local_nearest_m=self.completion_distance,
                    local_neighbors=int(self.get_parameter('probability_gp_neighbors').value)))
            if self.gp_async and self.probability_model=='marginal':
                self.prediction_worker=FeaturePredictionWorker(asdict(self.predictor.params))
        self.latest_odom: Optional[Odometry] = None
        self.latest_cloud: Optional[PointCloud2] = None
        self._input_lock = threading.Lock()
        self._processing_worker=None
        self._egress_worker=None
        self.processing_executor=str(self.get_parameter('processing_executor').value)
        # The predictor is advisory. Half-second retries can phase-lock to
        # simulator catch-up bursts and withhold every useful sensor slot.
        # Try a fresh latest input at least every two service periods (<=.2s);
        # the original input, encoded-output and delivery age gates still
        # reject every stale result. No source clock is refreshed here.
        self._source_budget=SourceBudget(min(.2,max(.05,2*self.min_update_period_sec)))
        if self.processing_executor not in ('thread_pool','single_ingress'):
            raise ValueError('processing_executor must be thread_pool or single_ingress')
        self._pending_map: Optional[SE2Grid] = None
        self._pending_received_wall = 0.
        self._cloud_history = deque(maxlen=8)
        self._reference_cloud_history = deque(maxlen=4)
        self._obstacle_memory_history = deque(maxlen=3)
        self._obstacle_memory_age = math.inf
        self._obstacle_memory_cells = 0
        self._obstacle_memory_rejections = 0
        self._input_group = MutuallyExclusiveCallbackGroup()
        self._compute_group = MutuallyExclusiveCallbackGroup()
        self.last_backend_warn = ""
        self._last_process_wall = 0.0
        self._next_process_wall = 0.0
        self._last_skip_log_wall = 0.0
        self._skipped_maps = 0
        self._deadline_input_drops = 0
        self._deadline_output_drops = 0
        self._invalid_serialized_inputs = 0
        self._process_stage_ms = {}

        self.map_sub = self.create_subscription(
            SE2Grid,
            self.input_fused_map_topic,
            self._map_callback,
            QoSProfile(depth=1),
            callback_group=self._input_group,
            raw=bool(self.get_parameter('raw_map_ingress').value),
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            QoSProfile(depth=5),
            callback_group=self._input_group,
        )
        if self.startup_mode=='stable_stance':
            # Listen to the planner acknowledgement, not the user goal topic.
            # Otherwise our subscriber alone could satisfy a goal publisher's
            # discovery wait before the planner exists, losing the actual goal.
            self.startup_goal_sub=self.create_subscription(PoseStamped,'/terrain_nav/received_goal',
                self._startup_goal_callback,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL),
                callback_group=self._input_group)
            self.startup_command_sub=self.create_subscription(Twist,'/cmd_vel',
                lambda m:self.startup_support.observe_command(m.linear.x),10,callback_group=self._input_group)
            self.startup_geometry_sub=self.create_subscription(DiagnosticArray,'/terrain_nav/chance_status',
                self._startup_geometry_callback,1,callback_group=self._input_group)
        if self.subscribe_cloud:
            self.cloud_sub = self.create_subscription(
                PointCloud2,
                self.input_cloud_topic,
                self._cloud_callback,
                QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE if self.get_parameter('cloud_reliable').value
                           else qos_profile_sensor_data.reliability),
                callback_group=self._input_group,
            )
        else:
            self.cloud_sub = None
        self.reference_cloud_sub=(self.create_subscription(
            PointCloud2,'/cloud_registered_visual',self._reference_cloud_callback,
            QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE if self.get_parameter('cloud_reliable').value
                       else qos_profile_sensor_data.reliability),
            callback_group=self._input_group) if self.clearance_floor_history is not None else None)
        memory_topic = str(self.get_parameter('obstacle_memory_topic').value)
        if memory_topic == 'none':
            memory_topic = ''
        self.obstacle_memory_sub = self.create_subscription(
            PointCloud2, memory_topic, self._obstacle_memory_callback,
            # Accumulated clouds exceed a UDP packet by orders of magnitude.
            # Reliable fragment retransmission is required, unlike live scans.
            QoSProfile(depth=3),
            callback_group=self._input_group) if memory_topic else None

        self.map_pub = self.create_publisher(
            SE2Grid,
            self.output_fused_map_topic,
            QoSProfile(depth=1),
        )
        self.runtime_pub = self.create_publisher(
            DiagnosticArray, self.output_fused_map_topic.rsplit('/', 1)[0] + '/runtime', 1)
        # Ingress stays responsive during CUDA/NumPy work. Keep one pending
        # map and one compute callback; no executor or application backlog.
        self.map_timer = self.create_timer(
            .01, self._map_timer_callback, callback_group=self._compute_group,
            clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.bgk_cloud_pub = self.create_publisher(
            PointCloud2,
            self.bgk_cloud_topic,
            QoSProfile(depth=1),
        )
        self.seb_cloud_pub = self.create_publisher(
            PointCloud2,
            self.seb_cloud_topic,
            QoSProfile(depth=1),
        )

        self.get_logger().info(
            f"map transport: rmw={rclpy.get_rmw_implementation_identifier()} "
            f"requested_fastdds_mode={os.environ.get('RMW_FASTRTPS_PUBLICATION_MODE', 'default')} "
            f"xml_qos_override={os.environ.get('RMW_FASTRTPS_USE_QOS_FROM_XML', '0')} "
            "reliable keep_last=1; source timestamps unchanged")

        self.get_logger().info(
            f"fsgp_bgk_se2 ready: input={self.input_fused_map_topic} "
            f"output={self.output_fused_map_topic} backend={self.estimator.backend_name} "
            f"gpu_requested={self.params.prefer_gpu} max_update={self.max_update_rate_hz:.2f}Hz "
            f"diagnostic_layers={self.include_diagnostic_layers} "
            f"prediction={self.prediction_backend} refine_observed={self.gp_refine_observed} "
            f"reason={self.estimator.backend_reason}"
        )

    def _startup_goal_callback(self,msg):
        stamp=msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec
        if math.isfinite(stamp) and stamp>=0 and np.isfinite([msg.pose.position.x,msg.pose.position.y]).all():
            # /clock and the reliable ack may be discovered independently.
            # Qualify against the NEXT processed terrain source stamp rather
            # than permanently discarding a one-shot ack when this clock is 0.
            self._pending_startup_goal=(stamp,time.monotonic())

    def _qualify_startup_goal(self,terrain_stamp):
        pending=self._pending_startup_goal
        if pending is None:return
        stamp,received=pending;age=terrain_stamp-stamp;wall_age=time.monotonic()-received
        if age>2. or wall_age>2.:
            self._pending_startup_goal=None
            self.get_logger().warn(f'expired initial goal acknowledgement: source_age={age:.3f}s wall_age={wall_age:.3f}s')
        elif 0<=age<=2.:
            self.startup_support.arm_initial_goal();self._pending_startup_goal=None
            self.get_logger().info(f'initial goal acknowledged against terrain source: age={age:.3f}s')

    def _startup_geometry_callback(self,msg):
        for status in msg.status:
            if status.name!='terrain_nav/chance_constraints':continue
            values={item.key:item.value for item in status.values}
            try:
                self.startup_support.set_footprint(float(values['collision_front_m']),
                    float(values['collision_back_m']),float(values['collision_protected_width_m'])/2)
            except (KeyError,ValueError):
                self.startup_support.revoked=True;self.startup_support.reason='invalid planner body geometry; revoked'

    def _declare_parameters(self) -> None:
        self.declare_parameter("input_fused_map_topic", "/car_0_terrain_analyzer/fused_map")
        self.declare_parameter("input_cloud_topic", "/cloud_registered")
        # Optional mapper-owned, visibility-cleaned geometry. Never GP input.
        self.declare_parameter('obstacle_memory_topic', '')
        self.declare_parameter('obstacle_memory_max_age_sec', 1.25)
        self.declare_parameter('obstacle_memory_required', False)
        self.declare_parameter('obstacle_memory_role', 'hard')
        self.declare_parameter('overhead_filter_enabled', False)
        self.declare_parameter('vehicle_envelope_height_m', .65)
        self.declare_parameter('collision_body_front_m', .58)
        self.declare_parameter('collision_body_back_m', .48)
        self.declare_parameter('collision_body_half_width_m', .40)
        self.declare_parameter('overhead_clearance_margin_m', .10)
        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("output_fused_map_topic", "/car_0_fsgp_bgk/fused_map")
        self.declare_parameter("bgk_cloud_topic", "/car_0_fsgp_bgk/bgk_traversability_cloud")
        self.declare_parameter("seb_cloud_topic", "/car_0_fsgp_bgk/seb_risk_cloud")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("subscribe_cloud", False)

        self.declare_parameter("use_gpu", True)
        self.declare_parameter("gpu_backend", "auto")
        self.declare_parameter("auto_gpu_min_work_items", 100_000)
        self.declare_parameter("yaw_bins", 0)
        self.declare_parameter("bgk_sigma_m", 0.45)
        self.declare_parameter("bgk_radius_m", 1.20)
        self.declare_parameter("variance_free", 0.01)
        self.declare_parameter("variance_hard", 0.20)
        self.declare_parameter("slope_free_rad", 0.10)
        self.declare_parameter("max_slope_rad", 0.55)
        self.declare_parameter("step_free_m", 0.04)
        self.declare_parameter("max_step_m", 0.18)
        self.declare_parameter("max_pitch_rad", 0.55)
        self.declare_parameter("max_roll_rad", 0.45)
        self.declare_parameter("pitch_free_rad", 0.08)
        self.declare_parameter("roll_free_rad", 0.08)
        self.declare_parameter("min_support_ratio", 0.70)
        self.declare_parameter("footprint_front", 0.42)
        self.declare_parameter("footprint_back", 0.42)
        self.declare_parameter("footprint_half_width", 0.313)
        self.declare_parameter("footprint_margin", 0.10)
        self.declare_parameter("footprint_samples_x", 5)
        self.declare_parameter("footprint_samples_y", 5)
        self.declare_parameter("risk_fusion_mode", "weighted")
        self.declare_parameter("baseline_weight", 0.45)
        self.declare_parameter("baseline_is_pose_risk", True)
        self.declare_parameter("seb_weight", 0.45)
        self.declare_parameter("uncertainty_weight", 0.10)
        self.declare_parameter("unknown_risk", 0.65)
        self.declare_parameter("use_observed_geometry", True)
        self.declare_parameter("residual_step_scale", 1.0)
        self.declare_parameter("geometry_guard_risk", .90)
        self.declare_parameter("include_diagnostic_layers", False)
        self.declare_parameter("publish_debug_clouds", True)
        self.declare_parameter("async_map_egress", True)
        self.declare_parameter("debug_yaw_index", 0)
        self.declare_parameter("debug_cloud_stride", 2)
        self.declare_parameter("max_update_rate_hz", 8.0)
        self.declare_parameter('cloud_reliable',False)
        self.declare_parameter('raw_map_ingress',False)
        self.declare_parameter('processing_executor','thread_pool')
        self.declare_parameter("max_input_source_age_sec", 0.0)
        self.declare_parameter("max_output_source_age_sec", 0.0)
        self.declare_parameter("output_delivery_reserve_sec", 0.0)
        self.declare_parameter("log_period_sec", 2.0)
        self.declare_parameter("prediction_backend", "none")
        self.declare_parameter("gp_device", "auto")
        self.declare_parameter("gp_training_steps", 12)
        self.declare_parameter("gp_query_chunk", 2048)
        self.declare_parameter("gp_fit_period_s", 1.0)
        self.declare_parameter("gp_async", True)
        self.declare_parameter("gp_prediction_max_age_s", 2.0)
        self.declare_parameter("gp_support_radius_m", 1.2)
        self.declare_parameter("gp_accept_std_m", 0.10)
        self.declare_parameter("gp_refine_observed", False)
        self.declare_parameter("gp_local_conditioning", True)
        self.declare_parameter("gp_complete_supported", False)
        self.declare_parameter("publish_probability", True)
        self.declare_parameter("probability_height_std_floor_m", 0.005)
        self.declare_parameter('probability_roughness_limit_m',0.)
        self.declare_parameter('vehicle_ground_clearance_m',.135)
        self.declare_parameter('underbody_clearance_margin_m',.01)
        self.declare_parameter("probability_variance_scale", 1.0)
        self.declare_parameter("probability_use_gp_completion", False)
        self.declare_parameter("probability_gp_max_age_s", 0.25)
        self.declare_parameter('shared_gp_detail_sources',64)
        self.declare_parameter('shared_gp_posterior_radius_m',3.)
        self.declare_parameter('shared_gp_basis_spacing_m',1.)
        self.declare_parameter('shared_gp_refinement_radius_m',0.)
        self.declare_parameter('shared_gp_condition_full_residual',False)
        self.declare_parameter('shared_gp_residual_inducing_points',0)
        self.declare_parameter('shared_gp_residual_lengthscale_m',.35)
        self.declare_parameter('shared_gp_independent_model_floor',False)
        self.declare_parameter('shared_gp_bounded_pending',False)
        self.declare_parameter('probability_model','marginal')
        self.declare_parameter('probability_gp_max_distance_m',.35)
        self.declare_parameter('probability_gp_max_extrapolation_m',0.)
        self.declare_parameter("probability_gp_neighbors", 64)
        self.declare_parameter('probability_ground_history_s',0.)
        self.declare_parameter('probability_ground_process_std_mps',.01)
        self.declare_parameter("startup_support_mode", "unknown")
        self.declare_parameter('startup_support_directional',False)

    def _load_parameters(self) -> None:
        self.prediction_backend = str(self.get_parameter("prediction_backend").value)
        self.publish_probability = _as_bool(self.get_parameter("publish_probability").value)
        if self.prediction_backend not in ("none", "feature_sgp"):
            raise ValueError("prediction_backend must be none or feature_sgp")
        self.gp_refine_observed = _as_bool(self.get_parameter("gp_refine_observed").value)
        self.gp_complete_supported = _as_bool(self.get_parameter("gp_complete_supported").value)
        self.probability_use_gp_completion = _as_bool(self.get_parameter("probability_use_gp_completion").value)
        if self.probability_use_gp_completion and not self.gp_complete_supported:
            raise ValueError("probability GP support requires explicit gp_complete_supported")
        if self.gp_complete_supported and self.prediction_backend!="feature_sgp":
            raise ValueError("supported completion requires feature_sgp")
        self.gp_fit_period_s = max(0.2, float(self.get_parameter("gp_fit_period_s").value))
        self.gp_async=_as_bool(self.get_parameter("gp_async").value)
        self.gp_prediction_max_age_s=float(self.get_parameter("gp_prediction_max_age_s").value)
        if not math.isfinite(self.gp_prediction_max_age_s) or self.gp_prediction_max_age_s<=0:
            raise ValueError("gp prediction age must be finite and positive")
        self.input_fused_map_topic = self.get_parameter("input_fused_map_topic").value
        self.input_cloud_topic = self.get_parameter("input_cloud_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.output_fused_map_topic = self.get_parameter("output_fused_map_topic").value
        self.bgk_cloud_topic = self.get_parameter("bgk_cloud_topic").value
        self.seb_cloud_topic = self.get_parameter("seb_cloud_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.subscribe_cloud = _as_bool(self.get_parameter("subscribe_cloud").value)
        self.include_diagnostic_layers = _as_bool(
            self.get_parameter("include_diagnostic_layers").value
        )
        self.publish_debug_clouds = _as_bool(self.get_parameter("publish_debug_clouds").value)
        self.debug_yaw_index = int(self.get_parameter("debug_yaw_index").value)
        self.debug_cloud_stride = max(1, int(self.get_parameter("debug_cloud_stride").value))
        self.max_update_rate_hz = max(0.0, float(self.get_parameter("max_update_rate_hz").value))
        self.max_input_source_age_sec = float(self.get_parameter("max_input_source_age_sec").value)
        self.max_output_source_age_sec = float(self.get_parameter("max_output_source_age_sec").value)
        self.output_delivery_reserve_sec = float(self.get_parameter("output_delivery_reserve_sec").value)
        self.output_admission_age_sec = self._output_admission_deadline(
            self.max_output_source_age_sec,self.output_delivery_reserve_sec)
        if not all(math.isfinite(value) and value >= 0. for value in
                   (self.max_input_source_age_sec, self.max_output_source_age_sec)):
            raise ValueError("source age limits must be finite and nonnegative (zero disables)")
        self.min_update_period_sec = (
            1.0 / self.max_update_rate_hz if self.max_update_rate_hz > 0.0 else 0.0
        )
        self.log_period_sec = max(0.2, float(self.get_parameter("log_period_sec").value))

        self.configured_yaw_bins = int(self.get_parameter("yaw_bins").value)
        self.params = BgkSebParams(
            resolution=0.20,
            yaw_bins=max(1, self.configured_yaw_bins or 72),
            prefer_gpu=_as_bool(self.get_parameter("use_gpu").value),
            gpu_backend=str(self.get_parameter("gpu_backend").value),
            auto_gpu_min_work_items=int(self.get_parameter("auto_gpu_min_work_items").value),
            bgk_sigma_m=float(self.get_parameter("bgk_sigma_m").value),
            bgk_radius_m=float(self.get_parameter("bgk_radius_m").value),
            variance_free=float(self.get_parameter("variance_free").value),
            variance_hard=float(self.get_parameter("variance_hard").value),
            slope_free_rad=float(self.get_parameter("slope_free_rad").value),
            max_slope_rad=float(self.get_parameter("max_slope_rad").value),
            step_free_m=float(self.get_parameter("step_free_m").value),
            max_step_m=float(self.get_parameter("max_step_m").value),
            max_pitch_rad=float(self.get_parameter("max_pitch_rad").value),
            max_roll_rad=float(self.get_parameter("max_roll_rad").value),
            pitch_free_rad=float(self.get_parameter("pitch_free_rad").value),
            roll_free_rad=float(self.get_parameter("roll_free_rad").value),
            min_support_ratio=float(self.get_parameter("min_support_ratio").value),
            footprint_front=float(self.get_parameter("footprint_front").value),
            footprint_back=float(self.get_parameter("footprint_back").value),
            footprint_half_width=float(self.get_parameter("footprint_half_width").value),
            footprint_margin=float(self.get_parameter("footprint_margin").value),
            footprint_samples_x=int(self.get_parameter("footprint_samples_x").value),
            footprint_samples_y=int(self.get_parameter("footprint_samples_y").value),
            fusion_mode=str(self.get_parameter("risk_fusion_mode").value),
            baseline_weight=float(self.get_parameter("baseline_weight").value),
            baseline_is_pose_risk=_as_bool(self.get_parameter("baseline_is_pose_risk").value),
            seb_weight=float(self.get_parameter("seb_weight").value),
            uncertainty_weight=float(self.get_parameter("uncertainty_weight").value),
            unknown_risk=float(self.get_parameter("unknown_risk").value),
            use_observed_geometry=_as_bool(self.get_parameter("use_observed_geometry").value),
            residual_step_scale=float(self.get_parameter("residual_step_scale").value),
            geometry_guard_risk=float(self.get_parameter("geometry_guard_risk").value),
            store_diagnostic_slices=self.include_diagnostic_layers or self.publish_debug_clouds,
        )

    def _odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def _cloud_callback(self, msg: PointCloud2) -> None:
        with self._input_lock:
            if self._cloud_history and self._stamp_ns(msg.header.stamp) < self._stamp_ns(self._cloud_history[-1].header.stamp):
                self._cloud_history.clear()
                self._obstacle_memory_history.clear()
            self.latest_cloud = msg
            self._cloud_history.append(msg)

    def _reference_cloud_callback(self,msg:PointCloud2)->None:
        with self._input_lock:
            if self._reference_cloud_history and self._stamp_ns(msg.header.stamp)<self._stamp_ns(self._reference_cloud_history[-1].header.stamp):
                self._reference_cloud_history.clear()
            self._reference_cloud_history.append(msg)

    def _reference_points(self,msg,fallback,fallback_age):
        """Same-pose full scan only for measured clearance references, never GP training."""
        with self._input_lock:history=tuple(self._reference_cloud_history)
        stamp=self._stamp_ns(msg.info.header.stamp)
        matches=[cloud for cloud in history if cloud.header.frame_id==msg.info.header.frame_id
                 and 0<=stamp-self._stamp_ns(cloud.header.stamp)<=150000000]
        if not matches:return fallback,fallback_age,False
        cloud=max(matches,key=lambda candidate:self._stamp_ns(candidate.header.stamp))
        values=self._decode_points(cloud)
        if values is None:return fallback,fallback_age,False
        return values,(stamp-self._stamp_ns(cloud.header.stamp))*1e-9,True

    def destroy_node(self):
        if self._processing_worker is not None:
            if not self._processing_worker.close():
                raise RuntimeError('terrain compute still active; cannot destroy its ROS resources')
            self._processing_worker=None
        if self._egress_worker is not None:
            if not self._egress_worker.close():
                raise RuntimeError('terrain publication still active; cannot destroy its ROS resources')
            self._egress_worker=None
        if self.prediction_worker is not None:
            self.prediction_worker.close()
            self.prediction_worker=None
        return super().destroy_node()

    def _prediction_for_map(self,msg,elevation,observed,x,y):
        key=(elevation.shape,float(x[0,0]),float(y[0,0]),float(msg.info.pos_resolution))
        stamp=msg.info.header.stamp.sec+msg.info.header.stamp.nanosec*1e-9
        query=np.column_stack((x.ravel(),y.ravel()))
        if self.prediction_worker is not None:
            training=np.column_stack((x[observed],y[observed],elevation[observed]))
            snapshot=self.prediction_worker.update(key,stamp,(training,query),period=self.gp_fit_period_s,
                max_age=self.gp_prediction_max_age_s)
            if snapshot is None:
                previous=self.prediction_worker.latest
                if previous is None or not 0<=stamp-previous.source_stamp<=self.gp_prediction_max_age_s: return None
                shifted=shift_prediction(previous.value,previous.key,key)
                if shifted is None: return None
                self._gp_source_stamp_s=previous.source_stamp
                self._gp_prediction_fallbacks=previous.fallbacks
                return shifted
            self._gp_source_stamp_s=snapshot.source_stamp
            self._gp_prediction_fallbacks=snapshot.fallbacks
            return snapshot.value
        refitted=False
        if self.predictor.model is None or time.monotonic()-self._gp_fit_wall>=self.gp_fit_period_s:
            self.predictor.fit(np.column_stack((x[observed],y[observed],elevation[observed])))
            self._gp_fit_wall=time.monotonic();self._gp_source_stamp_s=stamp;refitted=True
        if refitted or getattr(self,"_gp_query_key",None)!=key:
            self._gp_prediction=self.predictor.predict(query);self._gp_query_key=key
        return self._gp_prediction

    def _map_callback(self, msg: SE2Grid) -> None:
        with self._input_lock:
            if self._pending_map is not None:
                self._skipped_maps += 1
            self._pending_map = msg
            self._pending_received_wall = time.monotonic()
        if self._processing_worker is not None:self._processing_worker.notify()

    def start_processing_worker(self):
        """Call before spinning, so the timer and worker can never overlap."""
        if self._processing_worker is not None:raise RuntimeError('terrain worker already started')
        self.map_timer.cancel()
        if self.get_parameter('async_map_egress').value:
            self._egress_worker=BoundedEgress(self._publish_prepared_map,
                lambda exc:self.get_logger().error(f'terrain egress failed: {exc}'))
            self._egress_worker.start()
            self.get_logger().info('terrain egress: one publisher + one replaceable prepared map; final source-age gate retained')
        self._processing_worker=ProcessingLoop(self._map_timer_callback,self._worker_ready_at,
            lambda exc:self.get_logger().error(f'terrain processing loop failed: {exc}'))
        self._processing_worker.start()

    def _worker_ready_at(self):
        with self._input_lock:
            if self._pending_map is None or (self.max_input_source_age_sec<=0 and
                    self._pending_received_wall<self._next_process_wall):return None
            return self._next_process_wall

    def _map_timer_callback(self) -> None:
        now = time.monotonic()
        deadline=map_deadline(now,self._next_process_wall,self.min_update_period_sec)
        if deadline is None:
            return
        with self._input_lock:
            msg = self._pending_map
            # With an explicit source-age gate, the latest pending frame can
            # still be fresh even if it arrived during the previous compute.
            # Requiring arrival after the rate deadline adds an avoidable
            # idle scan. The source-age check below remains mandatory. Without
            # that bounded clock contract, retain the fresh-arrival policy.
            if msg is None or (self.max_input_source_age_sec<=0 and
                               self._pending_received_wall < self._next_process_wall):
                return
            self._pending_map = None
            skipped = self._skipped_maps
            if now - self._last_skip_log_wall >= self.log_period_sec:
                self._skipped_maps = 0
                self._last_skip_log_wall = now
            else:
                skipped = 0
        if skipped:
            self.get_logger().info(
                f"FSGP-BGK/SEB replaced {skipped} pending input maps; "
                f"max_update={self.max_update_rate_hz:.2f}Hz")

        serialized=isinstance(msg,(bytes,bytearray,memoryview))
        if serialized:
            try:
                timing_msg=SE2Grid();timing_msg.info.header=serialized_header(msg)
            except ValueError as exc:
                self._invalid_serialized_inputs+=1
                self.get_logger().warn(f'invalid serialized terrain input: {exc}',throttle_duration_sec=1.)
                return
        else:timing_msg=msg
        input_age = self._source_age(timing_msg.info.header.stamp)
        if self._age_expired(input_age, self.max_input_source_age_sec):
            self._deadline_input_drops += 1
            self._process_stage_ms = {}
            self._publish_runtime(timing_msg, input_age, input_age, 0., 0.,
                                  published=False, drop_reason='input deadline')
            # Wait for the next sensor frame without consuming an output slot.
            return
        if self.max_input_source_age_sec>0 and not self._source_budget.start(
                input_age,self.output_admission_age_sec,now):
            self._deadline_input_drops+=1
            self._process_stage_ms={}
            self._publish_runtime(timing_msg,input_age,input_age,0.,0.,
                                  published=False,drop_reason='insufficient processing budget')
            # No output slot consumed. A newer source may arrive immediately.
            return
        self._last_process_wall = now
        self._next_process_wall=deadline
        try:
            start = time.perf_counter()
            if serialized:msg=deserialize_grid(msg)
            decoded=time.perf_counter()
            result, out_msg = self._process_map(msg)
            self._process_stage_ms['ingress_decode']=(decoded-start)*1000.
            processed = time.perf_counter()
            processed_age=self._source_age(msg.info.header.stamp)
            job=(msg,result,out_msg,input_age,start,processed,dict(self._process_stage_ms),processed_age)
            if self._egress_worker is not None:
                self._egress_worker.submit(job)
            else:self._publish_prepared_map(job)
        except Exception as exc:
            self.get_logger().warn(
                f"FSGP-BGK/SEB update failed: {exc}",
                throttle_duration_sec=1.0,
            )

    def _publish_prepared_map(self,job):
        msg,result,out_msg,input_age,start,processed,stage_snapshot,processed_age=job
        egress_started=time.perf_counter()
        stage_snapshot['egress_wait']=(egress_started-processed)*1000.
        pending_age=self._source_age(msg.info.header.stamp)
        if self._age_expired(pending_age,self.output_admission_age_sec):
            self._deadline_output_drops+=1
            self._publish_runtime(msg,input_age,pending_age,(egress_started-start)*1000.,0.,
                published=False,drop_reason='egress queue deadline',stages=stage_snapshot)
            return
        # Prepare the EXACT ROS payload before the last age check. The
        # structured publish path hides Python->C conversion/serialization
        # AFTER that check. Serialized publication avoids doing it twice
        # and leaves only the separately budgeted transport operation.
        payload=serialize_grid(out_msg)
        encoded=time.perf_counter()
        stage_snapshot['egress_encode']=(encoded-egress_started)*1000.
        output_age = self._source_age(msg.info.header.stamp)
        self._source_budget.observe(encoded-start,output_age-input_age,
            waiting_wall_seconds=max(0.,egress_started-processed),
            waiting_source_seconds=max(0.,pending_age-processed_age))
        if self._age_expired(output_age, self.output_admission_age_sec):
            self._deadline_output_drops += 1
            self._publish_runtime(msg, input_age, output_age, (encoded-start)*1000., 0.,
                                  published=False, drop_reason='output deadline', stages=stage_snapshot)
            self.get_logger().warn(
                f'terrain result withheld: source_age={output_age:.3f}s '
                f'admission={self.output_admission_age_sec:.3f}s delivery_reserve={self.output_delivery_reserve_sec:.3f}s; '
                f'input_drops={self._deadline_input_drops} output_drops={self._deadline_output_drops}',
                throttle_duration_sec=1.)
            return
        self.map_pub.publish(payload)
        published = time.perf_counter()
        backend_state = f"{result.backend}: {result.backend_reason}"
        if backend_state != self.last_backend_warn:
            self.get_logger().info(f"FSGP-BGK compute backend: {backend_state}")
            self.last_backend_warn = backend_state
        if self.publish_debug_clouds:
            self._publish_debug_clouds(msg, result)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stages=(f"process={(processed-start)*1000:.1f}ms "
                f"publish={(published-processed)*1000:.1f}ms "
                f"post={elapsed_ms-(published-start)*1000:.1f}ms")
        output_age = self._source_age(msg.info.header.stamp)
        self._publish_runtime(msg, input_age, output_age, elapsed_ms,
                              (published-processed)*1000., stages=stage_snapshot)
        if elapsed_ms > 180. or output_age > .30:
            detail=" ".join(f"{name}={duration:.1f}ms" for name,duration in stage_snapshot.items())
            self.get_logger().warn(f"slow terrain callback: {elapsed_ms:.1f}ms ({stages}; {detail}); "
                                   f"source_age_in={input_age:.3f}s out={output_age:.3f}s; source stamp retained",
                                   throttle_duration_sec=.5)
        self.get_logger().info(
            f"FSGP-BGK/SEB map: {result.bgk_mean.shape[1]}x{result.bgk_mean.shape[0]} "
            f"yaw={len(result.final_risk_slices)} backend={result.backend} {elapsed_ms:.1f}ms ({stages})",
            throttle_duration_sec=self.log_period_sec,
        )
        if self._runtime_gc_warmup is not None:
            warmup,self._runtime_gc_warmup=self._runtime_gc_warmup,None
            warmup()
            self.get_logger().info('runtime GC warmed: long-lived initialization graph frozen; new cycles remain collectable')

    @staticmethod
    def _stamp_ns(stamp):
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    @staticmethod
    def _output_admission_deadline(limit,reserve):
        if (not math.isfinite(limit) or limit<0 or not math.isfinite(reserve) or reserve<0 or
                reserve>0 and (limit<=0 or reserve>=limit)):
            raise ValueError('output delivery reserve requires a finite, larger source-age deadline')
        return limit-reserve if limit>0 else 0.

    def _source_age(self, stamp):
        now_ns = self.get_clock().now().nanoseconds
        source_ns=self._stamp_ns(stamp)
        if now_ns<=0:return math.nan
        clock_age=(now_ns-source_ns)*1e-9
        # Preserve the clock-bootstrap/future-stamp rejection: a candidate's
        # own stamp must never be used to legitimize a reset or future input.
        if clock_age<-.11:return clock_age
        if self.get_parameter('use_sim_time').value:
            # /clock callbacks can lag other executor groups during startup.
            # Independent, already received map-frame sensor/odom stamps are
            # lower bounds on simulation time. They may ONLY increase age;
            # no wall-time extrapolation, timestamp rewrite or stale bypass.
            with self._input_lock:
                references=[self.latest_odom,self.latest_cloud]
                if self._reference_cloud_history:references.append(self._reference_cloud_history[-1])
            for reference in references:
                if reference is not None and reference.header.frame_id==self.frame_id:
                    now_ns=max(now_ns,self._stamp_ns(reference.header.stamp))
        return (now_ns-source_ns)*1e-9

    @staticmethod
    def _age_expired(age, limit):
        return limit > 0. and (not math.isfinite(age) or age < -.11 or age > limit)

    def _publish_runtime(self, msg, input_age, output_age, elapsed_ms, publish_ms,
                         *, published=True, drop_reason='', stages=None):
        report = DiagnosticArray()
        report.header.stamp = msg.info.header.stamp
        report.header.frame_id = msg.info.header.frame_id
        status = DiagnosticStatus(name='fsgp_bgk_se2/runtime', hardware_id='terrain_pipeline')
        status.level = DiagnosticStatus.WARN if not published or output_age > .30 else DiagnosticStatus.OK
        status.message = drop_reason or ('source age high' if output_age > .30 else 'processed latest input')
        values = dict(input_age_s=input_age, output_age_s=output_age,
                      output_admission_age_s=self.output_admission_age_sec,
                      output_delivery_reserve_s=self.output_delivery_reserve_sec,
                      callback_ms=elapsed_ms, publish_ms=publish_ms,
                      obstacle_memory_age_s=self._obstacle_memory_age,
                      obstacle_memory_cells=self._obstacle_memory_cells,
                      obstacle_memory_rejections=self._obstacle_memory_rejections,
                      result_published=int(published),
                      deadline_input_drops=self._deadline_input_drops,
                      deadline_output_drops=self._deadline_output_drops,
                      invalid_serialized_inputs=self._invalid_serialized_inputs,
                      egress_async=int(self._egress_worker is not None),
                      egress_replaced=0 if self._egress_worker is None else self._egress_worker.replaced,
                      **{name+'_ms':ms for name,ms in (self._process_stage_ms if stages is None else stages).items()})
        status.values = [KeyValue(key=name, value=str(value)) for name,value in values.items()]
        report.status = [status]
        self.runtime_pub.publish(report)

    def _current_points(self, msg):
        with self._input_lock:
            history = tuple(self._cloud_history)
            cloud = self.latest_cloud
        stamp_ns = self._stamp_ns(msg.info.header.stamp)
        matches = [candidate for candidate in history
                   if candidate.header.frame_id == msg.info.header.frame_id
                   and self._stamp_ns(candidate.header.stamp) <= stamp_ns]
        if matches:
            cloud = max(matches, key=lambda candidate:self._stamp_ns(candidate.header.stamp))
        if cloud is None or cloud.header.frame_id != msg.info.header.frame_id:
            return None, math.inf
        # Subtract integer ROS stamps before conversion. Equal stamps must not
        # round to a negative age and invalidate current-scan support.
        age = ((msg.info.header.stamp.sec-cloud.header.stamp.sec)*1_000_000_000
               + msg.info.header.stamp.nanosec-cloud.header.stamp.nanosec)*1e-9
        if abs(age) > .35:
            return None, age
        return self._decode_points(cloud), age

    @staticmethod
    def _decode_points(cloud):
        fields = {f.name: f for f in cloud.fields}
        if not all(name in fields and fields[name].datatype == PointField.FLOAT32
                   for name in ("x", "y", "z")):
            return None
        dtype = np.dtype({"names": ["x", "y", "z"],
            "formats": [">f4" if cloud.is_bigendian else "<f4"]*3,
            "offsets": [fields[n].offset for n in ("x", "y", "z")],
            "itemsize": cloud.point_step})
        data = np.ndarray((cloud.height, cloud.width), dtype=dtype, buffer=cloud.data,
                          strides=(cloud.row_step, cloud.point_step))
        return np.column_stack([data[n].ravel() for n in ("x", "y", "z")])

    def _obstacle_memory_callback(self, msg):
        with self._input_lock:
            history = self._obstacle_memory_history
            if history and (msg.header.frame_id != history[-1].header.frame_id or
                    self._stamp_ns(msg.header.stamp) < self._stamp_ns(history[-1].header.stamp)):
                history.clear()
            history.append(msg)

    def _memory_points(self, msg):
        with self._input_lock:
            history = tuple(self._obstacle_memory_history)
        stamp = self._stamp_ns(msg.info.header.stamp)
        candidates = [cloud for cloud in history if cloud.header.frame_id == msg.info.header.frame_id
                      and self._stamp_ns(cloud.header.stamp) <= stamp]
        if not candidates:
            return None, math.inf
        cloud = max(candidates, key=lambda item:self._stamp_ns(item.header.stamp))
        age = (stamp-self._stamp_ns(cloud.header.stamp))*1e-9
        limit = float(self.get_parameter('obstacle_memory_max_age_sec').value)
        if not math.isfinite(limit) or not 0 < limit <= 2. or age > limit:
            return None, age
        points = self._decode_points(cloud)
        if points is not None:
            # Crop before GPU transfer. A global point cloud is not a global
            # planning grid, and cannot increase this callback's grid extent.
            p = msg.info.pose.position
            inside = (np.isfinite(points).all(axis=1) &
                      (np.abs(points[:,0]-p.x) < msg.info.length_x/2) &
                      (np.abs(points[:,1]-p.y) < msg.info.length_y/2))
            points = points[inside]
        return points, age

    def _process_map(self, msg: SE2Grid) -> Tuple[BgkSebResult, SE2Grid]:
        self._process_stage_ms={}
        stage_start=time.perf_counter()

        def stage_done(name):
            nonlocal stage_start
            now=time.perf_counter()
            self._process_stage_ms[name]=(now-stage_start)*1000.
            stage_start=now

        if self.frame_id and msg.info.header.frame_id != self.frame_id:
            raise ValueError(f"terrain frame {msg.info.header.frame_id!r} differs from {self.frame_id!r}; "
                             "a frame rename cannot replace a transform")
        template = first_data_template(msg)
        if template is None:
            raise ValueError("input SE2Grid has no data layers")

        shape = matrix_shape(template)
        elevation, elevation_template = get_layer_2d(msg, "elevation", 0.0, shape)
        variance, _ = get_layer_2d(msg, "var", 0.0, elevation.shape)
        refs = layer_refs(msg)
        if "valid" in refs:
            valid, _ = get_layer_2d(msg, "valid", 0.0, elevation.shape)
            valid = np.nan_to_num(valid, nan=0.0) >= 0.5
        else:
            # The native CUDA terrain analyzer encodes unobserved cells as NaN
            # in elevation and intentionally omits a separate validity layer.
            valid = np.isfinite(elevation)

        risk_slices, _, risk_template = get_layer_slices(msg, "risk")
        if not risk_slices:
            risk_slices = [np.zeros_like(elevation, dtype=np.float32)]
        yaws = self.configured_yaw_bins or yaw_slices(msg)
        yaws = max(1, int(yaws))

        elevation = grid_buffer_to_world(elevation, msg)
        variance = grid_buffer_to_world(variance, msg)
        valid = grid_buffer_to_world(valid, msg)
        valid &= np.isfinite(elevation)
        risk_slices = [grid_buffer_to_world(layer, msg) for layer in risk_slices]
        observed_elevation = elevation
        observed_variance = variance
        # A missing variance layer cannot silently become zero uncertainty.
        probability_variance = variance if "var" in refs else np.full_like(variance, np.nan)
        observed_valid=valid.copy()
        probability_observed = observed_valid.copy()
        if "ground_observed" in refs:
            measured, _ = get_layer_2d(msg, "ground_observed", 0.0, elevation.shape)
            probability_observed &= grid_buffer_to_world(measured, msg) >= 0.5
        native_floor_observed=probability_observed.copy()
        ground_pending=np.zeros(elevation.shape,bool)
        if 'ground_pending' in refs:
            pending,_=get_layer_2d(msg,'ground_pending',1.,elevation.shape)
            pending=grid_buffer_to_world(pending,msg)
            ground_pending=~np.isfinite(pending)|(pending>=.5)
            probability_observed &= ~ground_pending
        self.params.resolution = max(float(msg.info.pos_resolution), 1.0e-3)
        self.params.yaw_bins = yaws
        baseline_risk_slices = repeat_or_trim_slices(
            risk_slices,
            yaws,
            np.zeros_like(elevation, dtype=np.float32),
        )

        gp_layers = {}
        points, cloud_age = self._current_points(msg)
        origin=(msg.info.pose.position.x-msg.info.length_x/2,msg.info.pose.position.y-msg.info.length_y/2)
        ceiling=np.full(elevation.shape,np.inf,np.float32)
        clearance_reference=None
        clearance_reference_max_age=0.
        reference_full_scan=False
        reference_age=cloud_age
        clearance_backend='cpu'
        if self.get_parameter('overhead_filter_enabled').value:
            floor_reference=probability_observed&np.isfinite(observed_variance)&(observed_variance>=0)&(observed_variance<=.02**2)
            reference_current,reference_age=points,cloud_age
            if self.clearance_floor_history is not None:
                reference_current,reference_age,reference_full_scan=self._reference_points(msg,points,cloud_age)
                stamp=msg.info.header.stamp.sec+msg.info.header.stamp.nanosec*1e-9
                current=(floor_witness_points(observed_elevation,floor_reference,reference_current,origin,self.params.resolution)
                         if 0<=reference_age<=.25 else np.empty((0,3),np.float32))
                clearance_reference,ages=self.clearance_floor_history.update(current,stamp,
                    msg.info.header.frame_id,(msg.info.pose.position.x,msg.info.pose.position.y),
                    source_age_s=max(0.,reference_age) if math.isfinite(reference_age) else 0.)
                clearance_reference_max_age=float(ages.max()) if len(ages) else 0.
            clearance_backend='cupy' if self.estimator.backend_name=='cupy' and not self.clearance_gpu_failed else 'cpu'
            envelope_options=dict(vehicle_height_m=float(self.get_parameter('vehicle_envelope_height_m').value),
                margin_m=float(self.get_parameter('overhead_clearance_margin_m').value),reference_points=clearance_reference,
                reference_error_m=.01*clearance_reference_max_age,
                measured_fallback=self.clearance_floor_history is not None)
            try:
                ceiling=measured_ceiling(observed_elevation,floor_reference,reference_current,origin,self.params.resolution,
                    backend=clearance_backend,**envelope_options)
            except Exception as exc:
                if clearance_backend=='cpu':raise
                self.clearance_gpu_failed=True
                self.get_logger().warn(f'clearance GPU unavailable; using CPU until restart: {exc}')
                ceiling=measured_ceiling(observed_elevation,floor_reference,reference_current,origin,self.params.resolution,
                    **envelope_options)
            points=clip_overhead(points,ceiling,origin,self.params.resolution)
        # Only marginal GP conditioning consumes the legacy any-yaw mask.
        # Shared-GP uses independent raw column/ground evidence below.
        protected = (np.maximum.reduce([np.nan_to_num(r, nan=0.0)
            for r in baseline_risk_slices]) >= 1.0-1e-6
            if self.predictor is not None and self.probability_model=='marginal' else None)
        stage_done("decode")
        if self.predictor is not None and self.probability_model=='marginal' and np.count_nonzero(valid & np.isfinite(elevation)) >= 8:
            resolution = self.params.resolution
            left = msg.info.pose.position.x - msg.info.length_x / 2
            bottom = msg.info.pose.position.y - msg.info.length_y / 2
            x, y = np.meshgrid(left + (np.arange(elevation.shape[1]) + 0.5) * resolution,
                               bottom + (np.arange(elevation.shape[0]) + 0.5) * resolution)
            observed = probability_observed & np.isfinite(elevation)
            try:
                prediction=self._prediction_for_map(msg,elevation,observed,x,y)
                if prediction is not None:
                    accepted=prediction.accepted.reshape(elevation.shape)
                    gp_layers={"gp_mean":prediction.mean.reshape(elevation.shape),
                        "gp_latent_var":prediction.latent_var.reshape(elevation.shape),
                        "gp_observation_var":prediction.observation_var.reshape(elevation.shape),
                        "gp_observed":observed.astype(np.float32),
                        "gp_supported":prediction.supported.reshape(elevation.shape).astype(np.float32),
                        "gp_accepted":accepted.astype(np.float32),
                        "gp_candidate":(accepted & ~observed).astype(np.float32),
                        "gp_prediction_age":np.full_like(elevation,
                            msg.info.header.stamp.sec+msg.info.header.stamp.nanosec*1e-9-self._gp_source_stamp_s)}
                    if self.gp_refine_observed:
                        refine=observed & accepted
                        gp_variance=gp_layers["gp_latent_var"]
                        obs_variance=np.maximum(np.nan_to_num(variance,nan=.05),1e-4)
                        weight=obs_variance/(obs_variance+np.maximum(gp_variance,1e-6))
                        elevation=np.where(refine,(1-weight)*elevation+weight*gp_layers["gp_mean"],elevation)
                    if self.gp_complete_supported and not self.probability_use_gp_completion:
                        elevation,variance,valid,completed=supported_completion(
                            elevation,variance,observed_valid,prediction,blocked=protected)
                        gp_layers["gp_completed"]=completed.astype(np.float32)
                    self.get_logger().info(
                        f"feature_sgp: device={prediction.device} fit={prediction.fit_ms:.1f}ms "
                        f"predict={prediction.predict_ms:.1f}ms observed={np.count_nonzero(observed)} "
                        f"candidate={np.count_nonzero(gp_layers['gp_candidate'])} async={self.gp_async} "
                        f"generic_fallbacks={self._gp_prediction_fallbacks if self.prediction_worker is not None else self.predictor.prediction_fallbacks}",
                        throttle_duration_sec=self.log_period_sec)
            except Exception as exc:
                if self.prediction_worker is None: self.predictor.model = None
                self.get_logger().warn(f"feature_sgp failed; retaining observed-only BGK/SEB: {exc}",
                                       throttle_duration_sec=self.log_period_sec)

        stage_done("posterior")
        if self.completion_predictor is not None:
            from .fresh_completion import fresh_local_prediction
            resolution = self.params.resolution
            x, y = np.meshgrid(msg.info.pose.position.x-msg.info.length_x/2+
                (np.arange(elevation.shape[1])+.5)*resolution,
                msg.info.pose.position.y-msg.info.length_y/2+
                (np.arange(elevation.shape[0])+.5)*resolution)
            try:
                current = fresh_local_prediction(self.completion_predictor,
                    observed_elevation, probability_variance, probability_observed, x, y,
                    points, cloud_age, resolution, blocked=protected,
                    max_age=float(self.get_parameter("probability_gp_max_age_s").value),
                    max_step=self.params.max_step_m,history=self.ground_history,
                    stamp=msg.info.header.stamp.sec+1e-9*msg.info.header.stamp.nanosec,
                    frame=msg.info.header.frame_id,pose_xy=(msg.info.pose.position.x,msg.info.pose.position.y),
                    process_std_mps=float(self.get_parameter('probability_ground_process_std_mps').value))
                elevation, variance, _, completed = supported_completion(
                    elevation, variance, probability_observed, current, blocked=protected,
                    max_distance_m=self.completion_distance)
                valid |= completed
                gp_layers["gp_completed"] = completed.astype(np.float32)
                for name, values in (("gp_completion_mean", current.mean),
                    ("gp_completion_variance", current.latent_var),
                    ("gp_conditioning_age",current.conditioning_age),
                    ("gp_completion_nearest_m",current.nearest_distance),
                    ("gp_completion_distance_limit_m",np.full_like(elevation,self.completion_distance)),
                    ("gp_completion_age", np.full_like(elevation, cloud_age))):
                    gp_layers[name] = values
                self.get_logger().info(f"current local GP: device={current.device} "
                    f"condition={current.predict_ms:.1f}ms sources={current.training_points} "
                    f"neighbors={self.completion_predictor.params.local_neighbors} "
                    f"completed={np.count_nonzero(completed)} source_age={cloud_age:.3f}s",
                    throttle_duration_sec=self.log_period_sec)
            except Exception as exc:
                self.get_logger().warn(f"current GP unavailable; retaining measured support: {exc}",
                                       throttle_duration_sec=self.log_period_sec)

        stage_done("current_gp")
        result = self.estimator.compute(elevation, variance, valid, baseline_risk_slices,
                                        observed_elevation=observed_elevation)
        stage_done("bgk_seb")
        collision = np.zeros_like(valid, dtype=bool)
        memory_collision = np.zeros_like(collision)
        if points is not None:
            xp = np
            if result.backend == "cupy":
                import cupy as xp
            mask, _ = vertical_columns(points, valid.shape,
                (msg.info.pose.position.x-msg.info.length_x/2,
                 msg.info.pose.position.y-msg.info.length_y/2), self.params.resolution,
                min_height=self.params.max_step_m+1e-6 if self.probability_model=='shared_gp' else .32,
                min_hits=1 if self.probability_model=='shared_gp' else 2,
                ground_elevation=np.where(observed_valid, observed_elevation, np.nan), xp=xp)
            collision = xp.asnumpy(mask) if xp is not np else mask
        memory_points, memory_age = self._memory_points(msg)
        self._obstacle_memory_age = memory_age
        if memory_points is None and self.get_parameter('obstacle_memory_required').value:
            self._obstacle_memory_rejections += 1
            raise ValueError(f'required mapper obstacle snapshot unavailable, future, wrong-frame or expired (age={memory_age:.3f}s)')
        if memory_points is not None:
            memory_points=clip_overhead(memory_points,ceiling,origin,self.params.resolution)
            xp = np
            if result.backend == 'cupy':
                import cupy as xp
            mask, _ = vertical_columns(memory_points, valid.shape,
                (msg.info.pose.position.x-msg.info.length_x/2,
                 msg.info.pose.position.y-msg.info.length_y/2), self.params.resolution,
                min_hits=1, ground_elevation=np.where(observed_valid, observed_elevation, np.nan), xp=xp)
            memory_collision = xp.asnumpy(mask) if xp is not np else mask
        self._obstacle_memory_cells = int(np.count_nonzero(memory_collision))
        current_collision = collision.copy()
        memory_role = str(self.get_parameter('obstacle_memory_role').value)
        if memory_role not in ('hard', 'advisory'):
            raise ValueError('obstacle_memory_role must be hard or advisory')
        # Historical returns have no per-point dynamic provenance. In advisory
        # mode they bias route selection only, never ground or a probability.
        if memory_role == 'hard':
            collision |= memory_collision
        collision_points=points
        if memory_role=='hard' and memory_points is not None:
            collision_points=memory_points if points is None else np.vstack((points,memory_points))
        # Only genuine measured ground can distinguish a floor return from a
        # column's XY extent. The original coarse occupancy remains mandatory;
        # no current witness means full-square fallback, never clearing.
        collision_floor=np.where(probability_observed&np.isfinite(observed_variance)&
            (observed_variance>=0)&(observed_variance<=.02**2),observed_elevation,np.nan)
        witness_floor=collision_floor;witness_gradient=None
        if memory_role=='advisory':
            from .obstacle_columns import measured_collision_floor_witness
            witness_floor,witness_gradient=measured_collision_floor_witness(
                observed_elevation,observed_variance,probability_observed,collision,self.params.resolution,
                max_step_m=self.params.max_step_m)
            gp_layers['collision_floor_witness_interpolated']=(~np.isfinite(collision_floor)&np.isfinite(witness_floor)).astype(np.float32)
        subcell_backend=('cupy' if result.backend=='cupy' and
                         not getattr(self,'_collision_subcell_gpu_failed',False) else 'cpu')
        subcell_options=dict(measured_floor=witness_floor,measured_floor_gradient=witness_gradient,
            max_step_m=self.params.max_step_m,factor=4,return_bounds=True)
        shape_points=collision_points
        if self.probability_model=='shared_gp' and memory_role=='advisory' and points is not None:
            from .obstacle_columns import measured_floor_return_mask
            floor_returns=measured_floor_return_mask(points,observed_elevation,observed_variance,
                probability_observed,collision,origin,self.params.resolution,max_step_m=self.params.max_step_m,
                max_slope_rad=max(self.params.max_pitch_rad,self.params.max_roll_rad))
            shape_points=points[~floor_returns]
            # Classification changes only padded collision XY witnesses.
            # Original scan, coarse occupancy and GP conditioning stay intact.
        try:
            subcells,subcell_bounds=collision_subcells(shape_points,collision,origin,self.params.resolution,
                backend=subcell_backend,**subcell_options)
        except Exception as exc:
            if subcell_backend=='cpu':raise
            self._collision_subcell_gpu_failed=True
            self.get_logger().warn(f'fine collision raster GPU unavailable; retaining CPU boxes and bounds: {exc}')
            subcells,subcell_bounds=collision_subcells(shape_points,collision,origin,self.params.resolution,
                **subcell_options)
        memory_subcells=collision_subcells(memory_points,memory_collision,origin,self.params.resolution,
            measured_floor=collision_floor,max_step_m=self.params.max_step_m)
        witness_low,witness_high=occupied_column_extents(collision_points,collision,origin,self.params.resolution)
        # The occupancy detector also conservatively flags negative returns.
        # A PIT must retain its original measured-floor contradiction witness;
        # it is not a positive vertical object whose side height is demoted.
        positive_columns=collision&(~probability_observed|~np.isfinite(observed_elevation)|
            (np.isfinite(witness_high)&(witness_high>observed_elevation+self.params.max_step_m)))
        # A static column does not become terrain when the current scan is
        # occluded. Historical vertical geometry is a ground-label exclusion
        # only, never a new hard obstacle or a GP ground measurement.
        ground_columns=positive_columns|memory_collision
        if self.probability_model=='shared_gp':
            gp_layers['gp_column_label_candidates']=ground_columns.astype(np.float32)
            from .obstacle_columns import separate_measured_floor_labels
            ground_columns,measured_floor_preserved=separate_measured_floor_labels(
                ground_columns,observed_elevation,observed_variance,probability_observed,
                self.params.resolution,max_step_m=self.params.max_step_m,
                max_ground_slope_rad=max(self.params.max_pitch_rad,self.params.max_roll_rad))
            gp_layers['gp_measured_floor_preserved']=measured_floor_preserved.astype(np.float32)
            # A pending positive column-side height is not evidence that the
            # floor itself moved. Keep the original frontend layer intact and
            # expose both the effective veto and its independent provenance.
            # Current occupied volumes, negative evidence and GP covariance do
            # not change. Missing pending metadata retains the original veto.
            if 'ground_pending_elevation' in refs and 'ground_pending_variance' in refs:
                from .obstacle_columns import separate_pending_column_heights
                pending_z,_=get_layer_2d(msg,'ground_pending_elevation',np.nan,elevation.shape)
                pending_v,_=get_layer_2d(msg,'ground_pending_variance',np.nan,elevation.shape)
                ground_pending,column_pending=separate_pending_column_heights(
                    ground_pending,grid_buffer_to_world(pending_z,msg),grid_buffer_to_world(pending_v,msg),
                    positive_columns,observed_elevation,observed_variance,probability_observed&~ground_columns,
                    self.params.resolution,max_step_m=self.params.max_step_m,
                    max_ground_slope_rad=max(self.params.max_pitch_rad,self.params.max_roll_rad),
                    ground_exclusions=ground_columns)
            else:column_pending=np.zeros_like(ground_pending)
            gp_layers['ground_pending_column_side']=column_pending.astype(np.float32)
            gp_layers['ground_pending_effective']=ground_pending.astype(np.float32)
            gp_layers['probability_pending_column_separation']=np.full_like(elevation,2.)
            gp_layers['gp_model_current_positive_columns']=positive_columns.astype(np.float32)
        preserve_collision_risk(result, collision)
        if "gp_completed" in gp_layers:
            completed = gp_layers["gp_completed"] >= .5
            rejected = completed & collision
            elevation = np.where(rejected, observed_elevation, elevation)
            variance = np.where(rejected, observed_variance, variance)
            valid = np.where(rejected, observed_valid, valid)
            gp_layers["gp_completed"][rejected] = 0
        if "gp_candidate" in gp_layers:
            gp_layers["gp_candidate"]=np.where(collision,0,gp_layers["gp_candidate"])
        stage_done("collision")
        out_msg = self._build_output_message(
            msg,
            result,
            # Occupied cells are observed even when the ground estimator has
            # insufficient support; otherwise RViz filters out the red column.
            (valid | collision).astype(np.float32, copy=False),
            elevation_template or risk_template or template,
        )
        for name,values in (('collision_subcell_bits',subcells.astype(np.float32)),
                            ('collision_subcell_factor',np.full_like(elevation,4.)),
                            ('collision_subcell_contract',np.full_like(elevation,3.)),
                            ('collision_subcell_xy_bounds_contract',np.full_like(elevation,1.))):
            self._append_layer(out_msg,name,(False,[world_to_grid_buffer(values,msg)]),template)
        for k,values in enumerate(subcell_bounds):
            self._append_layer(out_msg,f'collision_subcell_xy_bounds_{k}',
                (False,[world_to_grid_buffer(values.astype(np.float32),msg)]),template)
        self._append_layer(out_msg, "collision_risk",
                           (False, [world_to_grid_buffer(collision.astype(np.float32), msg)]),
                           elevation_template or risk_template or template)
        self._append_layer(out_msg,'collision_ceiling_elevation',
            (False,[world_to_grid_buffer(ceiling,msg)]),elevation_template or risk_template or template)
        self._append_layer(out_msg,'collision_floor_reference_max_age_s',
            (False,[world_to_grid_buffer(np.full_like(elevation,clearance_reference_max_age),msg)]),template)
        self._append_layer(out_msg,'collision_floor_reference_height_bound_m',
            (False,[world_to_grid_buffer(np.full_like(elevation,.01*clearance_reference_max_age),msg)]),template)
        self._append_layer(out_msg,'collision_floor_reference_full_scan',
            (False,[world_to_grid_buffer(np.full_like(elevation,float(reference_full_scan)),msg)]),template)
        for name,value in (('collision_floor_reference_source_age_s',reference_age),
                           ('collision_ceiling_reference_radius_max_m',3. if self.clearance_floor_history is not None else .9),
                           ('collision_ceiling_reference_residual_limit_m',.10 if self.clearance_floor_history is not None else .04),
                           ('collision_ceiling_gpu_used',float(clearance_backend=='cupy' and not self.clearance_gpu_failed))):
            self._append_layer(out_msg,name,(False,[world_to_grid_buffer(np.full_like(elevation,value),msg)]),template)
        for name,value in (('collision_height_contract',1.),
                ('collision_height_filter_enabled',float(self.get_parameter('overhead_filter_enabled').value)),
                ('collision_vehicle_height_m',float(self.get_parameter('vehicle_envelope_height_m').value)),
                ('collision_headroom_margin_m',float(self.get_parameter('overhead_clearance_margin_m').value))):
            self._append_layer(out_msg,name,(False,[world_to_grid_buffer(np.full_like(elevation,value),msg)]),template)
        for name, mask in (('collision_current', current_collision), ('collision_memory', memory_collision),
                           ('historical_obstacle_advisory', memory_collision if memory_role == 'advisory'
                            else np.zeros_like(memory_collision))):
            self._append_layer(out_msg, name,
                (False, [world_to_grid_buffer(mask.astype(np.float32), msg)]),
                elevation_template or risk_template or template)
        # Historical extent refines only the optimistic route guide. Missing
        # or malformed bits in old maps retain the full coarse history square.
        for name,value in (('historical_obstacle_subcell_bits',memory_subcells.astype(np.float32)),
                           ('historical_obstacle_subcell_factor',np.full_like(elevation,2.)),
                           ('collision_witness_z_min',witness_low),('collision_witness_z_max',witness_high),
                           ('collision_measured_floor_refinement',np.full_like(elevation,1.))):
            self._append_layer(out_msg,name,(False,[world_to_grid_buffer(value,msg)]),template)
        # Provenance for ROUTING refresh, not a terrain observation or a
        # probability certificate. Missing/expired mapper input must not be
        # mistaken for a positive observation that old geometry disappeared.
        for name,value in (('historical_obstacle_source_valid',float(memory_points is not None)),
                           ('historical_obstacle_source_age_s',memory_age)):
            self._append_layer(out_msg,name,
                (False,[world_to_grid_buffer(np.full_like(elevation,value),msg)]),template)
        stage_done("message_base")
        if self.publish_probability:
            shared_prediction=None
            probability_elevation = observed_elevation
            probability_support = probability_observed.copy()
            gp_used = np.zeros_like(observed_valid)
            probability_ground_observed=probability_observed&~ground_columns if self.probability_model=='shared_gp' else probability_observed
            if self.probability_use_gp_completion and "gp_completed" in gp_layers:
                map_stamp_s = msg.info.header.stamp.sec + msg.info.header.stamp.nanosec*1e-9
                age = cloud_age
                max_age = float(self.get_parameter("probability_gp_max_age_s").value)
                if 0 <= age <= max_age:
                    gp_used = gp_layers["gp_completed"] >= 0.5
                    probability_support |= gp_used
                    probability_elevation = np.where(gp_used, elevation, observed_elevation)
                    probability_variance = np.where(gp_used, variance, probability_variance)
            if self.probability_model=='shared_gp':
                # Column-side/top heights are not the underlying road. Keep
                # them in the independent occupied volume, not as a terrain
                # step extending over the complete contact stencil.
                probability_support &= ~ground_columns
                probability_elevation=np.where(ground_columns,np.nan,probability_elevation)
                probability_variance=np.where(ground_columns,np.nan,probability_variance)
                gp_used &= ~ground_columns
            probability_params = ProbabilityParams(
                resolution=self.params.resolution, yaw_bins=yaws,
                front=self.params.footprint_front, back=self.params.footprint_back,
                half_width=self.params.footprint_half_width, margin=self.params.footprint_margin,
                max_pitch=self.params.max_pitch_rad, max_roll=self.params.max_roll_rad,
                max_step=self.params.max_step_m,
                height_std_floor=float(self.get_parameter("probability_height_std_floor_m").value),
                variance_scale=float(self.get_parameter("probability_variance_scale").value),
                internal_steps=self.startup_mode=="stable_stance" or self.probability_model=='shared_gp',
                exact_footprint=True,
                internal_step_model='height_jump' if self.probability_model=='shared_gp' else 'second_difference',
                independent_model_floor=self.independent_model_floor,
                max_roughness=(float(self.get_parameter('probability_roughness_limit_m').value)
                    if self.probability_model=='shared_gp' and float(self.get_parameter('probability_roughness_limit_m').value)>0 else None))
            if self.startup_mode=='stable_stance' and result.backend=='cupy':
                # First map, before a mission: remove the initial conditional
                # kernel's lazy geometry/JIT stall from the motion interval.
                # Compilation supplies no support evidence or free cells.
                from .probability_cuda import prepare_conditioned_geometry
                try:prepare_conditioned_geometry(probability_params)
                except Exception as exc:
                    self.get_logger().warn(f'conditional kernel preparation unavailable; retaining normal CPU fallback: {exc}',
                        throttle_duration_sec=self.log_period_sec)
            clearance=float(self.get_parameter('vehicle_ground_clearance_m').value)
            clearance_margin=float(self.get_parameter('underbody_clearance_margin_m').value)
            if self.independent_model_floor and probability_params.max_roughness is None:
                raise ValueError('independent floor contract requires explicit body clearance constraints')
            if probability_params.max_roughness is not None and (
                not math.isfinite(clearance) or not math.isfinite(clearance_margin) or clearance_margin<0 or
                probability_params.roughness_limit>clearance-clearance_margin+1e-8):
                raise ValueError('terrain residual span exceeds declared body ground clearance minus its margin')
            probability_contract=(8 if self.independent_model_floor else 7 if probability_params.max_roughness is not None else
                6 if self.probability_model=='shared_gp' else 4 if probability_params.internal_steps else 2)
            bounded_pending=np.zeros_like(probability_support)
            bounded_radius=None
            prior=np.zeros_like(probability_support)
            resolution=self.params.resolution
            x,y=np.meshgrid(msg.info.pose.position.x-msg.info.length_x/2+(np.arange(elevation.shape[1])+.5)*resolution,
                msg.info.pose.position.y-msg.info.length_y/2+(np.arange(elevation.shape[0])+.5)*resolution)
            if self.startup_mode=="stable_stance":
                self._qualify_startup_goal(msg.info.header.stamp.sec+1e-9*msg.info.header.stamp.nanosec)
                odom=self.latest_odom;pose=None
                if odom is not None and odom.header.frame_id==msg.info.header.frame_id:
                    q=odom.pose.pose.orientation;n=q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z
                    if n>1e-12:
                        pose=Stance(odom.pose.pose.position.x,odom.pose.pose.position.y,
                            math.atan2(2*(q.w*q.z+q.x*q.y)/n,1-2*(q.y*q.y+q.z*q.z)/n),
                            math.atan2(2*(q.w*q.x+q.y*q.z)/n,1-2*(q.x*q.x+q.y*q.y)/n),
                            math.asin(np.clip(2*(q.w*q.y-q.z*q.x)/n,-1,1)),
                            float(np.linalg.norm([odom.twist.twist.linear.x,odom.twist.twist.linear.y])),
                            odom.header.stamp.sec+1e-9*odom.header.stamp.nanosec,odom.pose.pose.position.z)
                prior,prior_height,prior_variance=self.startup_support.update(observed_elevation,probability_variance,
                    probability_observed,collision,x,y,pose,msg.info.header.stamp.sec+1e-9*msg.info.header.stamp.nanosec,probability_params)
                # Keep the existing fixed-stance certificate intact. Replacing
                # its shared plane by separate GP marginals can lose a safe
                # path through destroyed error cancellation, despite more data.
                gp_used &= ~prior
                if prior_height is not None:
                    probability_elevation=np.where(prior,prior_height,probability_elevation)
                    probability_variance=np.where(prior,prior_variance,probability_variance)
                    probability_support |= prior
                self.get_logger().info(f"startup support: {self.startup_support.reason}; cells={np.count_nonzero(prior)}",
                    throttle_duration_sec=self.log_period_sec)
            if self.probability_model=='shared_gp':
                from .shared_terrain import SharedTerrainParams,shared_terrain_prediction,raw_measured_evidence,initial_plane_context
                raw_veto=np.zeros_like(probability_observed)
                ground_clear=np.zeros_like(probability_observed)
                stamp=msg.info.header.stamp.sec+1e-9*msg.info.header.stamp.nanosec
                self.shared_veto_history.update(np.empty((0,3)),stamp,msg.info.header.frame_id,
                    (msg.info.pose.position.x,msg.info.pose.position.y))
                source_stamp=self.shared_veto_history.clear_stamp
                # A return ABOVE an existing measured floor is additive
                # occupancy, not proof that the underlying floor vanished.
                # Every return above the max step is checked by the separate
                # current hard column layer. Negative (pit) evidence still
                # removes support immediately and retains its history.
                positive_support_veto_height=None
                strong_ground_clear=None
                if 0<=cloud_age<=float(self.get_parameter('probability_gp_max_age_s').value):
                    raw_veto,ground_clear=raw_measured_evidence(observed_elevation,probability_ground_observed,x,y,points,
                        self.params.resolution,self.params.max_step_m,positive_support_veto_height)
                    source_stamp=stamp-cloud_age
                    # This frontend witness is valid for its CURRENT scan,
                    # never a held flag paired with another conditioning scan.
                    if abs(cloud_age)<=1e-6 and 'ground_reobserved' in refs:
                        witness,_=get_layer_2d(msg,'ground_reobserved',0.,elevation.shape)
                        witness=grid_buffer_to_world(witness,msg)
                        strong_ground_clear=np.isfinite(witness)&(witness>=.5)&~ground_pending
                # Distinguish a NEW contradictory return from a held veto.
                # Missing rays must not be mistaken for new pit observations
                # when diagnosing support disappearing under the moving car.
                gp_layers['probability_current_negative']=raw_veto.astype(np.float32)
                gp_layers['probability_current_ground_clear']=ground_clear.astype(np.float32)
                raw_veto=self.shared_veto_history.veto_mask(x,y,raw_veto,ground_clear,source_stamp,
                    strong_ground_clear=strong_ground_clear)
                gp_layers['probability_history_veto']=raw_veto.astype(np.float32)
                support_veto=raw_veto|ground_pending
                probability_support &= ~support_veto
                prior &= ~support_veto
                gp_used &= ~support_veto
                probability_elevation=np.where(support_veto,np.nan,probability_elevation)
                probability_variance=np.where(support_veto,np.nan,probability_variance)
                gp_layers['probability_raw_veto']=raw_veto.astype(np.float32)
                # The upstream SE(2) cost describes an entire oriented vehicle,
                # not occupancy at this ground cell. Reusing its any-yaw veto
                # here dilates obstacles twice and discards measured flat ground.
                # Raw columns, height contradictions and the full-pose geometric
                # certificate remain independent hard checks.
                gp_layers['gp_model_initial_support']=prior.astype(np.float32)
                gp_layers['gp_model_ground_observed']=probability_ground_observed.astype(np.float32)
                gp_layers['gp_model_positive_columns']=ground_columns.astype(np.float32)
                gp_layers['gp_model_initial_elevation']=np.where(prior,prior_height,np.nan) if prior.any() else np.full_like(elevation,np.nan)
                plane_context=initial_plane_context(self.startup_support,probability_params) if prior.any() else None
                if plane_context is not None:
                    gp_layers['gp_model_initial_area']=plane_context.area.astype(np.float32)
                    for group,values in (('anchor',plane_context.anchor_xy),('coefficient',plane_context.coefficients),
                                         ('std_bound',plane_context.coefficient_std_bound)):
                        for i,value in enumerate(values):gp_layers[f'gp_model_initial_{group}_{i}']=np.broadcast_to(np.asarray(value,np.float32),elevation.shape)
                    for i in range(3):
                        for j in range(3):
                            gp_layers[f'gp_model_initial_covariance_{i}_{j}']=np.broadcast_to(np.asarray(plane_context.coefficient_covariance_bound[i,j],np.float32),elevation.shape)
                shared_diagnostics={}
                shared_prediction=shared_terrain_prediction(observed_elevation,observed_variance,
                    probability_ground_observed,x,y,points,cloud_age,self.params.resolution,support_veto,
                    device=str(self.predictor.device),
                    params=SharedTerrainParams(radius_m=self.shared_posterior_radius,basis_spacing_m=self.shared_basis_spacing,
                        max_distance_m=self.completion_distance,
                        max_extrapolation_m=float(self.get_parameter('probability_gp_max_extrapolation_m').value),
                        # The fixed plane is now a separate, source-preserving
                        # field. Gating the GP to a free affine extrapolation
                        # inside it biased the neighboring unobserved ground.
                        flat_initial_basis=False,
                        max_detail_sources=int(self.get_parameter('shared_gp_detail_sources').value),
                        condition_local_residual=True,
                        condition_full_residual=self.condition_full_residual,
                        residual_inducing_points=int(self.get_parameter('shared_gp_residual_inducing_points').value),
                        residual_lengthscale_m=float(self.get_parameter('shared_gp_residual_lengthscale_m').value),
                        bounded_queries=True,
                        complete_footprint_halo=True,
                        current_floor_column_seams=True,
                        condition_measured_column_seams=True,
                        predict_ground_under_columns=True,
                        column_shadow_floor_reference=True,
                        observation_halo_m=height_support_radius(probability_params)*self.params.resolution),
                    max_age=float(self.get_parameter('probability_gp_max_age_s').value),max_step=self.params.max_step_m,
                    initial_support=prior if prior.any() else None,
                    initial_height=prior_height if prior.any() else None,
                    plane_context=plane_context,
                    positive_veto_height=positive_support_veto_height,columns=ground_columns,
                    diagnostics=shared_diagnostics)
                if shared_prediction is None:
                    self.get_logger().warn(f'shared terrain GP unavailable: {shared_diagnostics}; '
                        'measured terrain retained; no unsupported completion admitted',throttle_duration_sec=self.log_period_sec)
                if shared_prediction is not None:
                    if self.bounded_pending and not prior.any() and all(name in refs for name in (
                            'ground_observed','ground_pending_elevation','ground_pending_variance')):
                        from .bounded_pending import condition_pending
                        pending_height,_=get_layer_2d(msg,'ground_pending_elevation',np.nan,elevation.shape)
                        pending_variance,_=get_layer_2d(msg,'ground_pending_variance',np.nan,elevation.shape)
                        shared_prediction=condition_pending(shared_prediction,observed_elevation,observed_variance,
                            grid_buffer_to_world(pending_height,msg),grid_buffer_to_world(pending_variance,msg),
                            ground_pending,native_floor_observed,raw_veto,ground_columns,max_step=self.params.max_step_m,
                            measured=probability_ground_observed,measured_height=observed_elevation,
                            measured_variance=observed_variance)
                        if shared_prediction.bounded_mean_factor is not None:
                            probability_contract=9
                            bounded_pending=shared_prediction.bounded_pending
                            bounded_radius=shared_prediction.bounded_mean_factor.abs().sum(dim=-1).cpu().numpy()
                            gp_layers['bounded_pending_used']=bounded_pending.astype(np.float32)
                            gp_layers['bounded_mean_radius_m']=bounded_radius
                            gp_layers['bounded_mean_lower']=shared_prediction.mean-bounded_radius
                            gp_layers['bounded_mean_upper']=shared_prediction.mean+bounded_radius
                    from .conditioned_probability import merge_shared_field
                    probability_elevation,probability_variance,probability_support,joint=merge_shared_field(
                        probability_elevation,probability_variance,probability_support,prior,shared_prediction)
                    if probability_contract==9:
                        # Every released pose must refer to this ONE family;
                        # no fallback to a different native-height mean outside
                        # its admissible domain, or to a fixed-stance source.
                        probability_support &= shared_prediction.valid
                        probability_elevation=np.where(probability_support,probability_elevation,np.nan)
                        probability_variance=np.where(probability_support,probability_variance,np.nan)
                    gp_layers['gp_completion_discontinuity_veto']=shared_prediction.discontinuity_veto.astype(np.float32)
                    gp_layers['gp_column_shadow_floor_supported']=shared_prediction.column_shadow_floor_supported.astype(np.float32)
                    contradicted=self.shared_veto_history.veto_mask(x,y,shared_prediction.contradicted,
                        ground_clear,source_stamp,strong_ground_clear=strong_ground_clear)
                    probability_support &= ~contradicted
                    gp_used &= ~contradicted
                    prior &= ~contradicted
                    probability_elevation=np.where(contradicted,np.nan,probability_elevation)
                    probability_variance=np.where(contradicted,np.nan,probability_variance)
                    gp_used |= shared_prediction.completed&~prior
                    gp_layers['gp_completed']=(shared_prediction.completed&~prior).astype(np.float32)
                    gp_layers['gp_completion_mean']=shared_prediction.mean
                    gp_layers['gp_completion_variance']=shared_prediction.variance
                    gp_layers['gp_completion_valid']=shared_prediction.valid.astype(np.float32)
                    if shared_prediction.relative_variance is not None:
                        gp_layers['gp_completion_relative_variance']=shared_prediction.relative_variance
                    gp_layers['gp_completion_nearest_m']=shared_prediction.nearest
                    gp_layers['gp_completion_age']=np.full_like(elevation,cloud_age)
                    # The frontend currently supplies no per-cell observation
                    # timestamps. Zero would falsely label fused history fresh.
                    gp_layers['gp_conditioning_age']=np.full_like(elevation,np.nan)
                    gp_layers['probability_joint_used']=joint.astype(np.float32)
                    gp_layers['probability_raw_veto']=(contradicted|raw_veto).astype(np.float32)
                    # Display the same current posterior that supplies navigation.
                    # A second asynchronous GP would consume GPU time and show a
                    # different height model at the same map coordinates.
                    candidate=shared_prediction.candidate
                    gp_layers['gp_mean']=np.where(candidate,shared_prediction.mean,np.nan)
                    gp_layers['gp_latent_var']=np.where(candidate,shared_prediction.variance,np.nan)
                    gp_layers['gp_observation_var']=gp_layers['gp_latent_var']+.02**2
                    gp_layers['gp_candidate']=(candidate&~probability_observed&~prior).astype(np.float32)
                    gp_layers['gp_accepted']=joint.astype(np.float32)
                    gp_layers['gp_supported']=joint.astype(np.float32)
                    gp_layers['gp_observed']=probability_ground_observed.astype(np.float32)
                    gp_layers['gp_prediction_age']=np.full_like(elevation,cloud_age)
                    self.get_logger().info(f'shared terrain GP: device={shared_prediction.factor.device} '
                        f'sources={shared_prediction.training_points} fit={shared_prediction.fit_ms:.1f}ms '
                        f'completed={np.count_nonzero(shared_prediction.completed)} '
                        f'discrepancy={shared_prediction.discrepancy:.4f}m',throttle_duration_sec=self.log_period_sec)
            # A changed surface under confirmation is neither a collision
            # measurement nor permission for the GP/initial hypothesis to fill
            # it. Both probability backends keep it explicitly unsupported.
            unsupported_pending=ground_pending&~bounded_pending
            probability_support &= ~unsupported_pending
            gp_used &= ~unsupported_pending;prior &= ~ground_pending
            probability_elevation=np.where(unsupported_pending,np.nan,probability_elevation)
            probability_variance=np.where(unsupported_pending,np.nan,probability_variance)
            for name,values in (("startup_prior_used",prior.astype(np.float32)),
                                ('startup_support_retired',np.full_like(elevation,float(self.startup_support.retired))),
                                ("probability_elevation",probability_elevation),("probability_variance",probability_variance)):
                self._append_layer(out_msg,name,(False,[world_to_grid_buffer(values,msg)]),template)
            if prior.any():
                for name,values in (('initial_assumption_elevation',prior_height),('initial_assumption_variance',prior_variance)):
                    self._append_layer(out_msg,name,(False,[world_to_grid_buffer(np.where(prior,values,np.nan),msg)]),template)
                if self.startup_support.raster_extra is not None:
                    self._append_layer(out_msg,'startup_raster_support',
                        (False,[world_to_grid_buffer((prior&self.startup_support.raster_extra).astype(np.float32),msg)]),template)
            speed_limit=self.startup_support.speed_limits(x,y)
            self._append_layer(out_msg,"startup_speed_limit",(False,[world_to_grid_buffer(speed_limit,msg)]),template)
            xp = np
            if result.backend == "cupy":
                import cupy as xp
            geometry_device='cuda' if result.backend=='cupy' else 'cpu'
            if shared_prediction is not None and shared_prediction.factor.device.type!=geometry_device:
                from .shared_terrain import prediction_on_device
                fit_device=str(shared_prediction.factor.device)
                shared_prediction=prediction_on_device(shared_prediction,geometry_device)
                self.get_logger().info(f'shared terrain allocation: fit={fit_device} geometry={geometry_device}',
                    throttle_duration_sec=self.log_period_sec)
            stage_done("support")
            # Terrain uncertainty and deterministic obstacle geometry have
            # separate discretizations. Coarse columns must not be expanded
            # over the entire vehicle a second time inside the height proof.
            terrain_collision=np.zeros_like(collision)
            try:
                probability_layers = geometric_probability(
                    probability_elevation, probability_variance, probability_support,
                    probability_params, collision=terrain_collision, xp=xp, diagnostics=self.include_diagnostic_layers,
                    bounded_radius=bounded_radius)
            except Exception as exc:
                if xp is np:
                    raise
                xp=np
                self.get_logger().warn(f"probability GPU failed; retrying CPU: {exc}",
                                       throttle_duration_sec=self.log_period_sec)
                probability_layers = geometric_probability(
                    probability_elevation, probability_variance, probability_support,
                    probability_params, collision=terrain_collision, diagnostics=self.include_diagnostic_layers,
                    bounded_radius=bounded_radius)
            stage_done("probability")
            if self.startup_mode=="stable_stance":
                probability_layers["probability_pose_event"]=[np.zeros_like(elevation) for _ in range(yaws)]
                if prior.any():
                    from .conditioned_probability import condition_initial_plane
                    try:
                        probability_layers=condition_initial_plane(probability_layers,probability_elevation,probability_variance,
                            probability_support,terrain_collision,prior,x,y,self.startup_support,probability_params,xp=xp)
                    except Exception as exc:
                        if xp is np:raise
                        self.get_logger().warn(f"conditional probability GPU failed; retrying CPU: {exc}",
                            throttle_duration_sec=self.log_period_sec)
                        probability_layers=condition_initial_plane(probability_layers,probability_elevation,probability_variance,
                            probability_support,terrain_collision,prior,x,y,self.startup_support,probability_params)
            if self.probability_model=='shared_gp':
                probability_layers.setdefault('probability_pose_event',[np.zeros_like(elevation) for _ in range(yaws)])
                # The marginal/Cauchy calculation is itself a complete pose
                # event bound on the SAME selected height field. Its valid
                # bound can use the same union-event path accounting. A shared
                # covariance proof is an optional tightening, never required
                # just to recognize that a full-pose bound is a pose event.
                for k in range(yaws):
                    qualified=(probability_layers['probability_valid'][k]>.5)&np.isfinite(
                        probability_layers['failure_probability_upper'][k])
                    probability_layers['probability_pose_event'][k][qualified]=1
                if shared_prediction is not None:
                    if prior.any() and not shared_prediction.initial_plane_used:
                        from .mixed_terrain import mixed_geometric_probability
                        # Keep covariance inside the GP group, while bounding
                        # unknown correlation to the original plane sources.
                        mixed=mixed_geometric_probability(shared_prediction,prior,prior_height,
                            self.startup_support,probability_params,terrain_collision,
                            (x[0,0]-.5*self.params.resolution,y[0,0]-.5*self.params.resolution),
                            heading=pose.yaw if pose is not None else None)
                        for k,(r,c,upper) in enumerate(mixed):
                            probability_layers['failure_probability_upper'][k][r,c]=np.minimum(
                                probability_layers['failure_probability_upper'][k][r,c],upper)
                            probability_layers['probability_pose_event'][k][r,c]=1
                    from .shared_terrain import shared_geometric_probability
                    # The selected field keeps the fixed plane at prior cells.
                    # A pure-GP proof over those cells would describe a different
                    # terrain field. Mixed footprints were bounded above using
                    # shared plane sources plus arbitrary GP/source correlation.
                    certificate_prediction=(shared_prediction if shared_prediction.initial_plane_used else
                        replace(shared_prediction,valid=shared_prediction.valid&~prior))
                    # Keep a valid inexpensive bound when it is already below
                    # 0.05% per pose. Skipping its optional refinement can only
                    # retain a MORE conservative bound; covariance, admissible
                    # support, state/path/braking budgets are unchanged.
                    refine=np.stack([values>.0005 for values in probability_layers['failure_probability_upper']])
                    # Only an OPTIONAL covariance tightening is localized.
                    # Every unqueried pose retains its original valid bound;
                    # posterior, support, geometry and source ages are unchanged.
                    with self._input_lock:refinement_odom=self.latest_odom
                    center=None;pose_stamp=None
                    if refinement_odom is not None and refinement_odom.header.frame_id==msg.info.header.frame_id:
                        center=(refinement_odom.pose.pose.position.x,refinement_odom.pose.pose.position.y)
                        pose_stamp=refinement_odom.header.stamp.sec+1e-9*refinement_odom.header.stamp.nanosec
                    refine=refinement_query_mask(refine,x,y,radius_m=self.shared_refinement_radius,
                        center=center,pose_stamp=pose_stamp,
                        map_stamp=msg.info.header.stamp.sec+1e-9*msg.info.header.stamp.nanosec)
                    for k,(r,c,upper) in enumerate(shared_geometric_probability(certificate_prediction,
                            probability_params,terrain_collision,query_mask=refine)):
                        probability_layers['failure_probability_upper'][k][r,c]=np.minimum(
                            probability_layers['failure_probability_upper'][k][r,c],upper)
                        probability_layers['probability_pose_event'][k][r,c]=1
            body=[max(float(self.get_parameter(name).value),contact)+self.params.footprint_margin
                  for name,contact in (('collision_body_front_m',self.params.footprint_front),
                                       ('collision_body_back_m',self.params.footprint_back),
                                       ('collision_body_half_width_m',self.params.footprint_half_width))]
            pose_backend='cupy' if result.backend=='cupy' and not getattr(self,'_collision_pose_gpu_failed',False) else 'cpu'
            try:
                occupied_poses=collision_pose_masks(subcells,self.params.resolution,*body,yaws,backend=pose_backend,factor=4)
            except Exception as exc:
                if pose_backend=='cpu':raise
                self._collision_pose_gpu_failed=True
                self.get_logger().warn(f'fine collision GPU unavailable; retaining exact CPU geometry: {exc}')
                occupied_poses=collision_pose_masks(subcells,self.params.resolution,*body,yaws,factor=4)
            # Keep the height proof separately: a rounded grid-centre/yaw-bin
            # collision overlay is useful for display but cannot veto an
            # actually clear continuous pose. Native v3 composes this bound
            # with its exact body/subcell check at the queried x,y,yaw.
            self._append_layer(out_msg,'terrain_failure_probability_upper',
                (True,[world_to_grid_buffer(value,msg) for value in
                       probability_layers['failure_probability_upper']]),template)
            for k,mask in enumerate(occupied_poses):
                probability_layers['failure_probability_upper'][k][mask]=1.
            for name,value in zip(('collision_body_front','collision_body_back','collision_body_half_width'),body):
                self._append_constant_layer(out_msg,name,value,elevation.shape,template)
            stage_done("conditional")
            for name, slices in probability_layers.items():
                self._append_layer(out_msg, name,
                    (True, [world_to_grid_buffer(m, msg) for m in slices]), template)
            supported_count = sum(np.count_nonzero(s) for s in probability_layers["probability_valid"])
            self.get_logger().info(
                f"probability contract={probability_contract} model={self.probability_model} "
                f"risk_valid_poses={supported_count}/{yaws*elevation.size} "
                f"gp_cells_used={np.count_nonzero(gp_used)}; legacy risk remains heuristic",
                throttle_duration_sec=self.log_period_sec)
            self._append_layer(out_msg, "probability_gp_used",
                (False, [world_to_grid_buffer(gp_used.astype(np.float32), msg)]), template)
            # Contract 6 uses real adjacent height jumps (not a second
            # difference mislabeled with the physical step-height limit).
            # Both retain union accounting and explicit source permission.
            # Scalars repeat because SE2Grid has no scalar metadata field.
            for name, value in (("probability_contract",float(probability_contract)),
                                ('probability_bounded_pending_model',float(probability_contract==9)),
                                ('probability_bounded_pending_sources',float(np.count_nonzero(bounded_pending))),
                                ('probability_refinement_radius_m',self.shared_refinement_radius),
                                ('probability_shared_posterior_radius_m',self.shared_posterior_radius),
                                ('probability_shared_basis_spacing_m',self.shared_basis_spacing),
                                ('probability_step_model',float(self.probability_model=='shared_gp')),
                                ('probability_floor_covariance_model',float(self.independent_model_floor)),
                                ('probability_footprint_geometry',2.),
                                ('probability_conditioned_local_residual',float(self.probability_model=='shared_gp')),
                                ('probability_conditioned_full_residual',float(self.condition_full_residual)),
                                ('probability_residual_inducing_points',float(self.get_parameter('shared_gp_residual_inducing_points').value)),
                                ('probability_residual_lengthscale_m',float(self.get_parameter('shared_gp_residual_lengthscale_m').value)),
                                ('probability_bounded_posterior_queries',float(self.probability_model=='shared_gp')),
                                ('probability_conditioned_measurement_std_m',.01),
                                ('probability_joint_model',float(self.probability_model=='shared_gp')),
                                ('probability_initial_source_preserved',float(self.probability_model=='shared_gp')),
                                ('probability_mixed_source_bound',float(self.probability_model=='shared_gp')),
                                ('probability_initial_plane_source_bound',float(self.probability_model=='shared_gp' and
                                    shared_prediction is not None and shared_prediction.initial_plane_used)),
                                ('probability_covariance_is_envelope',float(self.probability_model=='shared_gp' and
                                    shared_prediction is not None and shared_prediction.initial_plane_used)),
                                ('probability_source_contrast_bound',float(self.probability_model=='shared_gp' and
                                    shared_prediction is not None and shared_prediction.plane_components>0)),
                                ('probability_initial_affine_basis',float(self.probability_model=='shared_gp' and
                                    shared_prediction is not None and shared_prediction.flat_initial_basis_used)),
                                ('probability_detail_max_sources',float(self.get_parameter('shared_gp_detail_sources').value)),
                                ('probability_observation_halo_m',height_support_radius(probability_params)*self.params.resolution if self.probability_model=='shared_gp' else 0.),
                                ('probability_completion_footprint_halo',float(self.probability_model=='shared_gp')),
                                ('probability_current_floor_column_seams',float(self.probability_model=='shared_gp')),
                                ('probability_condition_measured_column_seams',float(self.probability_model=='shared_gp')),
                                ('probability_predict_ground_under_columns',float(self.probability_model=='shared_gp')),
                                ('probability_column_shadow_floor_reference',float(self.probability_model=='shared_gp')),
                                ("probability_startup_prior_enabled",float(self.startup_mode!="unknown")),
                                ("probability_height_radius_cells", height_support_radius(probability_params)),
                                ("probability_front", probability_params.front+probability_params.margin),
                                ("probability_back", probability_params.back+probability_params.margin),
                                ("probability_half_width", probability_params.half_width+probability_params.margin),
                                ("probability_max_pitch", probability_params.max_pitch),
                                ("probability_max_roll", probability_params.max_roll),
                                ("probability_max_step", probability_params.max_step),
                                ('probability_max_roughness',probability_params.roughness_limit),
                                ('probability_ground_clearance',clearance),
                                ('probability_underbody_margin',clearance_margin),
                                ("probability_height_std_floor", probability_params.height_std_floor),
                                ("probability_variance_scale", probability_params.variance_scale),
                                ('probability_positive_support_veto_height',0. if self.probability_model=='shared_gp' else 1.5),
                                ("probability_gp_enabled", float(self.probability_use_gp_completion))):
                self._append_constant_layer(out_msg,name,value,elevation.shape,template)
        if self.gp_complete_supported:
            for name,values in (("elevation",elevation),("var",variance)):
                ref=layer_refs(out_msg).get(name);matrix=world_to_grid_buffer(values,msg)
                if ref:
                    out_msg.data[ref.data_indices[0]]=matrix_to_multiarray_like(matrix,elevation_template or template)
                else:self._append_layer(out_msg,name,(False,[matrix]),template)
        if self.gp_complete_supported or self.gp_refine_observed:
            for name,values in (("gp_elevation_observed",observed_elevation),("gp_variance_observed",observed_variance)):
                self._append_layer(out_msg,name,(False,[world_to_grid_buffer(np.where(probability_observed,values,np.nan),msg)]),template)
        for name, matrix in gp_layers.items():
            self._append_layer(out_msg, name, (False, [world_to_grid_buffer(matrix, msg)]),
                               elevation_template or risk_template or template)
        self._append_layer(out_msg,"terrain_observed",
            (False,[world_to_grid_buffer(probability_observed.astype(np.float32),msg)]),template)
        stage_done("message_layers")
        return result, out_msg

    def _build_output_message(
        self,
        msg: SE2Grid,
        result: BgkSebResult,
        valid: np.ndarray,
        template: Float32MultiArray,
    ) -> SE2Grid:
        overrides: Dict[str, Tuple[bool, List[np.ndarray]]] = {
            "valid": (False, [valid]),
            "risk": (True, result.final_risk_slices),
            "baseline_risk": (False, [result.baseline_risk]),
            "bgk_mean": (False, [result.bgk_mean]),
            "bgk_var": (False, [result.bgk_var]),
            "bgk_traversability": (False, [result.bgk_traversability]),
        }
        if self.include_diagnostic_layers:
            overrides.update({
                "seb_risk": (True, result.seb_risk_slices),
                "seb_pitch_risk": (True, result.seb_pitch_risk_slices),
                "seb_roll_risk": (True, result.seb_roll_risk_slices),
                "seb_step_risk": (True, result.seb_step_risk_slices),
                "seb_support": (True, result.seb_support_slices),
            })
        overrides = {
            name: (so2, [world_to_grid_buffer(matrix, msg) for matrix in matrices])
            for name, (so2, matrices) in overrides.items()
        }

        refs = layer_refs(msg)
        out = SE2Grid()
        out.info = copy.deepcopy(msg.info)
        out.info.header.stamp = msg.info.header.stamp
        if self.frame_id:
            out.info.header.frame_id = self.frame_id
        yaw_count = max(1, len(result.final_risk_slices))
        out.info.yaw_resolution = 2.0 * math.pi / yaw_count
        out.info.length_yaw = 2.0 * math.pi
        out.start_index_row = msg.start_index_row
        out.start_index_col = msg.start_index_col

        emitted = set()
        for layer_name in msg.layers:
            if layer_name in overrides:
                self._append_layer(out, layer_name, overrides[layer_name], template)
                emitted.add(layer_name)
                continue

            ref = refs[layer_name]
            if not self.include_diagnostic_layers and ref.has_so2:
                continue
            matrices = [
                msg.data[idx]
                for idx in ref.data_indices
                if idx < len(msg.data)
            ]
            if not matrices:
                continue
            out.layers.append(layer_name)
            out.has_so2.append(ref.has_so2)
            if ref.has_so2 and len(matrices) != len(result.final_risk_slices):
                np_mats = [multiarray_to_matrix(m) for m in matrices]
                np_mats = repeat_or_trim_slices(
                    np_mats,
                    len(result.final_risk_slices),
                    np_mats[0],
                )
                out.data.extend(matrix_to_multiarray_like(m, template) for m in np_mats)
            else:
                # Unchanged layers have immutable input ownership. Publishing
                # converts/copies their buffers into the C message while this
                # output retains the Python references; no received array is
                # edited in-place. Avoid copying every unchanged numeric buffer.
                out.data.extend(matrices)
            emitted.add(layer_name)

        for layer_name in (
            "valid",
            "baseline_risk",
            "bgk_mean",
            "bgk_var",
            "bgk_traversability",
        ):
            if layer_name not in emitted:
                self._append_layer(out, layer_name, overrides[layer_name], template)
                emitted.add(layer_name)

        if self.include_diagnostic_layers:
            for layer_name in (
                "seb_risk",
                "seb_pitch_risk",
                "seb_roll_risk",
                "seb_step_risk",
                "seb_support",
            ):
                if layer_name not in emitted:
                    self._append_layer(out, layer_name, overrides[layer_name], template)
                    emitted.add(layer_name)

        return out

    def _append_constant_layer(self,out,name,value,world_shape,template):
        # Permuting a constant cannot change a cell. With a known template,
        # repeat one Float32 in its owning ROS primitive buffer directly;
        # retain the exact layout, including row/column storage convention.
        if template is not None and len(template.layout.dim)>=2:
            packed=matrix_to_multiarray_like(np.asarray([[value]],np.float32),template)
            packed.data*=int(np.prod(world_shape))
            out.layers.append(name);out.has_so2.append(False);out.data.append(packed)
        else:
            native=np.broadcast_to(np.asarray(value,np.float32),tuple(reversed(world_shape)))
            self._append_layer(out,name,(False,[native]),template)

    def _append_layer(
        self,
        out: SE2Grid,
        layer_name: str,
        layer_data: Tuple[bool, List[np.ndarray]],
        template: Float32MultiArray,
    ) -> None:
        has_so2, matrices = layer_data
        out.layers.append(layer_name)
        out.has_so2.append(has_so2)
        for matrix in matrices:
            out.data.append(matrix_to_multiarray_like(matrix, template))

    def _publish_debug_clouds(self, msg: SE2Grid, result: BgkSebResult) -> None:
        self.bgk_cloud_pub.publish(
            self._grid_to_cloud(
                msg,
                result.bgk_mean,
                result.bgk_traversability,
                "bgk_traversability",
            )
        )
        if not result.seb_risk_slices:
            return
        yaw_idx = max(0, min(self.debug_yaw_index, len(result.seb_risk_slices) - 1))
        self.seb_cloud_pub.publish(
            self._grid_to_cloud(
                msg,
                result.bgk_mean,
                result.seb_risk_slices[yaw_idx],
                "seb_risk",
            )
        )

    def _grid_to_cloud(
        self,
        msg: SE2Grid,
        z_layer: np.ndarray,
        intensity_layer: np.ndarray,
        cloud_name: str,
    ) -> PointCloud2:
        rows, cols = z_layer.shape
        res = float(msg.info.pos_resolution)
        center_x = float(msg.info.pose.position.x)
        center_y = float(msg.info.pose.position.y)
        start_x = center_x - 0.5 * float(msg.info.length_x) + 0.5 * res
        start_y = center_y - 0.5 * float(msg.info.length_y) + 0.5 * res

        points = []
        stride = self.debug_cloud_stride
        for r in range(0, rows, stride):
            y = start_y + r * res
            for c in range(0, cols, stride):
                z = float(z_layer[r, c])
                intensity = float(intensity_layer[r, c])
                if not np.isfinite(z) or not np.isfinite(intensity):
                    continue
                x = start_x + c * res
                points.append((x, y, z + 0.05, intensity))

        header = Header()
        header.stamp = msg.info.header.stamp
        header.frame_id = msg.info.header.frame_id or self.frame_id
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud = point_cloud2.create_cloud(header, fields, points)
        self.get_logger().debug(f"{cloud_name} cloud points={len(points)}")
        return cloud


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    # The executable owns this process. Freeze initial imports here; the
    # first published map seals lazy imports/geometry once, then normal GC
    # handles all subsequent allocations. Shutdown releases the owned freeze.
    with initialized_module_gc() as warmup:
        try:
            _spin_terrain_node(warmup)
        finally:
            if rclpy.ok():rclpy.shutdown()


def _spin_terrain_node(warmup):
    node = FsgpBgkSe2Node()
    node._runtime_gc_warmup=warmup
    profile_path=os.environ.get('HUNTER_TERRAIN_COMPUTE_PROFILE')
    profiler=None
    if profile_path:
        from .processing_loop import OneShotComputeProfile
        if os.path.exists(profile_path):
            node.destroy_node()
            raise ValueError('terrain diagnostic profile requires a new output path')
        profiler=OneShotComputeProfile();compute=node._process_map
        node._process_map=lambda msg:profiler.run(node.startup_support.anchor is not None,compute,msg)
        node.get_logger().warn('one measured-stance compute will be profiled; instrumented run cannot establish uninstrumented real-time performance')
    executor=(SingleThreadedExecutor() if node.processing_executor=='single_ingress'
              else MultiThreadedExecutor(num_threads=3))
    executor.add_node(node)
    if node.processing_executor=='single_ingress':node.start_processing_worker()
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if profiler is not None:profiler.save(profile_path)


if __name__ == "__main__":
    main()
