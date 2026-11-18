#!/usr/bin/env python3
"""Passive, source-matched dynamic-certificate diagnosis.

No goals, commands or truth inputs. Retain raw bounded buffers during motion;
decode/serialize to disk only once the owning test has stopped. An instrumented
run is not used to assert uninstrumented timing performance.
"""
import argparse
from collections import deque
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import time

NUMBER=r'[-+0-9.eE]+'
EVENT=re.compile(r'dynamic yield: no certified escape .*?\[source_s=('+NUMBER+r') sample=\(('
    +NUMBER+'),('+NUMBER+'),('+NUMBER+r')\) t=('+NUMBER+r') speed=('+NUMBER+
    r') path_upper=('+NUMBER+r') brake_upper=('+NUMBER+r')(?: conditional_dynamic_upper=('+NUMBER+r'))?\]')


class StatusSelection:
    """Select a refusal after an accepted moving action, using diagnostics only."""
    def __init__(self,after_action=False):
        self.after_action=after_action
        self.had_action=False

    def accept(self,values):
        try:speed=float(values.get('measured_speed_mps','nan'))
        except ValueError:speed=math.nan
        if (values.get('dynamic_yield_state') in ('advance','retreat') and
                values.get('command_accepted')=='1' and math.isfinite(speed) and abs(speed)>.05):
            self.had_action=True
        return (values.get('dynamic_yield_state')=='refused' and values.get('command_accepted')=='0' and
                (not self.after_action or self.had_action))


class RejectionBuffers:
    """Bounded raw observations; first match by default, latest complete match by explicit option."""
    def __init__(self,reason_substring='',latest=False):
        self.reason_substring=reason_substring
        self.latest=latest
        self.queues={name:deque(maxlen=6 if name in ('map','front') else 40)
                     for name in ('map','front','cloud','odom')}
        self.pending=None;self.captured=None
        self.recent_captures=deque(maxlen=16)
        self.unmatched_events=0

    def retain(self,name,stamp,payload):
        self.queues[name].append((stamp,payload));self._match()

    def event(self,line):
        if self.reason_substring not in line:return
        if not self.latest and (self.pending is not None or self.captured is not None):return
        match=EVENT.search(line)
        if not match:return
        self.pending=dict(map_source_stamp_ns=int(Decimal(match[1])*1_000_000_000),
            rejected_pose=[float(match[i]) for i in (2,3,4)],trajectory_time_s=float(match[5]),
            speed_mps=float(match[6]),path_upper_at_rejection=float(match[7]),
            braking_upper_at_rejection=float(match[8]),original_log=line.strip())
        if match[9] is not None:self.pending['conditional_dynamic_upper_at_rejection']=float(match[9])
        self._match()

    def status(self,source_ns,values):
        if self.reason_substring not in values.get('dynamic_yield_reason',''):return
        if not self.latest and (self.pending is not None or self.captured is not None):return
        self.pending=dict(map_source_stamp_ns=source_ns,
            control_root=[float(values[k])for k in ('control_state_x_m','control_state_y_m','control_state_yaw_rad')],
            speed_mps=float(values.get('control_state_speed_mps',values['measured_speed_mps'])),
            original_status=dict(values),selection=('latest' if self.latest else 'first')+' dynamic refusal with explicit reported map source')
        # The callback started before selection/final validation. A refreshed
        # root must not be replayed at that older clock. Preserve the explicit
        # root clock, or reconstruct it from the recorded source + prediction.
        if 'control_state_stamp_s' in values:
            root_time=Decimal(values['control_state_stamp_s'])
        elif 'odometry_source_stamp_s' in values and 'control_prediction_sec' in values:
            root_time=Decimal(values['odometry_source_stamp_s'])+Decimal(values['control_prediction_sec'])
        else:
            root_time=None
        if root_time is not None and root_time.is_finite():
            self.pending['control_root_stamp_s']=str(root_time)
        self._match()

    def _match(self):
        if self.pending is None or self.captured is not None and not self.latest:return
        source=self.pending['map_source_stamp_ns']
        maps=[(t,b)for t,b in self.queues['map']if abs(t-source)<=5]
        fronts=[(t,b)for t,b in self.queues['front']if abs(t-source)<=5]
        if not maps or not fronts:
            # A pre-discovery or dropped source may never enter the bounded
            # buffer. Preserve its miss count, then allow the next ORIGINAL
            # status to be selected; never substitute a newer map for it.
            if any(len(self.queues[n])==self.queues[n].maxlen and
                   min(t for t,b in self.queues[n])>source+5 for n in ('map','front')):
                self.pending=None;self.unmatched_events+=1
            return
        live_stamp=self.pending.get('original_status',{}).get('reported_live_source_stamp_s')
        if live_stamp is not None:
            value=Decimal(live_stamp)
            if not value.is_finite():return
            live_ns=int(value*1_000_000_000)
            if not any(abs(t-live_ns)<=5 for t,b in self.queues['cloud']):return
        self.captured=dict(event=dict(self.pending),map=maps[-1],front=fronts[-1],
            clouds=list(self.queues['cloud']),odometry=list(self.queues['odom']))
        if self.latest:
            self.recent_captures.append(self.captured);self.pending=None

    def before_stop(self,stop_stamp):
        """Never mistake a post-assessor-stop refusal for a failure cause."""
        candidates=self.recent_captures if self.latest else [self.captured]
        for capture in reversed(candidates):
            if capture is None:continue
            value=capture['event'].get('control_root_stamp_s')
            if value is not None and Decimal(value)<=Decimal(str(stop_stamp)):
                return capture
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=240.)
    p.add_argument('--reason-substring',default='',help='Select a recorded failure reason; no change to navigation or its acceptance')
    p.add_argument('--status-only',action='store_true',help='Require full posterior diagnostics instead of a text-log event')
    p.add_argument('--confirmed-motion-only',action='store_true',help='Ignore startup refusals before any confirmed observed motion model')
    p.add_argument('--after-executed-action',action='store_true',help='Select a refusal only after an accepted dynamic action with nonzero measured vehicle speed; implies status-only')
    p.add_argument('--latest',action='store_true',help='Retain the last successfully source-matched refusal; unmatched newer input never destroys the last complete capture')
    a=p.parse_args()
    if a.output.exists() or not 0<a.timeout<=600:p.error('new output and bounded timeout required')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from se2_grid_msgs.msg import SE2Grid
    from diagnostic_msgs.msg import DiagnosticArray
    from fsgp_bgk_se2.se2_grid_tools import get_layer_slices
    from serialized_header import leading_stamp_ns
    buffers=RejectionBuffers(a.reason_substring,latest=a.latest);selection=StatusSelection(a.after_executed_action)
    rclpy.init();node=rclpy.create_node('dynamic_certificate_observer')
    def status(message):
        for row in message.status:
            values={v.key:v.value for v in row.values}
            if not selection.accept(values):continue
            if a.confirmed_motion_only and not any(k.startswith('motion_object_') and 'confirmed_identity=1' in v
                                                   for k,v in values.items()):continue
            if not all(k in values for k in ('reported_map_source_stamp_s','control_state_x_m','control_state_y_m',
                'control_state_yaw_rad','measured_speed_mps')):continue
            try:source=int(Decimal(values['reported_map_source_stamp_s'])*1_000_000_000)
            except (ValueError,OverflowError):continue
            buffers.status(source,values)
    node.create_subscription(DiagnosticArray,'/terrain_nav/chance_status',status,10)
    for name,cls,topic in (('map',SE2Grid,'/car_0_fsgp_bgk/fused_map'),
            ('front',SE2Grid,'/car_0_terrain_analyzer/fused_map'),
            ('cloud',PointCloud2,'/cloud_registered'),('odom',Odometry,'/Odometry')):
        def retain(payload,n=name):buffers.retain(n,leading_stamp_ns(payload),payload)
        node.create_subscription(cls,topic,retain,1 if cls is SE2Grid else qos_profile_sensor_data,raw=True)
    deadline=time.monotonic()+a.timeout;offset=0;carry=''
    try:
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.02)
            log=a.run/'planning.log'
            if log.exists():
                with log.open()as f:f.seek(offset);carry+=f.read();offset=f.tell()
                lines=carry.split('\n');carry=lines.pop()
                if not a.status_only and not a.after_executed_action:
                    for line in lines:buffers.event(line)
            # The owning trial sends its final stop before writing this file.
            if (a.run/'navigation_data.json').exists():break
        else:
            raise RuntimeError('owner did not finish; refusing disk-heavy capture during motion')
        stop_file=a.run/'navigation_motion_end.json'
        stop_stamp=None
        if a.latest and stop_file.exists():
            stop_stamp=json.loads(stop_file.read_text())['last_truth_stamp_s']
            buffers.captured=buffers.before_stop(stop_stamp)
        a.output.mkdir(parents=True)
        if buffers.captured is None:
            record=dict(scope=__doc__,status='no_source_matched_rejection',event=buffers.pending)
        else:
            capture=buffers.captured;stamp,payload=capture['map']
            msg=deserialize_message(payload,SE2Grid)
            ages=get_layer_slices(msg,'gp_completion_age')[0]
            source=stamp-round(float(ages[0].flat[0])*1e9)if ages else stamp
            exact=[b for t,b in capture['clouds']if abs(t-source)<=5]
            data={'map.cdr':payload,'front.cdr':capture['front'][1]}
            if exact:data['completion_cloud.cdr']=exact[-1]
            # Current collision evidence can be newer than the GP conditioning
            # cloud. Preserve both clocks and original bytes separately.
            live_source=capture['event'].get('original_status',{}).get('reported_live_source_stamp_s')
            live_ns=None
            if live_source is not None:
                value=Decimal(live_source)
                if value.is_finite():live_ns=int(value*1_000_000_000)
            live_cloud=[b for t,b in capture['clouds']if live_ns is not None and abs(t-live_ns)<=5]
            if live_cloud:data['live_cloud.cdr']=live_cloud[-1]
            # Nearest odometry is explicitly separate from exact map/scan.
            if capture['odometry']:
                odom_stamp,odom=min(capture['odometry'],key=lambda x:abs(x[0]-stamp))
                data['nearby_odom.cdr']=odom
            else:odom_stamp=None
            for name,b in data.items():(a.output/name).write_bytes(b)
            record=dict(scope=__doc__,status='captured',event=capture['event'],
                map_source_stamp_ns=stamp,conditioning_stamp_ns=source,
                exact_conditioning_cloud=bool(exact),nearby_odom_stamp_ns=odom_stamp,
                live_source_stamp_ns=live_ns,exact_live_cloud=bool(live_cloud),
                source_run=str(a.run.resolve()),
                sha256={name:hashlib.sha256(b).hexdigest()for name,b in data.items()},
                limitation='logged rejected rollout sample; no reconstructed actuator history or ground-truth feedback')
        record['unmatched_events']=buffers.unmatched_events
        record['selection_mode']='latest source-matched refusal' if a.latest else 'first source-matched refusal'
        record['owner_truth_stop_stamp_s']=stop_stamp
        (a.output/'capture.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        print(record['status'],flush=True);return 0
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':raise SystemExit(main())
