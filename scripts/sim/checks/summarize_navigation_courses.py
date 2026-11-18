#!/usr/bin/env python3
"""Plot explicit physical qualification attempts, retaining failures and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from plot_style import configure_chinese
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/isaac_sim_hunter_livo2/scripts'))
from navigation_courses import front_sign_obstacles
import boundary_courses
from clearance_metrics import static_course_clearances,sampled_body_clearances


NAMES = {'staggered': '连续错列障碍', 'corridor': '双墙通道', 'ramp_detour': '坡面绕障',
         'overhead_pass':'高横梁下穿', 'overhead_block':'低横梁拒行',
         'overhead_plain':'无特征横梁走廊',
         'ramp': '坡面通行', 'crossing': '平路动态横穿', 'ramp_crossing': '坡面动态横穿',
         'reserve_overlap': '额外膨胀区脱困后导航', 'boundary':'边界场景'}


def case_name(summary, data=None):
    if summary['scenario']!='boundary':return NAMES[summary['scenario']]
    kind=summary['kind']
    if kind=='rubble' and data is not None:
        return f"刚性低凸条（最高 {100*data['spec']['height_m']:g} 厘米）"
    return {'multi_dynamic':'三个运动物体场景','rubble':'刚性低凸条',
            'long_corridor':'长走廊','side_goal_corridor':'长墙侧后目标',
            'dense':'密集立柱','chicane':'连续错位门'}.get(kind,kind)


def boundary_geometry(path, data):
    """Refuse to reconstruct old physical fixtures with a changed generator."""
    with zipfile.ZipFile(path/'source_snapshot.zip') as archive:
        frozen=archive.read('src/isaac_sim_hunter_livo2/scripts/boundary_courses.py')
    if hashlib.sha256(frozen).digest()!=hashlib.sha256(Path(boundary_courses.__file__).read_bytes()).digest():
        raise ValueError('fixture generator changed: use the archived generator before plotting')
    return boundary_courses.fixtures(data['spec'])


def planning_statistics(path):
    values = [float(value) for value in re.findall(
        r'terrain_nav planner: .*?elapsed=([\d.]+)ms', (path / 'planning.log').read_text())]
    with zipfile.ZipFile(path / 'source_snapshot.zip') as archive:
        source = archive.read('src/planner/terrain_nav/src/terrain_nav_node.cpp').decode()
    match = re.search(r'RCLCPP_INFO(_THROTTLE)?\([^;]*"terrain_nav planner:', source)
    if match is None:
        raise ValueError(f'cannot establish timing-log sampling policy: {path}')
    return dict(scope=('throttled_log_samples_only' if match.group(1) else 'all_completed_planner_jobs'),
                count=len(values), elapsed_ms=values,
                p50_ms=float(np.percentile(values, 50)) if values else None,
                p95_ms=float(np.percentile(values, 95)) if values else None,
                max_ms=max(values) if values else None)


def runtime_figure(records, output):
    figure, axes = plt.subplots(4, 1, figsize=(13, 12), layout='constrained', sharex=True)
    x = np.arange(len(records))
    colors = ['#197d89' if r['overall_passed'] else '#b64c5f' for r in records]
    for i, record in enumerate(records):
        stats = record['planning_statistics']
        if stats['count']:
            box = axes[0].boxplot([stats['elapsed_ms']], positions=[i], widths=.5,
                                  patch_artist=True, manage_ticks=False,
                                  medianprops=dict(color='black'))
            box['boxes'][0].set(facecolor=colors[i], alpha=.6)
            suffix = '*' if stats['scope'] != 'all_completed_planner_jobs' else ''
            axes[0].annotate(f"n={stats['count']}{suffix}", (i, stats['max_ms']),
                             xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
        streams = record['stream_statistics']
        for topic, offset, color in (('/cloud_registered', -.15, '#4875a5'),
                                      ('/car_0_fsgp_bgk/fused_map', .15, '#197d89')):
            stream = streams[topic]
            p95, maximum = 1000*stream['age_p95'], 1000*stream['age_max']
            axes[1].plot([i+offset, i+offset], [p95, maximum], color=color, linewidth=2)
            axes[1].scatter(i+offset, p95, color=color, marker='o', s=25)
            axes[1].scatter(i+offset, maximum, color=color, marker='_', s=60)
        summary = record['summary']
        minima = list((summary.get('course_clearance_minima_m') or {}).values())
        minima += list((summary.get('front_sign_clearance_minima_m') or {}).values())
        minima += list((summary.get('static_clearance_minima_m') or {}).values())
        minima += [v['minimum_m'] for v in (summary.get('moving_clearance') or {}).values() if v['minimum_m'] is not None]
        if summary.get('sampled_body_clearance_min_m') is not None:
            minima.append(summary['sampled_body_clearance_min_m'])
        if minima:
            minimum=min(minima)
            axes[2].scatter(i, minimum, color=colors[i], s=45)
            axes[2].annotate(f'{minimum:.5f}', (i, minimum),
                             xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
            if .085<=minimum<=.175:
                axes[3].scatter(i,minimum,color=colors[i],s=45)
                axes[3].annotate(f'{minimum:.5f}',(i,minimum),xytext=(0,8),
                    textcoords='offset points',ha='center',fontsize=9)
            else:
                direction='↑' if minimum>.175 else '↓'
                axes[3].text(i,.170 if minimum>.175 else .087,
                    f'{direction} {minimum:.5f} 米',ha='center',fontsize=8,color='#525e64')
    axes[0].set(title='规划任务墙钟耗时分布（含失败任务；不是目标到达耗时）', ylabel='耗时（毫秒）')
    axes[0].margins(y=.25)
    axes[0].set_ylim(bottom=0)
    axes[0].text(.01, .98, '*：早期版本日志限频，只是被记录的子样本，不用于全任务分位数结论',
                 transform=axes[0].transAxes, va='top', fontsize=9)
    axes[1].axhline(350, color='#b64c5f', linestyle='--', label='源时间年龄验收上限')
    axes[1].plot([], [], 'o-', color='#4875a5', label='配准点云：圆点 P95，端点最大值')
    axes[1].plot([], [], 'o-', color='#197d89', label='概率风险图：圆点 P95，端点最大值')
    axes[1].set(title='运动监测器接收时的源时间年龄（模拟时钟；不含持有期增长）', ylabel='年龄（毫秒）')
    axes[1].legend(fontsize=9)
    axes[2].axhline(.1, color='#b64c5f', linestyle='--', label='验收净距 0.10 米')
    axes[2].set(title='独立真值采样车身净距下界（仅针对记录的专项/背景障碍）', ylabel='净距（米）')
    axes[2].legend(fontsize=9)
    axes[3].axhline(.1,color='#b64c5f',linestyle='--',label='验收净距 0.10000 米')
    axes[3].set(ylim=(.085,.175),title='净距门限附近放大（范围外用箭头标注；完整数值见上图）',ylabel='净距（米）')
    axes[3].legend(fontsize=8,loc='upper left')
    labels = []
    for record in records:
        summary = record['summary']
        direction = {'uphill': '上', 'downhill': '下'}.get(summary.get('ramp_direction'), '')
        short = Path(record['directory']).name.rsplit('_', 1)[-1]
        digest = record['method_hashes']['install/terrain_nav/lib/terrain_nav/terrain_nav_node'][:7]
        mapping=record['method_hashes'].get('install/fast_lio/lib/fast_lio/fastlio_mapping','unknown')[:7]
        labels.append(f"{short}\n{record.get('case_name',case_name(summary))}{direction}\nN:{digest}\nL:{mapping}")
    axes[3].set_xticks(x, labels, fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    figure.suptitle('定向整链资格记录：按试次与二进制标识展示，不能混算为导航成功率', fontsize=13)
    for extension in ('png', 'pdf'):
        figure.savefig(output.with_suffix('.'+extension), dpi=180)
    plt.close(figure)


def index_attempt(path):
    result = json.loads((path / 'result.json').read_text())
    metadata = json.loads((path / 'run_metadata.json').read_text())
    artifacts = {}
    for name in ('result.json','run_metadata.json','source_snapshot.zip','navigation_data.json',
                 'motion_streams.json','return_motion_streams.json','goal_trace.json','return_goal_trace.json'):
        file = path / name
        if file.is_file():
            artifacts[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    if not metadata.get('source_snapshot_sha256') or metadata['source_snapshot_sha256'] != artifacts.get('source_snapshot.zip'):
        raise ValueError(f'source snapshot missing or hash mismatch: {path}')
    summaries = {}
    for name in ('navigation_data.json','goal_trace.json','return_goal_trace.json'):
        if name in artifacts:
            summaries[name] = json.loads((path / name).read_text()).get('summary')
    return dict(directory=str(path.resolve()), result=result, metadata=metadata,
                summaries=summaries, artifact_sha256=artifacts)


def load_run(path):
    index_attempt(path)  # Validate archived provenance before deriving new figures.
    data_path = path / 'navigation_data.json'
    data = json.loads(data_path.read_text())
    result = json.loads((path / 'result.json').read_text())
    streams = json.loads((path / 'motion_streams.json').read_text())
    metadata = json.loads((path / 'run_metadata.json').read_text())
    summary = data['summary']
    if summary['scenario'] not in NAMES:
        raise ValueError('unsupported navigation-course attempt')
    return dict(directory=str(path.resolve()), data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        overall_passed=result['passed'], navigation_accepted=data['accepted'],
        streams_passed=streams['passed'], failure_reason=result.get('failure_reason'),
        arguments=metadata['arguments'], method_hashes=metadata['method_hashes'],
        summary=summary, stream_statistics=streams['summaries'],
        case_name=case_name(summary,data),planning_statistics=planning_statistics(path)), data


def plot_run(record, data, output):
    truth = np.asarray(data['truth'], dtype=float)
    if len(truth) < 2:
        raise ValueError('trajectory missing')
    t = truth[:, 6]-truth[0, 6]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    ax = axes[0]
    if record['summary'].get('ramp_rise_m') is not None:
        ax.axvspan(1.5,5.5,color='#d9dfe2',alpha=.5,label='实际坡段 X 范围')
    geometry=list(record['summary'].get('course_geometry_world_m', []))
    if record['summary']['scenario']=='boundary':
        geometry=boundary_geometry(Path(record['directory']),data)
    if data.get('front_sign_body_clearances'):
        geometry.extend(front_sign_obstacles())
    for obstacle in geometry:
        if obstacle['shape'] == 'cylinder':
            patch = Circle((obstacle['x'], obstacle['y']), obstacle['radius'])
        else:
            patch = Rectangle((obstacle['x']-obstacle['length']/2., obstacle['y']-obstacle['width']/2.),
                              obstacle['length'], obstacle['width'])
        patch.set(facecolor='#dbc9aa' if obstacle.get('terrain') else '#a4abb0', edgecolor='#4e555c', linewidth=.7)
        ax.add_patch(patch)
    for name,rows in data.get('moving_obstacles',{}).items():
        values=np.asarray(rows,float)
        if len(values):
            ax.plot(values[:,1],values[:,2],linestyle=':',linewidth=1.2,label=f'{name} 运动轨迹')
            ax.text(values[-1,1],values[-1,2],name,fontsize=7)
    obstacles = np.asarray(data.get('obstacle', []), dtype=float)
    if record['summary']['scenario'] == 'reserve_overlap':
        obstacles = np.empty((0, 3))  # Navigation uses the stationary box above; preparation is separately archived.
    if len(obstacles):
        ax.plot(obstacles[:, 1], obstacles[:, 2], color='#b06948', linestyle=':',
                linewidth=1.2, label='动态障碍物轨迹')
        for obstacle in obstacles[::max(1,len(obstacles)//6)]:
            ax.add_patch(Circle(obstacle[1:3], .65, facecolor='#b06948',
                                edgecolor='#80513b', alpha=.15, linewidth=.8))
    ax.plot(truth[:, 1], truth[:, 2], color='#187e8a', linewidth=2., label='独立真值轨迹')
    ax.scatter(*data['truth_goal'], marker='*', s=95, color='#c66335', label='目标')
    # Show the actual sampled body envelope, not only a center-line clearance.
    for sample in truth[::max(1, len(truth)//18)]:
        c, s = np.cos(sample[7]), np.sin(sample[7])
        local = np.array([[.58, .4], [-.48, .4], [-.48, -.4], [.58, -.4], [.58, .4]])
        xy = local @ np.array([[c, s], [-s, c]]) + sample[1:3]
        ax.plot(xy[:, 0], xy[:, 1], color='#187e8a', alpha=.25, linewidth=.65)
    ax.set(title='真值轨迹与专项障碍（非完整环境）', xlabel='世界 X（米）', ylabel='世界 Y（米）', aspect='equal')
    ax.legend(loc='best', fontsize=8)
    clearance_series = data.get('course_body_clearances') or {}
    clearance_series = dict(clearance_series, **(data.get('front_sign_body_clearances') or {}))
    if record['summary']['scenario']=='boundary':
        clearance_series=static_course_clearances(data['truth'],[o for o in geometry if not o['terrain']])
        for obstacle in boundary_courses.movers(data['spec'],0.):
            name=obstacle['name']
            clearance_series[name]=sampled_body_clearances(data['truth'],data['moving_obstacles'][name],
                radius=obstacle['radius'],front=.58,back=.48,half_width=.4)
    if len(obstacles):
        clearance_series = dict(clearance_series, moving=data['sampled_body_clearances'])
    pilaster_labeled=False
    for index, (name, samples) in enumerate(clearance_series.items(), 1):
        values = np.asarray(samples)
        label={'FrontSignPanel':'标牌面板（空间分离下界）','FrontSignPost':'背景标牌支柱',
               'moving':'动态横穿圆柱','SlopePillar':'坡面立柱','FrontReserve':'前方静态箱体',
               'Pillar_0':'错列立柱 1','Pillar_1':'错列立柱 2','Pillar_2':'错列立柱 3',
               'OverheadBeam':'横梁（空间分离下界）',
               'CrossA':'横穿物体 A','CrossB':'横穿物体 B','Lead':'纵向物体',
               'Lanem1':'通道侧墙 1','Lane1':'通道侧墙 2',
               'Wall_0':'通道墙 1','Wall_1':'通道墙 2'}.get(name,f'专项障碍 {index}')
        if name.startswith('Pilaster_'):
            # Retain every per-object curve, without an unreadable 11-item
            # legend or silently averaging away the closest obstacle.
            axes[1].plot(values[:,0]-truth[0,6],values[:,1],color='#a5afb6',alpha=.55,linewidth=.8,
                         label='附加立柱（逐柱细线）' if not pilaster_labeled else '_nolegend_')
            pilaster_labeled=True
        else:
            axes[1].plot(values[:, 0]-truth[0, 6], values[:, 1], linewidth=1.4, label=label)
    axes[1].axhline(.10, color='#bf4850', linestyle='--', linewidth=1., label='验收净距 0.10 米')
    axes[1].set(title='每个障碍的采样车身净距', xlabel='模拟时间（秒）', ylabel='净距（米；负值为重叠）')
    axes[1].legend(fontsize=8)
    if not clearance_series:
        axes[1].clear()
        axes[1].plot(t,np.degrees(np.abs(truth[:,4])),color='#437ea1',label='实际俯仰绝对值')
        axes[1].plot(t,np.degrees(np.abs(truth[:,5])),color='#b06948',label='实际横滚绝对值')
        axes[1].axhline(np.degrees(.55),color='#bf4850',linestyle='--',linewidth=1.,label='俯仰限制')
        axes[1].axhline(np.degrees(.45),color='#725d8c',linestyle=':',linewidth=1.,label='横滚限制')
        axes[1].set(title='实际车身姿态与限制（无任务障碍）',xlabel='模拟时间（秒）',ylabel='绝对角度（度）')
        axes[1].legend(fontsize=8)
    axes[2].plot(t, truth[:, 3]-truth[0, 3], color='#437ea1', label='相对起点的实际高度')
    axes[2].set(title='实际高度、带符号命令与速率大小', xlabel='模拟时间（秒）', ylabel='相对高度（米）')
    heights = truth[:, 3]-truth[0, 3]
    # Do not visually magnify submillimetre flat-ground suspension noise.
    span = max(.05, float(np.ptp(heights)))
    axes[2].set_ylim(float(min(heights))-.1*span, float(max(heights))+.9*span if np.ptp(heights)<.05
                     else float(max(heights))+.1*span)
    control_ax = axes[2].twinx()
    controls = np.asarray(data['controls'])
    if len(controls):
        control_ax.plot(controls[:, 0]-truth[0, 6], controls[:, 1], color='#c66335', alpha=.75,
                        linewidth=1., label='命令速度')
    unique=np.r_[True,np.diff(truth[:,6])>1e-6]
    positions=truth[unique]
    if len(positions)>1:
        speed=np.hypot(np.diff(positions[:,1]),np.diff(positions[:,2]))/np.diff(positions[:,6])
        control_ax.plot(positions[1:,6]-truth[0,6],speed,color='#525e64',
                        linewidth=.8,alpha=.75,linestyle='--',label='真值水平速率大小（位移差分）')
    control_ax.set_ylabel('速度 / 速率（米/秒）')
    axes[2].legend(loc='upper left', fontsize=8)
    control_ax.legend(loc='upper right', fontsize=8)
    encounter=record['summary'].get('held_velocity_encounter')
    response=record['summary'].get('event_stop_response')
    if encounter is not None:
        event=encounter['event_stamp_s']-truth[0,6]
        stop=response.get('first_sustained_truth_stop_delay_s') if response else None
        for index in (1,2):
            axes[index].axvline(event,color='#725d8c',linestyle=':',linewidth=1.)
            if stop is not None:
                axes[index].axvspan(event,event+stop,color='#725d8c',alpha=.10)
        zero=response.get('first_zero_command_delay_s') if response else None
        clearance=record['summary'].get('sampled_body_clearance_min_m')
        label=f'横穿开始 {event:.2f} 秒'
        if zero is not None:label+=f'；零速命令延迟 {zero:.3f} 秒'
        if stop is not None:label+=f'\n持续实际停稳延迟 {stop:.3f} 秒'
        if clearance is not None:label+=f'；最小净距 {clearance:.3f} 米'
        axes[1].text(.02,.98,label,transform=axes[1].transAxes,fontsize=8,va='top',
                     bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=3))
        values=np.asarray(data['sampled_body_clearances'])
        minimum=values[np.argmin(values[:,1])]
        axes[1].scatter(minimum[0]-truth[0,6],minimum[1],s=18,color='#bf4850',zorder=4)
    for ax in axes:
        ax.grid(alpha=.18)
    status = '通过' if record['overall_passed'] else '未通过（失败记录保留）'
    direction = {'uphill':'上坡','downhill':'下坡'}.get(record['summary'].get('ramp_direction'),'')
    rise=record['summary'].get('ramp_rise_m')
    grade=f'；坡面抬升 {rise:.2f} 米' if rise is not None else ''
    figure.suptitle(f"{record.get('case_name',case_name(record['summary']))}{direction}{grade}：{status}；{Path(record['directory']).name}\n"
                   '受控场景资格测试；不是成功率估计，也不是连续碰撞证明', fontsize=12)
    for extension in ('png', 'pdf'):
        figure.savefig(output.with_suffix('.'+extension), dpi=180)
    plt.close(figure)


def constraint_figure(record,data,output):
    """Display reported model checks, including gaps; never replace NaN by zero."""
    origin=data['truth'][0][6];end=data['truth'][-1][6]
    rows=[(s['stamp_s'],v['values']) for s in data.get('safety_status',[])
          for v in s['status'] if v['name']=='terrain_nav/chance_constraints' and origin<=s['stamp_s']<=end]
    if not rows:return
    t=np.array([r[0]-origin for r in rows])
    def values(key):
        result=[]
        for _,row in rows:
            try:value=float(row.get(key,'nan'))
            except (ValueError,TypeError):value=np.nan
            result.append(value if np.isfinite(value) else np.nan)
        return np.asarray(result)
    figure,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
    for key,label,color in (('state_probability_upper','当前姿态模型上界','#197d89'),
                            ('braking_probability_upper','所报制动模型上界','#b06948')):
        axes[0].plot(t,values(key),marker='.',markersize=2,linewidth=.8,label=label,color=color)
    axes[0].plot(t,values('state_limit'),linestyle='--',color='#b64c5f',label='姿态预算')
    axes[0].plot(t,values('braking_budget'),linestyle=':',color='#62568a',label='制动预算')
    axes[0].set_yscale('symlog',linthresh=1e-4,linscale=.5)
    axes[0].set(ylim=(0,1.1),yticks=[0,.0001,.001,.01,.05,.2,1],
                yticklabels=['0%','0.01%','0.1%','1%','5%','20%','100%'],
                ylabel='条件模型上界（非线性百分比刻度）',
                title='对数轴、零附近线性；缺失保留断点；姿态通过不能替代路径、制动及碰撞检查')
    axes[0].legend(fontsize=9,ncol=2)
    axes[1].plot(t,1000*values('map_age_s'),color='#4875a5',linewidth=1.)
    axes[1].set(ylabel='源时间年龄（毫秒）',title='控制状态发布时的风险图年龄（模拟时钟，含消息持有期）',ylim=(0,None))
    for index,(key,label,color) in enumerate((('state_support_known','当前足迹支撑已知','#197d89'),
                  ('command_accepted','本次命令获准','#4875a5'),('provisional_motion_brake','临时运动候选制动','#c58b3d'))):
        mask=values(key)>.5
        axes[2].scatter(t[mask],np.full(mask.sum(),index),marker='|',s=45,color=color)
    axes[2].set(yticks=range(3),yticklabels=['当前足迹支撑已知','本次命令获准','临时运动候选制动'],
                ylim=(-.6,2.6),xlabel='模拟时间（秒）',title='每根短线对应一次标志为 1；空白包括 0 或缺失')
    for ax in axes:ax.grid(alpha=.2)
    figure.suptitle(f"{record['case_name']}：概率约束与消息时效\n条件高斯工程模型，尚非实地失效概率校准；{Path(record['directory']).name}",fontsize=12)
    for extension in ('png','pdf'):figure.savefig(output.with_suffix('.'+extension),dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index-only',action='store_true',help='Inventory all supplied attempts, including infrastructure and performance failures')
    args = parser.parse_args()
    if len({path.resolve() for path in args.run}) != len(args.run):
        parser.error('duplicate run directories are not independent attempts')
    if args.index_only:
        records = [index_attempt(path) for path in args.run]
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'inventory.json').write_text(json.dumps(dict(
            scope='explicit engineering qualification inventory; mixed revisions and infrastructure failures; not a navigation success-rate estimate',
            attempts=records), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(args.output),attempts=len(records))))
        return
    configure_chinese()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for path in args.run:
        record, data = load_run(path)
        records.append(record)
        plot_run(record, data, args.output / path.name)
        constraint_figure(record,data,args.output/(path.name+'_constraints'))
    runtime_figure(records, args.output / 'runtime_qualification')
    (args.output / 'summary.json').write_text(json.dumps(dict(
        scope='explicit selected qualification attempts including supplied failures; not a statistical campaign',
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runs=records), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), runs=len(records),
                         passed=sum(r['overall_passed'] for r in records)), ensure_ascii=False))


if __name__ == '__main__':
    main()
