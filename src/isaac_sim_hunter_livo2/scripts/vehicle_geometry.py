"""Measured specification and conservative CAD wheel sweep, without Isaac/ROS.

Datasheet dimensions describe the straight vehicle. A steering tyre can extend
outside that rectangle. Never replace the complete swept body by 626mm just
to pass a gap. Axle geometry is explicitly the existing CAD, not a measurement
invented from the datasheet's turning radius. Axle spacing/track are fitted
from the user's dimension drawing before Isaac imports the CAD joints.
"""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


SPEC_FILE = Path(__file__).resolve().parents[3] / 'config/vehicles/neos.json'


def load_spec():
    spec = json.loads(SPEC_FILE.read_text())
    for field in ('length_m', 'width_m', 'height_m', 'empty_mass_kg',
                  'center_turn_radius_m', 'wheelbase_m', 'track_width_m', 'max_climb_deg', 'vertical_obstacle_m'):
        if not math.isfinite(spec[field]) or spec[field] <= 0:
            raise ValueError('invalid vehicle specification: '+field)
    return spec


def geometry(spec=None):
    p = load_spec() if spec is None else spec
    length = p['wheelbase_m']
    centre = length/2 - p['rear_axle_offset_m']
    radius = math.sqrt(p['center_turn_radius_m']**2 - (length/2)**2)
    return dict(wheelbase=length, centre_x=centre,
                front=p['length_m']/2+centre, back=p['length_m']/2-centre,
                half_width=p['width_m']/2, rear_radius=radius,
                max_steer=math.atan2(length, radius))


def tyre_rectangles(steer):
    """CAD tyre XY rectangles in base_link, using actual axial mesh offsets."""
    from wheel_control import ackermann_joint_targets
    p = geometry()
    spec=load_spec();half_track=spec['track_width_m']/2;half_axle=p['wheelbase']/2
    ys=(half_track-.0214,-half_track+.0210)
    angles, _, _, _ = ackermann_joint_targets(0., steer, p['wheelbase'],
        (p['wheelbase'],p['wheelbase']),ys,(half_track,-half_track),.69)
    result = []
    for x, y, offset, angle in zip((half_axle,half_axle),ys,(.0214,-.0210),angles):
        c, s = math.cos(angle), math.sin(angle)
        cx, cy = x-offset*s, y+offset*c
        result.append(tuple((cx+u*c-v*s, cy+u*s+v*c)
            for u in (-.1348,.1348)for v in (-spec['wheel_width_m']/2,spec['wheel_width_m']/2)))
    for y in (half_track,-half_track):
        result.append(tuple((-half_axle+u,y+v)for u in (-.13495,.13495)for v in (-spec['wheel_width_m']/2,spec['wheel_width_m']/2)))
    return result


def fit_neos_urdf(text):
    """Adapt expanded CAD joint origins AND tyre collisions before importing.

    Steering pivots differ from wheel centres because the CAD tyre meshes have
    axial offsets. The labelled 548mm track refers to tyre centres, not pivots.
    """
    root=ET.fromstring(text);p=load_spec();x=p['wheelbase_m']/2;y=p['track_width_m']/2
    for joint in root.iter('joint'):
        name=joint.attrib['name'].removeprefix('${prefix}');parent=joint.find('parent')
        if parent is None or parent.attrib['link'].removeprefix('${prefix}')!='base_link':continue
        target=None
        if name in ('fr_susp_left_joint','fr_steer_left_joint'):target=(x,y-.0214)
        elif name in ('fr_susp_right_joint','fr_steer_right_joint'):target=(x,-y+.0210)
        elif name in ('re_susp_left_joint','re_left_joint'):target=(-x,y+.0145)
        elif name in ('re_susp_right_joint','re_right_joint'):target=(-x,-y-.0145)
        if target is not None:
            origin=joint.find('origin');z=float(origin.attrib['xyz'].split()[2])
            origin.set('xyz',f'{target[0]:.9f} {target[1]:.9f} {z:.9f}')
    for link in root.iter('link'):
        if link.attrib['name'].removeprefix('${prefix}') not in ('fr_left_link','fr_right_link','re_left_link','re_right_link'):continue
        for shape in link.findall('collision/geometry/cylinder'):shape.set('length',str(p['wheel_width_m']))
    return ET.tostring(root,encoding='unicode')


def chassis_vertical_bounds():
    p=load_spec();return p['ground_clearance_m']-p['nominal_base_height_m'],p['height_m']-p['nominal_base_height_m']


def swept_bounds():
    p = geometry()
    xs = [-p['back'], p['front']];ys = [-p['half_width'], p['half_width']]
    for i in range(401):
        angle=p['max_steer']*(2*i/400-1)
        for rectangle in tyre_rectangles(angle):
            xs.extend(q[0] for q in rectangle);ys.extend(q[1] for q in rectangle)
    # One millimetre outward rounding encloses sampling/float conversion.
    up=lambda x: math.ceil((x+.0001)*1000)/1000
    return dict(front=up(max(xs)), back=up(-min(xs)), half_width=up(max(map(abs,ys))))
