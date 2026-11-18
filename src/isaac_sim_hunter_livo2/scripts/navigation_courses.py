"""Controlled physical fixtures; used only by the simulator and truth evaluator.

Never publish these obstacle coordinates to perception or the navigation node.
All dimensions are metres in the simulator world frame.
"""

OVERHEAD_SCENARIOS = ('overhead_pass', 'overhead_block', 'overhead_plain')
COMPOSITE_SCENARIOS = ('staggered', 'corridor', 'ramp_detour', 'reserve_overlap', *OVERHEAD_SCENARIOS)
RAMP_SCENARIOS = ('ramp', 'ramp_detour', 'ramp_crossing')
CROSSING_SCENARIOS = ('crossing', 'ramp_crossing')


def front_sign_obstacles():
    """Background sign geometry, shared by scene authoring and truth assessment."""
    return (dict(name='FrontSignPost', shape='cylinder', x=5.4, y=-3.25, z=.48,
                 radius=.035, height=.84),
            dict(name='FrontSignPanel', shape='box', x=5.4, y=-3.25, z=1.04,
                 length=.04, width=.76, height=.54))


def default_obstacle_trigger_distance(scenario):
    return 2.0 if scenario == 'ramp_crossing' else 1.10


def course_obstacles(scenario):
    if scenario in OVERHEAD_SCENARIOS:
        bottom=.35 if scenario=='overhead_block' else .95
        # Keep the unfeatured version as an explicit localization stress case.
        # Physical pilasters in the paired clearance fixture add longitudinal
        # LiDAR normals; they narrow the channel instead of removing hazards.
        pilasters=() if scenario=='overhead_plain' else tuple(
            dict(name=f'Pilaster_{"Left" if side>0 else "Right"}_{i}',shape='box',x=x,y=side*1.25,z=.73,
                 length=.30,width=.30,height=1.50)
            for side,xs in ((-1,(-1.5,.5,2.5,5.5,8.,10.)),(1,(-.5,1.5,3.3,6.5,9.)))
            for i,x in enumerate(xs))
        return (*tuple(dict(name=f'Wall_{i}',shape='box',x=3.5,y=y,z=.73,
                            length=20.,width=.3,height=1.5) for i,y in enumerate((-1.35,1.35))),
                *pilasters,
                dict(name='OverheadBeam',shape='box',x=4.,y=0.,z=bottom+.13,
                     bottom_m=bottom,length=.4,width=3.,height=.30))
    if scenario == 'staggered':
        return tuple(dict(name=f'Pillar_{i}', shape='cylinder', x=x, y=y,
                          radius=.65, height=1.8)
                     for i, (x, y) in enumerate(((3., .5), (6., -.5), (9., .5))))
    if scenario == 'corridor':
        return tuple(dict(name=f'Wall_{i}', shape='box', x=4.5, y=y,
                          length=5., width=.3, height=1.5)
                     for i, y in enumerate((-1.35, 1.35)))
    if scenario == 'ramp_detour':
        return (dict(name='SlopePillar', shape='cylinder', x=3.5, y=0.,
                     radius=.65, height=1.8),)
    if scenario == 'reserve_overlap':
        return (dict(name='FrontReserve', shape='box', x=3.1, y=0.,
                     length=.2, width=.3, height=1.2),)
    return ()


def course_distance(scenario):
    return {'staggered': 12., 'corridor': 9., 'ramp_detour': 7., 'ramp_crossing': 7.,
            'reserve_overlap': 4.}.get(scenario, 7.)


def minimum_course_distance(scenario):
    if scenario in OVERHEAD_SCENARIOS:
        # The task crosses the beam inside the channel, not the ends of its
        # long side walls. Both cases use the same goal and world geometry.
        return 4.2+.48+.30
    fixtures = course_obstacles(scenario)
    end = max(o['x'] + (o['radius'] if o['shape'] == 'cylinder' else o['length']/2.)
              for o in fixtures) if fixtures else 0.
    if scenario in RAMP_SCENARIOS:
        end = max(end, 5.5)
    return end + .48 + .30
