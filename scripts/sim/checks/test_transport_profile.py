import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('overrides,expected', [({}, 'ASYNCHRONOUS'),
    ({'RMW_FASTRTPS_PUBLICATION_MODE':'SYNCHRONOUS'}, 'SYNCHRONOUS'),
    ({'HUNTER_ISAAC_DDS_PUBLICATION_MODE':'auto'}, 'AUTO'),
    ({'HUNTER_ISAAC_DDS_PUBLICATION_MODE':'asynchronous',
      'RMW_FASTRTPS_PUBLICATION_MODE':'SYNCHRONOUS'}, 'ASYNCHRONOUS')])
def test_sim_transport_defaults_and_explicit_overrides(overrides, expected):
    env = {k:v for k,v in os.environ.items() if k not in
           ('HUNTER_ISAAC_DDS_PUBLICATION_MODE','RMW_FASTRTPS_PUBLICATION_MODE')}
    env.update(overrides)
    command = 'source scripts/lib/runtime.sh\nRMW_IMPLEMENTATION=rmw_fastrtps_cpp\nhunter_configure_sim_transport\nprintenv RMW_FASTRTPS_PUBLICATION_MODE'
    result = subprocess.run(['bash','-ec',command],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    assert result.stdout.strip() == expected


def test_invalid_mode_fails_and_non_fastdds_transport_is_untouched():
    command = 'source scripts/lib/runtime.sh\nRMW_IMPLEMENTATION=rmw_fastrtps_cpp\nHUNTER_ISAAC_DDS_PUBLICATION_MODE=invalid\nhunter_configure_sim_transport'
    assert subprocess.run(['bash','-ec',command],cwd=ROOT,capture_output=True).returncode == 2
    command = 'source scripts/lib/runtime.sh\nRMW_IMPLEMENTATION=rmw_cyclonedds_cpp\nexport RMW_FASTRTPS_PUBLICATION_MODE=keep\nhunter_configure_sim_transport\nprintenv RMW_FASTRTPS_PUBLICATION_MODE'
    result = subprocess.run(['bash','-ec',command],cwd=ROOT,capture_output=True,text=True,check=True)
    assert result.stdout.strip() == 'keep'


@pytest.mark.parametrize('overrides,expected',[
    ({},str(ROOT/'scripts/sim/config/fastdds_large_maps.xml')),
    ({'FASTRTPS_DEFAULT_PROFILES_FILE':'/user/custom.xml'},'/user/custom.xml'),
    ({'HUNTER_ISAAC_DDS_PROFILE':'builtin'},None),
    ({'HUNTER_ISAAC_DDS_PROFILE':'builtin','FASTRTPS_DEFAULT_PROFILES_FILE':'/user/custom.xml'},'/user/custom.xml'),
])
def test_launchers_and_qualification_use_the_same_profile_without_overriding_user_xml(overrides,expected):
    from pipeline_smoke import configure_transport
    env={k:v for k,v in os.environ.items() if k not in
        ('HUNTER_ISAAC_DDS_PROFILE','FASTRTPS_DEFAULT_PROFILES_FILE','HUNTER_ISAAC_DDS_PUBLICATION_MODE')}
    env.update(overrides,RMW_IMPLEMENTATION='rmw_fastrtps_cpp')
    result=configure_transport(env)
    assert result.get('FASTRTPS_DEFAULT_PROFILES_FILE')==expected
    assert result['RMW_IMPLEMENTATION']=='rmw_fastrtps_cpp'


def test_invalid_profile_does_not_silently_change_transport():
    from pipeline_smoke import configure_transport
    with pytest.raises(subprocess.CalledProcessError):
        configure_transport(dict(os.environ,RMW_IMPLEMENTATION='rmw_fastrtps_cpp',HUNTER_ISAAC_DDS_PROFILE='invalid'))


def test_large_map_profile_is_accepted_by_the_installed_fastdds_and_has_payload_headroom():
    import xml.etree.ElementTree as ET
    profile=ROOT/'scripts/sim/config/fastdds_large_maps.xml'
    root=ET.parse(profile).getroot();ns={'d':'http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles'}
    descriptors=root.findall('d:transport_descriptors/d:transport_descriptor',ns)
    shm=next(d for d in descriptors if d.find('d:type',ns).text=='SHM')
    message=int(shm.find('d:maxMessageSize',ns).text)
    assert message>800_000*32  # bounded accumulated PCL map, not just one scan
    assert int(shm.find('d:segment_size',ns).text)>=4*message
    assert any(d.find('d:type',ns).text=='UDPv4' for d in descriptors)
    env=dict(os.environ,ROS_DOMAIN_ID='86',RMW_IMPLEMENTATION='rmw_fastrtps_cpp',
             FASTRTPS_DEFAULT_PROFILES_FILE=str(profile))
    code='import rclpy; rclpy.init(); n=rclpy.create_node("ptr_xml_qualification"); n.destroy_node(); rclpy.shutdown()'
    result=subprocess.run(['python3','-c',code],env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert 'XMLPARSER' not in result.stderr and 'XMLPARSER' not in result.stdout
