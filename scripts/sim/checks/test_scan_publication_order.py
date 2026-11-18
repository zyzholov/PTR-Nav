"""Keep navigation scan publication ahead of nonessential accumulated-map work."""
from pathlib import Path


def test_corrected_navigation_scan_precedes_increment_and_history_maintenance():
    source=(Path(__file__).resolve().parents[3]/'src/FAST_LIO/src/laserMapping.cpp').read_text()
    start=source.index('publish_odometry(pubOdomAftMapped_, tf_broadcaster_);')
    finish=source.index('void map_publish_callback()',start)
    callback=source[start:finish]
    assert callback.count('publish_frame_world(pubLaserCloudFull_)')==1
    assert callback.index('publish_frame_world(pubLaserCloudFull_)')<callback.index('map_incremental();')
    assert callback.index('map_incremental();')<callback.index('accumulate_map_frame();')
    assert 'Slow mapping frame: stamp=' in callback


def test_full_resolution_collision_reference_is_gated_and_precedes_history():
    source=(Path(__file__).resolve().parents[3]/'src/FAST_LIO/src/laserMapping.cpp').read_text()
    start=source.index('publish_odometry(pubOdomAftMapped_, tf_broadcaster_);')
    callback=source[start:source.index('void map_publish_callback()',start)]
    assert callback.index('publish_frame_world(pubLaserCloudFull_)')<callback.index('publish_registered_cloud(feats_undistort, pubLaserCloudVisual_)')
    assert callback.index('publish_registered_cloud(feats_undistort, pubLaserCloudVisual_)')<callback.index('map_incremental();')
    assert 'visual_scan_pub_en && pubLaserCloudVisual_->get_subscription_count()>0' in callback
    assert 'publish_registered_cloud(dense_pub_en ? feats_undistort : feats_down_body, pubLaserCloudFull)' in source
    assert '"/cloud_registered_visual", 1' in source
