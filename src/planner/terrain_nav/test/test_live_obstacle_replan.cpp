// Standalone CPU regression using the real production control callback.
#define TERRAIN_NAV_BENCHMARK
#ifdef TERRAIN_NAV_TEST_NODE_SOURCE
#include TERRAIN_NAV_TEST_NODE_SOURCE
#else
#include "../src/terrain_nav_node.cpp"
#endif
#include <rcl/time.h>
#include <iostream>
#include <stdexcept>

struct TerrainNavBenchmark
{
  static void require(bool condition, const std::string &message)
  {
    if (!condition) throw std::runtime_error(message);
  }

  static rclcpp::NodeOptions options()
  {
    rclcpp::NodeOptions out;
    out.parameter_overrides({rclcpp::Parameter("use_sim_time", true),
      rclcpp::Parameter("live_obstacle_enable", true),
      rclcpp::Parameter("cmd_topic", "/planning_diagnostics/replan_guard/cmd_vel"),
      rclcpp::Parameter("replan_min_interval", .4)});
    return out;
  }

  static void clock(TerrainNavNode &node, double now)
  {
    auto *handle = node.get_clock()->get_clock_handle();
    require(rcl_enable_ros_time_override(handle) == RCL_RET_OK, "clock enable failed");
    require(rcl_set_ros_time_override(handle, static_cast<int64_t>(now * 1e9)) == RCL_RET_OK,
            "clock set failed");
    node.odom_.stamp = node.now();
    auto map = std::make_shared<ControlMapSnapshot>();
    map->stamp = node.now();
    std::atomic_store(&node.control_map_snapshot_, std::shared_ptr<const ControlMapSnapshot>(map));
    auto live = std::make_shared<LiveObstacleSnapshot>();
    live->min_x = live->min_y = -5.;
    live->resolution = .1;
    live->width = live->height = 100;
    live->stamp_sec = now;
    live->hard.assign(10000, 0);
    for (int iy = 46; iy <= 54; ++iy)
      for (int ix = 68; ix <= 72; ++ix) live->hard[iy * 100 + ix] = 1;
    std::atomic_store(&node.live_snapshot_, std::shared_ptr<const LiveObstacleSnapshot>(live));
  }

  static void prepare(TerrainNavNode &node, double now, double last, bool changed, bool running)
  {
    node.have_goal_ = true;
    node.goal_seq_ = 1;
    node.raw_goal_ = Pose2{8., 0., 0., 0.};
    node.active_goal_ = node.raw_goal_;
    node.map_ready_ = true;
    node.map_dirty_ = false;
    node.odom_.valid = true;
    node.odom_.x = node.odom_.y = node.odom_.yaw = node.odom_.v = 0.;
    node.planner_running_.store(running);
    node.planner_wakeup_ = false;
    node.plan_requested_.store(false);
    node.last_plan_time_sec_.store(last);
    node.live_dirty_.store(changed);
    node.last_cmd_v_ = .1;
    node.last_cmd_steer_ = .2;
    node.path_ = {Pose2{0., 0., 0., 0.}, Pose2{4., 0., 0., 4.}};
    node.have_path_ = true;
    clock(node, now);
  }

  static void check(bool changed, bool running, double now, double last, bool queued)
  {
    TerrainNavNode node(options(), false);
    prepare(node, now, last, changed, running);
    require(node.livePathAheadUnsafe(node.path_, 0), "test path does not intersect live mask");
    node.controlTimer();
    require(node.path_.empty() && !node.have_path_, "unsafe path retained");
    require(node.last_cmd_v_ == 0. && node.last_cmd_steer_ == 0., "stop was not immediate");
    require(node.planner_wakeup_ == queued, "unexpected live-obstacle request state");
    if (queued) require(node.planner_reason_ == "new live obstacle", "wrong request reason");
  }

  static void retry_after_cooldown()
  {
    TerrainNavNode node(options(), false);
    prepare(node, 10.02, 10., false, false);
    node.controlTimer();
    require(!node.planner_wakeup_, "cooldown did not suppress initial rejected replacement");
    clock(node, 10.42);
    node.controlTimer();
    require(node.planner_wakeup_, "cleared path never retried after cooldown");
    require(node.last_cmd_v_ == 0., "retry did not remain stopped");
  }

  static void repeated_rejections()
  {
    TerrainNavNode node(options(), false);
    double last = -1.;
    int requests = 0;
    for (int i = 0; i < 2000; ++i) {
      const double now = 10. + i * .02;
      prepare(node, now, last, i == 0, false);
      node.controlTimer();
      require(node.path_.empty() && node.last_cmd_v_ == 0. && node.last_cmd_steer_ == 0.,
              "50 Hz unsafe-path rejection failed to stop");
      if (node.planner_wakeup_) {
        ++requests;
        last = now;
      }
    }
    std::cout << "[replan guard] 40 simulated seconds, 2000 rejected replacements, "
              << requests << " requests at interval 0.4s" << std::endl;
    require(requests > 1 && requests <= 100, "unchanged-mask replans exceed cooldown bound");
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rcutils_logging_set_logger_level("terrain_nav_node", RCUTILS_LOG_SEVERITY_ERROR);
  int failed = 0;
  const auto run = [&](const char *name, auto test) {
    try { test(); std::cout << "PASS " << name << std::endl; }
    catch (const std::exception &error) {
      ++failed;
      std::cerr << "FAIL " << name << ": " << error.what() << std::endl;
    }
  };
  run("unchanged_mask_respects_cooldown", [] { TerrainNavBenchmark::check(false, false, 10.02, 10., false); });
  run("changed_mask_requests_immediately", [] { TerrainNavBenchmark::check(true, false, 10.02, 10., true); });
  run("unchanged_mask_does_not_queue_while_running", [] { TerrainNavBenchmark::check(false, true, 11., 10., false); });
  run("changed_mask_does_not_queue_while_running", [] { TerrainNavBenchmark::check(true, true, 11., 10., false); });
  run("unchanged_mask_retries_after_cooldown", [] { TerrainNavBenchmark::check(false, false, 10.42, 10., true); });
  run("initial_plan_is_not_suppressed", [] { TerrainNavBenchmark::check(false, false, 10., -1., true); });
  run("cleared_path_eventually_retries", [] { TerrainNavBenchmark::retry_after_cooldown(); });
  run("forty_second_request_bound", [] { TerrainNavBenchmark::repeated_rejections(); });
  rclcpp::shutdown();
  std::cout << "[replan guard] 8 checks, " << failed << " failed" << std::endl;
  return failed ? 1 : 0;
}
