#pragma once

#include "mpc_controller/msg/se2_traj.hpp"
#include "mpc_controller/msg/poly_traj.hpp"
#include "mpc_controller/msg/single_poly_ac.hpp"
#include "mpc_controller/msg/poly_traj_ac.hpp"
#include "mpc_controller/msg/d_ptraj_container.hpp"
#include "utils/minco_traj.hpp"
#include "utils/dptraj.hpp"
#include <rclcpp/rclcpp.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

class TrajPoint
{
public:
    double x = 0;
    double y = 0;
    double v = 0;
    double cur = 0;
    double steer_vel = 0;
    double theta = 0;
    double a = 0;
    double phi = 0;
};

class TrajAnalyzer{
private:
    mpc_utils::SE2Traj minco_traj;
    mpc_utils::FlatTraj2d flat_traj;
    mpc_utils::MINCO_SE2 minco_anal;
    dptraj_mpc::UgvTrajectory ugv_traj;

    
    double traj_duration;
    bool use_flat = false;
    bool use_dp = false;
    std::shared_ptr<rclcpp::Node> node_;

    double currentRelativeTime() const
    {
      rclcpp::Time time_now;
      if (node_) {
          time_now = node_->get_clock()->now();
      } else {
          time_now = rclcpp::Clock().now();
      }
      return static_cast<double>(time_now.nanoseconds() - start_time.nanoseconds()) * 1e-9;
    }

    TrajPoint sampleAt(double t)
    {
      TrajPoint tp;
      const double temp = std::clamp(t, 0.0, traj_duration);
      if (use_flat)
      {
        Eigen::Vector2d po = flat_traj.getPos(temp);
        tp.x = po[0];
        tp.y = po[1];
        tp.v = flat_traj.getLonVel(temp);
        tp.a = flat_traj.getLonAcc(temp);
        tp.cur = flat_traj.getCur(temp);
        tp.theta = flat_traj.getAngle(temp);
      }
      else if (use_dp)
      {
        Eigen::Vector2d po = ugv_traj.getPos(temp);
        tp.x = po[0];
        tp.y = po[1];
        tp.v = ugv_traj.getVelItem(temp);
        tp.cur = ugv_traj.getCur(temp);
        tp.phi = ugv_traj.getPhi(temp);
        tp.theta = ugv_traj.getYaw(temp);
      }
      else
      {
        Eigen::Vector2d po = minco_traj.pos_traj.getPos(temp);
        tp.x = po[0];
        tp.y = po[1];
        tp.theta = minco_traj.angle_traj.getPos(temp)[0];
      }
      return tp;
    }

    double nearestTimeTo(double x, double y, double sample_dt)
    {
      if (traj_duration <= 0.0)
        return 0.0;

      const double step = std::max(0.03, sample_dt);
      double best_t = 0.0;
      double best_d2 = std::numeric_limits<double>::infinity();
      for (double t = 0.0; t <= traj_duration + 1e-6; t += step)
      {
        const TrajPoint p = sampleAt(std::min(t, traj_duration));
        const double dx = p.x - x;
        const double dy = p.y - y;
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d2)
        {
          best_d2 = d2;
          best_t = std::min(t, traj_duration);
        }
      }

      // Always evaluate the exact terminal point; the loop may miss it when
      // duration is not an integer multiple of sample_dt.
      const TrajPoint tail = sampleAt(traj_duration);
      const double dx = tail.x - x;
      const double dy = tail.y - y;
      const double d2 = dx * dx + dy * dy;
      if (d2 < best_d2)
        best_t = traj_duration;

      return best_t;
    }

    double nearestTimeToInRange(double x, double y, double sample_dt, double t_min, double t_max)
    {
      if (traj_duration <= 0.0)
        return 0.0;

      const double lo = std::clamp(std::min(t_min, t_max), 0.0, traj_duration);
      const double hi = std::clamp(std::max(t_min, t_max), 0.0, traj_duration);
      if (hi <= lo + 1e-6)
        return lo;

      const double step = std::max(0.03, sample_dt);
      double best_t = lo;
      double best_d2 = std::numeric_limits<double>::infinity();
      for (double t = lo; t <= hi + 1e-6; t += step)
      {
        const TrajPoint p = sampleAt(std::min(t, hi));
        const double dx = p.x - x;
        const double dy = p.y - y;
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d2)
        {
          best_d2 = d2;
          best_t = std::min(t, hi);
        }
      }

      const TrajPoint tail = sampleAt(hi);
      const double dx = tail.x - x;
      const double dy = tail.y - y;
      const double d2 = dx * dx + dy * dy;
      if (d2 < best_d2)
        best_t = hi;

      return best_t;
    }

public:
    rclcpp::Time start_time;
    bool at_goal = false;

    TrajAnalyzer() : node_(nullptr) {}
    
    double nearestTimeToPublic(double x, double y, double sample_dt) {
        return nearestTimeTo(x, y, sample_dt);
    }

    void setNode(std::shared_ptr<rclcpp::Node> node) {
        node_ = node;
    }

    void setTraj(std::shared_ptr<const mpc_controller::msg::SE2Traj> msg)
    {
      auto clock_type = node_ ? node_->get_clock()->get_clock_type() : RCL_SYSTEM_TIME;
      start_time = rclcpp::Time(msg->start_time, clock_type);

      Eigen::MatrixXd posP(2, msg->pos_pts.size() - 2);
      Eigen::MatrixXd angleP(1, msg->angle_pts.size() - 2);
      Eigen::VectorXd posT(msg->pos_t_pts.size());
      Eigen::VectorXd angleT(msg->angle_t_pts.size());
      Eigen::MatrixXd initS, tailS;

      for (int i = 1; i < (int)msg->pos_pts.size() - 1; i++)
      {
        posP(0, i - 1) = msg->pos_pts[i].x;
        posP(1, i - 1) = msg->pos_pts[i].y;
      }

      for (int i = 1; i < (int)msg->angle_pts.size() - 1; i++)
      {
        angleP(0, i - 1) = msg->angle_pts[i].x;
      }

      for (int i = 0; i < (int)msg->pos_t_pts.size(); i++)
      {
        posT(i) = msg->pos_t_pts[i];
      }

      for (int i = 0; i < (int)msg->angle_t_pts.size(); i++)
      {
        angleT(i) = msg->angle_t_pts[i];
      }

      initS.setZero(2, 3);
      tailS.setZero(2, 3);
      initS.col(0) = Eigen::Vector2d(msg->pos_pts[0].x, msg->pos_pts[0].y);
      initS.col(1) = Eigen::Vector2d(msg->init_v.x, msg->init_v.y);
      initS.col(2) = Eigen::Vector2d(msg->init_a.x, msg->init_a.y);
      tailS.col(0) = Eigen::Vector2d(msg->pos_pts.back().x, msg->pos_pts.back().y);
      tailS.col(1) = Eigen::Vector2d::Zero();
      tailS.col(2) = Eigen::Vector2d::Zero();
      minco_anal.pos_anal.reset(initS, msg->pos_pts.size() - 1);
      minco_anal.pos_anal.generate(posP, tailS, posT);
      minco_traj.pos_traj = minco_anal.pos_anal.getTraj();

      initS.setZero(1, 3);
      tailS.setZero(1, 3);
      initS(0, 0) = msg->angle_pts[0].x;
      initS(0, 1) = msg->init_v.z;
      initS(0, 2) = msg->init_a.z;
      tailS(0, 0) = msg->angle_pts.back().x;
      tailS(0, 1) = 0.0;
      tailS(0, 1) = 0.0;
      minco_anal.angle_anal.reset(initS, msg->angle_pts.size() - 1);
      minco_anal.angle_anal.generate(angleP, tailS, angleT);
      minco_traj.angle_traj = minco_anal.angle_anal.getTraj();

      traj_duration = minco_traj.pos_traj.getTotalDuration();
      use_flat = false;
      use_dp = false;
    }

    void setTraj(std::shared_ptr<const mpc_controller::msg::PolyTraj> msg)
    {
      auto clock_type = node_ ? node_->get_clock()->get_clock_type() : RCL_SYSTEM_TIME;
      start_time = rclcpp::Time(msg->start_time, clock_type);

      std::vector<double> durs;
      std::vector<double> sgls;
      std::vector<CoefficientMat<2>> cMats;
      for (int i=0; i<(int)msg->trajs.size(); i++)
      {
        CoefficientMat<2> c;
        for (int j=0; j<6; j++)
        {
          c(0, j) = msg->trajs[i].coef_x[j];
          c(1, j) = msg->trajs[i].coef_y[j];
        }
        cMats.push_back(c);
        double s = msg->trajs[i].reverse?-1.0:1.0;
        sgls.push_back(s);
        durs.push_back(msg->trajs[i].duration);
      }

      flat_traj = mpc_utils::FlatTraj2d(durs, cMats, sgls);
      traj_duration = flat_traj.getTotalDuration();
      use_flat = true;
      use_dp = false;
      // ROS_WARN("get flat traj!");
    }

    void setTraj(std::shared_ptr<const mpc_controller::msg::DPtrajContainer> msg)
    {
      auto clock_type = node_ ? node_->get_clock()->get_clock_type() : RCL_SYSTEM_TIME;
      start_time = rclcpp::Time(msg->start_time, clock_type);
      dptraj_mpc::UgvTrajectory ugvTraj_tmp;
      for(int i = 0; i <msg->traj_container.size(); i++){
        std::vector<double> dts;
        std::vector<double> dss;
        std::vector<CoefficientMat<2>> c2Mats;
        std::vector<CoefficientMat<1>> c1Mats;
        for(int j = 0; j < msg->traj_container[i].trajs.size(); j++){
            CoefficientMat<2> c2;
            CoefficientMat<1> c1;
            for (int k=0; k<6; k++)
            {
              c2(0, k) = msg->traj_container[i].trajs[j].coef_x[k];
              c2(1, k) = msg->traj_container[i].trajs[j].coef_y[k];
              c1(0, k) = msg->traj_container[i].trajs[j].coef_s[k];
            }
            c2Mats.push_back(c2);
            c1Mats.push_back(c1);
            dts.push_back( msg->traj_container[i].trajs[j].dt);
            dss.push_back( msg->traj_container[i].trajs[j].ds);
        }
        dptraj_mpc::PolyTrajectory<2> r(dss, c2Mats);
        dptraj_mpc::PolyTrajectory<1> s(dts, c1Mats);
        dptraj_mpc::DpTrajectory dptraj;
        dptraj.posTraj = r;
        dptraj.tTraj = s;
        ugvTraj_tmp.Traj_container.push_back(dptraj);
        if(msg->reverse[i]>0)
          ugvTraj_tmp.etas.push_back(-1);
        else
          ugvTraj_tmp.etas.push_back(1);
        // msg->traj_container[i].trajs
      }
      ugv_traj = ugvTraj_tmp;
      traj_duration = ugv_traj.getTotalDuration();
      use_flat = false;
      use_dp = true;
    }

    std::vector<TrajPoint> getTajWps(double dt)
    {
      std::vector<TrajPoint> P;
      TrajPoint tp;

      if (use_flat)
      {
        for (double t=0.0; t<=traj_duration; t+=dt)
        {
          Eigen::Vector2d po = flat_traj.getPos(t);
          tp.x = po[0];
          tp.y = po[1];
          tp.theta = flat_traj.getAngle(t);
          P.push_back(tp);
        }
      }
      else if(use_dp){
        for (double t=0.0; t<=traj_duration; t+=dt)
        {
          Eigen::Vector2d po = ugv_traj.getPos(t);
          tp.x = po[0];
          tp.y = po[1];
          tp.theta = ugv_traj.getYaw(t);
          P.push_back(tp);
        }
      }
      else
      {
        for (double t=0.0; t<=traj_duration; t+=dt)
        {
          Eigen::Vector2d po = minco_traj.pos_traj.getPos(t);
          tp.x = po[0];
          tp.y = po[1];
          tp.theta = minco_traj.angle_traj.getPos(t)[0];
          P.push_back(tp);
        }
      }

      return P;
    }

    std::vector<TrajPoint> getFutureWps(double dt, double horizon_sec)
    {
      std::vector<TrajPoint> P;
      const double sample_dt = std::max(0.03, dt);
      const double horizon = std::max(sample_dt, horizon_sec);
      double t_cur = currentRelativeTime();
      if (t_cur < 0.0)
        t_cur = 0.0;
      if (traj_duration <= 0.0 || t_cur > traj_duration + 0.2)
        return P;

      const double t_end = std::min(traj_duration, t_cur + horizon);
      for (double t = t_cur; t <= t_end + 1e-6; t += sample_dt)
      {
        P.push_back(sampleAt(t));
      }
      if (P.size() < 2 && t_cur <= traj_duration)
      {
        P.push_back(sampleAt(std::min(traj_duration, t_cur + sample_dt)));
      }
      return P;
    }

    std::vector<TrajPoint> getFutureWpsFromNearest(double dt, double horizon_sec, double x, double y)
    {
      std::vector<TrajPoint> P;
      if (traj_duration <= 0.0)
        return P;

      const double sample_dt = std::max(0.03, dt);
      const double horizon = std::max(sample_dt, horizon_sec);
      const double t_cur = nearestTimeTo(x, y, std::max(0.05, sample_dt));
      const double t_end = std::min(traj_duration, t_cur + horizon);

      for (double t = t_cur; t <= t_end + 1e-6; t += sample_dt)
      {
        P.push_back(sampleAt(t));
      }
      if (P.size() < 2)
      {
        P.push_back(sampleAt(t_cur));
        P.push_back(sampleAt(std::min(traj_duration, t_cur + sample_dt)));
      }
      return P;
    }

    std::vector<TrajPoint> getFutureWpsFromTime(double dt, double horizon_sec, double t_start)
    {
      std::vector<TrajPoint> P;
      if (traj_duration <= 0.0)
        return P;

      const double sample_dt = std::max(0.03, dt);
      const double horizon = std::max(sample_dt, horizon_sec);
      const double t_cur = std::clamp(t_start, 0.0, traj_duration);
      const double t_end = std::min(traj_duration, t_cur + horizon);

      for (double t = t_cur; t <= t_end + 1e-6; t += sample_dt)
      {
        P.push_back(sampleAt(t));
      }
      if (P.size() < 2)
      {
        P.push_back(sampleAt(t_cur));
        P.push_back(sampleAt(std::min(traj_duration, t_cur + sample_dt)));
      }
      return P;
    }

    std::vector<TrajPoint> getRefPointsFromNearest(const int T, double dt, double x, double y)
    {
      std::vector<TrajPoint> P;
      if (traj_duration <= 0.0 || T <= 0)
      {
        at_goal = true;
        return P;
      }

      const double sample_dt = std::max(0.05, std::min(0.20, dt));
      const double nearest_t = nearestTimeTo(x, y, sample_dt);
      const TrajPoint tail = sampleAt(traj_duration);
      const double tail_dist = std::hypot(tail.x - x, tail.y - y);
      at_goal = (nearest_t >= traj_duration - 0.10) && (tail_dist < 0.18);

      for (int j = 0; j < T; ++j)
      {
        const double t = std::min(traj_duration, nearest_t + (j + 1) * dt);
        P.push_back(sampleAt(t));
      }
      return P;
    }

    std::vector<TrajPoint> getRefPointsFromTime(const int T, double dt, double t_start, double x, double y)
    {
      std::vector<TrajPoint> P;
      if (traj_duration <= 0.0 || T <= 0)
      {
        at_goal = true;
        return P;
      }

      const double t_cur = std::clamp(t_start, 0.0, traj_duration);
      const TrajPoint tail = sampleAt(traj_duration);
      const double tail_dist = std::hypot(tail.x - x, tail.y - y);
      const double tail_lon =
        std::cos(tail.theta) * (x - tail.x) +
        std::sin(tail.theta) * (y - tail.y);
      at_goal =
        (t_cur >= traj_duration - 0.10 && tail_dist < 0.22) ||
        (t_cur >= traj_duration - 0.25 && tail_dist < 0.55 && tail_lon > 0.12);

      for (int j = 0; j < T; ++j)
      {
        const double t = std::min(traj_duration, t_cur + j * dt);
        P.push_back(sampleAt(t));
      }
      return P;
    }

    std::vector<TrajPoint> getRefPointsFromTimeSpatial(const int T, double spatial_step, double t_start, double x, double y)
    {
      std::vector<TrajPoint> P;
      if (traj_duration <= 0.0 || T <= 0)
      {
        at_goal = true;
        return P;
      }

      const double t_cur = std::clamp(t_start, 0.0, traj_duration);
      const TrajPoint tail = sampleAt(traj_duration);
      const double tail_dist = std::hypot(tail.x - x, tail.y - y);
      const double tail_lon =
        std::cos(tail.theta) * (x - tail.x) +
        std::sin(tail.theta) * (y - tail.y);
      at_goal =
        (t_cur >= traj_duration - 0.10 && tail_dist < 0.22) ||
        (t_cur >= traj_duration - 0.25 && tail_dist < 0.55 && tail_lon > 0.12);

      const double ds = std::clamp(spatial_step, 0.12, 0.45);
      const double sample_dt = 0.04;
      auto interp_point = [](const TrajPoint &a, const TrajPoint &b, double r) {
        r = std::clamp(r, 0.0, 1.0);
        TrajPoint q;
        q.x = a.x + r * (b.x - a.x);
        q.y = a.y + r * (b.y - a.y);
        q.v = a.v + r * (b.v - a.v);
        q.cur = a.cur + r * (b.cur - a.cur);
        q.steer_vel = a.steer_vel + r * (b.steer_vel - a.steer_vel);
        q.a = a.a + r * (b.a - a.a);
        q.phi = a.phi + r * (b.phi - a.phi);
        double dyaw = b.theta - a.theta;
        while (dyaw > M_PI) dyaw -= 2.0 * M_PI;
        while (dyaw < -M_PI) dyaw += 2.0 * M_PI;
        q.theta = a.theta + r * dyaw;
        return q;
      };

      P.reserve(static_cast<size_t>(T));
      TrajPoint prev = sampleAt(t_cur);
      P.push_back(prev);

      double accum_s = 0.0;
      double next_s = ds;
      for (double t = t_cur + sample_dt; t <= traj_duration + 1e-6 && static_cast<int>(P.size()) < T; t += sample_dt)
      {
        const double tc = std::min(t, traj_duration);
        const TrajPoint cur = sampleAt(tc);
        const double seg = std::hypot(cur.x - prev.x, cur.y - prev.y);
        while (seg > 1e-6 && accum_s + seg >= next_s && static_cast<int>(P.size()) < T)
        {
          const double r = (next_s - accum_s) / seg;
          P.push_back(interp_point(prev, cur, r));
          next_s += ds;
        }
        accum_s += seg;
        prev = cur;
        if (tc >= traj_duration - 1e-6)
          break;
      }

      while (static_cast<int>(P.size()) < T)
        P.push_back(tail);
      return P;
    }

    double getRelativeTime() const
    {
      return currentRelativeTime();
    }

    double getNearestTime(double x, double y, double sample_dt)
    {
      return nearestTimeTo(x, y, sample_dt);
    }

    double getNearestTimeInRange(double x, double y, double sample_dt, double t_min, double t_max)
    {
      return nearestTimeToInRange(x, y, sample_dt, t_min, t_max);
    }

    TrajPoint getFinalPoint()
    {
      return sampleAt(traj_duration);
    }

    bool usesDpTrajectory() const
    {
      return use_dp;
    }

    double getDuration() const
    {
      return traj_duration;
    }

    void setTestTraj(double max_vel)
    {
      Eigen::MatrixXd eight(10, 3);
      eight << 2.34191,      -0.382897,  0.127306, 
              3.09871,      0.936706,   1.72168,
              1.99125,      2.68782,    2.53673,
              0.394621,     3.877,      2.25751,
              -0.0799935,   5.86051,    1.13753,
              1.90338,      6.56037,    -0.143977,
              3.17197,      5.21122,    -1.66021,
              2.12699,      3.42104,    -2.48622,
              0.48492,      2.23846,    -2.31336,
              -0.00904252,  0.365258,   -1.55525;
      Eigen::MatrixXd posP = eight.transpose();
      Eigen::VectorXd T(11);
      T << 3.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0;
      T *= 1.5 / max_vel;
      Eigen::MatrixXd initS, tailS;

      initS.setZero(2, 3);
      tailS.setZero(2, 3);
      minco_anal.pos_anal.reset(initS, T.size());
      minco_anal.pos_anal.generate(posP.block<2, 10>(0, 0), tailS, T);
      minco_traj.pos_traj = minco_anal.pos_anal.getTraj();

      initS.setZero(1, 3);
      tailS.setZero(1, 3);
      initS(0, 0) = -0.257661;
      tailS(0, 0) = -1.54148 ;

      minco_anal.angle_anal.reset(initS, T.size());
      minco_anal.angle_anal.generate(posP.block<1, 10>(2, 0), tailS, T);
      minco_traj.angle_traj = minco_anal.angle_anal.getTraj();

      traj_duration = minco_traj.pos_traj.getTotalDuration();
     if (node_) {
       start_time = node_->get_clock()->now();
     } else {
       start_time = rclcpp::Clock().now();
     }
      use_flat = false;
    }

    std::vector<TrajPoint> getRefPoints(const int T, double dt)
    {
      std::vector<TrajPoint> P;
      P.clear();
      TrajPoint tp;
      rclcpp::Time time_now;
      if (node_) {
          time_now = node_->get_clock()->now();  // 使用 node 的 clock
      } else {
          time_now = rclcpp::Clock().now();  // fallback，但可能有时间源问题
      }
        double t_cur = static_cast<double>(time_now.nanoseconds() - start_time.nanoseconds()) * 1e-9;
      int j=0;

      // Fix: clamp t_cur to [0, traj_duration] to handle time synchronization issues
      if (t_cur < 0.0) {
        t_cur = 0.0;  // Start from beginning if time is negative
      }

      if (t_cur > traj_duration + 1.0)
      {
        at_goal = true;
        return P;
      }
      else
      {
        at_goal = false;
      }

      if (use_flat)
      {
        for (double t=t_cur+dt; j<T; j++, t+=dt)
        {
          double temp = t;
          if (temp <= traj_duration)
          {
            Eigen::Vector2d po = flat_traj.getPos(temp);
            tp.x = po[0];
            tp.y = po[1];
            tp.v = flat_traj.getLonVel(temp);
            tp.a = flat_traj.getLonAcc(temp);
            tp.cur = flat_traj.getCur(temp);
            tp.theta = flat_traj.getAngle(temp);
            P.push_back(tp);
          }
          else
          {
            Eigen::Vector2d po =flat_traj.getPos(traj_duration);
            tp.x = po[0];
            tp.y = po[1];
            tp.v = flat_traj.getLonVel(traj_duration);
            tp.a = flat_traj.getLonAcc(traj_duration);
            tp.cur = flat_traj.getCur(traj_duration);
            tp.theta = flat_traj.getAngle(traj_duration);
            P.push_back(tp);
          }
        }
      }
      else if(use_dp){
        for (double t=t_cur+dt; j<T; j++, t+=dt)
        {
          double temp = t;
          if (temp <= traj_duration)
          {
            Eigen::Vector2d po = ugv_traj.getPos(temp);
            tp.x = po[0];
            tp.y = po[1];
            tp.v = ugv_traj.getVelItem(temp);
            tp.cur = ugv_traj.getCur(temp);
            tp.phi = ugv_traj.getPhi(temp);
            tp.theta = ugv_traj.getYaw(temp);
            P.push_back(tp);
          }
          else
          {
            Eigen::Vector2d po =ugv_traj.getPos(traj_duration);
            tp.x = po[0];
            tp.y = po[1];
            tp.v = ugv_traj.getVelItem(traj_duration);
            tp.cur = ugv_traj.getCur(traj_duration);
            tp.phi = ugv_traj.getPhi(traj_duration);
            tp.theta = ugv_traj.getYaw(traj_duration);
            P.push_back(tp);
          }
        }
      }
      else
      {
        for (double t=t_cur+dt; j<T; j++, t+=dt)
        {
          double temp = t;
          if (temp <= traj_duration)
          {
            Eigen::Vector2d po = minco_traj.pos_traj.getPos(temp);
            tp.x = po[0];
            tp.y = po[1];
            tp.theta = minco_traj.angle_traj.getPos(temp)[0];
            P.push_back(tp);
          }
          else
          {
            Eigen::Vector2d po = minco_traj.pos_traj.getPos(traj_duration);
            tp.x = po[0];
            tp.y = po[1];
            tp.theta = minco_traj.angle_traj.getPos(traj_duration)[0];
            P.push_back(tp);
          }
        }
      }
      
      return P;
    }

    TrajPoint getRefPoint()
    {
      TrajPoint tp;
      rclcpp::Time time_now;
      if (node_) {
          time_now = node_->get_clock()->now();
      } else {
          time_now = rclcpp::Clock().now();
      }
        double t_cur = static_cast<double>(time_now.nanoseconds() - start_time.nanoseconds()) * 1e-9;

      double temp = t_cur;
      if (temp < 0)
        return tp;
      if (use_flat)
      {
        if (temp <= traj_duration)
        {
          Eigen::Vector2d po = flat_traj.getPos(temp);
          tp.x = po[0];
          tp.y = po[1];
          // tp.v = flat_traj.getLonVel(temp);
          // tp.a = flat_traj.getLonAcc(temp);
          // tp.cur = flat_traj.getCur(temp);
          tp.theta = flat_traj.getAngle(temp);
          // tp.phi = flat_traj.getPhi(temp);
          // tp.steer_vel = flat_traj.getOmega(temp);
        }
        else
        {
          Eigen::Vector2d po =flat_traj.getPos(traj_duration);
          tp.x = po[0];
          tp.y = po[1];
          // tp.v = flat_traj.getLonVel(traj_duration);
          // tp.a = flat_traj.getLonAcc(traj_duration);
          // tp.cur = flat_traj.getCur(traj_duration);
          tp.theta = flat_traj.getAngle(traj_duration);
          // tp.phi = flat_traj.getPhi(traj_duration);
          // tp.steer_vel = flat_traj.getOmega(traj_duration);
        }
      }
      else if (use_dp)
      {
        if (temp <= traj_duration)
        {
          Eigen::Vector2d po = ugv_traj.getPos(temp);
          tp.x = po[0];
          tp.y = po[1];
          // tp.v = ugv_traj.getVelItem(temp);
          // tp.a = ugv_traj.getLonAcc(temp);
          // tp.cur = ugv_traj.getCur(temp);
          tp.theta = ugv_traj.getYaw(temp);
          // tp.phi = ugv_traj.getPhi(temp);
          // tp.steer_vel = ugv_traj.getOmega(temp);
        }
        else
        {
          Eigen::Vector2d po =ugv_traj.getPos(traj_duration);
          tp.x = po[0];
          tp.y = po[1];
          // tp.v = ugv_traj.getVelItem(traj_duration);
          // tp.a = ugv_traj.getLonAcc(traj_duration);
          // tp.cur = ugv_traj.getCur(traj_duration);
          tp.theta = ugv_traj.getYaw(traj_duration);
          // tp.phi=  ugv_traj.getPhi(traj_duration);
          // tp.steer_vel = ugv_traj.getOmega(traj_duration);
        }
      }

      return tp;
    }
   
    ~TrajAnalyzer() {}
};
