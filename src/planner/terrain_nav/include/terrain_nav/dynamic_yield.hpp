#pragma once
#include <stdexcept>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>
#include <utility>
#include "terrain_nav/body_kinematics.hpp"

namespace terrain_nav {
// Advisory clearance envelope: v*delay + v^2/(2*a) <= distance-reserve.
// This does not replace swept-body collision or terrain probability checks.
inline double clearanceSpeedLimit(double distance,double reserve,double delay,double deceleration)
{
  if(std::isnan(distance) || !std::isfinite(reserve) || !std::isfinite(delay) ||
     !std::isfinite(deceleration) || reserve<0 || delay<0 || deceleration<=0) return 0.;
  const double available=std::max(0.,distance-reserve);
  if(!std::isfinite(available))return available;
  // Rationalized quadratic root avoids cancellation for near-zero clearance.
  const double ad=deceleration*delay;
  const double denominator=std::sqrt(ad*ad+2.*deceleration*available)+ad;
  return denominator>0 ? 2.*deceleration*available/denominator : 0.;
}

struct YieldCommand { double t=0,speed=0,steering=0; };
struct YieldState { double t=0,x=0,y=0,yaw=0,speed=0,steering=0; };
struct YieldModel {
  double acceleration=1.2,deceleration=1.2,delay=.35,wheelbase=.549,steering_rate=4,step=.02;
  double rear_axle_offset=0.;
  // Pure input dead times, distinct from the explicit actuator slew below.
  // -1 inherits the legacy common delay, preserving existing configurations.
  double speed_dead_time=-1.,steering_dead_time=-1.;
};
inline double yieldSpeedDelay(const YieldModel &m){return m.speed_dead_time<0?m.delay:m.speed_dead_time;}
inline double yieldSteeringDelay(const YieldModel &m){return m.steering_dead_time<0?m.delay:m.steering_dead_time;}
inline double maximumYieldDelay(const YieldModel &m){return std::max(yieldSpeedDelay(m),yieldSteeringDelay(m));}
inline bool validYieldDeadTimes(const YieldModel &m) {
  const auto axis=[](double v){return std::isfinite(v)&&(v==-1.||(v>=0.&&v<=1.));};
  return std::isfinite(m.delay)&&m.delay>=0&&axis(m.speed_dead_time)&&axis(m.steering_dead_time);
}

// Actual publications, in plant input coordinates. Planned (but not yet
// published) commands must never enter this queue. Keep a predecessor of the
// retention boundary so a constant input remains defined across that boundary.
class IssuedYieldHistory {
 public:
  bool record(YieldCommand command,double retention=2.) {
    if(!std::isfinite(command.t)||!std::isfinite(command.speed)||!std::isfinite(command.steering)||
       !std::isfinite(retention)||retention<=0)return false;
    if(!commands_.empty()&&command.t<commands_.back().t)commands_.clear();
    if(!commands_.empty()&&command.t==commands_.back().t)commands_.back()=command;
    else commands_.push_back(command);
    while(commands_.size()>2&&commands_[1].t<command.t-retention)commands_.pop_front();
    // A resource bound, not a claim of evidence before the oldest sample.
    while(commands_.size()>4096)commands_.pop_front();
    return true;
  }
  std::vector<YieldCommand> snapshot() const {return {commands_.begin(),commands_.end()};}
 private:
  std::deque<YieldCommand> commands_;
};

// Advisory anticipation before velocity confirmation. Raw measured compact
// bodies may start moving; slowing only AFTER strict velocity confirmation
// can leave no feasible escape under source/actuator latency. This does not
// assert that a cue is dynamic, safe, or traversable, and never grants motion.
inline double compactEncounterSpeed(double requested,double x,double y,double yaw,
    const std::vector<std::array<double,4>>&bounds,double limit)
{
  if(limit<=0 || requested<=0)return requested;
  if(!std::isfinite(limit) || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(yaw))return 0.;
  const double c=std::cos(yaw),s=std::sin(yaw);
  for(const auto &b:bounds) {
    if(!std::all_of(b.begin(),b.end(),[](double v){return std::isfinite(v);}) || b[0]>b[2] || b[1]>b[3])return 0.;
    const double dx=(b[0]+b[2])*.5-x,dy=(b[1]+b[3])*.5-y;
    const double along=c*dx+s*dy,side=-s*dx+c*dy;
    const double hx=(b[2]-b[0])*.5,hy=(b[3]-b[1])*.5;
    const double a=std::abs(c)*hx+std::abs(s)*hy,l=std::abs(s)*hx+std::abs(c)*hy;
    if(along+a>=-.6 && along-a<=4. && std::abs(side)-l<=3.5)return std::min(requested,limit);
  }
  return requested;
}

// Include the ACTUAL momentum, sign change, reverse travel, delayed stop and
// a stationary tail. A fixed three seconds silently removed most retreats
// while the car was moving quickly. This is a bounded proposal horizon,
// never a collision certificate or an extension of source freshness.
inline double interactionHorizon(const YieldState&initial,const YieldModel&model,
                                 double reverse_speed,double reverse_distance)
{
  if(!std::isfinite(initial.speed) || !std::isfinite(reverse_speed) || reverse_speed<=0 ||
     !std::isfinite(reverse_distance) || reverse_distance<=0 ||
     !std::isfinite(model.acceleration) || model.acceleration<=0 ||
     !std::isfinite(model.deceleration) || model.deceleration<=0 ||
     !validYieldDeadTimes(model))return 0.;
  return std::clamp(2*maximumYieldDelay(model)+std::max(0.,initial.speed)/model.deceleration+
      reverse_speed/model.acceleration+reverse_distance/reverse_speed+
      reverse_speed/model.deceleration+.25,3.,6.);
}

inline YieldCommand yieldCommandAt(const std::vector<YieldCommand> &commands,double t)
{
  YieldCommand result;
  for(const auto &command:commands) {if(command.t>t+1e-9)break;result=command;}
  return result;
}
inline YieldCommand delayedYieldCommand(const YieldState &state,const YieldState &held,
    const std::vector<YieldCommand> &commands,const YieldModel &model)
{
  const double speed_time=state.t-yieldSpeedDelay(model),steering_time=state.t-yieldSteeringDelay(model);
  const auto speed=speed_time<0?YieldCommand{0,held.speed,held.steering}:yieldCommandAt(commands,speed_time);
  const auto steering=steering_time==speed_time?speed:
    steering_time<0?YieldCommand{0,held.speed,held.steering}:yieldCommandAt(commands,steering_time);
  return {0,speed.speed,steering.steering};
}

// Rebase only the UNISSUED suffix for a receding decision. The active input
// is explicit at zero, future switches retain their absolute deadlines, and
// already published/pending inputs remain in IssuedYieldHistory separately.
// In particular, refreshing the root must never restart an old drive pulse.
inline std::vector<YieldCommand> remainingYieldCommands(const std::vector<YieldCommand>&commands,double elapsed)
{
  if(commands.empty()||!std::isfinite(elapsed)||elapsed<0)return {};
  for(size_t i=0;i<commands.size();++i)
    if(!std::isfinite(commands[i].t)||commands[i].t<0||!std::isfinite(commands[i].speed)||
       !std::isfinite(commands[i].steering)||(i&&commands[i].t<commands[i-1].t))return {};
  auto active=yieldCommandAt(commands,elapsed);active.t=0.;
  std::vector<YieldCommand> result{active};
  for(auto command:commands)if(command.t>elapsed+1e-9){command.t-=elapsed;result.push_back(command);}
  return result;
}

// Keep the stop and the eventual forward motion in one time-indexed proposal.
// The caller must prove the swept path, including the stationary interval.
inline std::vector<YieldCommand> waitThenAdvance(const std::vector<YieldCommand> &advance,
    double wait,double steering)
{
  if(advance.empty()||advance.front().t!=0.||!(advance.front().speed>0.)||
     !std::isfinite(wait)||wait<=0.||!std::isfinite(steering))return {};
  std::vector<YieldCommand> schedule{{0.,0.,steering}};
  for(auto command:advance) {
    command.t+=wait;
    schedule.push_back(command);
  }
  return schedule;
}

inline YieldState advanceYield(YieldState state,const YieldCommand &command,
                              const YieldModel &model,double dt)
{
  double next=state.speed;
  if(state.speed*command.speed<0) {
    const double stop_time=std::abs(state.speed)/model.deceleration;
    next=dt<=stop_time ? std::copysign(std::max(0.,std::abs(state.speed)-model.deceleration*dt),state.speed) :
      std::copysign(std::min(std::abs(command.speed),model.acceleration*(dt-stop_time)),command.speed);
  } else {
    const double rate=std::abs(command.speed)<std::abs(state.speed) ? model.deceleration : model.acceleration;
    next+=std::clamp(command.speed-state.speed,-rate*dt,rate*dt);
  }
  const double steer=state.steering+std::clamp(command.steering-state.steering,
                                             -model.steering_rate*dt,model.steering_rate*dt);
  const double speed=.5*(state.speed+next),steering=.5*(state.steering+steer);
  const double turn=speed*std::tan(steering)/model.wheelbase*dt;
  state=integrateBodyVelocity(state,speed,turn/dt,dt,model.rear_axle_offset);
  state.speed=next;state.steering=steer;state.t+=dt;
  return state;
}

// Times are relative to selection. Delayed commands retain the measured initial
// motion until the request reaches the plant; reversal always crosses zero.
inline std::vector<YieldState> rolloutYield(YieldState state,const YieldState &held_inputs,
    const std::vector<YieldCommand> &commands,const YieldModel &model,double end)
{
  if(!validYieldDeadTimes(model)||!std::isfinite(model.step)||model.step<=0||
     !std::isfinite(end)||!std::isfinite(state.t))return {};
  std::vector<YieldState> result{state};
  while(state.t<end-1e-9) {
    const auto command=delayedYieldCommand(state,held_inputs,commands,model);
    state=advanceYield(state,command,model,std::min(model.step,end-state.t));
    result.push_back(state);
  }
  return result;
}

// Causal replay of the pending plant input pipeline. State/plan times are
// relative to origin; issued times are absolute on that SAME clock. Before
// root+delay, use only commands actually published before the root. Thereafter
// use the proposed schedule. This is essential on receding checks: a command
// from the old plan may have been refused, replaced by zero, or never sent.
// Without an older publication, the explicit fallback is measured root input
// (the legacy hold assumption); this is not a worst-case actuator certificate.
inline std::vector<YieldState> rolloutQueuedYield(YieldState state,
    const std::vector<YieldCommand>&commands,const YieldModel&model,double end,
    double origin,const std::vector<YieldCommand>&issued)
{
  const auto finite_command=[](const YieldCommand&c){return std::isfinite(c.t)&&
    std::isfinite(c.speed)&&std::isfinite(c.steering);};
  const auto ordered=[&](const auto&v) {
    for(size_t i=0;i<v.size();++i)if(!finite_command(v[i])||(i&&v[i].t<v[i-1].t))return false;
    return true;
  };
  if(!std::isfinite(state.t)||!std::isfinite(state.x)||!std::isfinite(state.y)||
     !std::isfinite(state.yaw)||!std::isfinite(state.speed)||!std::isfinite(state.steering)||
     !std::isfinite(origin)||!std::isfinite(end)||end<state.t||
     !validYieldDeadTimes(model)||!std::isfinite(model.step)||model.step<=0||
     !std::isfinite(model.acceleration)||model.acceleration<=0||
     !std::isfinite(model.deceleration)||model.deceleration<=0||
     !std::isfinite(model.wheelbase)||model.wheelbase<=0||
     !std::isfinite(model.steering_rate)||model.steering_rate<0||
     !std::isfinite(model.rear_axle_offset)||!ordered(commands)||!ordered(issued))return {};
  const double root=state.t;
  const YieldCommand fallback{0,state.speed,state.steering};
  std::vector<YieldState> result{state};
  while(state.t<end-1e-9) {
    const auto at_delay=[&](double delay) {
      const double input_time=state.t-delay;YieldCommand applied=fallback;double next_change=end;
      if(input_time<root-1e-9) {
        const auto next=std::upper_bound(issued.begin(),issued.end(),origin+input_time+1e-9,
          [](double t,const YieldCommand&c){return t<c.t;});
        if(next!=issued.begin())applied=*std::prev(next);
        next_change=std::min(next_change,root+delay);
        // Publications after the root are not evidence of a past input.
        if(next!=issued.end()&&next->t<origin+root-1e-9)
          next_change=std::min(next_change,next->t-origin+delay);
      } else {
        applied=yieldCommandAt(commands,input_time);
        const auto next=std::upper_bound(commands.begin(),commands.end(),input_time+1e-9,
          [](double t,const YieldCommand&c){return t<c.t;});
        if(next!=commands.end())next_change=std::min(next_change,next->t+delay);
      }
      return std::make_pair(applied,next_change);
    };
    const double speed_delay=yieldSpeedDelay(model),steering_delay=yieldSteeringDelay(model);
    const auto speed=at_delay(speed_delay),steering=steering_delay==speed_delay?speed:at_delay(steering_delay);
    const YieldCommand applied{0,speed.first.speed,steering.first.steering};
    const double next_change=std::min(speed.second,steering.second);
    // Split at input events; a non-grid-aligned publication must not acquire
    // another full integration tick of artificial response delay.
    const double dt=std::min({model.step,end-state.t,next_change-state.t});
    if(dt<=0)return {};
    state=advanceYield(state,applied,model,dt);result.push_back(state);
  }
  return result;
}

// A short finite forward escape must include its COMPLETE deceleration and a
// guarded stationary tail. Reusing a horizon sized for a 1.2m reverse maneuver
// adds arbitrary waiting to a different action; growing prediction error over
// that wait can eliminate every shorter endpoint. This remains a finite,
// repeatedly revalidated prediction, not indefinite invariant-set safety.
inline std::vector<YieldState> finiteAdvanceRollout(std::vector<YieldState> states,
    double stationary_tail=1.2,double minimum_horizon=3.)
{
  if(states.empty() || !std::isfinite(stationary_tail) || stationary_tail<1.2 ||
     !std::isfinite(minimum_horizon) || minimum_horizon<3. ||
     !std::isfinite(states.back().speed) || std::abs(states.back().speed)>.001)return {};
  double last_motion=states.front().t;
  for(size_t i=0;i<states.size();++i) {
    if(!std::isfinite(states[i].t) || !std::isfinite(states[i].speed) ||
       (i && states[i].t<=states[i-1].t))return {};
    if(std::abs(states[i].speed)>.001)last_motion=states[i].t;
  }
  const double required=std::max(states.front().t+minimum_horizon,last_motion+stationary_tail);
  // Insufficient stopped-tail evidence cannot be repaired by truncation.
  if(required>states.back().t+1e-8)return {};
  const auto end=std::lower_bound(states.begin(),states.end(),std::min(required,states.back().t),
    [](const YieldState &state,double t){return state.t<t;});
  states.erase(end+1,states.end());
  return states;
}

// Keep at least three FUTURE seconds on every continuation. The original
// command schedule still stops at its fixed deadline; only its already
// stopped tail is extended. A receding check must not shrink to a few ticks
// just because the selected finite action is about to finish.
inline std::vector<YieldState> continuationYieldRollout(YieldState actual,const YieldState &initial,
    const std::vector<YieldCommand> &commands,const YieldModel &model,double nominal_end)
{
  if(!std::isfinite(actual.t) || !std::isfinite(nominal_end) || actual.t<0)return {};
  // Receding kinematic-hold model: the newly measured root is the held
  // motion until the scheduled response. Do not restore an old observed
  // velocity/steering. Callers testing an EXPLICIT pending-command hypothesis
  // retain rolloutYield(actual,pending,...) independently (e.g. retirement).
  (void)initial;
  return rolloutYield(actual,actual,commands,model,std::max(nominal_end,actual.t+3.));
}
// Conditional joint-risk mode: at least three seconds AND a complete stop
// with >=200ms predicted zero-speed tail. A longer finite escape is not discarded
// merely because its physical braking finishes after a fixed 3s boundary.
inline std::vector<YieldState> finiteJointRollout(std::vector<YieldState> states)
{
  if(states.empty()||!std::isfinite(states.back().speed)||std::abs(states.back().speed)>.001)return {};
  size_t stopped=0;
  for(size_t i=0;i<states.size();++i) {
    if(!std::isfinite(states[i].t)||!std::isfinite(states[i].speed)||
       (i&&states[i].t<=states[i-1].t))return {};
    if(std::abs(states[i].speed)>.001)stopped=i+1;
  }
  if(stopped>=states.size())return {};
  const double required=std::max(states.front().t+3.,states[stopped].t+.20);
  if(required>states.back().t+1e-8)return {};
  const auto end=std::lower_bound(states.begin(),states.end(),std::min(required,states.back().t),
    [](const auto &s,double t){return s.t<t;});
  states.erase(end+1,states.end());return states;
}

// Conservative PRIORITY test only: stronger/earlier braking can leave the
// vehicle at an earlier point of its braking path. Test those possible rest
// positions against the full future obstacle reservation. This point set is
// not a time-ordered executable trajectory and may never authorize a command.
inline std::vector<YieldState> possibleEarlyStops(const std::vector<YieldState>&braking)
{
  if(braking.empty())return {};
  std::vector<YieldState> result{braking.front()};
  result.reserve(braking.size()+1);
  for(auto state:braking) {state.t=braking.back().t;state.speed=0.;result.push_back(state);}
  return result;
}

inline std::vector<YieldCommand> makeYieldCommands(const YieldState &initial,
    const YieldModel &model,double distance,double speed,double steering,double horizon)
{
  std::vector<YieldCommand> commands{{0,0,0}};
  if(distance<=0)return commands;
  commands[0]={0,-speed,steering};
  auto state=initial;double reversed=0;
  while(state.t<horizon) {
    const auto applied=delayedYieldCommand(state,initial,commands,model);
    const auto next=advanceYield(state,applied,model,model.step);
    reversed+=std::max(0.,-.5*(state.speed+next.speed)*model.step);
    state=next;
    const double v=std::max(0.,-state.speed);
    if(v>0 &&
       reversed+v*yieldSpeedDelay(model)+v*v/(2*model.deceleration)+v*model.step>=distance) {
      commands.push_back({state.t,0,steering});
      return commands;
    }
  }
  return {};
}

// Short forward avoidance candidate with an explicit stop. This is a motion
// proposal only: callers must certify the entire swept body, moving obstacles,
// terrain/path risk, and stopping envelope before publishing any command.
inline std::vector<YieldCommand> makeAdvanceCommands(const YieldState &initial,
    const YieldModel &model,double distance,double speed,double steering,double horizon)
{
  if(!std::isfinite(distance) || !std::isfinite(speed) || !std::isfinite(steering) ||
     !std::isfinite(horizon) || distance<=0 || speed<=0 || horizon<=0 ||
     model.deceleration<=0 || model.step<=0)return {};
  std::vector<YieldCommand> commands{{0,speed,steering}};
  auto state=initial;double advanced=0;
  while(state.t<horizon) {
    const auto applied=delayedYieldCommand(state,initial,commands,model);
    const auto next=advanceYield(state,applied,model,model.step);
    advanced+=std::max(0.,.5*(state.speed+next.speed)*model.step);state=next;
    const double v=std::max(0.,state.speed);
    if(v>0 && advanced+v*yieldSpeedDelay(model)+v*v/(2*model.deceleration)+v*model.step>=distance) {
      commands.push_back({state.t,0,steering});
      const auto rollout=rolloutYield(initial,initial,commands,model,horizon);
      return !rollout.empty() && std::abs(rollout.back().speed)<.001 ? commands : std::vector<YieldCommand>{};
    }
  }
  return {};
}

// Two opposite steering phases followed by straightening. Match the integrated
// heading, not equal phase durations: during acceleration, equally long left
// and right commands traverse different distances and do NOT cancel curvature.
// Include the delayed command and steering slew in the straightening prediction.
// This is a proposal only; no geometry, support or risk checks are bypassed.
inline std::vector<YieldCommand> makeBypassCommands(const YieldState &initial,
    const YieldModel &model,double distance,double speed,double steering,
    double phase_duration,double horizon)
{
  if(!std::isfinite(phase_duration) || phase_duration<=0 ||
     !std::isfinite(distance) || distance<=0 || !std::isfinite(speed) || speed<=0 ||
     !std::isfinite(steering) || model.deceleration<=0 || model.step<=0 ||
     !validYieldDeadTimes(model)||!std::isfinite(model.steering_rate)||model.steering_rate<=0||
     !std::isfinite(model.wheelbase)||model.wheelbase<=0||
     !std::isfinite(horizon) || horizon<=0)return {};
  std::vector<YieldCommand> commands{{0,speed,steering},
    {phase_duration,speed,-steering}};
  auto state=initial;double advanced=0;bool straightened=false;
  while(state.t<horizon) {
    const auto applied=delayedYieldCommand(state,initial,commands,model);
    const auto next=advanceYield(state,applied,model,model.step);
    advanced+=std::max(0.,.5*(state.speed+next.speed)*model.step);state=next;
    // Decide in publication time, before the queued opposite turn arrives.
    // Waiting until the measured steering has changed sign adds another
    // whole actuator delay and can already overshoot the desired heading.
    if(!straightened && state.t>phase_duration+1e-9) {
      auto straight=commands;straight.push_back({state.t,speed,0.});
      const double settling=yieldSteeringDelay(model)+2.*std::abs(steering)/model.steering_rate+2.*model.step;
      const auto recovery=rolloutYield(state,initial,straight,model,state.t+settling);
      if(!recovery.empty() && steering*std::remainder(recovery.back().yaw-initial.yaw,2.*M_PI)<=0.) {
        commands=std::move(straight);straightened=true;
      }
    }
    const double v=std::max(0.,state.speed);
    if(v>0 && advanced+v*yieldSpeedDelay(model)+v*v/(2*model.deceleration)+v*model.step>=distance) {
      commands.erase(std::remove_if(commands.begin(),commands.end(),
        [&](const auto &c){return c.t>=state.t;}),commands.end());
      commands.push_back({state.t,0,0});
      const auto rollout=rolloutYield(initial,initial,commands,model,horizon);
      return !rollout.empty() && std::abs(rollout.back().speed)<.001 ? commands : std::vector<YieldCommand>{};
    }
  }
  return {};
}

struct ForwardYieldProposal {
  std::vector<YieldCommand> commands;
  double distance=0.;
};
// Turn, release steering, then brake. A constant arc or an S-shaped lane
// change cannot represent a short diagonal escape followed by a straight
// coast. All times are publication times: rolloutQueuedYield applies the
// actual independent actuator dead times and slew. This bounded library is
// a proposal source only; it has no map, ownership or risk permissions.
inline std::vector<ForwardYieldProposal> turnReleaseYieldProposals(
    double maximum_speed,double maximum_steering)
{
  std::vector<ForwardYieldProposal> out;
  if(!std::isfinite(maximum_speed)||maximum_speed<=0||!std::isfinite(maximum_steering)||
     maximum_steering<=0||maximum_steering>=1.57)return out;
  for(double speed:{.7*maximum_speed,maximum_speed})
    for(double steering:{-maximum_steering,maximum_steering,-.5*maximum_steering,.5*maximum_steering})
      for(double turn_time:{.7,1.1})for(double coast_time:{.3,.7,1.1})
        out.push_back({{{0.,speed,steering},{turn_time,speed,0.},{turn_time+coast_time,0.,0.}},0.});
  return out;
}
inline std::vector<ForwardYieldProposal> reverseYieldProposals(const YieldState &initial,
    const YieldModel &model,double maximum_speed,double maximum_steering,double horizon,bool joint)
{
  std::vector<ForwardYieldProposal> out;
  if(!std::isfinite(maximum_speed)||maximum_speed<=0||!std::isfinite(maximum_steering)||
      maximum_steering<0||maximum_steering>=1.57||!std::isfinite(horizon)||horizon<=0)return out;
  const std::vector<double> distances=joint?std::vector<double>{2.,1.6,1.2,.8,.4}:
    std::vector<double>{1.2,.8,.4};
  // The longitudinal model and stop time are identical across steering choices.
  for(double distance:distances) {
    auto commands=makeYieldCommands(initial,model,distance,maximum_speed,0.,horizon);
    if(commands.empty())continue;
    const auto states=rolloutYield(initial,initial,commands,model,horizon);
    if(!(states.size()>=11 && std::abs(states[states.size()-11].speed)<=.001 &&
        std::abs(states.back().speed)<=.001))continue;
    for(double steering:{0.,-maximum_steering,maximum_steering}) {
      commands.front().steering=commands.back().steering=steering;
      out.push_back({commands,distance});
    }
  }
  return out;
}
// Input is already sorted by utility. Give each available behavior family
// one full-proof opportunity before spending the remaining bounded search
// on similar variants. This is only an evaluation schedule, never permission.
inline std::vector<size_t> interactionBranchOrder(const std::vector<int> &families)
{
  std::vector<size_t> first,rest;std::array<bool,3> seen{};
  for(size_t i=0;i<families.size();++i) {
    if(families[i]<-1||families[i]>1)throw std::invalid_argument("invalid interaction family");
    const size_t kind=size_t(families[i]+1);
    if(!seen[kind]){seen[kind]=true;first.push_back(i);}else rest.push_back(i);
  }
  first.insert(first.end(),rest.begin(),rest.end());return first;
}
// A small proof budget needs a robust fallback before utility refinement.
// Inputs stay utility-sorted. Seed each available family with its smallest
// SCREEN estimate (ties keep utility order), then visit remaining policies by
// utility. This is a proof schedule only; a screen can never grant motion.
inline std::vector<size_t> interactionProofOrder(const std::vector<int> &families,
                                               const std::vector<double> &estimates,
                                               bool threatened_stop=false)
{
  if(families.size()!=estimates.size())throw std::invalid_argument("interaction screen size mismatch");
  const size_t none=std::numeric_limits<size_t>::max();
  std::array<size_t,3> chosen{none,none,none};
  for(size_t i=0;i<families.size();++i) {
    if(families[i]<-1||families[i]>1||!std::isfinite(estimates[i])||estimates[i]<0||estimates[i]>1)
      throw std::invalid_argument("invalid interaction screen ordering");
    auto &best=chosen[size_t(families[i]+1)];
    if(best==none||estimates[i]<estimates[best])best=i;
  }
  std::vector<size_t> out;std::vector<bool> used(families.size(),false);std::array<bool,3> seeded{};
  for(int family:families) {
    const size_t kind=size_t(family+1);
    if(!seeded[kind]) {seeded[kind]=true;out.push_back(chosen[kind]);used[chosen[kind]]=true;}
  }
  const size_t family_seeds=out.size();
  for(size_t i=0;i<families.size();++i)if(!used[i])out.push_back(i);
  if(threatened_stop) {
    // When waiting itself is threatened, check the most separated remaining
    // policies first. Goal progress alone can otherwise fill the entire
    // proof budget with forward arcs after one failed retreat.
    std::stable_sort(out.begin()+family_seeds,out.end(),[&](size_t a,size_t b) {
      return estimates[a]<estimates[b];
    });
  }
  return out;
}
// Fair evaluation order is NOT utility order. A safe brake/retreat found
// while giving each family a turn must remain a fallback, not suppress a
// higher-utility forward policy that can still be proved within the budget.
// Permissions come exclusively from prove(). A finite budget can return the
// best PROVED fallback, without claiming that unchecked actions are optimal.
template<class Prove,class MayProve,class CommitReady>
size_t bestCertifiedInteraction(const std::vector<double> &utility,
    const std::vector<size_t> &order,Prove prove,MayProve may_prove,CommitReady commit_ready)
{
  const size_t none=std::numeric_limits<size_t>::max();
  if(order.size()!=utility.size())return none;
  std::vector<bool> seen(utility.size(),false);
  for(size_t i:order) {
    if(i>=utility.size() || seen[i] || !std::isfinite(utility[i]))return none;
    seen[i]=true;
  }
  size_t best=none;
  for(size_t i:order) {
    if(best!=none && utility[i]<=utility[best])continue;
    if(!may_prove())break;
    if(prove(i)) {
      best=i;
      // A fully proved useful maneuver may be worth dispatching now rather
      // than spending its source-age margin comparing more alternatives.
      // This preference is consulted ONLY AFTER the complete proof succeeds.
      if(commit_ready(i))break;
    }
  }
  return best;
}
template<class Prove,class MayProve>
size_t bestCertifiedInteraction(const std::vector<double> &utility,
    const std::vector<size_t> &order,Prove prove,MayProve may_prove)
{
  return bestCertifiedInteraction(utility,order,prove,may_prove,[](size_t){return false;});
}
// Preference only: the screen's finite-sample estimate is NOT a safety
// certificate. Independent full-risk and terrain/stop checks remain required.
inline double interactionRiskPriority(double utility,double estimate,double limit,double weight)
{
  if(!std::isfinite(utility)||!std::isfinite(estimate)||estimate<0||estimate>1||
     !std::isfinite(limit)||limit<=0||limit>=1||!std::isfinite(weight)||weight<0)return -INFINITY;
  return utility-weight*estimate/limit;
}
// One library shared by the controller and regression qualification. These
// are proposals, never certificates. A rear approach can make a short stop
// unsafe; longer gentle arcs must not vanish merely because their length
// exceeds the old 3m short-library boundary.
inline std::vector<ForwardYieldProposal> forwardYieldProposals(const YieldState &initial,
    const YieldModel &model,double maximum_speed,double maximum_steering,double horizon,bool joint)
{
  std::vector<ForwardYieldProposal> out;
  if(!std::isfinite(maximum_speed)||maximum_speed<=0||!std::isfinite(maximum_steering)||
     maximum_steering<0||maximum_steering>=1.57||!std::isfinite(horizon)||horizon<=0)return out;
  const auto advance=[&](double distance,double speed) {
    auto commands=makeAdvanceCommands(initial,model,distance,speed,0.,horizon);
    if(commands.empty())return;
    // Steering affects the later footprint check, not the stop schedule.
    for(double steering:{0.,-.5*maximum_steering,.5*maximum_steering,-maximum_steering,maximum_steering}) {
      commands.front().steering=commands.back().steering=steering;
      out.push_back({commands,distance});
    }
  };
  const auto bypass=[&](double distance,double speed,double steering,double phase) {
    auto c=makeBypassCommands(initial,model,distance,speed,steering,phase,horizon);
    if(!c.empty())out.push_back({std::move(c),distance});
  };
  std::vector<double> speeds={maximum_speed,std::min(.6,maximum_speed)};
  if(std::abs(speeds[0]-speeds[1])<1e-6)speeds.resize(1);
  for(double speed:speeds)for(int bin=3;bin<=10;++bin) {
    const double distance=.3*bin;
    advance(distance,speed);
    if(distance>=3. || (joint&&bin>=5))for(double steering:{-maximum_steering,maximum_steering})
      for(double phase:{.6,.9})bypass(distance,speed,steering,phase);
  }
  if(joint)for(double distance:{3.6,4.2,4.8,5.4,6.}) {
    advance(distance,maximum_speed);
    // A long constant arc can reach a safe dynamic separation only by
    // turning into a roadside wall. Recover the heading instead: include
    // gentler curvature and enough first-phase time to achieve a useful
    // lateral displacement. Short S phases alone cover little side motion.
    // These remain proposals, subject to the SAME complete stop/terrain/body
    // and independently sampled multi-object certificate in the controller.
    for(double steering:{-.5*maximum_steering,.5*maximum_steering,-maximum_steering,maximum_steering})
      for(double phase:{.45,.6,.9,1.2,1.5})
      bypass(distance,maximum_speed,steering,phase);
  }
  return out;
}

// Comparable utility for ALL maneuver families. Selection still requires a
// certificate. Progress is toward the actual task goal, never toward a truth
// waypoint or the currently estimated obstacle departure point.
inline double interactionUtility(const YieldState &initial,const std::vector<YieldState>&states,
    double goal_x,double goal_y)
{
  if(states.empty() || !std::isfinite(goal_x) || !std::isfinite(goal_y))return -INFINITY;
  const auto &end=states.back();
  double travel=0,reverse=0,steering_change=0,progress_integral=0,motion_time=0;
  const double initial_distance=std::hypot(goal_x-initial.x,goal_y-initial.y);
  for(size_t i=1;i<states.size();++i) {
    const double ds=std::hypot(states[i].x-states[i-1].x,states[i].y-states[i-1].y);
    travel+=ds;if(states[i].speed<0)reverse+=ds;
    steering_change+=std::abs(states[i].steering-states[i-1].steering);
    const double dt=states[i].t-states[i-1].t;
    progress_integral+=(initial_distance-std::hypot(goal_x-states[i].x,goal_y-states[i].y))*dt;
    if(std::abs(states[i].speed)>.03)motion_time=states[i].t;
  }
  const double progress=std::hypot(goal_x-initial.x,goal_y-initial.y)-std::hypot(goal_x-end.x,goal_y-end.y);
  const double heading=std::remainder(std::atan2(goal_y-end.y,goal_x-end.x)-end.yaw,2*M_PI);
  const double early_progress=progress_integral/std::max(.01,end.t-states.front().t);
  return 3*progress+.5*early_progress-.15*motion_time-.2*travel-.5*reverse-
    .35*std::abs(heading)-.1*steering_change;
}
}  // namespace terrain_nav
