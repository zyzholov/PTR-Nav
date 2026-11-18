// Offline input/state identification. Recorded physical poses are assessment
// only; no ROS node, navigation command, pose correction or live truth input.
#include "terrain_nav/dynamic_yield.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>
using J=nlohmann::json;
using namespace terrain_nav;

int main(int argc,char **argv) {
  try {
    if((argc!=4&&argc!=5)||std::filesystem::exists(argv[3])||(argc==5&&std::string(argv[4])!="--separate"))
      throw std::runtime_error("actuator_model_audit RESPONSE_JSON VEHICLE_JSON NEW_OUTPUT_JSON [--separate]");
    J data,spec;std::ifstream(argv[1])>>data;std::ifstream(argv[2])>>spec;
    const auto &truth=data.at("truth");const double start=data.at("start");
    if(truth.size()<50||data.at("commands").size()<20)throw std::runtime_error("incomplete response record");
    std::vector<YieldCommand> published;
    for(const auto &u:data.at("commands")) {
      YieldCommand c{u.at(0),u.at(1),u.at(2)};
      if(!std::isfinite(c.t)||!std::isfinite(c.speed)||!std::isfinite(c.steering)||
         (!published.empty()&&c.t<published.back().t))throw std::runtime_error("invalid input schedule");
      published.push_back(c);
    }
    for(size_t i=0;i<truth.size();++i) {
      if(truth[i].size()!=5)throw std::runtime_error("invalid physical pose row");
      for(const auto &v:truth[i])if(!v.is_number()||!std::isfinite(v.get<double>()))throw std::runtime_error("nonfinite physical pose");
      if(i&&truth[i][0].get<double>()<=truth[i-1][0].get<double>())throw std::runtime_error("nonmonotonic physical pose");
    }
    J rows=J::array();double best=INFINITY,best_delay=NAN,best_speed=NAN,best_steering=NAN;
    std::vector<std::pair<double,double>> settings;
    for(double speed:{0.,.01,.02,.03,.04,.05,.06,.08,.10,.15,.20,.35})
      for(double steering:{0.,.01,.02,.03,.04,.05,.06,.08,.10,.15,.20,.35})
        if(argc==5||speed==steering)settings.emplace_back(speed,steering);
    for(const auto &[speed_delay,steering_delay]:settings) {
      const double delay=std::max(speed_delay,steering_delay);
      YieldModel model;model.delay=delay;model.step=.005;
      model.speed_dead_time=speed_delay;model.steering_dead_time=steering_delay;
      model.wheelbase=spec.at("wheelbase_m");model.rear_axle_offset=spec.at("rear_axle_offset_m");
      model.acceleration=model.deceleration=1.2;model.steering_rate=4.;
      J events=J::array();double total=0.;
      for(const auto &entry:std::vector<std::pair<double,double>>{{4.5,1.5},{7.,1.5},{9.5,2.}}) {
        size_t first=1;
        while(first+1<truth.size()&&truth[first][0].get<double>()<start+entry.first-.2)++first;
        if(first+1>=truth.size())throw std::runtime_error("missing pre-event physical root");
        const double time=truth[first][0],yaw=truth[first][4];
        const auto &a=truth[first-1],&b=truth[first+1];const double dt=b[0].get<double>()-a[0].get<double>();
        if(dt>.12)throw std::runtime_error("source gap around physical root");
        const double speed=((b[1].get<double>()-a[1].get<double>())*std::cos(yaw)+
          (b[2].get<double>()-a[2].get<double>())*std::sin(yaw))/dt;
        const double rate=std::remainder(b[4].get<double>()-a[4].get<double>(),2*M_PI)/dt;
        YieldState state{0.,truth[first][1],truth[first][2],yaw,speed,
          std::abs(speed)>.05?std::atan(rate*model.wheelbase/speed):0.};
        std::vector<YieldCommand> commands,issued;
        for(auto c:published) {
          if(c.t<=time)issued.push_back(c);
          else {c.t-=time;commands.push_back(c);}
        }
        auto active=yieldCommandAt(published,time);active.t=0.;commands.insert(commands.begin(),active);
        const auto path=rolloutQueuedYield(state,commands,model,entry.second+.2,time,issued);
        if(path.empty())throw std::runtime_error("input replay rejected");
        double xy2=0.,yaw2=0.,maximum=0.;size_t count=0,position=0;
        for(size_t i=first;i<truth.size();++i) {
          const double elapsed=truth[i][0].get<double>()-time;if(elapsed>path.back().t)break;
          while(position+1<path.size()&&path[position+1].t<elapsed)++position;
          if(position+1>=path.size())break;
          const auto &p=path[position],&q=path[position+1];const double f=(elapsed-p.t)/(q.t-p.t);
          const double x=p.x+f*(q.x-p.x),y=p.y+f*(q.y-p.y),
            heading=p.yaw+f*std::remainder(q.yaw-p.yaw,2*M_PI);
          const double error=std::hypot(x-truth[i][1].get<double>(),y-truth[i][2].get<double>()),
            angle=std::remainder(heading-truth[i][4].get<double>(),2*M_PI);
          xy2+=error*error;yaw2+=angle*angle;maximum=std::max(maximum,error);++count;
        }
        if(count<20)throw std::runtime_error("insufficient matched physical samples");
        const double xy=std::sqrt(xy2/count),heading=std::sqrt(yaw2/count);
        total+=xy+.5*heading;
        events.push_back({{"event_elapsed_s",entry.first},{"xy_rmse_m",xy},{"yaw_rmse_rad",heading},
          {"xy_max_m",maximum},{"samples",count}});
      }
      rows.push_back({{"pure_input_delay_s",delay},{"speed_dead_time_s",speed_delay},
        {"steering_dead_time_s",steering_delay},{"loss",total},{"events",events}});
      if(total<best){best=total;best_delay=delay;best_speed=speed_delay;best_steering=steering_delay;}
    }
    std::ofstream(argv[3])<<J{{"scope","offline full-pose input-delay identification with fixed 1.2m/s2 slew and 4rad/s steering slew; response threshold is not pure dead time"},
      {"limitation","one unloaded flat trace; best fit is a nominal parameter, not a worst-case latency or closed-loop safety bound"},
      {"source",argv[1]},{"separate_actuators",argc==5},{"best_pure_input_delay_s",best_delay},
      {"best_speed_dead_time_s",best_speed},{"best_steering_dead_time_s",best_steering},{"rows",rows}}.dump(2)<<'\n';
    std::cout<<"best nominal speed/steering dead times="<<best_speed<<"/"<<best_steering<<" s; loss="<<best<<'\n';return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
