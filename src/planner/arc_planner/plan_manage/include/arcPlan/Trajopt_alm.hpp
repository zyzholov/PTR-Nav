#ifndef TRAJOPT_HPP
#define TRAJOPT_HPP
#include "Polytraj.hpp"
#include "lbfgs.hpp"
#include "geoutils2d.hpp"
#include <tools/config.hpp>
#include <algorithm>
#include <chrono>
#include <limits>
// using namespace std;

namespace PolyTrajOpt
{
class TrajOpt{
    private:
        Eigen::VectorXi piecenums;
        int trajnum;
        double gslar_ = 0.05;
        double wei_time_ = 500.0;
        double esdfWei = 1000.0;
        double riskWei_ = 0.85;
        double risk_safe_threshold_ = 0.75;
        double kmax = 0.45;
        double vmax = 3.0;
        double latAccmax = 3.0;
        double lonAccmax = 3.0;
        double accRatemax = 3.0;
        double phimax = 0.785;
        double wheelbase = 0.6;
        int traj_res = 16;
        int iter_wxx = 0;
        int skip_iter_vis = 10;
        int skip_iter_duration_ms = 10;
        bool once_optimization = false;
        std::vector<Eigen::MatrixXd> cfgHs_;
        std::vector<Eigen::Vector2d> conpts;
        /*lots of alm param*/
        double scaling_wf_ = 1.0;
        Eigen::VectorXd scaling_eqwc_, scaling_uneqwc_;
        double scaling_wf_min_ = 0.01, scaling_wc_min_ = 0.01;
        int eqc_num = 0, neqc_num = 0, totalc_num = 0;
        Eigen::VectorXd lambda, mu, hx, gx;
        double rho = 1.0, rho_max =1000.0;
        double gamma = 1;
        double cons_eps_ = 0.20;
        double safeMargin = 0.15;
        Eigen::VectorXd gradx;
        double almTotalcost;
        Eigen::MatrixXd headState2d_, tailState2d_;
        std::vector<int> singual_;
        std::shared_ptr<se2_grid::SE2Grid> map_itf_;
        std::shared_ptr<se2_grid::SE2Grid> sdf_map_itf_;
        int eqc_idx = 0;
        int uneqc_idx = 0;
        std::shared_ptr<visualization::Visualization> vis_tool_;
        bool isdebug_;
        bool isfix_;
        double backW_;
        double miniS = 0.1;
        bool isstart;
        double totalCost;
        std::string name_;
        std::vector<std::vector<Eigen::MatrixXd>> hPoly_container_;
        double lbfgs_start_wall_sec_ = 0.0;
        double lbfgs_time_budget_sec_ = 8.0;
        bool lbfgs_timeout_hit_ = false;
        double jerkWei_ = 0.1;
    public:
        TrajOpt(){

        };
        std::vector<PolyTrajOpt::MINCO_S3NU<2>> posOpts;
        std::vector<PolyTrajOpt::MINCO_S3NU<1>> tOpts;
        Eigen::MatrixXd finalInnerpoints;
        Eigen::VectorXd finalpieceTime;
        double min_dis_;
        double debugEnergy, debugRhoT,debugEsdf, debugCur, debugMinis;
        std::vector<Eigen::Matrix2Xd> robots_vis;
        inline int OptimizeSe2Trajectory(
            const Eigen::MatrixXd &headState2d, const Eigen::MatrixXd &tailState2d, Eigen::VectorXd rdTs_container,
            const std::vector<Eigen::MatrixXd> &inPs_container, const std::vector<Eigen::Vector2d> &gearPos,
             const std::vector<double> & angles, std::shared_ptr<se2_grid::SE2Grid> map_itf, std::shared_ptr<se2_grid::SE2Grid> sdf_map_itf,std::shared_ptr<Config> config_, const std::vector<int> singual,
            std::shared_ptr<visualization::Visualization> vis_tool,std::string name = "nn", std::vector<Eigen::VectorXd> ds_container = {})
        {
            //arcs is total length for each mincotraj
            //angles is the angle at the gear shifting state
            //inPs_container[i] is the wayPts for each mincotraj
            name_ = name;
            vis_tool_ = vis_tool;
            /*setparam*/
            wei_time_ = config_->wei_time_;
            kmax = config_->kmax;
            vmax = config_->vmax;
            latAccmax = config_->latAccmax;
            lonAccmax = config_->lonAccmax;
            accRatemax = config_->accRatemax;
            traj_res = config_->traj_res;
            rho = config_->rho;
            rho_max = config_->rho_max;
            gamma = config_->gamma;
            cons_eps_ = config_->cons_eps_;
            conpts = config_->conpts;
            singual_ = singual;
            isdebug_ = config_->isdebug;
            isfix_ = config_->isfixGear;
            backW_ = config_->backW;
            miniS = config_->miniS;
            safeMargin = config_->safeMargin;
            esdfWei = config_->esdfWei;
            riskWei_ = config_->riskWei;
            risk_safe_threshold_ = config_->risk_safe_threshold;
            jerkWei_ = std::clamp(config_->backend_jerk_weight, 0.03, 1.00);
            map_itf_ =  map_itf;
            sdf_map_itf_ = sdf_map_itf;
            phimax = config_->phimax;
            wheelbase = config_->wheel_base;
            kmax = tan(phimax) / wheelbase; 
            skip_iter_vis = config_->skip_iter_vis;
            skip_iter_duration_ms = config_->skip_iter_duration_ms;
            once_optimization = config_->once_optimization;
            isstart = true;
            for(int i = 0; i < rdTs_container.size(); i++){
                if(rdTs_container[i] <= gslar_){
                    std::cerr << "[ERROR] piece time <= gslar_" << std::endl;
                    rdTs_container[i] = gslar_ + 0.1;
                }
            }
            //totalt is total time
            headState2d_ = headState2d;
            tailState2d_ = tailState2d;
            trajnum = singual_.size();
            if (isdebug_)
                std::cout << "TrajOpt: trajnum: " << trajnum << std::endl;
            posOpts.resize(trajnum);piecenums.resize(trajnum);
            tOpts.resize(trajnum);
            finalInnerpoints.resize(2,trajnum-1);
            finalInnerpoints.setZero();
            finalpieceTime.resize(trajnum);
            finalpieceTime.setZero();
            int variable_num_ = 0;
            for(int i = 0; i < trajnum; i++) {
                piecenums[i] = inPs_container[i].cols() + 1;
                posOpts[i].reset(piecenums[i]);
                tOpts[i].reset(piecenums[i]);
                variable_num_ += 2 * (piecenums[i] - 1);
                variable_num_ += (piecenums[i]); // scale (ds
                eqc_num += piecenums[i] * (traj_res+1) * 0; // vnorm = 1
                neqc_num += piecenums[i] * (traj_res+1) * 8; //  vel>0.1 cur  s' >= 0   vel acc  omega
            }
            if (isdebug_)
                std::cout << "begin to optimize Arc Trajectory!" << std::endl;
            variable_num_ += trajnum;         // dts
            variable_num_ += 2 * (trajnum-1); //gear pos
            variable_num_ += 1 * (trajnum-1); //gear angle
            totalc_num = eqc_num + neqc_num;
            scaling_wf_ = 1.0;
            scaling_eqwc_.resize(eqc_num); scaling_eqwc_.setConstant(1.0);
            scaling_uneqwc_.resize(totalc_num); scaling_uneqwc_.setConstant(1.0);
            hx.resize(eqc_num); lambda.resize(eqc_num);
            hx.setZero();       lambda.setZero();
            gx.resize(neqc_num); mu.resize(neqc_num);
            gx.setZero();        mu.setZero(); 
            Eigen::VectorXd x;
            x.resize(variable_num_);
            int offset = 0;
            for(int i = 0; i<trajnum; i++){
                memcpy(x.data()+offset,inPs_container[i].data(), inPs_container[i].size() * sizeof(x[0]));
                offset += inPs_container[i].size();
            }
            Eigen::Map<Eigen::VectorXd> Vdts(x.data()+offset, rdTs_container.size());
            for(int i = 0; i < rdTs_container.size(); i++){
                double va;
                RealT2VirtualT(rdTs_container[i], va);
                Vdts[i] = va;
            }
            offset += rdTs_container.size();
            for(int i = 0; i < trajnum-1; i++){
                memcpy(x.data()+offset,gearPos[i].data(), 2 * sizeof(x[0]));
                offset += 2;
            }
            Eigen::Map<Eigen::VectorXd> angles_(x.data()+offset, trajnum-1);
            for(int i = 0; i < trajnum - 1; i++){
                angles_[i] = angles[i];
            }
            offset += trajnum-1;
            std::vector<Eigen::VectorXd> Rds_container;
            for(int i = 0; i < trajnum; i++){
                Eigen::Map<Eigen::VectorXd> Vds(x.data()+offset, piecenums[i]);
                Eigen::VectorXd Rds(piecenums[i]);
                if(ds_container.size() == 0){
                    {
                        Eigen::Vector2d tmpHead, tmpTail;
                        if(i == 0){
                            tmpHead << headState2d.col(0);
                        }
                        else{
                            tmpHead << gearPos[i-1];
                        }
                        if(i == trajnum - 1){
                            tmpTail << tailState2d.col(0);
                        }
                        else{
                            tmpTail << gearPos[i];
                        }
                        Eigen::MatrixXd tmpPts; tmpPts.resize(2, piecenums[i] + 1);
                        tmpPts.col(0) << tmpHead; tmpPts.col(piecenums[i]) << tmpTail;
                        for(int j = 0; j < piecenums[i] - 1; j++){
                            tmpPts.col(j+1) = inPs_container[i].col(j);
                        }

                        for(int j = 0; j < piecenums[i]; j++){
                            Rds[j] = (tmpPts.col(j) - tmpPts.col(j+1)).norm();
                        }
                    }     
                }
                else{
                    Rds = ds_container[i];
                }                           
                for(int j = 0; j < Rds.size(); j++){
                    if(Rds[j] <= gslar_){
                        std::cerr << "[ERROR] RT[i] <= gslar_" << std::endl;
                        Rds[j] = gslar_+ 0.1;
                    }
                }
                Rds_container.push_back(Rds);
                RealT2VirtualT(Rds, Vds);
                offset += piecenums[i];
            }
            //the variables beform optimization
            if (isdebug_)
                std::cout << "The variables beform optimization: " << std::endl;
            // for(int i = 0; i < trajnum; i++){
            //     std::cout << "trajid: "<<i<<" scales: "<<Rds_container[i].transpose()<<std::endl;
            //     std::cout << "trajid: "<<i<<" times: "<<rdTs_container[i]<<std::endl;
            //     std::cout << "trajid: "<<i<<" waypoints:\n"<<inPs_container[i]<<std::endl;
            //     std::cout << "trajid: "<<i<<" singuals: "<<singual_[i]<<std::endl;
            //     std::cout << "headState2d: \n"<<headState2d<<std::endl;
            //     std::cout << "tailState2d: \n"<<tailState2d<<std::endl;
            //     if(i < trajnum-1){
            //     std::cout << "gearPos: "<<gearPos[i].transpose()<<std::endl;
            //     std::cout << "angles: "<<angles[i]<<std::endl;}
            // }
            lbfgs::lbfgs_parameter_t lbfgs_params;
            lbfgs_params.mem_size = config_->mem_size;//128
            lbfgs_params.past = config_->past; //3 
            lbfgs_params.g_epsilon = config_->g_epsilon;
            lbfgs_params.min_step = config_->min_step;
            lbfgs_params.delta = config_->delta;
            lbfgs_params.max_iterations = config_->max_iterations;
            double final_cost;
            bool flag_success = true;
            bool emergency_accept = false;
            /*ALM*/
            iter_wxx = 0;
            double t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            lbfgs_start_wall_sec_ = t1;
            // Bound optimizer wall time for real-time replanning. The previous
            // lower bound (4s) caused severe control lag and trajectory stutter.
            lbfgs_time_budget_sec_ = std::min(0.80, std::max(0.25, rdTs_container.sum() * 0.06 + 0.12));
            lbfgs_timeout_hit_ = false;
            int iter = 0;
            while(true){
                int result = lbfgs::lbfgs_optimize(
                x,
                final_cost,
                TrajOpt::almCostFunctionCallback,
                NULL,
                TrajOpt::earlyExit,
                this,
                lbfgs_params);
                if (result == lbfgs::LBFGS_CONVERGENCE ||
                result == lbfgs::LBFGS_CANCELED ||
                result == lbfgs::LBFGS_STOP||result == lbfgs::LBFGSERR_MAXIMUMITERATION)
                {
                    if (result == lbfgs::LBFGS_CANCELED && lbfgs_timeout_hit_)
                    {
                        std::cerr << "[WARN] Lbfgs canceled by wall-time budget (" << lbfgs_time_budget_sec_ << "s)" << std::endl;
                        emergency_accept = true;
                        break;
                    }
                 }
                else if (result == lbfgs::LBFGSERR_MAXIMUMLINESEARCH ||
                         result == -1007)  // rounding error: step at machine precision, essentially converged
                {
                    std::cerr << "[WARN] Lbfgs line-search limit / rounding (code=" << result
                              << "), accepting current solution" << std::endl;
                    emergency_accept = true;
                    break;
                }
                else
                {
                    if (lbfgs_timeout_hit_)
                    {
                        std::cerr << "[WARN] Lbfgs interrupted by wall-time budget (" << lbfgs_time_budget_sec_ << "s), code=" << result << std::endl;
                        emergency_accept = true;
                        break;
                    }
                    std::cerr << "[WARN] Solver error " << result << ": " << lbfgs::lbfgs_strerror(result) << std::endl;
                    return 0;
                }
                updateDualVar();
                if(ifConvergence()){
                    if (isdebug_)
                        std::cout << "Convergence! iter: " << iter << std::endl;
                    break;
                }
                iter++;
                if(iter > config_->amlMaxIter){
                    std::cerr << "[WARN] Reach max iteration" << std::endl;
                    emergency_accept = true;
                    break;
                }
            }
            double t2 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            debugVis();
            if (emergency_accept)
            {
                std::cerr << "[WARN] Emergency-accept unconverged trajectory for continuity" << std::endl;
            }
            if (isdebug_)
            {
                printf("se2 trajectory generation success!time(ms)=%5.3f, totalCost=%5.3f\n", (t2-t1) * 1000.0,totalCost);
                std::cout << "dptraj debugEsdf: " << debugEsdf << std::endl;
            }
            return flag_success;
        }

        static inline int earlyExit(void *instance,
                            const Eigen::VectorXd &x,
                            const Eigen::VectorXd &g,
                            const double fx,
                            const double step,
                            const int k,
                            const int ls)
        {
            TrajOpt *opt = reinterpret_cast<TrajOpt *>(instance);
            (void)x;
            (void)g;
            (void)fx;
            (void)step;
            (void)k;
            (void)ls;
            double now = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            if (opt->lbfgs_start_wall_sec_ > 0.0 && (now - opt->lbfgs_start_wall_sec_) > opt->lbfgs_time_budget_sec_)
            {
                opt->lbfgs_timeout_hit_ = true;
                return 1;
            }
            // ROS1 debug output disabled in ROS2 migration
            // ROS_INFO_STREAM("x: " << x.transpose());
            // ROS_INFO_STREAM("g: " << g.transpose());
            // ROS_INFO_STREAM("step: " << step);
            // ROS_INFO_STREAM("");
            return 0;
        }
        static inline double almCostFunctionCallback(void *func_data, const Eigen::VectorXd &x, Eigen::VectorXd &grad){
            double smcost = 0.0, timecost = 0.0, almcost = 0.0;
            TrajOpt *opt = reinterpret_cast<TrajOpt *>(func_data);
            double now = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            if (opt->lbfgs_start_wall_sec_ > 0.0 && (now - opt->lbfgs_start_wall_sec_) > opt->lbfgs_time_budget_sec_)
            {
                opt->lbfgs_timeout_hit_ = true;
                grad.setZero();
                return std::numeric_limits<double>::infinity();
            }
            opt->totalCost = 0.0;
            opt->robots_vis.clear();

            int offset = 0;
            std::vector<Eigen::Map<const Eigen::MatrixXd>> P_container;
            std::vector<Eigen::Map<Eigen::MatrixXd>> gradP_container;;
            for(int trajid = 0; trajid < opt->trajnum; trajid++){
                Eigen::Map<const Eigen::MatrixXd> P(x.data()+offset, 2, opt->piecenums[trajid] - 1);
                Eigen::Map<Eigen::MatrixXd>gradP(grad.data()+offset, 2, opt->piecenums[trajid] - 1);
                offset += 2 * (opt->piecenums[trajid] - 1);
                gradP.setZero();
                P_container.push_back(P);
                gradP_container.push_back(gradP);
            }
            Eigen::Map<const Eigen::VectorXd> Vdts(x.data()+offset, opt->trajnum);
            Eigen::Map<Eigen::VectorXd>gradVdts(grad.data()+offset, opt->trajnum);
            offset += opt -> trajnum;
            Eigen::VectorXd dts(opt->trajnum);
            Eigen::VectorXd gradDts(opt->trajnum); gradDts.setZero();
            opt->VirtualT2RealT(Vdts, dts);

            std::vector<Eigen::Map<const Eigen::Vector2d>> Gear_container;
            std::vector<Eigen::Map<Eigen::Vector2d>> gradGear_container;
            for(int trajid = 0; trajid < opt->trajnum - 1; trajid++){
                Eigen::Map<const Eigen::Vector2d> Gear(x.data()+offset, 2);
                Eigen::Map<Eigen::Vector2d>gradGear(grad.data()+offset, 2);
                offset += 2;
                gradGear.setZero();
                Gear_container.push_back(Gear);
                gradGear_container.push_back(gradGear);
            }
            Eigen::Map<const Eigen::VectorXd> Angles(x.data()+offset, opt->trajnum-1);
            Eigen::Map<Eigen::VectorXd>gradAngles(grad.data()+offset, opt->trajnum-1);
            gradAngles.setZero();
            offset += opt->trajnum-1;
            std::vector<Eigen::Map<const Eigen::VectorXd>> Vs_container;
            std::vector<Eigen::Map<Eigen::VectorXd>> gradVs_container;;
            std::vector<Eigen::VectorXd> Rs_container;
            std::vector<Eigen::VectorXd> gradRs_container;;
            for(int trajid = 0; trajid < opt->trajnum; trajid++){
                Eigen::Map<const Eigen::VectorXd> Vs(x.data()+offset, opt->piecenums[trajid]);
                Eigen::Map<Eigen::VectorXd>gradVs(grad.data()+offset, opt->piecenums[trajid]);
                Vs_container.push_back(Vs);
                gradVs.setZero();
                gradVs_container.push_back(gradVs); 
                Eigen::VectorXd Rs(opt->piecenums[trajid]);
                Eigen::VectorXd gradRs(opt->piecenums[trajid]); gradRs.setZero();
                opt->VirtualT2RealT(Vs, Rs);
                Rs_container.push_back(Rs);
                gradRs_container.push_back(gradRs);
                offset += opt->piecenums[trajid];
            }
            std::vector<Eigen::MatrixXd> gdS2c_2d(opt->trajnum), gdS2c_1d(opt->trajnum), gdC2c_2d(opt->trajnum), gdC2c_1d(opt->trajnum), gdTotalc_2d(opt->trajnum), gdTotalc_1d(opt->trajnum);
            std::vector<Eigen::VectorXd> gdS2t_2d(opt->trajnum), gdC2t_2d(opt->trajnum), gdTotalt_2d(opt->trajnum), gdS2t_1d(opt->trajnum), gdC2t_1d(opt->trajnum), gdTotalt_1d(opt->trajnum);

            opt->eqc_idx = 0;
            opt->uneqc_idx = 0;
            opt->debugEnergy = 0.0;
            opt->debugEsdf = 0.0;
            opt->min_dis_ = INFINITY;
            opt->debugRhoT = 0.0;
            opt->debugCur = -INFINITY;
            opt->debugMinis = INFINITY;

            for(int trajid = 0; trajid < opt->trajnum; trajid++){
                double s = 1.0 * opt->singual_[trajid];
                Eigen::Matrix<double, 2, 3> IniState, FinState;
                if(trajid == 0){
                    IniState = opt->headState2d_;
                }
                else{
                    IniState.col(0) << Gear_container[trajid - 1];
                    IniState.col(1) << s * cos(Angles[trajid -1]), s * sin(Angles[trajid -1]);
                    IniState.col(2) << 0.0, 0.0;
                }
                if(trajid == opt->trajnum-1){
                    FinState = opt->tailState2d_;
                }
                else{
                    FinState.col(0) << Gear_container[trajid];
                    FinState.col(1) << s * cos(Angles[trajid]), s * sin(Angles[trajid]);
                    FinState.col(2) << 0.0, 0.0;
                }
                opt->posOpts[trajid].generate(P_container[trajid],Rs_container[trajid],IniState,FinState);
                opt->posOpts[trajid].initSmGradCost(gdS2c_2d[trajid], gdS2t_2d[trajid]);
                gdS2c_2d[trajid].setZero();
                gdS2t_2d[trajid].setZero();

                {
                    Eigen::MatrixXd wpTs, IniSt, FinSt;
                    wpTs.resize(1, opt->piecenums[trajid]-1); IniSt.resize(1,3); FinSt.resize(1,3);
                    for(int idx = 0; idx <  opt->piecenums[trajid]-1; idx++){
                        wpTs.col(idx) << Rs_container[trajid].segment(0, idx + 1).sum();
                    }
                    IniSt << 0.0, 0.0, 0.0;
                    FinSt << Rs_container[trajid].sum(), 0.0,0.0;
                    Eigen::VectorXd cdts(opt->piecenums[trajid]);
                    cdts.setConstant(dts[trajid]);
                    opt->tOpts[trajid].generate(wpTs, cdts, IniSt, FinSt);
                    opt->tOpts[trajid].initSmGradCost(gdS2c_1d[trajid], gdS2t_1d[trajid]);
                    gdS2c_1d[trajid].setZero();
                    gdS2t_1d[trajid].setZero();
                }
                opt->almGradCost2CT(almcost, opt->posOpts[trajid], opt->tOpts[trajid], gdC2c_2d[trajid], gdC2t_2d[trajid], gdC2c_1d[trajid], gdC2t_1d[trajid], trajid); // Time int cost
                gdTotalc_2d[trajid] = gdC2c_2d[trajid] + gdS2c_2d[trajid];
                gdTotalt_2d[trajid] = gdC2t_2d[trajid] + gdS2t_2d[trajid];
                gdTotalc_1d[trajid] = gdC2c_1d[trajid] + gdS2c_1d[trajid];
                gdTotalt_1d[trajid] = gdC2t_1d[trajid] + gdS2t_1d[trajid];
            }
            for(int trajid = 0; trajid < opt->trajnum; trajid++) {
                Eigen::MatrixXd propogate_gradp_2d, propogate_gradIni_2d, propogate_gradFin_2d;
                opt->posOpts[trajid].calGrads_PT(gdTotalc_2d[trajid], gdTotalt_2d[trajid], propogate_gradp_2d, propogate_gradIni_2d, propogate_gradFin_2d);
                double sg = 1.0 * opt->singual_[trajid];
                if(trajid > 0){
                    double theta = Angles[trajid-1];
                    gradGear_container[trajid-1] += propogate_gradIni_2d.col(0);
                    gradAngles[trajid-1] += propogate_gradIni_2d.col(1).transpose() * Eigen::Vector2d(-sg * sin(theta), sg*cos(theta));
                }
                if(trajid < opt->trajnum-1){
                    double theta = Angles[trajid];
                    gradGear_container[trajid] += propogate_gradFin_2d.col(0);
                    gradAngles[trajid] += propogate_gradFin_2d.col(1).transpose() * Eigen::Vector2d(-sg * sin(theta), sg*cos(theta));
                }
                gradP_container[trajid] = propogate_gradp_2d;
                double k;
                if(sg > 0){
                    k = 1.0;
                }
                else{
                    k = opt->backW_;
                }
                gradRs_container[trajid] = gdTotalt_2d[trajid];
                
                Eigen::MatrixXd propogate_gradp_1d, propogate_gradIni_1d, propogate_gradFin_1d;
                opt->tOpts[trajid].calGrads_PT(gdTotalc_1d[trajid], gdTotalt_1d[trajid], propogate_gradp_1d, propogate_gradIni_1d, propogate_gradFin_1d);
                //特殊
                for(int p1 = 0; p1 < opt->piecenums[trajid] - 1; p1++){
                    gradRs_container[trajid].segment(0, p1 + 1).array() += propogate_gradp_1d.col(p1)[0];
                }
                gradRs_container[trajid].array() += propogate_gradFin_1d.col(0)[0];
                gradDts[trajid] = gdTotalt_1d[trajid].sum();
                gradDts[trajid] += k *  opt->wei_time_ * opt->piecenums[trajid];
                timecost += k * opt->wei_time_ * dts[trajid]* opt->piecenums[trajid];
                opt->debugRhoT += k * opt->wei_time_ * dts[trajid]* opt->piecenums[trajid];
            }
            opt->Virtual2Grad(Vdts,  gradDts, gradVdts);
            for(int trajid = 0; trajid < opt->trajnum; trajid++){
                opt->Virtual2Grad(Vs_container[trajid],  gradRs_container[trajid], gradVs_container[trajid]);        
            }
            if(opt->isfix_){
                for(int trajid = 0; trajid < opt->trajnum - 1; trajid++){
                    gradGear_container[trajid].setZero();
                    gradAngles[trajid]  = 0.0;
                }
            }
            if(opt->isdebug_){
                // std::cout << "------------------------------------------------------------------\n";
                std::cout << "min_dis_: "<<opt->min_dis_ << std::endl;
                std::cout << "debugCur: "<<opt->debugCur << std::endl;
                std::cout << "debugMinis: "<<opt->debugMinis << std::endl;
                if(opt->debugEnergy > 1.0e10){
                    std::cout << "cost is too high!"<<"\n";
                }   
                else{
                    opt->debugVis();
                    // ros::Duration(0.01).sleep();  // ROS1 sleep disabled in ROS2 migration
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
            opt->totalCost += timecost;
            return smcost + timecost + almcost;
        }
        template <typename EIGENVEC>
        void VirtualT2RealT(const EIGENVEC &VT, Eigen::VectorXd &RT)
        {
            for (int i = 0; i < VT.size(); ++i)
            {
            RT(i) = VT(i) > 0.0 ? ((0.5 * VT(i) + 1.0) * VT(i) + 1.0) + gslar_
                                : 1.0 / ((0.5 * VT(i) - 1.0) * VT(i) + 1.0) + gslar_;
            }
        }
        void VirtualT2RealT(const  double & VT, double &RT){
            
            
            RT = VT > 0.0 ? ((0.5 * VT + 1.0) * VT + 1.0) + gslar_
                                : 1.0 / ((0.5 * VT - 1.0) * VT + 1.0) + gslar_;
        }
        template <typename EIGENVEC>
        inline void RealT2VirtualT(const Eigen::VectorXd &RT, EIGENVEC &VT)
        {
        for (int i = 0; i < RT.size(); ++i)
            {
                VT(i) = RT(i) > 1.0 + gslar_ 
              ? (sqrt(2.0 * RT(i) - 1.0 - 2 * gslar_) - 1.0)
              : (1.0 - sqrt(2.0 / (RT(i)-gslar_) - 1.0));
            }
        }
        inline void RealT2VirtualT(const double &RT, double &VT)
        {
        
            VT = RT > 1.0 + gslar_ 
            ? (sqrt(2.0 * RT - 1.0 - 2 * gslar_) - 1.0)
            : (1.0 - sqrt(2.0 / (RT-gslar_) - 1.0));
        }
        template <typename EIGENVEC, typename EIGENVECGD>
        void VirtualTGradCost(const Eigen::VectorXd &RT, const EIGENVEC &VT,const Eigen::VectorXd &gdRT, EIGENVECGD &gdVT,double &costT)
        {
            for (int i = 0; i < VT.size(); ++i)
            {
            double gdVT2Rt;
            if (VT(i) > 0)
            {
                gdVT2Rt = VT(i) + 1.0;
            }
            else
            {
                double denSqrt = (0.5 * VT(i) - 1.0) * VT(i) + 1.0;
                gdVT2Rt = (1.0 - VT(i)) / (denSqrt * denSqrt);
            }

            gdVT(i) = (gdRT(i) + wei_time_) * gdVT2Rt;
            }

            costT = RT.sum() * wei_time_;
        }
        void VirtualTGradCost(const double &RT, const double &VT, const double &gdRT, double &gdVT, double& costT){
            double gdVT2Rt;
            if (VT > 0)
            {
            gdVT2Rt = VT + 1.0;
            }
            else
            {
            double denSqrt = (0.5 * VT - 1.0) * VT + 1.0;
            gdVT2Rt = (1.0 - VT) / (denSqrt * denSqrt);
            }

            gdVT = (gdRT + wei_time_) * gdVT2Rt;
            costT = RT * wei_time_;
        }        
        void VirtualTGrad2t(const double &VT, const double &gdRT, double &gdVT){
            double gdVT2Rt;
            if (VT > 0)
            {
            gdVT2Rt = VT + 1.0;
            }
            else
            {
            double denSqrt = (0.5 * VT - 1.0) * VT + 1.0;
            gdVT2Rt = (1.0 - VT) / (denSqrt * denSqrt);
            }
            gdVT = gdRT * gdVT2Rt;
        }    
        template <typename EIGENVEC, typename EIGENVECGD>
        void Virtual2Grad(const EIGENVEC &VT, const Eigen::VectorXd &gdRT, EIGENVECGD &gdVT){
            for (int i = 0; i < VT.size(); ++i)
            {
            double gdVT2Rt;
            if (VT(i) > 0)
            {
                gdVT2Rt = VT(i) + 1.0;
            }
            else
            {
                double denSqrt = (0.5 * VT(i) - 1.0) * VT(i) + 1.0;
                gdVT2Rt = (1.0 - VT(i)) / (denSqrt * denSqrt);
            }

            gdVT(i) = (gdRT(i)) * gdVT2Rt;
            }
            
        }        
        double expC2(double t) {
            return t > 0.0 ? ((0.5 * t + 1.0) * t + 1.0)
                 : 1.0 / ((0.5 * t - 1.0) * t + 1.0);
        }
        double logC2(double T) {
            return T > 1.0 ? (sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T - 1.0));
        }
        void forwardT(const Eigen::Ref<const Eigen::VectorXd>& t, const double& sT, Eigen::Ref<Eigen::VectorXd> vecT) {
            int M = t.size();
            for (int i = 0; i < M; ++i) {
                vecT(i) = expC2(t(i));
            }
            vecT(M) = 0.0;
            vecT /= 1.0 + vecT.sum();
            vecT(M) = 1.0 - vecT.sum();
            vecT *= sT;
            return;
        }
        void backwardT(const Eigen::Ref<const Eigen::VectorXd>& vecT, Eigen::Ref<Eigen::VectorXd> t) {
            int M = t.size();
            t = vecT.head(M) / vecT(M);
            for (int i = 0; i < M; ++i) {
            t(i) = logC2(vecT(i));
            }
            return;
        }
        void almGradCost2CT(double &cost,const PolyTrajOpt::MINCO_S3NU<2>& posOpt, const PolyTrajOpt::MINCO_S3NU<1>& tOpt, Eigen::MatrixXd& gdC2c_2d, Eigen::VectorXd& gdC2t_2d, Eigen::MatrixXd& gdC2c_1d, Eigen::VectorXd& gdC2t_1d, int trajid){
          uneqc_idx = 0;
          int N = piecenums[trajid];
            double sg = 1.0 * singual_[trajid];
            double s, ds, dds, ddds, dddds, latAcc, lonAcc, vnorm, cur;
            Eigen::Vector2d pos, vel, acc, jerk, snap;
            Eigen::Vector2d realVel, realAcc, realJerk;
            double energy;
            double gradS, gradDs, gradDds, gradDdds;
            Eigen::Vector2d gradPos, gradVel, gradAcc, gradJerk;
            Eigen::Matrix<double, 6, 1> beta0_2d, beta1_2d, beta2_2d, beta3_2d, beta4_2d;
            Eigen::Matrix<double, 6, 1> beta0_1d, beta1_1d, beta2_1d, beta3_1d, beta4_1d;
            Eigen::Matrix2d rotR, drotR;
            Eigen::Vector2d outerNormal;
            Eigen::Matrix<double, 6, 1> t_2d, t_1d;
            double step, alpha, eqf;
            double vioCur, vioVel, vioLonAcc, vioLatAcc, vioAcc, vioYawdot, vioCor, vioDs;
            int corId = -1;
            double baset = 0.0;
            t_2d(0) = 1.0;
            t_1d(0) = 1.0;

            Eigen::Vector3d xb, yb;
            double lon_acc, lat_acc, vx, wz, ax_body, ay_body, dyaw;
            double inv_cos_vphix, sin_phix, inv_cos_vphiy, sin_phiy, cos_xi, inv_cos_xi, risk;
            double gravity = 9.81;
            Eigen::Vector3d grad_inv_cos_vphix, grad_sin_phix, grad_inv_cos_vphiy, grad_sin_phiy, grad_cos_xi, grad_inv_cos_xi, grad_risk;
            Eigen::Vector3d grad_se2;
            std::vector<double> terrain_values;
            std::vector<Eigen::Vector3d> terrain_grads;
            double vel_snorm, vel_snorm_inv, vel_snorm_sigl, vel_snorm_sigl_inv, vlon_sigl_inv, vel_norm;
            Eigen::Vector3d pos_se2;
            double delta_sigl = 1e-5;

            gdC2c_2d.resize(6 * N, 2);
            gdC2c_2d.setZero();
            gdC2t_2d.resize(N);
            gdC2t_2d.setZero();
            gdC2c_1d.resize(6 * N, 1);
            gdC2c_1d.setZero();
            gdC2t_1d.resize(N);
            gdC2t_1d.setZero();
            double baseS = 0.0;

            int pointid = -1;
            for (int i = 0; i < N; ++i)
            {
                const Eigen::Matrix<double, 6, 1> &c_1d = tOpt.getCoeffs().block<6, 1>(i * 6, 0);
                const Eigen::Matrix<double, 6, 2> &c_2d = posOpt.getCoeffs().block<6, 2>(i * 6, 0);
                step = tOpt.T1(i) / traj_res; // T_i /k
                for (int j = 0; j <= traj_res; ++j)
                {
                    t_1d(1) = step * j;
                    t_1d(2) = t_1d(1) * t_1d(1);
                    t_1d(3) = t_1d(2) * t_1d(1);
                    t_1d(4) = t_1d(2) * t_1d(2);
                    t_1d(5) = t_1d(4) * t_1d(1);
                    beta0_1d  = t_1d;
                    beta1_1d << 0.0, 1.0, 2.0 * t_1d(1), 3.0 * t_1d(2), 4.0 * t_1d(3), 5.0 * t_1d(4);
                    beta2_1d << 0.0, 0.0, 2.0, 6.0 * t_1d(1), 12.0 * t_1d(2), 20.0 * t_1d(3);
                    beta3_1d << 0.0, 0.0, 0.0, 6.0, 24.0 * t_1d(1), 60.0 * t_1d(2);
                    beta4_1d << 0.0, 0.0, 0.0, 0.0, 24.0, 120 * t_1d(1);
                    s = c_1d.transpose() * beta0_1d;
                    ds = c_1d.transpose() * beta1_1d;
                    dds = c_1d.transpose() * beta2_1d;
                    ddds = c_1d.transpose() * beta3_1d;
                    dddds = c_1d.transpose() * beta4_1d;
                    gradS = 0.0;
                    gradDs = 0.0;
                    gradDds = 0.0;
                    gradDdds = 0.0;
                    /*analyse pos*/
                    t_2d(1) = s - baseS;
                    t_2d(2) = t_2d(1) * t_2d(1);
                    t_2d(3) = t_2d(2) * t_2d(1);
                    t_2d(4) = t_2d(2) * t_2d(2);
                    t_2d(5) = t_2d(4) * t_2d(1);
                    beta0_2d  = t_2d;
                    beta1_2d << 0.0, 1.0, 2.0 * t_2d(1), 3.0 * t_2d(2), 4.0 * t_2d(3), 5.0 * t_2d(4);
                    beta2_2d << 0.0, 0.0, 2.0, 6.0 * t_2d(1), 12.0 * t_2d(2), 20.0 * t_2d(3);
                    beta3_2d << 0.0, 0.0, 0.0, 6.0, 24.0 * t_2d(1), 60.0 * t_2d(2);
                    beta4_2d << 0.0, 0.0, 0.0, 0.0, 24.0, 120 * t_2d(1);
                    pos = c_2d.transpose() * beta0_2d;
                    vel = c_2d.transpose() * beta1_2d;
                    acc = c_2d.transpose() * beta2_2d;
                    jerk = c_2d.transpose() * beta3_2d;
                    snap = c_2d.transpose() * beta4_2d;
                    gradPos.setZero();
                    gradVel.setZero();
                    gradAcc.setZero();
                    gradJerk.setZero();
                    grad_se2.setZero();
                    alpha = 1.0 / traj_res * j;
                    cur = (vel[0]*acc[1] - acc[0]*vel[1]);
                    rotR << vel[0], -vel[1],
                            vel[1],  vel[0];
                    rotR = (sg * rotR) / vel.norm();
                    realVel = vel * ds;
                    realAcc = dds * vel + ds*acc*ds;
                    realJerk = ddds * vel + acc * dds * ds +2.0 * ds * acc * dds + ds * ds * ds * jerk;
                    pointid++;
                    if(pos.norm() > 40){
                        pos = 40.0 / pos.norm() * pos;
                    }
                    if(std::isnan(pos(0))|| std::isnan(pos(1)) || std::isinf(pos(0)) || std::isinf(pos(1))){
                        std::cout<<"pos is nan: "<<pos.transpose()<<std::endl;
                        return;
                    }
                    // terrain
                    pos_se2.head(2) = pos;
                    pos_se2(2) = atan2(vel(1), vel(0));
                    getTPMWithGrad(pos_se2, terrain_values, terrain_grads);
                    inv_cos_vphix = terrain_values[0];
                    sin_phix = terrain_values[1];
                    inv_cos_vphiy = terrain_values[2];
                    sin_phiy = terrain_values[3];
                    cos_xi = terrain_values[4];
                    inv_cos_xi = terrain_values[5];
                    risk = terrain_values[6];
                    grad_inv_cos_vphix = terrain_grads[0];
                    grad_sin_phix = terrain_grads[1];
                    grad_inv_cos_vphiy = terrain_grads[2];
                    grad_sin_phiy = terrain_grads[3];
                    grad_cos_xi = terrain_grads[4];
                    grad_inv_cos_xi = terrain_grads[5];
                    grad_risk = terrain_grads[6];

                    Eigen::Vector2d vel_local_dt = vel * ds;
                    double vel_local_norm = vel_local_dt.norm() + delta_sigl;
                    
                    Eigen::Vector2d acc_local_dt = acc * ds * ds + vel * dds;
                    vx = ds * vel.norm() * inv_cos_vphix;
                    ax_body = acc_local_dt.dot(vel_local_dt) / vel_local_norm * inv_cos_vphix + gravity * sin_phix;

                    Eigen::Vector2d acc_cross(acc(1), - acc(0));
                    ay_body = vel.dot(acc_cross) / vel.norm() * ds * ds * inv_cos_vphiy + gravity * sin_phiy;
                    wz = ( vel_local_dt(0) * acc_local_dt(1) - vel_local_dt(1) * acc_local_dt(0)) / vel.squaredNorm() * inv_cos_xi;
                    {
                        debugMinis = std::min(debugMinis, vel.norm());
                        double vionegV = miniS * miniS - vel.dot(vel);
                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vionegV;
                        if(tmu + rho * swc * vionegV > 0){
                            cost += swc * vionegV * (tmu + 0.5 * rho * swc * vionegV);
                            Eigen::Vector2d gradViolanegv;
                            gradViolanegv = -2.0*vel;
                            gradVel += addAlmGradPro(swc, tmu, vionegV, gradViolanegv);
                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;

                    }
                    {
                        double curs = 100.0;
                        cur = (vel[0]*acc[1] - acc[0]*vel[1]) / (vel.norm() * vel.norm() * vel.norm()) * inv_cos_xi * inv_cos_vphix;
                        double cross = (vel[0]*acc[1] - acc[0]*vel[1]);
                        vioCur = curs*(cur*cur - kmax*kmax);
                        debugCur = std::max(debugCur, fabs(cur));

                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vioCur;
                        if(tmu + rho * swc * vioCur > 0){
                            cost += swc * vioCur * (tmu + 0.5 * rho * swc * vioCur);
                            Eigen::Vector2d gradViolaCv;
                            Eigen::Vector2d gradViolaCa;
                            
                            gradViolaCv = curs *(
                                          2*cross*Eigen::Vector2d(acc[1], -acc[0])/(vel.squaredNorm()*vel.squaredNorm()*vel.squaredNorm()) -
                                          6*Eigen::Vector2d(vel[0], vel[1])*cross*cross/(vel.squaredNorm()*vel.squaredNorm()*vel.squaredNorm()*vel.squaredNorm())
                                            )* inv_cos_xi * inv_cos_vphix; 
                            gradViolaCa = curs * 2 * Eigen::Vector2d(-vel[1], vel[0]) * cross / (vel.squaredNorm()*vel.squaredNorm()*vel.squaredNorm())* inv_cos_xi * inv_cos_vphix;
                            gradVel += addAlmGradPro(swc, tmu, vioCur, gradViolaCv);
                            gradAcc += addAlmGradPro(swc, tmu, vioCur, gradViolaCa);
                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;
                    }
                    {
                        
                        vioDs = -ds;
                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vioDs;
                        if(tmu + rho * swc * vioDs > 0){
                            cost += swc * vioDs * (tmu + 0.5 * rho * swc * vioDs);
                            gradDs += addAlmGradPro(swc, tmu, vioDs, -1.0);
                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;
                    }
                    double v_weight = 100.0;
                    double vel_norm = vel.norm() + delta_sigl;
                    {
                        vioVel = v_weight * (vx * vx - vmax * vmax);
                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vioVel;
                        if(tmu + rho * swc * vioVel > 0){
                            cost += swc * vioVel * (tmu + 0.5 * rho * swc * vioVel);
                            Eigen::Vector2d gradViolaVv;
                            Eigen::Vector3d gradVioSe2;
                            double gradViolaVds;
                            gradViolaVv = 2 * vx * ds * inv_cos_vphix * vel / vel_norm;
                            gradViolaVds = 2 * vx * vel.norm() * inv_cos_vphix;
                            gradVel += v_weight * addAlmGradPro(swc, tmu, vioVel, gradViolaVv);
                            gradDs += v_weight * addAlmGradPro(swc, tmu, vioVel, gradViolaVds);

                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;
                    }

                    {
                        vioLonAcc = ax_body * ax_body - lonAccmax * lonAccmax;
                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vioLonAcc;
                        if(tmu + rho * swc * vioLonAcc > 0){
                            cost += swc * vioLonAcc * (tmu + 0.5 * rho * swc * vioLonAcc);
                            Eigen::Vector2d gradViolaLonAv = 2 * ax_body * ((ds * ds * acc + 2 * dds * vel) / vel_norm * inv_cos_vphix 
                                    - vel * (ds * ds * vel.dot(acc) + dds * vel.dot(vel)) / pow( vel_norm,3) * inv_cos_vphix); 
                            Eigen::Vector2d gradViolaLonAa = 2 * ax_body * ds * ds * vel / vel_norm * inv_cos_vphix;
                            double gradViolaLonAds = 2.0 * ax_body * (2 * ds * acc.dot(vel) /  vel_norm) * inv_cos_vphix;
                            double gradViolaLonAdds = 2.0 * ax_body * vel.dot(vel) /  vel_norm * inv_cos_vphix;
                            
                            gradVel += addAlmGradPro(swc, tmu, vioLonAcc, gradViolaLonAv);
                            gradAcc += addAlmGradPro(swc, tmu, vioLonAcc, gradViolaLonAa);
                            gradDs += addAlmGradPro(swc, tmu, vioLonAcc, gradViolaLonAds);
                            gradDds += addAlmGradPro(swc, tmu, vioLonAcc, gradViolaLonAdds);

                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;
                    }
                    {
                        // Risk-based footprint collision checking (replaces ESDF/SDF cost).
                        // Use the actual risk-map gradient. The previous body-corner
                        // direction approximation could flip with vehicle yaw and twist the path.
                        for(const Eigen::Vector2d resPt : conpts){
                            Eigen::Vector2d absPt = pos + rotR * resPt;
                            if(absPt.norm() > 40){
                                absPt = 40.0 / absPt.norm() * pos;
                            }
                            if(std::isnan(absPt(0))|| std::isnan(absPt(1)) || std::isinf(absPt(0)) || std::isinf(absPt(1))){
                                continue;
                            }
                            double fp_risk = 0.0;
                            Eigen::Vector3d fp_grad = Eigen::Vector3d::Zero();
                            try {
                                fp_risk = map_itf_->getValueGrad(
                                    "risk", Eigen::Vector3d(absPt(0), absPt(1), 0.0),
                                    fp_grad, se2_grid::InterpolationMethods::INTER_LINEAR);
                            } catch (...) {
                                continue;
                            }
                            double vioRisk = fp_risk - risk_safe_threshold_;
                            if(vioRisk > 0){
                                double pena, penaD;
                                positiveSmoothedL1(vioRisk, pena, penaD);
                                const double risk_scale = std::clamp(step, 0.05, 0.20);
                                cost += riskWei_ * risk_scale * pena;
                                debugEsdf += riskWei_ * risk_scale * pena;
                                Eigen::Vector2d risk_grad_xy = fp_grad.head(2);
                                if (risk_grad_xy.allFinite()) {
                                    const double grad_norm = risk_grad_xy.norm();
                                    if (grad_norm > 3.0)
                                        risk_grad_xy *= 3.0 / grad_norm;
                                    if (risk_grad_xy.norm() > 1.0e-5)
                                        gradPos += riskWei_ * risk_scale * penaD * risk_grad_xy;
                                }
                            }
                        }
                    }

                    {
                        vioLatAcc = ay_body * ay_body - latAccmax * latAccmax;
                        double swc = scaling_uneqwc_[uneqc_idx];
                        double tmu = mu[uneqc_idx];
                        gx[uneqc_idx] = vioLatAcc;

                        if(tmu + rho * swc * vioLatAcc > 0){
                            cost += swc * vioLatAcc * (tmu + 0.5 * rho * swc * vioLatAcc);
                            Eigen::Vector2d gradViolaLonAv;
                            gradViolaLonAv = 2 * ay_body * (acc_cross * ds * ds / vel_norm * inv_cos_vphiy 
                                - vel * (vel.dot(acc_cross) * ds * ds / pow(vel_norm,3)) * inv_cos_vphiy);
                        
                            Eigen::Vector2d gradViolaLonAa; 
                            Eigen::Vector2d vel_cross(-vel(1), vel(0));
                            gradViolaLonAa = 2 * ay_body * (ds * ds * vel_cross / vel_norm * inv_cos_vphiy);
                            double gradViolaLonAds = 2.0 * ay_body * 2 * ds * (vel.dot(acc_cross)) / vel_norm * inv_cos_vphiy;                         
                            gradVel += addAlmGradPro(swc, tmu, vioLatAcc, gradViolaLonAv);
                            gradAcc += addAlmGradPro(swc, tmu, vioLatAcc, gradViolaLonAa);
                            gradDs += addAlmGradPro(swc, tmu, vioLatAcc, gradViolaLonAds);
                        }
                        else{
                            cost += -0.5 * tmu * tmu / rho;
                        }
                        uneqc_idx++;
                    }
                    {
                        double wpw = jerkWei_;
                        if(sg > 0){
                            wpw = jerkWei_;
                        }
                        else{
                            wpw = std::max(backW_, jerkWei_);
                        }
                        cost += wpw * step * realJerk.squaredNorm();
                        totalCost += wpw * step * realJerk.squaredNorm();
                        gradVel += wpw * step * 2.0 * ddds * realJerk;
                        gradAcc += wpw * step * 2.0 * 3.0 * ds * dds * realJerk;
                        gradJerk += wpw * step * 2.0 * ds * ds * ds * realJerk;
                        gradDs += wpw * step * 2.0 * realJerk.transpose() * (3*acc*dds + 3*ds*ds*jerk);
                        gradDds += wpw * step * 2.0 * realJerk.transpose() * (3*acc*ds);
                        gradDdds += wpw * step * 2.0 * realJerk.transpose() * (vel);
                        gdC2t_1d[i] += wpw * realJerk.squaredNorm() / traj_res;
                        
                        debugEnergy += wpw * step * realJerk.squaredNorm();

                    }

                    gradPos += grad_se2.head(2);
                    gradVel(0) -= grad_se2(2) * vel(1) / vel.squaredNorm();
                    gradVel(1) += grad_se2(2) * vel(0) / vel.squaredNorm();
                    double gradResS = (gradPos.dot(vel) +
                                 gradVel.dot(acc) +
                                 gradAcc.dot(jerk) +
                                 gradJerk.dot(snap));
                    gradS += gradResS;
                    
                    gdC2c_1d.block<6, 1>(i * 6, 0) += beta0_1d * gradS +
                                                    beta1_1d * gradDs +
                                                    beta2_1d * gradDds +
                                                    beta3_1d * gradDdds;
                    gdC2t_1d[i] += (gradS * (ds) +
                                 gradDs * (dds) +
                                 gradDds * (ddds) +
                                 gradDdds * (dddds)) *
                                    alpha;


                    gdC2c_2d.block<6, 2>(i * 6, 0) += beta0_2d * gradPos.transpose() +
                                                    beta1_2d * gradVel.transpose() +
                                                    beta2_2d * gradAcc.transpose() +
                                                    beta3_2d * gradJerk.transpose();

                    if(i > 0){
                        gdC2t_2d.segment(0, i).array() +=  -gradResS;
                    }

                                
                }
                baseS += posOpt.T1(i);
            }
        }        
        void positiveSmoothedL1(const double &x, double &f, double &df)
        {
                const double pe = 1.0e-4;
                const double half = 0.5 * pe;
                const double f3c = 1.0 / (pe * pe);
                const double f4c = -0.5 * f3c / pe;
                const double d2c = 3.0 * f3c;
                const double d3c = 4.0 * f4c;

                if (x < pe)
                {
                    f = (f4c * x + f3c) * x * x * x;
                    df = (d3c * x + d2c) * x * x;
                }
                else
                {
                    f = x - half;
                    df = 1.0;
                }
                return;
        }
        void positiveSmoothedL3(const double &x, double &f, double &df){
            df = x * x;
            f = df *x;
            df *= 3.0;
            return ;
        }
        void updateDualVar(){
            /*update lambda*/
            {
                for(int i = 0; i < eqc_num; i++){
                    lambda[i] += rho * scaling_eqwc_[i] * hx[i];
                }
            }
            /*update mu*/
            {
                for(int i = 0; i < neqc_num; i++){
                    if(mu[i] + rho * scaling_uneqwc_[i] * gx[i] >= 0){
                        mu[i] += rho * scaling_uneqwc_[i] * gx[i];
                    }
                    else{
                        mu[i] = 0;
                    }
                }
            }
            /*update rho*/
            {
                rho = std::min((1 + gamma) * rho, rho_max);
            }
        }
        bool ifConvergence(){
            Eigen::VectorXd maxgx(neqc_num), maxhx(eqc_num);
            for(int i =0 ;i < eqc_num; i++){
                maxhx[i] = hx[i];
            }
            for(int i = 0; i < neqc_num; i++){
                maxgx[i] = std::max(0.0, gx[i]);
            }
            double reshx;
            if(maxhx.size()==0){
                reshx = 0.0;
            }
            else{
                reshx = maxhx.cwiseAbs().maxCoeff();
            }
            int gxidx;
            double resgx = maxgx.cwiseAbs().maxCoeff(&gxidx);
            if (std::max(reshx, resgx) < cons_eps_ )
            {
                return true;
            }
            return false;
        }
        template <typename EIGENVEC>
        EIGENVEC addAlmGradPro(const double &ws, const double &lbd, const double &cons, const EIGENVEC &grad_){
            EIGENVEC output = ws * lbd * grad_ + rho * ws * ws * cons * grad_;
            return output;
        }
        double getAugmentedCost(double h_or_g, double lambda_or_mu)
        {
            return h_or_g * (lambda_or_mu + 0.5*rho*h_or_g);
        }
        // get ρh+λ or ρg+μ, the gradient of `getAugmentedCost`
        double getAugmentedGrad(double h_or_g, double lambda_or_mu)
        {
            return rho * h_or_g + lambda_or_mu;
        }
        std::vector<Eigen::Vector3d> getTraj(double dt){
            UgvTrajectory  optTraj = getOptTraj();
            std::vector<Eigen::Vector3d> poseTraj;
            for(int i = 0; i < trajnum; i++){
                PolyTrajOpt::PolyTrajectory<2> polytraj = posOpts[i].getTraj();
                PolyTrajectory<1> Ttraj = tOpts[i].getTraj();    
                for(double t = 0.0; t < Ttraj.getTotalDuration()-1.0e-3; t+=dt){
                    double s = Ttraj.getSigma(t)[0];
                    double ds = Ttraj.getdSigma(t)[0];
                    Eigen::Vector2d pos= polytraj.getSigma(s);
                    Eigen::Vector2d vel= polytraj.getdSigma(s);
                    Eigen::Vector3d grad_inpainted, pos_tmp;
                    pos_tmp.head(2) = pos;
                    pos_tmp(2) = 0.0;
                    double z = map_itf_->getValueGrad("smooth", pos_tmp, grad_inpainted, se2_grid::InterpolationMethods::INTER_LINEAR);
                    pos_tmp(2) = z;
                    poseTraj.push_back(pos_tmp);
                }   
            }
            return poseTraj;
        }
        double getLength(int i){
            double length = 0;
            PolyTrajOpt::PolyTrajectory<2> polytraj = posOpts[i].getTraj();
            PolyTrajectory<1> Ttraj = tOpts[i].getTraj();    
            for(double t = 0.0; t < Ttraj.getTotalDuration()-1.0e-3; t+=0.01){
                double s = Ttraj.getSigma(t)[0];
                double ds = Ttraj.getdSigma(t)[0];
                Eigen::Vector2d vel= polytraj.getdSigma(s);
                Eigen::Vector2d realV = vel * ds;
                length += 0.01 * realV.norm();
            }   
            return length;
        }
        double getTotalDuration(int i){
            PolyTrajOpt::PolyTrajectory<1> tTraj = tOpts[i].getTraj();
            return tTraj.getTotalDuration();
        }
        double getMaxCur(){
            double maxcur = 0.0;
            for(int i = 0; i < trajnum; i++){
                PolyTrajOpt::PolyTrajectory<2> polytraj = posOpts[i].getTraj();
                PolyTrajectory<1> Ttraj = tOpts[i].getTraj();    
                for(double t = 0.0; t < Ttraj.getTotalDuration()-1.0e-3; t+=0.01){
                    double s = Ttraj.getSigma(t)[0];
                    Eigen::Vector2d vel= polytraj.getdSigma(s);
                    Eigen::Vector2d acc= polytraj.getddSigma(s);
                    double cur = (vel[0]*acc[1] - acc[0]*vel[1]) / (vel.norm() * vel.norm() * vel.norm());
                    if(fabs(cur) > fabs(maxcur)){
                        maxcur = cur;
                    }
                }   
            }
            return maxcur;
        }
        double getMaxCurDot(){
            UgvTrajectory optTraj;
            for(int i = 0; i < trajnum; i++){
                PolyTrajOpt::PolyTrajectory<2> polytraj = posOpts[i].getTraj();
                PolyTrajectory<1> Ttraj = tOpts[i].getTraj();   
                DpTrajectory dpTraj;
                dpTraj.posTraj = polytraj;
                dpTraj.tTraj = Ttraj;
                optTraj.Traj_container.push_back(dpTraj);
                optTraj.etas.push_back(singual_[i]);
            }
            double maxcurdot = 0.0;
            for(double t = 0.0; t < optTraj.getTotalDuration(); t+=0.01){
                double curdot = optTraj.getCurDot(t);        
                if(fabs(curdot) > fabs(maxcurdot)){
                    maxcurdot = curdot;
                }
            }
            return maxcurdot;
        }
        
        double getMaxOmega(){
            UgvTrajectory optTraj;
            for(int i = 0; i < trajnum; i++){
                PolyTrajOpt::PolyTrajectory<2> polytraj = posOpts[i].getTraj();
                PolyTrajectory<1> Ttraj = tOpts[i].getTraj();   
                DpTrajectory dpTraj;
                dpTraj.posTraj = polytraj;
                dpTraj.tTraj = Ttraj;
                optTraj.Traj_container.push_back(dpTraj);
                optTraj.etas.push_back(singual_[i]);
            }
            double maxomega = 0.0;
            for(double t = 0.0; t < optTraj.getTotalDuration(); t+=0.01){
                double omega = optTraj.getOmega(t);        
                if(fabs(omega) > fabs(maxomega)){
                    maxomega = omega;
                }
            }
            return maxomega;
        }
        void getTPMWithGrad(const Eigen::Vector3d& pos, std::vector<double>& values, std::vector<Eigen::Vector3d>& grads)
        {
            double inv_cos_vphix, sin_phix, inv_cos_vphiy, sin_phiy, cos_xi, inv_cos_xi;
            Eigen::Vector3d grad_inv_cos_vphix, grad_sin_phix, grad_inv_cos_vphiy, grad_sin_phiy, grad_cos_xi, grad_inv_cos_xi;
            double risk;
            Eigen::Vector2d zb;
            Eigen::Vector3d grad_zbx;
            Eigen::Vector3d grad_zby;
            Eigen::Vector3d grad_zbz;
            Eigen::Vector3d grad_risk;
            zb(0) = map_itf_->getValueGrad("zbx", pos, grad_zbx, se2_grid::InterpolationMethods::INTER_LINEAR);
            zb(1) = map_itf_->getValueGrad("zby", pos, grad_zby, se2_grid::InterpolationMethods::INTER_LINEAR);
            risk = map_itf_->getValueGrad("risk", pos, grad_risk, se2_grid::InterpolationMethods::INTER_LINEAR);
            double c = sqrt(1.0 - zb(0)*zb(0) - zb(1)*zb(1));
            grad_zbz = -(grad_zbx * zb(0) + grad_zby * zb(1)) / c;
            double inv_c = 1.0 / c;
            double cyaw = cos(pos(2));
            double syaw = sin(pos(2));
            Eigen::Vector2d xyaw(cyaw, syaw);
            Eigen::Vector2d yyaw(-syaw, cyaw);
            double t = xyaw.dot(zb);
            double s = -yyaw.dot(zb);
            double sqrt_1_t2 = sqrt(1.0 - t*t);
            double inv_sqrt_1_t2 = 1.0 / sqrt_1_t2;
            double inv_sqrt_1_t2_3 = inv_sqrt_1_t2 * inv_sqrt_1_t2 * inv_sqrt_1_t2;
            Eigen::Vector3d dt = grad_zbx * cyaw + grad_zby * syaw;
            Eigen::Vector3d ds = grad_zbx * syaw - grad_zby * cyaw;
            dt(2) -= s;
            ds(2) += t;
            inv_cos_vphix = inv_sqrt_1_t2;
            sin_phix = -c * t * inv_sqrt_1_t2;
            inv_cos_vphiy = sqrt_1_t2 * inv_c;
            sin_phiy = s * inv_sqrt_1_t2;
            cos_xi = c;
            inv_cos_xi = inv_c; 
            grad_inv_cos_vphix = t * inv_sqrt_1_t2_3 * dt;
            grad_sin_phix = -(t * inv_sqrt_1_t2 * grad_zbz + inv_sqrt_1_t2_3 * c * dt);
            grad_inv_cos_vphiy = -inv_c * (t * inv_sqrt_1_t2 * dt + sqrt_1_t2 * inv_c * grad_zbz);
            grad_sin_phiy = inv_sqrt_1_t2 * ds + t * inv_sqrt_1_t2_3 * s * dt;
            grad_cos_xi = grad_zbz;
            grad_inv_cos_xi = -inv_cos_xi * inv_cos_xi * grad_zbz;
            values.clear();
            grads.clear();
            values.push_back(inv_cos_vphix);
            values.push_back(sin_phix);
            values.push_back(inv_cos_vphiy);
            values.push_back(sin_phiy);
            values.push_back(cos_xi);
            values.push_back(inv_cos_xi);
            values.push_back(risk);
            grads.push_back(grad_inv_cos_vphix);
            grads.push_back(grad_sin_phix);
            grads.push_back(grad_inv_cos_vphiy);
            grads.push_back(grad_sin_phiy);
            grads.push_back(grad_cos_xi);
            grads.push_back(grad_inv_cos_xi);
            grads.push_back(grad_risk);
            return;
        }

        inline void getTPM(const Eigen::Vector3d& pos, std::vector<double>& values)
            {
                double inv_cos_vphix, sin_phix, inv_cos_vphiy, sin_phiy, cos_xi, inv_cos_xi;

                double risk;
                Eigen::Vector2d zb;

                zb(0) = map_itf_->getValue("zbx", pos, se2_grid::InterpolationMethods::INTER_LINEAR);
                zb(1) = map_itf_->getValue("zby", pos, se2_grid::InterpolationMethods::INTER_LINEAR);
                risk = map_itf_->getValue("risk", pos, se2_grid::InterpolationMethods::INTER_LINEAR);
                double c = sqrt(1.0 - zb(0)*zb(0) - zb(1)*zb(1));
                double inv_c = 1.0 / c;
                double cyaw = cos(pos(2));
                double syaw = sin(pos(2));
                Eigen::Vector2d xyaw(cyaw, syaw);
                Eigen::Vector2d yyaw(-syaw, cyaw);
                double t = xyaw.dot(zb);
                double s = -yyaw.dot(zb);
                double sqrt_1_t2 = sqrt(1.0 - t*t);
                double inv_sqrt_1_t2 = 1.0 / sqrt_1_t2;

                inv_cos_vphix = inv_sqrt_1_t2;
                sin_phix = -c * t * inv_sqrt_1_t2;
                inv_cos_vphiy = sqrt_1_t2 * inv_c;
                sin_phiy = s * inv_sqrt_1_t2;
                cos_xi = c;
                inv_cos_xi = inv_c; 
                values.clear();
                values.push_back(inv_cos_vphix);
                values.push_back(sin_phix);
                values.push_back(inv_cos_vphiy);
                values.push_back(sin_phiy);
                values.push_back(cos_xi);
                values.push_back(inv_cos_xi);
                values.push_back(risk);
                return;
            }
        UgvTrajectory getOptTraj(){
            UgvTrajectory optTraj;
            for(int i = 0; i < trajnum; i++){
                PolyTrajOpt::PolyTrajectory<2> polytraj = posOpts[i].getTraj();
                PolyTrajectory<1> Ttraj = tOpts[i].getTraj();   
                DpTrajectory dpTraj;
                dpTraj.posTraj = polytraj;
                dpTraj.tTraj = Ttraj;
                optTraj.Traj_container.push_back(dpTraj);
                optTraj.etas.push_back(singual_[i]);
            }
            return optTraj;
        }
        void debugVis(){
            // Visualization disabled - complex template issues in ROS2 migration
            // In ROS1, this would publish debug visualization to RViz
            // For now, debugging is done through console output only
            if(isstart&&isdebug_){
                std::cout << "TrajOpt: initial Guess visualization (disabled in ROS2)" << std::endl;
            }
            // vis_tool_->visualize_path(getTraj(0.01), "/visualization/refinedTraj_"+name_);
            // vis_tool_->visualize_pointcloud(...);
            // vis_tool_->visualize_se2traj(...);
            if(isstart&&isdebug_){
                std::this_thread::sleep_for(std::chrono::milliseconds(7500));
                isstart = false; 
            }
        }
};
} 
#endif 
