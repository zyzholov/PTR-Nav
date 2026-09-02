#ifndef GRIDMAP_H#ifndef GRIDMAP_H#ifndef GRIDMAP_H

#define GRIDMAP_H

#define GRIDMAP_H#define GRIDMAP_H

#include <iostream>

#include <rclcpp/rclcpp.hpp>

#include <map>

#include <tools/config.hpp>#include <iostream>#include <iostream>

#include <decomp_basis/data_type.h>

#include <sensor_msgs/msg/point_cloud2.hpp>#include <rclcpp/rclcpp.hpp>#include <rclcpp/rclcpp.hpp>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>#include <map>#include <map>

#include <pcl/point_types.h>

#include <memory>#include <tools/config.hpp>#include <tools/config.hpp>

#include <functional>

#include <decomp_basis/data_type.h>#include <decomp_basis/data_type.h>

namespace map_util {

    using Tmap = std::vector<signed char>;#include <sensor_msgs/msg/point_cloud2.hpp>#include <sensor_msgs/msg/point_cloud2.hpp>

    

    template <int Dim> #include <pcl_conversions/pcl_conversions.h>#include <pcl_conversions/pcl_conversions.h>

    class MapUtil {

    public:#include <pcl/point_cloud.h>#include <pcl/point_cloud.h>

      MapUtil() {}

      Tmap getMap() { return map_; }#include <pcl/point_types.h>#include <pcl/point_types.h>

      bool has_map_(){return has_map;}

      double getRes() { return res_; }#include <memory>#include <memory>

      Veci<Dim> getDim() { return dim_; }

      Vecf<Dim> getOrigin() { return origin_d_; }namespace map_util{

      Vecf<Dim> getMapSize() { return map_size; }

      namespace map_util{    using Tmap = std::vector<signed char>;

      void setParam(ConfigPtr config_, std::shared_ptr<rclcpp::Node> node){

        res_ = config_->mapRes;    using Tmap = std::vector<signed char>;    template <int Dim> class MapUtil {

        int buffer_size;

        expand_size = config_->expandSize;    template <int Dim> class MapUtil {    public:

        priv_config_ = config_;

        node_ = node;    public:      ///Simple constructor

        

        if(Dim == 3){      ///Simple constructor      MapUtil() {}

            map_size(0) = config_->mapX;

            map_size(1) = config_->mapY;      MapUtil() {}      ///Get map data

            map_size(2) = config_->mapZ;

            origin_d_[0] = -map_size(0)/2;      ///Get map data      Tmap getMap() { return map_; }

            origin_d_[1] = -map_size(1)/2;

            origin_d_[2] = 0;      Tmap getMap() { return map_; }      //

            dim_(0) = map_size(0)/res_;

            dim_(1) = map_size(1)/res_;      //      bool has_map_(){return has_map;}

            dim_(2) = map_size(2)/res_;

            buffer_size = dim_(0)*dim_(1)*dim_(2);      bool has_map_(){return has_map;}      ///Get resolution

        }

        else if(Dim==2){      ///Get resolution      double getRes() { return res_; }

            map_size(0) = config_->mapX + 1.0e-4;

            map_size(1) = config_->mapY + 1.0e-4;      double getRes() { return res_; }      ///Get dimensions

            origin_d_[0] = -map_size(0)/2;

            origin_d_[1] = -map_size(1)/2;      ///Get dimensions      Veci<Dim> getDim() { return dim_; }

            dim_(0) = map_size(0)/res_;

            dim_(1) = map_size(1)/res_;      Veci<Dim> getDim() { return dim_; }      ///Get origin

            buffer_size = dim_(0)*dim_(1);

      ///Get origin      Vecf<Dim> getOrigin() { return origin_d_; }

            distance_buffer_ = std::vector<double>(buffer_size, 10000);

            distance_buffer_neg_ = std::vector<double>(buffer_size, 10000);      Vecf<Dim> getOrigin() { return origin_d_; }      Vecf<Dim> getMapSize() { return map_size; }

            distance_buffer_all_ = std::vector<double>(buffer_size, 10000);

            tmp_buffer1_ = std::vector<double>(buffer_size, 0);      Vecf<Dim> getMapSize() { return map_size; }      void setParam(ConfigPtr config_, std::shared_ptr<rclcpp::Node> node){

            occupancy_buffer_neg = std::vector<char>(buffer_size, 0);

            esdf_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/global_esdf", 10);      void setParam(ConfigPtr config_, std::shared_ptr<rclcpp::Node> node){        res_ = config_->mapRes;

            pcl_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/global_pcl", 10);

        }        res_ = config_->mapRes;        int buffer_size;

        else{

            RCLCPP_ERROR(node_->get_logger(), "grid map dimensions must be 2 or 3!");        int buffer_size;        expand_size = config_->expandSize;

        }

        map_.resize(buffer_size,val_free);        expand_size = config_->expandSize;        priv_config_ = config_;

        auto callback = std::bind(&MapUtil::MapBuild, this, std::placeholders::_1);

        point_cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(        priv_config_ = config_;        node_ = node;

            "/global_map", 10, callback);

      }        node_ = node;        if(Dim == 3){

      

      void setEnv(const sensor_msgs::msg::PointCloud2 & pointcloud_map){        if(Dim == 3){            map_size(0) = config_->mapX;

        int buffer_size = dim_(0)*dim_(1);

        distance_buffer_ = std::vector<double>(buffer_size, 10000);            map_size(0) = config_->mapX;            map_size(1) = config_->mapY;

        distance_buffer_neg_ = std::vector<double>(buffer_size, 10000);

        distance_buffer_all_ = std::vector<double>(buffer_size, 10000);            map_size(1) = config_->mapY;            map_size(2) = config_->mapZ;

        tmp_buffer1_ = std::vector<double>(buffer_size, 0);

        occupancy_buffer_neg = std::vector<char>(buffer_size, 0);            map_size(2) = config_->mapZ;            origin_d_[0] = -map_size(0)/2;

        std::fill(map_.begin(), map_.end(), val_free);

            origin_d_[0] = -map_size(0)/2;            origin_d_[1] = -map_size(1)/2;

        pcl::PointCloud<pcl::PointXYZ> cloud;

        pcl::fromROSMsg(pointcloud_map, cloud);            origin_d_[1] = -map_size(1)/2;            origin_d_[2] = 0;

        if( (int)cloud.points.size() == 0 ) return;

                    origin_d_[2] = 0;            dim_(0) = map_size(0)/res_;

        for (int idx = 0; idx < (int)cloud.points.size(); idx++){

          setObs(cloud.points[idx]);            dim_(0) = map_size(0)/res_;            dim_(1) = map_size(1)/res_;

        }

        has_map = true;            dim_(1) = map_size(1)/res_;            dim_(2) = map_size(2)/res_;

        updateESDF2d();

        publishESDF();            dim_(2) = map_size(2)/res_;            buffer_size = dim_(0)*dim_(1)*dim_(2);

        publishPCL();

      }            buffer_size = dim_(0)*dim_(1)*dim_(2);        }

      

      void MapBuild(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_map){        }        else if(Dim==2){

        if(has_map) {

          publishESDF();        else if(Dim==2){            map_size(0) = config_->mapX + 1.0e-4;

          return;

        }            map_size(0) = config_->mapX + 1.0e-4;            map_size(1) = config_->mapY + 1.0e-4;

        pcl::PointCloud<pcl::PointXYZ> cloud;

        pcl::fromROSMsg(*pointcloud_map, cloud);            map_size(1) = config_->mapY + 1.0e-4;            origin_d_[0] = -map_size(0)/2;

        if( (int)cloud.points.size() == 0 ) return;

                    origin_d_[0] = -map_size(0)/2;            origin_d_[1] = -map_size(1)/2;

        for (int idx = 0; idx < (int)cloud.points.size(); idx++){

          setObs(cloud.points[idx]);            origin_d_[1] = -map_size(1)/2;            dim_(0) = map_size(0)/res_;

        }

        has_map = true;            dim_(0) = map_size(0)/res_;            dim_(1) = map_size(1)/res_;

        updateESDF2d();

        publishESDF();            dim_(1) = map_size(1)/res_;            buffer_size = dim_(0)*dim_(1);

      }

                  buffer_size = dim_(0)*dim_(1);

      void setObs(const pcl::PointXYZ& pt){

        if(Dim==3){            /*ESDF*/

            Vec3f pt3(pt.x, pt.y, pt.z);

            Veci<Dim> index3i;            /*ESDF*/            distance_buffer_ = std::vector<double>(buffer_size, 10000);

            for(int i = 0; i < Dim; i++)

              index3i(i) = std::round((pt3(i) - origin_d_(i)) / res_ - 0.5);            distance_buffer_ = std::vector<double>(buffer_size, 10000);            distance_buffer_neg_ = std::vector<double>(buffer_size, 10000);

            if(isOutside(index3i))

                return;            distance_buffer_neg_ = std::vector<double>(buffer_size, 10000);            distance_buffer_all_ = std::vector<double>(buffer_size, 10000);

            for (int i=-expand_size;i<=expand_size;i++)

            for (int j=-expand_size;j<=expand_size;j++)            distance_buffer_all_ = std::vector<double>(buffer_size, 10000);            tmp_buffer1_ = std::vector<double>(buffer_size, 0);

            for (int k=-expand_size;k<=expand_size;k++){

                Veci<Dim> temp_index;            tmp_buffer1_ = std::vector<double>(buffer_size, 0);            occupancy_buffer_neg = std::vector<char>(buffer_size, 0);

                temp_index(0) = index3i(0)+i;

                temp_index(1) = index3i(1)+j;            occupancy_buffer_neg = std::vector<char>(buffer_size, 0);            esdf_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/global_esdf", 10);

                temp_index(2) = index3i(2)+k;

                if(isOutside(temp_index)) continue;            esdf_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/global_esdf", 10);            pcl_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/global_pcl", 10);

                map_[getIndex(temp_index)] = val_occ;

            }            pcl_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/global_pcl", 10);

        }

        else{        }

            Vec2f pt2(pt.x, pt.y);

            Veci<Dim> index2i;        }        else{

            for(int i = 0; i < Dim; i++)

              index2i(i) = std::round((pt2(i) - origin_d_(i)) / res_ - 0.5);        else{            RCLCPP_ERROR(node_->get_logger(), "grid map dimensions must be 2 or 3!");

            if(isOutside(index2i))

                return;            RCLCPP_ERROR(node_->get_logger(), "grid map dimensions must be 2 or 3!");        }

            for (int i=-expand_size;i<=expand_size;i++)

            for (int j=-expand_size;j<=expand_size;j++){        }        map_.resize(buffer_size,val_free);

                Veci<Dim> temp_index;

                temp_index(0) = index2i(0)+i;        map_.resize(buffer_size,val_free);        auto callback = std::bind(&MapUtil::MapBuild, this, std::placeholders::_1);

                temp_index(1) = index2i(1)+j;

                if(isOutside(temp_index)) continue;        auto callback = std::bind(&MapUtil::MapBuild, this, std::placeholders::_1);        point_cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(

                map_[getIndex(temp_index)] = val_occ;

            }        point_cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(            "/global_map", 10, callback);

        }

      }            "/global_map", 10, callback);      }

      

      int getIndex(const Veci<Dim>& pn) {      }      void setEnv(const sensor_msgs::msg::PointCloud2 & pointcloud_map){

          return Dim == 2 ? pn(0) + dim_(0) * pn(1) :

                            pn(0) + dim_(0) * pn(1) + dim_(0) * dim_(1) * pn(2);      void setEnv(const sensor_msgs::msg::PointCloud2 & pointcloud_map){        //only used in 2D environment

      }

              //only used in 2D environment        int buffer_size;

      bool isOutsideXYZ(const Veci<Dim> &n, int i) { return n(i) < 0 || n(i) >= dim_(i); }

      bool isFree(int idx) { return map_[idx] == val_free; }        int buffer_size;        buffer_size = dim_(0)*dim_(1);

      bool isUnknown(int idx) { return map_[idx] == val_unknown; }

      bool isOccupied(int idx) { return map_[idx] > val_free; }        buffer_size = dim_(0)*dim_(1);        distance_buffer_ = std::vector<double>(buffer_size, 10000);

      

      bool isOutside(const Veci<Dim> &pn) {        distance_buffer_ = std::vector<double>(buffer_size, 10000);        distance_buffer_neg_ = std::vector<double>(buffer_size, 10000);

        for(int i = 0; i < Dim; i++)

          if (pn(i) < 0 || pn(i) >= dim_(i))        distance_buffer_neg_ = std::vector<double>(buffer_size, 10000);        distance_buffer_all_ = std::vector<double>(buffer_size, 10000);

            return true;

        return false;        distance_buffer_all_ = std::vector<double>(buffer_size, 10000);        tmp_buffer1_ = std::vector<double>(buffer_size, 0);

      }

              tmp_buffer1_ = std::vector<double>(buffer_size, 0);        occupancy_buffer_neg = std::vector<char>(buffer_size, 0);

      bool isFree(const Veci<Dim> &pn) {

        if (isOutside(pn))        occupancy_buffer_neg = std::vector<char>(buffer_size, 0);        std::fill(map_.begin(), map_.end(), val_free);

          return false;

        else        std::fill(map_.begin(), map_.end(), val_free);    

          return isFree(getIndex(pn));

      }

      

      bool isOccupied(const Veci<Dim> &pn) {        pcl::PointCloud<pcl::PointXYZ> cloud;

        if (isOutside(pn))

          return true;        pcl::fromROSMsg(pointcloud_map, cloud);        pcl::PointCloud<pcl::PointXYZ> cloud;

        else

          return isOccupied(getIndex(pn));        if( (int)cloud.points.size() == 0 ) return;        pcl::fromROSMsg(pointcloud_map, cloud);

      }

              pcl::PointXYZ pt;        if( (int)cloud.points.size() == 0 ) return;

      bool isOccupied(const Vecf<Dim> &pt) {

        Veci<Dim> pn = floatToInt(pt);        for (int idx = 0; idx < (int)cloud.points.size(); idx++)      void MapBuild(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_map){

        if (isOutside(pn))

          return true;        {        if(has_map) {

        else

          return isOccupied(getIndex(pn));          pt = cloud.points[idx];          publishESDF();

      }

                setObs(pt);          return;}

      bool isUnknown(const Veci<Dim> &pn) {

        if (isOutside(pn))        }        pcl::PointCloud<pcl::PointXYZ> cloud;

          return false;

        return map_[getIndex(pn)] == val_unknown;        has_map = true;        pcl::fromROSMsg(*pointcloud_map, cloud);

      }

        // std::cout << "!map build\n";        if( (int)cloud.points.size() == 0 ) return;

      void setMap(const Vecf<Dim>& ori, const Veci<Dim>& dim, const Tmap &map, double res) {

        map_ = map;        updateESDF2d();        pcl::PointXYZ pt;

        dim_ = dim;

        origin_d_ = ori;        publishESDF();        // cloud.points.clear();//data generaion

        res_ = res;

      }        publishPCL();



      void info() {      }        for (int idx = 0; idx < (int)cloud.points.size(); idx++)

        Vecf<Dim> range = dim_.template cast<double>() * res_;

        std::cout << "MapUtil Info ========================== " << std::endl;      void MapBuild(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_map){        {

        std::cout << "   res: [" << res_ << "]" << std::endl;

        std::cout << "   origin: [" << origin_d_.transpose() << "]" << std::endl;        if(has_map) {          pt = cloud.points[idx];

        std::cout << "   range: [" << range.transpose() << "]" << std::endl;

        std::cout << "   dim: [" << dim_.transpose() << "]" << std::endl;          publishESDF();          setObs(pt);

      }

          return;}        }

      Veci<Dim> floatToInt(const Vecf<Dim> &pt) {

        Veci<Dim> pn;        pcl::PointCloud<pcl::PointXYZ> cloud;        has_map = true;

        for(int i = 0; i < Dim; i++)

          pn(i) = floor((pt(i) - origin_d_(i)) / res_);        pcl::fromROSMsg(*pointcloud_map, cloud);        updateESDF2d();

        return pn;

      }        if( (int)cloud.points.size() == 0 ) return;        publishESDF();

      

      Vecf<Dim> intToFloat(const Veci<Dim> &pn) {        pcl::PointXYZ pt;      }

        return (pn.template cast<double>() + Vecf<Dim>::Constant(0.5)) * res_ + origin_d_;

      }        // cloud.points.clear();//data generaion      void setObs(pcl::PointXYZ pt){



      vec_Veci<Dim> rayTrace(const Vecf<Dim> &pt1, const Vecf<Dim> &pt2) {        if(Dim==3){

        Vecf<Dim> diff = pt2 - pt1;

        double k = 0.8;        for (int idx = 0; idx < (int)cloud.points.size(); idx++)            double coord_x = pt.x;

        int max_diff = (diff / res_).template lpNorm<Eigen::Infinity>() / k;

        double s = 1.0 / max_diff;        {            double coord_y = pt.y;

        Vecf<Dim> step = diff * s;

          pt = cloud.points[idx];            double coord_z = pt.z;

        vec_Veci<Dim> pns;

        Veci<Dim> prev_pn = Veci<Dim>::Constant(-1);          setObs(pt);            Vec3f pt = Vec3f(coord_x,coord_y,coord_z);

        for (int n = 1; n < max_diff; n++) {

          Vecf<Dim> pt = pt1 + step * n;        }            Veci<Dim> index3i;

          Veci<Dim> new_pn = floatToInt(pt);

          if (isOutside(new_pn))        has_map = true;            for(int i = 0; i < Dim; i++)

            break;

          if (new_pn != prev_pn)        updateESDF2d();              index3i(i) = std::round((pt(i) - origin_d_(i)) / res_ - 0.5);

            pns.push_back(new_pn);

          prev_pn = new_pn;        publishESDF();            if(isOutside(index3i))

        }

        return pns;      }                return;

      }

      void setObs(pcl::PointXYZ pt){            for (int i=-expand_size;i<=expand_size;i++)

      bool isBlocked(const Vecf<Dim>& p1, const Vecf<Dim>& p2, int8_t val = 100) {

        vec_Veci<Dim> pns = rayTrace(p1, p2);        if(Dim==3){            for (int j=-expand_size;j<=expand_size;j++)

        for (const auto &pn : pns) {

          if (map_[getIndex(pn)] >= val)            double coord_x = pt.x;            for (int k=-expand_size;k<=expand_size;k++)

            return true;

        }            double coord_y = pt.y;                {

        return false;

      }            double coord_z = pt.z;                    Veci<Dim> temp_index;



      vec_Vecf<Dim> getCloud() {            Vec3f pt3 = Vec3f(coord_x,coord_y,coord_z);                    temp_index(0) = index3i(0)+i;

        vec_Vecf<Dim> cloud;

        Veci<Dim> n;            Veci<Dim> index3i;                    temp_index(1) = index3i(1)+j;

        if(Dim == 3) {

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {            for(int i = 0; i < Dim; i++)                    temp_index(2) = index3i(2)+k;

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

              for (n(2) = 0; n(2) < dim_(2); n(2)++) {              index3i(i) = std::round((pt3(i) - origin_d_(i)) / res_ - 0.5);                    if(isOutside(temp_index)) continue;

                if (isOccupied(getIndex(n)))

                  cloud.push_back(intToFloat(n));            if(isOutside(index3i))                    map_[getIndex(temp_index)] = val_occ;

              }

            }                return;                }

          }

        }            for (int i=-expand_size;i<=expand_size;i++)        }

        else if (Dim == 2) {

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {            for (int j=-expand_size;j<=expand_size;j++)        else{

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

              if (isOccupied(getIndex(n)))            for (int k=-expand_size;k<=expand_size;k++)            double coord_x = pt.x;

                cloud.push_back(intToFloat(n));

            }                {            double coord_y = pt.y;

          }

        }                    Veci<Dim> temp_index;            Vec2f pt = Vec2f(coord_x,coord_y);

        return cloud;

      }                    temp_index(0) = index3i(0)+i;            Veci<Dim> index2i;

      

      void updateESDF2d() {                    temp_index(1) = index3i(1)+j;            for(int i = 0; i < Dim; i++)

        if(Dim != 2){

          RCLCPP_ERROR(node_->get_logger(), "Dimension mismatch!");                    temp_index(2) = index3i(2)+k;              index2i(i) = std::round((pt(i) - origin_d_(i)) / res_ - 0.5);

        }

        Eigen::Vector2i min_esdf = Eigen::Vector2i(0,0);                    if(isOutside(temp_index)) continue;            if(isOutside(index2i))

        Eigen::Vector2i max_esdf = Eigen::Vector2i(dim_(0) - 1, dim_(1) - 1);

                    map_[getIndex(temp_index)] = val_occ;                return;

        for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {

          fillESDF(                }            for (int i=-expand_size;i<=expand_size;i++)

              [&](int y) {

                return isOccupied(Vec2i(x, y))?        }            for (int j=-expand_size;j<=expand_size;j++)

                    0 :

                    std::numeric_limits<double>::max();        else{                {

              },

              [&](int y, double val) {tmp_buffer1_[getIndex(Vec2i(x, y))] = val; }, min_esdf[1],            double coord_x = pt.x;                    Veci<Dim> temp_index;

              max_esdf[1], 1);

        }            double coord_y = pt.y;                    temp_index(0) = index2i(0)+i;



        for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {            Vec2f pt2 = Vec2f(coord_x,coord_y);                    temp_index(1) = index2i(1)+j;

          fillESDF([&](int x) { return tmp_buffer1_[getIndex(Vec2i(x, y))]; },

                    [&](int x, double val) {            Veci<Dim> index2i;                    if(isOutside(temp_index)) continue;

                      distance_buffer_[getIndex(Vec2i(x, y))] = res_ * std::sqrt(val);

                    },            for(int i = 0; i < Dim; i++)                    map_[getIndex(temp_index)] = val_occ;

                    min_esdf[0], max_esdf[0], 0);

        }              index2i(i) = std::round((pt2(i) - origin_d_(i)) / res_ - 0.5);                }



        for (int x = min_esdf(0); x <= max_esdf(0); ++x)            if(isOutside(index2i))

          for (int y = min_esdf(1); y <= max_esdf(1); ++y) {

            int idx = getIndex(Vec2i(x, y));                return;        }

            if (isFree(idx)) {

              occupancy_buffer_neg[idx] = 1;            for (int i=-expand_size;i<=expand_size;i++)        return;

            } else if (isOccupied(idx)) {

              occupancy_buffer_neg[idx] = 0;            for (int j=-expand_size;j<=expand_size;j++)

            } else {

              RCLCPP_ERROR(node_->get_logger(), "Unexpected map state!");                {      }

            }

          }                    Veci<Dim> temp_index;      ///Get index of a cell



        for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {                    temp_index(0) = index2i(0)+i;      int getIndex(const Veci<Dim>& pn) {

          fillESDF(

              [&](int y) {                    temp_index(1) = index2i(1)+j;          return Dim == 2 ? pn(0) + dim_(0) * pn(1) :

                return occupancy_buffer_neg[getIndex(Vec2i(x, y))] == 1 ?

                    0 :                    if(isOutside(temp_index)) continue;                            pn(0) + dim_(0) * pn(1) + dim_(0) * dim_(1) * pn(2);

                    std::numeric_limits<double>::max();

              },                    map_[getIndex(temp_index)] = val_occ;      }

              [&](int y, double val) { tmp_buffer1_[getIndex(Vec2i(x, y))] = val; }, min_esdf[1],

              max_esdf[1], 1);                }      ///Check if the given cell is outside of the map in i-the dimension

        }

      bool isOutsideXYZ(const Veci<Dim> &n, int i) { return n(i) < 0 || n(i) >= dim_(i); }

        for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {

          fillESDF([&](int x) { return tmp_buffer1_[getIndex(Vec2i(x, y))]; },        }      ///Check if the cell is free by index

                    [&](int x, double val) {

                      distance_buffer_neg_[getIndex(Vec2i(x, y))] = res_ * std::sqrt(val);        return;      bool isFree(int idx) { return map_[idx] == val_free; }

                    },

                    min_esdf[0], max_esdf[0], 0);      ///Check if the cell is unknown by index

        }

      }      bool isUnknown(int idx) { return map_[idx] == val_unknown; }

        for (int x = min_esdf(0); x <= max_esdf(0); ++x)

          for (int y = min_esdf(1); y <= max_esdf(1); ++y) {      ///Get index of a cell      ///Check if the cell is occupied by index

            int idx = getIndex(Vec2i(x, y));

            distance_buffer_all_[idx] = distance_buffer_[idx];      int getIndex(const Veci<Dim>& pn) {      bool isOccupied(int idx) { return map_[idx] > val_free; }



            if (distance_buffer_neg_[idx] > 0.0){          return Dim == 2 ? pn(0) + dim_(0) * pn(1) :      //free 0 occ 100 unknow -1

              distance_buffer_all_[idx] += (-distance_buffer_neg_[idx] + res_);

            }                            pn(0) + dim_(0) * pn(1) + dim_(0) * dim_(1) * pn(2);      ///Check if the cell is outside by coordinate

          }

      }      }      bool isOutside(const Veci<Dim> &pn) {

      

      template <typename F_get_val, typename F_set_val>      ///Check if the given cell is outside of the map in i-the dimension        for(int i = 0; i < Dim; i++)

      void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int d) {

        int v[dim_(d)];      bool isOutsideXYZ(const Veci<Dim> &n, int i) { return n(i) < 0 || n(i) >= dim_(i); }          if (pn(i) < 0 || pn(i) >= dim_(i))

        double z[dim_(d) + 1];

      ///Check if the cell is free by index            return true;

        int k = start;

        v[start] = start;      bool isFree(int idx) { return map_[idx] == val_free; }        return false;

        z[start] = -std::numeric_limits<double>::max();

        z[start + 1] = std::numeric_limits<double>::max();      ///Check if the cell is unknown by index      }



        for (int q = start + 1; q <= end; q++) {      bool isUnknown(int idx) { return map_[idx] == val_unknown; }      ///Check if the given cell is free by coordinate

          k++;

          double s;      ///Check if the cell is occupied by index      bool isFree(const Veci<Dim> &pn) {



          do {      bool isOccupied(int idx) { return map_[idx] > val_free; }        if (isOutside(pn))

            k--;

            s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);      //free 0 occ 100 unknow -1          return false;

          } while (s <= z[k]);

      ///Check if the cell is outside by coordinate        else

          k++;

      bool isOutside(const Veci<Dim> &pn) {          return isFree(getIndex(pn));

          v[k] = q;

          z[k] = s;        for(int i = 0; i < Dim; i++)      }

          z[k + 1] = std::numeric_limits<double>::max();

        }          if (pn(i) < 0 || pn(i) >= dim_(i))      ///Check if the given cell is occupied by coordinate



        k = start;            return true;      bool isOccupied(const Veci<Dim> &pn) {



        for (int q = start; q <= end; q++) {        return false;        if (isOutside(pn))

          while (z[k + 1] < q) k++;

          double val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);      }          return true;

          f_set_val(q, val);

        }      ///Check if the given cell is free by coordinate        else

      }

            bool isFree(const Veci<Dim> &pn) {          return isOccupied(getIndex(pn));

      inline double getDistance(const Eigen::Vector2d& pos) {

        Eigen::Vector2i id = floatToInt(pos);        if (isOutside(pn))      }

        boundIndex(id);

        return distance_buffer_all_[getIndex(id)];          return false;      bool isOccupied(const Vecf<Dim> &pt) {

      }

              else        Veci<Dim> pn = floatToInt(pt);

      inline double getDistance(const Eigen::Vector2i& id) {

        Eigen::Vector2i id1 = id;          return isFree(getIndex(pn));        if (isOutside(pn))

        boundIndex(id1);

        return distance_buffer_all_[getIndex(id1)];      }          return true;

      }

            ///Check if the given cell is occupied by coordinate        else

      inline void boundIndex(Eigen::Vector2i& id) {

        id(0) = std::max(std::min(id(0), dim_(0) - 1), 0);      bool isOccupied(const Veci<Dim> &pn) {          return isOccupied(getIndex(pn));

        id(1) = std::max(std::min(id(1), dim_(1) - 1), 0);

      }        if (isOutside(pn))      }

      

      void publishPCL(){          return true;      

        vec_Vecf<Dim> pcs = getCloud();

        pcl::PointCloud<pcl::PointXYZI> cloud;        else      

        pcl::PointXYZI pt;

        for(const auto& pc : pcs){          return isOccupied(getIndex(pn));      ///Check if the given cell is unknown by coordinate

          pt.x = pc[0];

          pt.y = pc[1];      }      bool isUnknown(const Veci<Dim> &pn) {

          pt.z = 0.0;

          cloud.push_back(pt);      bool isOccupied(const Vecf<Dim> &pt) {        if (isOutside(pn))

        }

        cloud.width = cloud.points.size();        Veci<Dim> pn = floatToInt(pt);          return false;

        cloud.height = 1;

        cloud.is_dense = true;        if (isOutside(pn))        return map_[getIndex(pn)] == val_unknown;

        cloud.header.frame_id = "world";

        sensor_msgs::msg::PointCloud2 cloud_msg;          return true;      }

        pcl::toMsg(cloud, cloud_msg);

        pcl_pub_->publish(cloud_msg);        else

      }

                return isOccupied(getIndex(pn));      /**

      void publishESDF() {

        pcl::PointCloud<pcl::PointXYZI> cloud;      }       * @brief Set map

        pcl::PointXYZI pt;

        const double min_dist = -4.0;             *

        const double max_dist = 4.0;

        Eigen::Vector2i min_cut = Eigen::Vector2i(0,0);             * @param ori origin position

        Eigen::Vector2i max_cut = Eigen::Vector2i(dim_(0)-1,dim_(1)-1);

      ///Check if the given cell is unknown by coordinate       * @param dim number of cells in each dimension

        for (int x = min_cut(0); x <= max_cut(0); ++x)

          for (int y = min_cut(1); y <= max_cut(1); ++y) {      bool isUnknown(const Veci<Dim> &pn) {       * @param map array of cell values

            Vec2f pos = intToFloat(Vec2i(x, y));

            double dist = getDistance(pos);        if (isOutside(pn))       * @param res map resolution

            pt.x = pos(0);

            pt.y = pos(1);          return false;       */

            pt.z = dist;

            pt.intensity = (dist - min_dist) / (max_dist - min_dist);        return map_[getIndex(pn)] == val_unknown;      void setMap(const Vecf<Dim>& ori, const Veci<Dim>& dim, const Tmap &map, double res) {

            cloud.push_back(pt);

          }      }        map_ = map;

        cloud.width = cloud.points.size();

        cloud.height = 1;        dim_ = dim;

        cloud.is_dense = true;

        cloud.header.frame_id = "world";      /**        origin_d_ = ori;

        sensor_msgs::msg::PointCloud2 cloud_msg;

        pcl::toMsg(cloud, cloud_msg);       * @brief Set map        res_ = res;

        esdf_pub_->publish(cloud_msg);

      }       *      }

      

      inline double getDistGrad(Eigen::Vector2d po, Eigen::Vector2d& grad) {       * @param ori origin position

        Eigen::Vector2d pos(po(0),po(1));

        if (isOutside(floatToInt(pos))) {       * @param dim number of cells in each dimension      ///Print basic information about the util

          grad.setZero();

          return 0;       * @param map array of cell values      void info() {

        }

       * @param res map resolution        Vecf<Dim> range = dim_.template cast<double>() * res_;

        Eigen::Vector2d pos_m = pos - 0.5 * res_ * Eigen::Vector2d::Ones();

       */        std::cout << "MapUtil Info ========================== " << std::endl;

        Eigen::Vector2i idx = floatToInt(pos_m);

        Eigen::Vector2d idx_pos = intToFloat(idx);      void setMap(const Vecf<Dim>& ori, const Veci<Dim>& dim, const Tmap &map, double res) {        std::cout << "   res: [" << res_ << "]" << std::endl;

        Eigen::Vector2d diff = (pos - idx_pos) / res_;

        map_ = map;        std::cout << "   origin: [" << origin_d_.transpose() << "]" << std::endl;

        double values[2][2];

        for (int x = 0; x < 2; x++) {        dim_ = dim;        std::cout << "   range: [" << range.transpose() << "]" << std::endl;

          for (int y = 0; y < 2; y++) {

              Eigen::Vector2i current_idx = idx + Eigen::Vector2i(x, y);        origin_d_ = ori;        std::cout << "   dim: [" << dim_.transpose() << "]" << std::endl;

              values[x][y] = getDistance(current_idx);

          }           res_ = res;      };

        }

        double v00 = (1-diff(0)) * values[0][0] + diff(0) * values[1][0];      }

        double v10 = (1-diff(0)) * values[0][1] + diff(0) * values[1][1];

        double v0  = (1-diff(1)) * v00 + diff(1) * v10;      ///Float position to discrete cell coordinate

        grad[1] = (v10 - v00) / res_;

        grad[0] = ((1-diff(1)) * (values[1][0]-values[0][0]) + diff(1) * (values[1][1]-values[0][1])) / res_;      ///Print basic information about the util      Veci<Dim> floatToInt(const Vecf<Dim> &pt) {



        return v0;      void info() {        Veci<Dim> pn;

      }

              Vecf<Dim> range = dim_.template cast<double>() * res_;        for(int i = 0; i < Dim; i++)

      vec_Vecf<Dim> getFreeCloud() {

        vec_Vecf<Dim> cloud;        std::cout << "MapUtil Info ========================== " << std::endl;          pn(i) = floor((pt(i) - origin_d_(i)) / res_);

        Veci<Dim> n;

        if(Dim == 3) {        std::cout << "   res: [" << res_ << "]" << std::endl;        return pn;

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {        std::cout << "   origin: [" << origin_d_.transpose() << "]" << std::endl;      }

              for (n(2) = 0; n(2) < dim_(2); n(2)++) {

                if (isFree(getIndex(n)))        std::cout << "   range: [" << range.transpose() << "]" << std::endl;      ///Discrete cell coordinate to float position

                  cloud.push_back(intToFloat(n));

              }        std::cout << "   dim: [" << dim_.transpose() << "]" << std::endl;      Vecf<Dim> intToFloat(const Veci<Dim> &pn) {

            }

          }      };        //return pn.template cast<double>() * res_ + origin_d_;

        }

        else if (Dim == 2) {        return (pn.template cast<double>() + Vecf<Dim>::Constant(0.5)) * res_ + origin_d_;

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {      ///Float position to discrete cell coordinate      }

              if (isFree(getIndex(n)))

                cloud.push_back(intToFloat(n));      Veci<Dim> floatToInt(const Vecf<Dim> &pt) {

            }

          }        Veci<Dim> pn;      ///Raytrace from float point pt1 to pt2

        }

        return cloud;        for(int i = 0; i < Dim; i++)      vec_Veci<Dim> rayTrace(const Vecf<Dim> &pt1, const Vecf<Dim> &pt2) {

      }

          pn(i) = floor((pt(i) - origin_d_(i)) / res_);        Vecf<Dim> diff = pt2 - pt1;

      vec_Vecf<Dim> getUnknownCloud() {

        vec_Vecf<Dim> cloud;        return pn;        double k = 0.8;

        Veci<Dim> n;

        if(Dim == 3) {      }        int max_diff = (diff / res_).template lpNorm<Eigen::Infinity>() / k;

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {      ///Discrete cell coordinate to float position        double s = 1.0 / max_diff;

              for (n(2) = 0; n(2) < dim_(2); n(2)++) {

                if (isUnknown(getIndex(n)))      Vecf<Dim> intToFloat(const Veci<Dim> &pn) {        Vecf<Dim> step = diff * s;

                  cloud.push_back(intToFloat(n));

              }        //return pn.template cast<double>() * res_ + origin_d_;

            }

          }        return (pn.template cast<double>() + Vecf<Dim>::Constant(0.5)) * res_ + origin_d_;        vec_Veci<Dim> pns;

        }

        else if (Dim == 2) {      }        Veci<Dim> prev_pn = Veci<Dim>::Constant(-1);

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {        for (int n = 1; n < max_diff; n++) {

              if (isUnknown(getIndex(n)))

                cloud.push_back(intToFloat(n));      ///Raytrace from float point pt1 to pt2          Vecf<Dim> pt = pt1 + step * n;

            }

          }      vec_Veci<Dim> rayTrace(const Vecf<Dim> &pt1, const Vecf<Dim> &pt2) {          Veci<Dim> new_pn = floatToInt(pt);

        }

        return cloud;        Vecf<Dim> diff = pt2 - pt1;          if (isOutside(new_pn))

      }

        double k = 0.8;            break;

      void dilate(const vec_Veci<Dim>& dilate_neighbor) {

        Tmap map = map_;        int max_diff = (diff / res_).template lpNorm<Eigen::Infinity>() / k;          if (new_pn != prev_pn)

        Veci<Dim> n = Veci<Dim>::Zero();

        if(Dim == 3) {        double s = 1.0 / max_diff;            pns.push_back(new_pn);

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {        Vecf<Dim> step = diff * s;          prev_pn = new_pn;

              for (n(2) = 0; n(2) < dim_(2); n(2)++) {

                if (isOccupied(getIndex(n))) {        }

                  for (const auto &it : dilate_neighbor) {

                    if (!isOutside(n + it))        vec_Veci<Dim> pns;        return pns;

                      map[getIndex(n + it)] = val_occ;

                  }        Veci<Dim> prev_pn = Veci<Dim>::Constant(-1);      }

                }

              }        for (int n = 1; n < max_diff; n++) {

            }

          }          Vecf<Dim> pt = pt1 + step * n;      ///Check if the ray from p1 to p2 is occluded

        }

        else if(Dim == 2) {          Veci<Dim> new_pn = floatToInt(pt);      bool isBlocked(const Vecf<Dim>& p1, const Vecf<Dim>& p2, int8_t val = 100) {

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {          if (isOutside(new_pn))        vec_Veci<Dim> pns = rayTrace(p1, p2);

              if (isOccupied(getIndex(n))) {

                for (const auto &it : dilate_neighbor) {            break;        for (const auto &pn : pns) {

                  if (!isOutside(n + it))

                    map[getIndex(n + it)] = val_occ;          if (new_pn != prev_pn)          if (map_[getIndex(pn)] >= val)

                }

              }            pns.push_back(new_pn);            return true;

            }

          }          prev_pn = new_pn;        }

        }

        map_ = map;        }        return false;

      }

        return pns;      }

      void freeUnknown() {

        Veci<Dim> n;      }

        if(Dim == 3) {

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {      ///Get occupied voxels

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

              for (n(2) = 0; n(2) < dim_(2); n(2)++) {      ///Check if the ray from p1 to p2 is occluded      vec_Vecf<Dim> getCloud() {

                if (isUnknown(getIndex(n)))

                  map_[getIndex(n)] = val_free;      bool isBlocked(const Vecf<Dim>& p1, const Vecf<Dim>& p2, int8_t val = 100) {        vec_Vecf<Dim> cloud;

              }

            }        vec_Veci<Dim> pns = rayTrace(p1, p2);        Veci<Dim> n;

          }

        }        for (const auto &pn : pns) {        if(Dim == 3) {

        else if (Dim == 2) {

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {          if (map_[getIndex(pn)] >= val)          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

              if (isUnknown(getIndex(n)))            return true;            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

                map_[getIndex(n)] = val_free;

            }        }              for (n(2) = 0; n(2) < dim_(2); n(2)++) {

          }

        }        return false;                if (isOccupied(getIndex(n)))

      }

            }                  cloud.push_back(intToFloat(n));

      void CheckIfCollisionUsingPosAndYaw(const Eigen::Vector3d &state, bool *res, double flat = 0.0){

        if(Dim==3){              }

          RCLCPP_ERROR(node_->get_logger(), "CheckIfCollisionUsingPosAndYaw Only works with Dimension = 2!");

          return;      ///Get occupied voxels            }

        }

        Eigen::Matrix2d rotM;      vec_Vecf<Dim> getCloud() {          }

        double yaw = state[2];

        rotM << cos(yaw), -sin(yaw),        vec_Vecf<Dim> cloud;        }

                sin(yaw),  cos(yaw);

        Eigen::Vector2d p = state.head(2);        Veci<Dim> n;        else if (Dim == 2) {

        for(int i = 0; i < priv_config_->conpts.size(); i++){

          Eigen::Vector2d pt = p + rotM * priv_config_->conpts[i];        if(Dim == 3) {          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

          Eigen::Vector2d grad;

          if(getDistGrad(pt, grad)<priv_config_->safeMargin + flat){          for (n(0) = 0; n(0) < dim_(0); n(0)++) {            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

            *res = true;

            return;            for (n(1) = 0; n(1) < dim_(1); n(1)++) {              if (isOccupied(getIndex(n)))

          }

        }              for (n(2) = 0; n(2) < dim_(2); n(2)++) {                cloud.push_back(intToFloat(n));

        *res = false;

      }                if (isOccupied(getIndex(n)))            }



      void CheckIfCollisionForBackEnd(const Eigen::Vector3d &state, bool *res){                  cloud.push_back(intToFloat(n));          }

        if(Dim==3){

          RCLCPP_ERROR(node_->get_logger(), "CheckIfCollisionUsingPosAndYaw Only works with Dimension = 2!");              }        }

          return;

        }            }

        Eigen::Matrix2d rotM;

        double yaw = state[2];          }        return cloud;

        rotM << cos(yaw), -sin(yaw),

                sin(yaw),  cos(yaw);        }      }

        Eigen::Vector2d p = state.head(2);

        for(int i = 0; i < priv_config_->conpts.size(); i++){        else if (Dim == 2) {      void updateESDF2d() {

          Eigen::Vector2d pt = p + rotM * priv_config_->conpts[i];

          Eigen::Vector2d grad;          for (n(0) = 0; n(0) < dim_(0); n(0)++) {        if(Dim != 2){

          if(getDistGrad(pt, grad)<priv_config_->safeMargin){

            *res = true;            for (n(1) = 0; n(1) < dim_(1); n(1)++) {          ROS_ERROR("Dimension mismatch!");

            return;

          }              if (isOccupied(getIndex(n)))        }

        }

        *res = false;                cloud.push_back(intToFloat(n));        Eigen::Vector2i min_esdf = Eigen::Vector2i(0,0);

      }

                  }        Eigen::Vector2i max_esdf = Eigen::Vector2i(dim_(0) - 1, dim_(1) - 1);

      Tmap map_;

      std::vector<double> distance_buffer_;          }

      std::vector<double> distance_buffer_neg_;

      std::vector<char> occupancy_buffer_neg;        }        /* ========== compute positive DT ========== */

      std::vector<double> distance_buffer_all_;

      std::vector<double> tmp_buffer1_;

      std::string world_frame_id;

        return cloud;        for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {

    protected:

      double res_;      }          fillESDF(

      Vecf<Dim> origin_d_;

      Veci<Dim> dim_;      void updateESDF2d() {              [&](int y) {

      Vecf<Dim> map_size;

      int8_t val_occ = 100;        if(Dim != 2){                return isOccupied(Vec2i(x, y))?

      int8_t val_free = 0;

      int8_t val_unknown = -1;          RCLCPP_ERROR(node_->get_logger(), "Dimension mismatch!");                    0 :

      int  expand_size = 1;

      bool has_map = false;        }                    std::numeric_limits<double>::max();

      std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>> point_cloud_sub_;

      ConfigPtr priv_config_;        Eigen::Vector2i min_esdf = Eigen::Vector2i(0,0);              },

      std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> esdf_pub_;

      std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> pcl_pub_;        Eigen::Vector2i max_esdf = Eigen::Vector2i(dim_(0) - 1, dim_(1) - 1);              [&](int y, double val) {tmp_buffer1_[getIndex(Vec2i(x, y))] = val; }, min_esdf[1],

      std::shared_ptr<rclcpp::Node> node_;

    };              max_esdf[1], 1);

    

    typedef MapUtil<2> OccMapUtil;        /* ========== compute positive DT ========== */        }

    typedef MapUtil<3> VoxelMapUtil;

}



#endif        for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {        for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {


          fillESDF(          fillESDF([&](int x) { return tmp_buffer1_[getIndex(Vec2i(x, y))]; },

              [&](int y) {                    [&](int x, double val) {

                return isOccupied(Vec2i(x, y))?                      distance_buffer_[getIndex(Vec2i(x, y))] = res_ * std::sqrt(val);

                    0 :                      //  min(mp_.resolution_ * std::sqrt(val),

                    std::numeric_limits<double>::max();                      //      md_.distance_buffer_[toAddress(x, y, z)]);

              },                    },

              [&](int y, double val) {tmp_buffer1_[getIndex(Vec2i(x, y))] = val; }, min_esdf[1],                    min_esdf[0], max_esdf[0], 0);

              max_esdf[1], 1);        }

        }

        /* ========== compute negative distance ========== */

        for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {        for (int x = min_esdf(0); x <= max_esdf(0); ++x)

          fillESDF([&](int x) { return tmp_buffer1_[getIndex(Vec2i(x, y))]; },          for (int y = min_esdf(1); y <= max_esdf(1); ++y)

                    [&](int x, double val) {          {

                      distance_buffer_[getIndex(Vec2i(x, y))] = res_ * std::sqrt(val);            int idx = getIndex(Vec2i(x, y));

                      //  min(mp_.resolution_ * std::sqrt(val),            if (isFree(idx)) {

                      //      md_.distance_buffer_[toAddress(x, y, z)]);              occupancy_buffer_neg[idx] = 1;

                    },

                    min_esdf[0], max_esdf[0], 0);            } else if (isOccupied(idx)) {

        }              occupancy_buffer_neg[idx] = 0;

            } else {

        /* ========== compute negative distance ========== */              RCLCPP_ERROR(node_->get_logger(), "what?");

        for (int x = min_esdf(0); x <= max_esdf(0); ++x)            }

          for (int y = min_esdf(1); y <= max_esdf(1); ++y)          }

          {

            int idx = getIndex(Vec2i(x, y));        for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {

            if (isFree(idx)) {          fillESDF(

              occupancy_buffer_neg[idx] = 1;              [&](int y) {

                return occupancy_buffer_neg[getIndex(Vec2i(x, y))] == 1 ?

            } else if (isOccupied(idx)) {                    0 :

              occupancy_buffer_neg[idx] = 0;                    std::numeric_limits<double>::max();

            } else {              },

              RCLCPP_ERROR(node_->get_logger(), "what?");              [&](int y, double val) { tmp_buffer1_[getIndex(Vec2i(x, y))] = val; }, min_esdf[1],

            }              max_esdf[1], 1);

          }        }



        for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {        for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {

          fillESDF(          fillESDF([&](int x) { return tmp_buffer1_[getIndex(Vec2i(x, y))]; },

              [&](int y) {                    [&](int x, double val) {

                return occupancy_buffer_neg[getIndex(Vec2i(x, y))] == 1 ?                      distance_buffer_neg_[getIndex(Vec2i(x, y))] = res_ * std::sqrt(val);

                    0 :                    },

                    std::numeric_limits<double>::max();                    min_esdf[0], max_esdf[0], 0);

              },        }

              [&](int y, double val) { tmp_buffer1_[getIndex(Vec2i(x, y))] = val; }, min_esdf[1],

              max_esdf[1], 1);        /* ========== combine pos and neg DT ========== */

        }        for (int x = min_esdf(0); x <= max_esdf(0); ++x)

          for (int y = min_esdf(1); y <= max_esdf(1); ++y)

        for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {          {

          fillESDF([&](int x) { return tmp_buffer1_[getIndex(Vec2i(x, y))]; },            int idx = getIndex(Vec2i(x, y));

                    [&](int x, double val) {            distance_buffer_all_[idx] = distance_buffer_[idx];

                      distance_buffer_neg_[getIndex(Vec2i(x, y))] = res_ * std::sqrt(val);

                    },            if (distance_buffer_neg_[idx] > 0.0){

                    min_esdf[0], max_esdf[0], 0);              distance_buffer_all_[idx] += (-distance_buffer_neg_[idx] + res_);

        }              }

          }

        /* ========== combine pos and neg DT ========== */

        for (int x = min_esdf(0); x <= max_esdf(0); ++x)      }

          for (int y = min_esdf(1); y <= max_esdf(1); ++y)      template <typename F_get_val, typename F_set_val>

          {      void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int d) {

            int idx = getIndex(Vec2i(x, y));        int v[dim_(d)];

            distance_buffer_all_[idx] = distance_buffer_[idx];        double z[dim_(d) + 1];



            if (distance_buffer_neg_[idx] > 0.0){        int k = start;

              distance_buffer_all_[idx] += (-distance_buffer_neg_[idx] + res_);        v[start] = start;

              }        z[start] = -std::numeric_limits<double>::max();

          }        z[start + 1] = std::numeric_limits<double>::max();



      }        for (int q = start + 1; q <= end; q++) {

      template <typename F_get_val, typename F_set_val>          k++;

      void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int d) {          double s;

        int v[dim_(d)];

        double z[dim_(d) + 1];          do {

            k--;

        int k = start;            s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);

        v[start] = start;          } while (s <= z[k]);

        z[start] = -std::numeric_limits<double>::max();

        z[start + 1] = std::numeric_limits<double>::max();          k++;



        for (int q = start + 1; q <= end; q++) {          v[k] = q;

          k++;          z[k] = s;

          double s;          z[k + 1] = std::numeric_limits<double>::max();

        }

          do {

            k--;        k = start;

            s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);

          } while (s <= z[k]);        for (int q = start; q <= end; q++) {

          while (z[k + 1] < q) k++;

          k++;          double val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);

          f_set_val(q, val);

          v[k] = q;        }

          z[k] = s;      }

          z[k + 1] = std::numeric_limits<double>::max();      inline double getDistance(const Eigen::Vector2d& pos) {

        }        Eigen::Vector2i id;

        id = floatToInt(pos);

        k = start;        boundIndex(id);

        return distance_buffer_all_[getIndex(id)];

        for (int q = start; q <= end; q++) {      }

          while (z[k + 1] < q) k++;      inline double getDistance(const Eigen::Vector2i& id) {

          double val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);      Eigen::Vector2i id1 = id;

          f_set_val(q, val);      boundIndex(id1);

        }      return distance_buffer_all_[getIndex(id1)];

      }      }

      inline double getDistance(const Eigen::Vector2d& pos) {      inline void boundIndex(Eigen::Vector2i& id) {

        Eigen::Vector2i id;        Eigen::Vector2i id1;

        id = floatToInt(pos);        id1(0) = std::max(std::min(id(0), dim_(0) - 1), 0);

        boundIndex(id);        id1(1) = std::max(std::min(id(1), dim_(1) - 1), 0);

        return distance_buffer_all_[getIndex(id)];        id = id1;

      }    }

      inline double getDistance(const Eigen::Vector2i& id) {    void publishPCL(){

      Eigen::Vector2i id1 = id;      vec_Vecf<Dim> pcs = getCloud();

      boundIndex(id1);      pcl::PointCloud<pcl::PointXYZI> cloud;

      return distance_buffer_all_[getIndex(id1)];      pcl::PointXYZI pt;

      }      for(const auto& pc : pcs){

      inline void boundIndex(Eigen::Vector2i& id) {        pt.x = pc[0];

        Eigen::Vector2i id1;        pt.y = pc[1];

        id1(0) = std::max(std::min(id(0), dim_(0) - 1), 0);        pt.z = 0.0;

        id1(1) = std::max(std::min(id(1), dim_(1) - 1), 0);        cloud.push_back(pt);

        id = id1;      }

    }      cloud.width = cloud.points.size();

    void publishPCL(){      cloud.height = 1;

      vec_Vecf<Dim> pcs = getCloud();      cloud.is_dense = true;

      pcl::PointCloud<pcl::PointXYZI> cloud;      cloud.header.frame_id = "world";

      pcl::PointXYZI pt;      sensor_msgs::msg::PointCloud2 cloud_msg;

      for(const auto& pc : pcs){      pcl::toMsg(cloud, cloud_msg);

        pt.x = pc[0];      pcl_pub_->publish(cloud_msg);

        pt.y = pc[1];

        pt.z = 0.0;    }

        cloud.push_back(pt);    void publishESDF() {

      }      double dist;

      cloud.width = cloud.points.size();      pcl::PointCloud<pcl::PointXYZI> cloud;

      cloud.height = 1;      pcl::PointXYZI pt;

      cloud.is_dense = true;      const double min_dist = -4.0;

      cloud.header.frame_id = "world";      const double max_dist = 4.0;

      sensor_msgs::msg::PointCloud2 cloud_msg;      Eigen::Vector2i min_cut = Eigen::Vector2i(0,0);

      pcl::toMsg(cloud, cloud_msg);      Eigen::Vector2i max_cut = Eigen::Vector2i(dim_(0)-1,dim_(1)-1);

      pcl_pub_->publish(cloud_msg);

      for (int x = min_cut(0); x <= max_cut(0); ++x)

    }        for (int y = min_cut(1); y <= max_cut(1); ++y) {

    void publishESDF() {          Vec2f pos;

      double dist;          pos = intToFloat(Vec2i(x, y));

      pcl::PointCloud<pcl::PointXYZI> cloud;          dist = getDistance(pos);

      pcl::PointXYZI pt;          // if(dist<min_dist) dist = min_dist;

      const double min_dist = -4.0;          // if(dist>max_dist) dist = max_dist;

      const double max_dist = 4.0;          pt.x = pos(0);

      Eigen::Vector2i min_cut = Eigen::Vector2i(0,0);          pt.y = pos(1);

      Eigen::Vector2i max_cut = Eigen::Vector2i(dim_(0)-1,dim_(1)-1);          pt.z = dist;

          pt.intensity = (dist - min_dist) / (max_dist - min_dist);

      for (int x = min_cut(0); x <= max_cut(0); ++x)          cloud.push_back(pt);

        for (int y = min_cut(1); y <= max_cut(1); ++y) {          // }

          Vec2f pos;        }

          pos = intToFloat(Vec2i(x, y));      cloud.width = cloud.points.size();

          dist = getDistance(pos);      cloud.height = 1;

          // if(dist<min_dist) dist = min_dist;      cloud.is_dense = true;

          // if(dist>max_dist) dist = max_dist;      cloud.header.frame_id = "world";

          pt.x = pos(0);      sensor_msgs::msg::PointCloud2 cloud_msg;

          pt.y = pos(1);      pcl::toMsg(cloud, cloud_msg);

          pt.z = dist;      esdf_pub_->publish(cloud_msg);

          pt.intensity = (dist - min_dist) / (max_dist - min_dist);    }

          cloud.push_back(pt);      inline double getDistGrad(Eigen::Vector2d po, Eigen::Vector2d& grad) {

          // }        Eigen::Vector2d pos(po(0),po(1));

        }        if (isOutside(floatToInt(pos))) {

      cloud.width = cloud.points.size();          grad.setZero();

      cloud.height = 1;          return 0;

      cloud.is_dense = true;        }

      cloud.header.frame_id = "world";

      sensor_msgs::msg::PointCloud2 cloud_msg;        /* use bilinear interpolation */

      pcl::toMsg(cloud, cloud_msg);        Eigen::Vector2d pos_m = pos - 0.5 * res_ * Eigen::Vector2d::Ones();

      esdf_pub_->publish(cloud_msg);

    }        Eigen::Vector2i idx;

      inline double getDistGrad(Eigen::Vector2d po, Eigen::Vector2d& grad) {        idx = floatToInt(pos_m);

        Eigen::Vector2d pos(po(0),po(1));        Eigen::Vector2d idx_pos, diff;

        if (isOutside(floatToInt(pos))) {        idx_pos = intToFloat(idx);

          grad.setZero();

          return 0;        diff = (pos - idx_pos) / res_;

        }

        double values[2][2];

        /* use bilinear interpolation */        for (int x = 0; x < 2; x++) {

        Eigen::Vector2d pos_m = pos - 0.5 * res_ * Eigen::Vector2d::Ones();          for (int y = 0; y < 2; y++) {

              Eigen::Vector2i current_idx = idx + Eigen::Vector2i(x, y);

        Eigen::Vector2i idx;              values[x][y] = getDistance(current_idx);

        idx = floatToInt(pos_m);          }   

        Eigen::Vector2d idx_pos, diff;        }

        idx_pos = intToFloat(idx);        double v00 = (1-diff(0)) * values[0][0] + diff(0) * values[1][0];

        double v10 = (1-diff(0)) * values[0][1] + diff(0) * values[1][1];

        diff = (pos - idx_pos) / res_;        double v0  = (1-diff(1)) * v00 + diff(1) * v10;

        // double dist = v0;

        double values[2][2];        grad[1] = (v10 - v00) / res_;

        for (int x = 0; x < 2; x++) {        grad[0] = ((1-diff(1)) * (values[1][0]-values[0][0]) + diff(1) * (values[1][1]-values[0][1])) / res_;

          for (int y = 0; y < 2; y++) {

              Eigen::Vector2i current_idx = idx + Eigen::Vector2i(x, y);        return v0;

              values[x][y] = getDistance(current_idx);      }

          }         

        }      ///Get free voxels

        double v00 = (1-diff(0)) * values[0][0] + diff(0) * values[1][0];      vec_Vecf<Dim> getFreeCloud() {

        double v10 = (1-diff(0)) * values[0][1] + diff(0) * values[1][1];        vec_Vecf<Dim> cloud;

        double v0  = (1-diff(1)) * v00 + diff(1) * v10;        Veci<Dim> n;

        // double dist = v0;        if(Dim == 3) {

        grad[1] = (v10 - v00) / res_;          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

        grad[0] = ((1-diff(1)) * (values[1][0]-values[0][0]) + diff(1) * (values[1][1]-values[0][1])) / res_;            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

              for (n(2) = 0; n(2) < dim_(2); n(2)++) {

        return v0;                if (isFree(getIndex(n)))

      }                  cloud.push_back(intToFloat(n));

                    }

      ///Get free voxels            }

      vec_Vecf<Dim> getFreeCloud() {          }

        vec_Vecf<Dim> cloud;        }

        Veci<Dim> n;        else if (Dim == 2) {

        if(Dim == 3) {          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {              if (isFree(getIndex(n)))

              for (n(2) = 0; n(2) < dim_(2); n(2)++) {                cloud.push_back(intToFloat(n));

                if (isFree(getIndex(n)))            }

                  cloud.push_back(intToFloat(n));          }

              }        }

            }

          }        return cloud;

        }      }

        else if (Dim == 2) {

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {      ///Get unknown voxels

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {      vec_Vecf<Dim> getUnknownCloud() {

              if (isFree(getIndex(n)))        vec_Vecf<Dim> cloud;

                cloud.push_back(intToFloat(n));        Veci<Dim> n;

            }        if(Dim == 3) {

          }          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

        }            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

              for (n(2) = 0; n(2) < dim_(2); n(2)++) {

        return cloud;                if (isUnknown(getIndex(n)))

      }                  cloud.push_back(intToFloat(n));

              }

      ///Get unknown voxels            }

      vec_Vecf<Dim> getUnknownCloud() {          }

        vec_Vecf<Dim> cloud;        }

        Veci<Dim> n;        else if (Dim == 2) {

        if(Dim == 3) {          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {              if (isUnknown(getIndex(n)))

              for (n(2) = 0; n(2) < dim_(2); n(2)++) {                cloud.push_back(intToFloat(n));

                if (isUnknown(getIndex(n)))            }

                  cloud.push_back(intToFloat(n));          }

              }        }

            }

          }        return cloud;

        }      }

        else if (Dim == 2) {

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {      ///Dilate occupied cells

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {      void dilate(const vec_Veci<Dim>& dilate_neighbor) {

              if (isUnknown(getIndex(n)))        Tmap map = map_;

                cloud.push_back(intToFloat(n));        Veci<Dim> n = Veci<Dim>::Zero();

            }        if(Dim == 3) {

          }          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

        }            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

              for (n(2) = 0; n(2) < dim_(2); n(2)++) {

        return cloud;                if (isOccupied(getIndex(n))) {

      }                  for (const auto &it : dilate_neighbor) {

                    if (!isOutside(n + it))

      ///Dilate occupied cells                      map[getIndex(n + it)] = val_occ;

      void dilate(const vec_Veci<Dim>& dilate_neighbor) {                  }

        Tmap map = map_;                }

        Veci<Dim> n = Veci<Dim>::Zero();              }

        if(Dim == 3) {            }

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {          }

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {        }

              for (n(2) = 0; n(2) < dim_(2); n(2)++) {        else if(Dim == 2) {

                if (isOccupied(getIndex(n))) {          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

                  for (const auto &it : dilate_neighbor) {            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

                    if (!isOutside(n + it))              if (isOccupied(getIndex(n))) {

                      map[getIndex(n + it)] = val_occ;                for (const auto &it : dilate_neighbor) {

                  }                  if (!isOutside(n + it))

                }                    map[getIndex(n + it)] = val_occ;

              }                }

            }              }

          }            }

        }          }

        else if(Dim == 2) {        }

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {        map_ = map;

              if (isOccupied(getIndex(n))) {      }

                for (const auto &it : dilate_neighbor) {

                  if (!isOutside(n + it))      ///Free unknown voxels

                    map[getIndex(n + it)] = val_occ;      void freeUnknown() {

                }        Veci<Dim> n;

              }        if(Dim == 3) {

            }          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

          }            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

        }              for (n(2) = 0; n(2) < dim_(2); n(2)++) {

                if (isUnknown(getIndex(n)))

        map_ = map;                  map_[getIndex(n)] = val_free;

      }              }

            }

      ///Free unknown voxels          }

      void freeUnknown() {        }

        Veci<Dim> n;        else if (Dim == 2) {

        if(Dim == 3) {          for (n(0) = 0; n(0) < dim_(0); n(0)++) {

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {            for (n(1) = 0; n(1) < dim_(1); n(1)++) {

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {              if (isUnknown(getIndex(n)))

              for (n(2) = 0; n(2) < dim_(2); n(2)++) {                map_[getIndex(n)] = val_free;

                if (isUnknown(getIndex(n)))            }

                  map_[getIndex(n)] = val_free;          }

              }        }

            }      }

          }      void CheckIfCollisionUsingPosAndYaw(const Eigen::Vector3d &state, bool *res, double flat = 0.0){

        }        if(Dim==3){

        else if (Dim == 2) {          RCLCPP_ERROR(node_->get_logger(), "CheckIfCollisionUsingPosAndYaw Only works with Dimension = 2!");

          for (n(0) = 0; n(0) < dim_(0); n(0)++) {          return;

            for (n(1) = 0; n(1) < dim_(1); n(1)++) {        }

              if (isUnknown(getIndex(n)))        else{

                map_[getIndex(n)] = val_free;          Eigen::Matrix2d rotM;

            }          double yaw = state[2];

          }          rotM << cos(yaw), -sin(yaw),

        }                  sin(yaw),  cos(yaw);

      }          Eigen::Vector2d p = state.head(2);

      void CheckIfCollisionUsingPosAndYaw(const Eigen::Vector3d &state, bool *res, double flat = 0.0){          for(int i = 0; i < priv_config_->conpts.size(); i++){

        if(Dim==3){            // Eigen::Vector2d rep1 = priv_config_->conpts[i];

          RCLCPP_ERROR(node_->get_logger(), "CheckIfCollisionUsingPosAndYaw Only works with Dimension = 2!");            // Eigen::Vector2d rep2 = priv_config_->conpts[(i+1)%priv_config_->conpts.size()];

          return;            // if(rep1[0] > 0){

        }            //   rep1[0] += res_;

        else{            // }

          Eigen::Matrix2d rotM;            // else{

          double yaw = state[2];            //   rep1[0] -= res_;

          rotM << cos(yaw), -sin(yaw),            // }

                  sin(yaw),  cos(yaw);            // if(rep1[1] > 0){

          Eigen::Vector2d p = state.head(2);            //   rep1[1] += res_;

          for(int i = 0; i < priv_config_->conpts.size(); i++){            // }

            Eigen::Vector2d pt = p + rotM * priv_config_->conpts[i];            // else{

            Eigen::Vector2d grad;            //   rep1[1] -= res_;

            if(getDistGrad(pt, grad)<priv_config_->safeMargin + flat){            // }

              *res = true;            // if(rep2[0] > 0){

              return;            //   rep2[0] += res_;

            }            // }

            // else{

          }            //   rep2[0] -= res_;

        }            // }

        *res = false;            // if(rep2[1] > 0){

        return;            //   rep2[1] += res_;

      }            // }

            // else{

            //   rep2[1] -= res_;

      void CheckIfCollisionForBackEnd(const Eigen::Vector3d &state, bool *res){            // }

        if(Dim==3){            // Eigen::Vector2d pt1 = p + rotM * rep1;

          RCLCPP_ERROR(node_->get_logger(), "CheckIfCollisionUsingPosAndYaw Only works with Dimension = 2!");            // Eigen::Vector2d pt2 = p + rotM * rep2;

          return;            // if(isBlocked(pt1, pt2)){

        }            //   *res = true;

        else{            //   return;

          Eigen::Matrix2d rotM;            // }

          double yaw = state[2];            

          rotM << cos(yaw), -sin(yaw),            Eigen::Vector2d pt = p + rotM * priv_config_->conpts[i];

                  sin(yaw),  cos(yaw);            Eigen::Vector2d grad;

          Eigen::Vector2d p = state.head(2);            // if(flat > 0){

          for(int i = 0; i < priv_config_->conpts.size(); i++){            // std::cout << "dis: "<<getDistGrad(pt, grad)<<"\n";

            Eigen::Vector2d pt = p + rotM * priv_config_->conpts[i];            // }

            Eigen::Vector2d grad;            if(getDistGrad(pt, grad)<priv_config_->safeMargin + flat){

            if(getDistGrad(pt, grad)<priv_config_->safeMargin){              *res = true;

              *res = true;              return;

              return;            }

            }

          }

          }        }

        }        *res = false;

        *res = false;        return;

        return;      }

      }

      ///Map entity

      Tmap map_;      void CheckIfCollisionForBackEnd(const Eigen::Vector3d &state, bool *res){

      std::vector<double> distance_buffer_;        if(Dim==3){

      std::vector<double> distance_buffer_neg_;          RCLCPP_ERROR(node_->get_logger(), "CheckIfCollisionUsingPosAndYaw Only works with Dimension = 2!");

      std::vector<char> occupancy_buffer_neg;          return;

      std::vector<double> distance_buffer_all_;        }

      std::vector<double> tmp_buffer1_;        else{

      std::string world_frame_id;          Eigen::Matrix2d rotM;

          double yaw = state[2];

    protected:          rotM << cos(yaw), -sin(yaw),

      ///Resolution                  sin(yaw),  cos(yaw);

      double res_;          Eigen::Vector2d p = state.head(2);

      ///Origin, float type          for(int i = 0; i < priv_config_->conpts.size(); i++){

      Vecf<Dim> origin_d_;            // Eigen::Vector2d rep1 = priv_config_->conpts[i];

      ///Dimension, int type            // Eigen::Vector2d rep2 = priv_config_->conpts[(i+1)%priv_config_->conpts.size()];

      Veci<Dim> dim_;            // if(rep1[0] > 0){

      Vecf<Dim> map_size;            //   rep1[0] += res_;

      ///Assume occupied cell has value 100            // }

      int8_t val_occ = 100;            // else{

      ///Assume free cell has value 0            //   rep1[0] -= res_;

      int8_t val_free = 0;            // }

      ///Assume unknown cell has value -1            // if(rep1[1] > 0){

      int8_t val_unknown = -1;            //   rep1[1] += res_;

      ///inflate size            // }

      int  expand_size = 1;            // else{

      bool has_map = false;            //   rep1[1] -= res_;

      std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>> point_cloud_sub_;            // }

      ConfigPtr priv_config_;            // if(rep2[0] > 0){

      std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> esdf_pub_;            //   rep2[0] += res_;

      std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> pcl_pub_;            // }

      std::shared_ptr<rclcpp::Node> node_;            // else{

  };            //   rep2[0] -= res_;

  typedef MapUtil<2> OccMapUtil;            // }

  typedef MapUtil<3> VoxelMapUtil;            // if(rep2[1] > 0){

            //   rep2[1] += res_;

            // }

}            // else{

            //   rep2[1] -= res_;

            // }

#endif            // Eigen::Vector2d pt1 = p + rotM * rep1;

            // Eigen::Vector2d pt2 = p + rotM * rep2;
            // if(isBlocked(pt1, pt2)){
            //   *res = true;
            //   return;
            // }
            
            Eigen::Vector2d pt = p + rotM * priv_config_->conpts[i];
            Eigen::Vector2d grad;
            if(getDistGrad(pt, grad)<priv_config_->safeMargin){
              *res = true;
              return;
            }

          }
        }
        *res = false;
        return;
      }
      ///Map entity
      Tmap map_;
      std::vector<double> distance_buffer_;
      std::vector<double> distance_buffer_neg_;
      std::vector<char> occupancy_buffer_neg;
      std::vector<double> distance_buffer_all_;
      std::vector<double> tmp_buffer1_;
      std::string world_frame_id;

    protected:
      ///Resolution
      double res_;
      ///Origin, float type
      Vecf<Dim> origin_d_;
      ///Dimension, int type
      Veci<Dim> dim_;
      Vecf<Dim> map_size;
      ///Assume occupied cell has value 100
      int8_t val_occ = 100;
      ///Assume free cell has value 0
      int8_t val_free = 0;
      ///Assume unknown cell has value -1
      int8_t val_unknown = -1;
      ///inflate size
      int  expand_size = 1;
      bool has_map = false;
      std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>> point_cloud_sub_;
      ConfigPtr priv_config_;
      std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> esdf_pub_;
      std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> pcl_pub_;
      std::shared_ptr<rclcpp::Node> node_;
  };
  typedef MapUtil<2> OccMapUtil;
  typedef MapUtil<3> VoxelMapUtil;


}


#endif
