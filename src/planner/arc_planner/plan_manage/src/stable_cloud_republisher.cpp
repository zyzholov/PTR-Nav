#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

class StableCloudRepublisher : public rclcpp::Node
{
public:
    StableCloudRepublisher()
        : rclcpp::Node("stable_cloud_republisher")
    {
        input_topic_ = declare_parameter<std::string>("input_topic", "/cloud_local_map");
        output_topic_ = declare_parameter<std::string>("output_topic", "/cloud_local_map_stable");
        min_points_ = declare_parameter<int>("min_points", 200);
        min_point_ratio_ = declare_parameter<double>("min_point_ratio", 0.35);
        publish_rate_ = declare_parameter<double>("publish_rate", 0.0);
        max_points_ = declare_parameter<int>("max_points", 0);

        min_points_ = std::max(1, min_points_);
        min_point_ratio_ = std::clamp(min_point_ratio_, 0.0, 1.0);
        publish_rate_ = std::max(0.0, publish_rate_);
        max_points_ = std::max(0, max_points_);

        auto sub_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
        auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, sub_qos,
            std::bind(&StableCloudRepublisher::cloudCallback, this, std::placeholders::_1));
        pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, pub_qos);
        if (publish_rate_ > 1e-3)
        {
            const auto period = std::chrono::duration<double>(1.0 / publish_rate_);
            timer_ = create_wall_timer(
                std::chrono::duration_cast<std::chrono::nanoseconds>(period),
                std::bind(&StableCloudRepublisher::publishLatestIfNew, this));
        }

        RCLCPP_INFO(get_logger(),
                    "Stable cloud republisher: %s -> %s, min_points=%d, min_ratio=%.2f, rate=%.1f Hz, max_points=%d",
                    input_topic_.c_str(), output_topic_.c_str(),
                    min_points_, min_point_ratio_, publish_rate_, max_points_);
    }

private:
    static std::size_t pointCount(const sensor_msgs::msg::PointCloud2 &msg)
    {
        return static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
    }

    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        const std::size_t count = pointCount(*msg);
        ++received_frames_;
        if (count < static_cast<std::size_t>(min_points_))
        {
            ++dropped_sparse_frames_;
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1500,
                                 "Drop sparse %s frame: points=%zu < min_points=%d",
                                 input_topic_.c_str(), count, min_points_);
            logStatusThrottled();
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (min_point_ratio_ > 1e-6 &&
                has_latest_ && latest_points_ > 0 &&
                count < static_cast<std::size_t>(static_cast<double>(latest_points_) * min_point_ratio_))
            {
                ++dropped_ratio_frames_;
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1500,
                                     "Drop transient %s frame: points=%zu < %.2f * last=%zu",
                                     input_topic_.c_str(), count, min_point_ratio_, latest_points_);
                logStatusThrottled();
                return;
            }

            latest_ = msg;
            latest_points_ = count;
            has_latest_ = true;
            ++latest_seq_;
            ++accepted_frames_;
        }

        if (publish_rate_ <= 1e-3)
            publishLatestIfNew();
        logStatusThrottled();
    }

    void publishLatestIfNew()
    {
        sensor_msgs::msg::PointCloud2::SharedPtr latest;
        sensor_msgs::msg::PointCloud2 out;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!has_latest_ || latest_seq_ == published_seq_)
                return;
            latest = latest_;
            published_seq_ = latest_seq_;
        }

        if (!latest)
            return;
        out = makeDisplayCloud(*latest);
        pub_->publish(out);
        last_published_points_ = pointCount(out);
        ++published_frames_;
    }

    std::size_t sourceOffset(const sensor_msgs::msg::PointCloud2 &msg, std::size_t linear_index) const
    {
        if (msg.width == 0 || msg.point_step == 0)
            return msg.data.size();
        const std::size_t row = linear_index / static_cast<std::size_t>(msg.width);
        const std::size_t col = linear_index % static_cast<std::size_t>(msg.width);
        return row * static_cast<std::size_t>(msg.row_step) + col * static_cast<std::size_t>(msg.point_step);
    }

    sensor_msgs::msg::PointCloud2 makeDisplayCloud(const sensor_msgs::msg::PointCloud2 &in) const
    {
        const std::size_t input_count = pointCount(in);
        if (max_points_ <= 0 || input_count <= static_cast<std::size_t>(max_points_) || in.point_step == 0)
            return in;

        const std::size_t target = static_cast<std::size_t>(max_points_);
        const std::size_t stride = std::max<std::size_t>(1, (input_count + target - 1) / target);

        sensor_msgs::msg::PointCloud2 out = in;
        out.height = 1;
        out.width = 0;
        out.row_step = 0;
        out.data.clear();
        out.data.reserve(target * static_cast<std::size_t>(in.point_step));

        for (std::size_t i = 0; i < input_count; i += stride)
        {
            const std::size_t offset = sourceOffset(in, i);
            const std::size_t point_step = static_cast<std::size_t>(in.point_step);
            if (offset + point_step > in.data.size())
                continue;
            out.data.insert(out.data.end(), in.data.begin() + offset, in.data.begin() + offset + point_step);
        }

        out.width = static_cast<uint32_t>(out.data.size() / static_cast<std::size_t>(out.point_step));
        out.row_step = out.width * out.point_step;
        return out;
    }

    void logStatusThrottled()
    {
        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Stable cloud status: rx=%lu accepted=%lu published=%lu sparse_drop=%lu ratio_drop=%lu latest_points=%zu published_points=%zu latest_seq=%lu",
            received_frames_, accepted_frames_, published_frames_,
            dropped_sparse_frames_, dropped_ratio_frames_,
            latest_points_, last_published_points_, latest_seq_);
    }

    std::string input_topic_;
    std::string output_topic_;
    int min_points_ = 200;
    double min_point_ratio_ = 0.35;
    double publish_rate_ = 0.0;
    int max_points_ = 0;

    std::mutex mutex_;
    sensor_msgs::msg::PointCloud2::SharedPtr latest_;
    std::size_t latest_points_ = 0;
    std::size_t last_published_points_ = 0;
    std::uint64_t latest_seq_ = 0;
    std::uint64_t published_seq_ = 0;
    bool has_latest_ = false;
    std::uint64_t received_frames_ = 0;
    std::uint64_t accepted_frames_ = 0;
    std::uint64_t published_frames_ = 0;
    std::uint64_t dropped_sparse_frames_ = 0;
    std::uint64_t dropped_ratio_frames_ = 0;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StableCloudRepublisher>());
    rclcpp::shutdown();
    return 0;
}
