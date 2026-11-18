#pragma once
#include <chrono>
#include <map>
#include <mutex>
#include <QLabel>
#include <QTimer>
#include <QComboBox>
#include <QTabWidget>
#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>

namespace ptr_navigation_rviz
{
class StatusPanel : public rviz_common::Panel
{
  Q_OBJECT
public:
  explicit StatusPanel(QWidget *parent = nullptr);
  void onInitialize() override;
  void load(const rviz_common::Config &config) override;
  void save(rviz_common::Config config) const override;
private:
  void refresh();
  QLabel *status_;
  std::map<std::string,QLabel *> fields_;
  QTimer *timer_;
  QComboBox *view_;
  QTabWidget *tabs_;
  QWidget *risk_legend_;
  void applyView(int mode);
  std::mutex mutex_;
  std::map<std::string,std::string> values_;
  std::chrono::steady_clock::time_point received_{};
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr subscription_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr view_state_pub_;
  int loaded_display_count_=-1,stable_display_ticks_=0;
  bool view_sync_pending_=true;
};
}
