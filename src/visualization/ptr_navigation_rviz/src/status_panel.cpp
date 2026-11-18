#include "ptr_navigation_rviz/status_panel.hpp"
#include <algorithm>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QScrollArea>
#include <QGridLayout>
#include <QApplication>
#include <QPalette>
#include <QStyleFactory>
#include <rviz_common/display_context.hpp>
#include <rviz_common/display_group.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace ptr_navigation_rviz
{
StatusPanel::StatusPanel(QWidget *parent) : Panel(parent)
{
  setMinimumWidth(260);
  setAttribute(Qt::WA_StyledBackground,true);
  setStyleSheet("QWidget {background:#23272d; color:#dce1e7;} QLabel {font-size:12px;}"
                "QLabel[section=\"true\"] {color:#88ced3; font-weight:bold; margin-top:12px;}");
  auto *root=new QVBoxLayout(this); root->setContentsMargins(0,0,0,0);
  auto *scroll=new QScrollArea(this); scroll->setWidgetResizable(true); scroll->setFrameShape(QFrame::NoFrame);
  auto *content=new QWidget(scroll); scroll->setWidget(content); root->addWidget(scroll);
  auto *layout=new QVBoxLayout(content);
  layout->setContentsMargins(12,8,12,12);
  status_=new QLabel("等待实时导航与地形数据",this);
  status_->setWordWrap(true); status_->setMinimumHeight(46);
  layout->addWidget(status_);
  view_=new QComboBox(this);
  view_->addItems({"地形风险分量与原始占据","实测与预测高程","累计点云与实时回波","地形证据来源","整车姿态风险（诊断）"});
  view_->setMinimumHeight(28);layout->addWidget(view_);
  connect(view_,QOverload<int>::of(&QComboBox::currentIndexChanged),this,[this](int mode){applyView(mode);});
  tabs_=new QTabWidget(this);layout->addWidget(tabs_);
  auto *navigation=new QWidget(tabs_);auto *terrain=new QWidget(tabs_);
  tabs_->addTab(navigation,"导航状态");tabs_->addTab(terrain,"高程与补全");
  auto *body=new QVBoxLayout(navigation);
  const auto section=[&](const char *name) {
    auto *title=new QLabel(name,this); title->setProperty("section",true); body->addWidget(title);
  };
  const auto form=[&](std::initializer_list<std::pair<const char *,const char *>> rows) {
    auto *box=new QFormLayout(); box->setSpacing(8);
    for (const auto &row : rows) {
      auto *value=new QLabel("--",this); value->setWordWrap(true);
      value->setMinimumWidth(105); value->setTextInteractionFlags(Qt::TextSelectableByMouse);
      fields_[row.first]=value; box->addRow(row.second,value);
    }
    body->addLayout(box);
  };
  section("控制与跟踪");
  form({{"controller","跟踪器"},{"state","控制状态"},{"speed","实测速度"},
        {"steering","前轮转角命令"},{"track_error","平面跟踪误差"},{"map_age","地形消息年龄"},
        {"recovery","退让恢复阶段"},{"recovery_attempts","恢复次数（连续/总计）"}});
  section("地图与观测证据");
  form({{"accumulated_points","累计点云点数"},{"measured_cells","实测地面格数"},
        {"candidate_cells","GP 候选格数"},{"navigation_prediction_cells","可用于导航的 GP"},
        {"startup_cells","起始假设格数"},
        {"unknown_cells","未确定格数"},{"prediction_age","后台 GP 年龄"},
        {"completion_age","补全扫描年龄"},{"conditioning_age","训练观测最大年龄"}});
  body->addStretch();body=new QVBoxLayout(terrain);
  section("高程模型与查询位置");
  form({{"height_scale","高程色标范围"},{"probe_xy","查询位置 XY"},{"source","地面证据来源"},
        {"risk_reason","查询位置风险来源"},{"height","展示层高程"},{"sigma","展示层高程 σ"},
        {"interval","展示层95%区间宽"},{"navigation_height","导航格高程均值"},
        {"navigation_sigma","导航格高程 σ"},{"pending_height","待确认观测高程"},
        {"pose_upper","姿态风险模型上界"},{"heading","车辆航向"}});
  auto *height_scale=new QLabel(this);height_scale->setFixedHeight(12);
  height_scale->setStyleSheet("background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #4867b8,stop:0.5 #36a4b0,stop:1 #a8dca6);");
  body->addWidget(height_scale);
  section("车型与地形约束");
  form({{"probability_contract","概率地图契约版本"},{"local_step_limit","局部台阶上限"},
        {"roughness_span_limit","整足迹残差跨度上限"},{"ground_clearance","车型离地间隙"},
        {"underbody_margin","底盘净空余量"},{"floor_model","几何误差相关性模型"}});
  auto *note=new QLabel("高程由低到高：蓝 → 青 → 浅绿。\n来源：青色为实测，淡紫为预测候选，薄荷绿为导航合格预测。\n候选预测不等于允许通行；未知地面不能当作安全。\n区间宽度是高程上下界之差，非地图横向宽度。\n模型概率尚未完成实地校准。",this);
  note->setWordWrap(true); note->setStyleSheet("color:#aab6c2; font-size:11px; margin-top:10px;");
  body->addWidget(note);body->addStretch();
  risk_legend_=new QWidget(content);
  auto *risk_layout=new QVBoxLayout(risk_legend_);risk_layout->setContentsMargins(0,0,0,0);
  body=risk_layout;
  section("风险模型上界色标（不等于通行证）");
  const char *colors[]={"#3a5e66","#49827c","#91a88c","#e8b25b","#d6495b","#4c5763"};
  const char *labels[]={"0%","0.1%","1%","5%","100%","未知"};
  auto *legend=new QGridLayout();legend->setVerticalSpacing(6);
  for (int i=0;i<6;++i) {
    auto *swatch=new QLabel(this);
    swatch->setFixedSize(22,13); swatch->setStyleSheet(QString("background:%1;").arg(colors[i]));
    legend->addWidget(swatch,i/2,(i%2)*2);legend->addWidget(new QLabel(labels[i],this),i/2,(i%2)*2+1);
  }
  risk_layout->addLayout(legend);layout->addWidget(risk_legend_);
  body=layout;
  section("图层说明");
  auto *key=new QLabel("主图颜色：地形足迹的条件风险分量，非障碍外形。\n珊瑚轮廓：实测占据边界；高度见原始点云。\n整车包络风险另见诊断视图。\n柱下预测仍是预测，不能穿过柱体。\n薄荷绿：合格补全；琥珀：仿真起始假设。\n青线：规划；金线：实际行驶轨迹。\n当前回波：一帧真实校正回波，过期即清空。\n近处金色、8 米薄荷绿、30 米外蓝色。\n颜色表示距车辆的水平距离，非反射强度。\n圆状等色线不是扫描形状，点位置才是回波。\n蓝灰背景：按高程着色的累计点云。",this);
  key->setWordWrap(true); key->setStyleSheet("color:#b5c1cc; font-size:11px;"); layout->addWidget(key);
  layout->addStretch();
  timer_=new QTimer(this);
  connect(timer_,&QTimer::timeout,this,[this] {refresh();}); timer_->start(200);
}

void StatusPanel::onInitialize()
{
  // Palette changes are confined to this RViz process; no desktop settings
  // or Ogre colours/data are changed. Match the property tree to the canvas.
  QApplication::setStyle(QStyleFactory::create("Fusion"));
  auto palette=QApplication::palette();
  palette.setColor(QPalette::Window,QColor("#232b34"));
  palette.setColor(QPalette::WindowText,QColor("#dce5ee"));
  palette.setColor(QPalette::Base,QColor("#18212a"));
  palette.setColor(QPalette::AlternateBase,QColor("#232e38"));
  palette.setColor(QPalette::Text,QColor("#dce5ee"));
  palette.setColor(QPalette::Button,QColor("#2c3743"));
  palette.setColor(QPalette::ButtonText,QColor("#dce5ee"));
  palette.setColor(QPalette::Highlight,QColor("#326977"));
  palette.setColor(QPalette::HighlightedText,QColor("#ffffff"));
  palette.setColor(QPalette::Disabled,QPalette::Text,QColor("#8494a3"));
  QApplication::setPalette(palette);
  // RViz's ordinary property labels carry explicit black foreground brushes.
  // Keep them readable on the dark tree; status icons/text remain present.
  qApp->setStyleSheet(
      "QTreeView::item {color:#dce5ee;} QToolTip {color:#18212a; background:#e3ebf2;}");
  auto abstraction=getDisplayContext()->getRosNodeAbstraction().lock();
  if (!abstraction) return;
  auto node=abstraction->get_raw_node();
  view_state_pub_=node->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "/terrain_nav/rviz_view_state",rclcpp::QoS(1).reliable().transient_local());
  subscription_=node->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
    "/terrain_nav/display_status",rclcpp::QoS(1),
    [this](diagnostic_msgs::msg::DiagnosticArray::ConstSharedPtr msg) {
      std::lock_guard<std::mutex> guard(mutex_);
      for (const auto &status : msg->status) if (status.name=="ptr_nav/presentation") {
        values_.clear();
        for (const auto &item : status.values) values_[item.key]=item.value;
        received_=std::chrono::steady_clock::now();
      }
    });
}

void StatusPanel::applyView(int mode)
{
  if (!getDisplayContext()) return;
  risk_legend_->setVisible(mode==0 || mode==4);
  tabs_->setCurrentIndex(mode==1 || mode==3 ? 1 : 0);
  auto *root=getDisplayContext()->getRootDisplayGroup();
  for (int i=0;i<root->numDisplays();++i) {
    auto *display=root->getDisplayAt(i);const auto name=display->getName();
    if (name=="Probability Surface (Gray Is Unknown)" || name=="地形风险与原始障碍（无车身膨胀；灰色未知）" || name=="地形风险分量与原始占据（灰色未知）") display->setEnabled(mode==0);
    else if(name=="整车姿态风险面（含车身包络，仅诊断）")display->setEnabled(mode==4);
    else if (name=="Measured Occupied Cells (No Body Expansion)" || name=="原始障碍轮廓（不含车身包络）") display->setEnabled(mode==0 || mode==3 || mode==4);
    else if (name=="Live LiDAR Returns (Range Color)") {
      display->setEnabled(mode==0 || mode==2);
      if(auto *alpha=display->subProp("Alpha"))alpha->setValue(mode==2 ? .90 : .16);
      if(auto *pixels=display->subProp("Size (Pixels)"))pixels->setValue(mode==2 ? 2 : 1);
    }
    else if (name=="Probability Cells (Current Heading)" || name=="整车姿态风险点（当前航向，仅诊断）") display->setEnabled(false);
    else if (name=="Measured Elevation Surface" || name=="Predicted Height (Candidates)") display->setEnabled(mode==1);
    else if (name=="GP Height (Navigation Qualified)") display->setEnabled(mode==1 || mode==3);
    else if (name=="Measured Height") display->setEnabled(false);
    else if (name=="Initial Assumed Height (Simulation)") display->setEnabled(mode!=2);
    else if (name=="Prediction Support (Unqualified)") display->setEnabled(false);
    else if (name=="Measured / GP Sources") display->setEnabled(mode==3);
    else if (name=="FAST-LIO Accumulated Map") {
      display->setEnabled(true);
      if (auto *alpha=display->subProp("Alpha")) alpha->setValue(mode==2 ? .90 : mode==1 ? .20 : .40);
      if (auto *pixels=display->subProp("Size (Pixels)")) pixels->setValue(mode==2 ? 3 : 2);
    }
  }
  if(view_state_pub_) {
    diagnostic_msgs::msg::DiagnosticArray report;diagnostic_msgs::msg::DiagnosticStatus state;
    state.name="ptr_nav/rviz_view_state";
    const auto field=[&](const std::string &key,const std::string &value) {
      diagnostic_msgs::msg::KeyValue entry;entry.key=key;entry.value=value;state.values.push_back(entry);
    };
    field("mode",std::to_string(mode));field("display_count",std::to_string(root->numDisplays()));
    for(int i=0;i<root->numDisplays();++i) {
      const auto *display=root->getDisplayAt(i);const auto name=display->getName();
      const char *key=nullptr;
      if(name=="地形风险分量与原始占据（灰色未知）")key="terrain_risk_surface";
      else if(name=="整车姿态风险面（含车身包络，仅诊断）")key="body_pose_risk_surface";
      else if(name=="整车姿态风险点（当前航向，仅诊断）")key="body_pose_risk_cloud";
      if(key)field(key,display->isEnabled()?"1":"0");
    }
    report.status.push_back(state);view_state_pub_->publish(report);
  }
}

void StatusPanel::load(const rviz_common::Config &config)
{
  Panel::load(config);int mode=0;config.mapGetInt("ViewMode",&mode);
  view_->setCurrentIndex(std::clamp(mode,0,4));
  view_sync_pending_=true;stable_display_ticks_=0;
}

void StatusPanel::save(rviz_common::Config config) const
{
  Panel::save(config);config.mapSetValue("ViewMode",view_->currentIndex());
}

void StatusPanel::refresh()
{
  // Panel loading can finish before the display group. Apply a preset only
  // after that group has settled; a single early timer could miss the very
  // diagnostic overlays it was meant to turn off. Later manual toggles stay
  // usable until the user changes preset or loads another display group.
  if(getDisplayContext()) {
    auto *root=getDisplayContext()->getRootDisplayGroup();const int count=root->numDisplays();
    if(count!=loaded_display_count_) {loaded_display_count_=count;stable_display_ticks_=0;view_sync_pending_=true;}
    if(view_sync_pending_ && count>0 && ++stable_display_ticks_>=3) {
      applyView(view_->currentIndex());view_sync_pending_=false;
    }
  }
  std::map<std::string,std::string> values;
  double age;
  {std::lock_guard<std::mutex> guard(mutex_); values=values_;
    age=std::chrono::duration<double>(std::chrono::steady_clock::now()-received_).count();}
  const bool fresh=age<=1.;
  const auto banner=values.find("banner");
  status_->setText(fresh && banner!=values.end() ? QString::fromStdString(banner->second) : "无实时数据 / 显示数据已过期");
  status_->setStyleSheet(fresh ? "color:#e9ca83;" : "color:#f2a386;");
  for (const auto &entry : fields_) {
    const auto found=values.find(entry.first);
    entry.second->setText(fresh && found!=values.end() ? QString::fromStdString(found->second) : "--");
  }
}
}
PLUGINLIB_EXPORT_CLASS(ptr_navigation_rviz::StatusPanel,rviz_common::Panel)
