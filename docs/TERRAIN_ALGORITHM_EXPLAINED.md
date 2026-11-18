# 地形分析模块：原理、公式推导与当前工程实现

日期：2026-10-16。

版本提示（2026-10-20）：下文解释旧启发式 `risk` 分支；新增独立概率上界、局部路径和制动预算见[概率导航接口说明](PROBABILISTIC_NAVIGATION.md)。新版本本轮只做编译/静态核查，未运行验收，不将旧实验结果移作新实现的证据。

本文对照当前源码和默认启动配置说明算法，不把论文方法、工程适配、可选功能及未来目标混为一谈。这里只解释地形分析及其规划接口，不修改算法、FAST-LIO 参数或冻结实验数据。

建议阅读顺序：先看第 1、2、7、8、9 节建立整体概念，再看第 3 至 6 节的统计推导。

## 1. 首先明确：到底要估计什么？

模块并不是简单地给点云着色，而是回答三个不同问题。

| 问题 | 数学对象 | 单位/含义 |
| --- | --- | --- |
| 这里的地面有多高？ | $h(x,y)$ | 米 |
| 这个高程估计有多不确定？ | $v_h(x,y)$ | 平方米；标准差为 $\sqrt{v_h}$ |
| 这辆车以这个朝向经过这里，代价/风险多大？ | $R(x,y,\theta)$ | 无量纲工程分数，不是已校准概率 |

第一、二项主要是地形表示与估计问题；第三项还取决于车辆尺寸、车头方向和通过限制。

### 1.1 为什么不能只做二维“可通行地图”？

假设地面是 $h(x,y)=0.5x$。该位置的总体坡度不随车头变化，但车沿 X 方向行驶时主要承受纵坡，沿 Y 方向行驶时主要承受横坡。车辆纵向和横向允许的倾角不同，风险就应当不同。

所以需要保留状态

$$
q=(x,y,\theta)\in SE(2),
$$

而不是把所有朝向压成一个 $R(x,y)$。其中 $SE(2)$ 表示平面位置与朝向，并不是完整三维姿态空间 $SE(3)$。

### 1.2 2.5D 高程图不是完整三维世界

每个 XY 格通常只能保存一个代表性高程。因此桥面与桥下地面、多层楼板、悬空结构、负障碍及真实轮胎接触都不能仅靠一张高程图完整描述。保留点云占据和其他保护层仍然必要。

## 2. 当前实际链路与模块关系

默认在线流程如下：

    FAST-LIO 注册点云 /cloud_registered + 位姿 /Odometry
        |
        v
    terrain_analyzer_gpu
      滚动高程/方差更新、几何分析、基线 SE(2) 风险 R0
        |
        v
    fsgp_bgk_se2_node
      可选 Feature-SGP 高程预测分支（默认关闭）
      空间核平滑及局部离散程度
      定向车辆足迹评估、几何保护
      当前扫描的立体障碍保护
        |
        v
    增强 SE2Grid /car_0_fsgp_bgk/fused_map
        |
        v
    terrain_nav：地图约束、规划、控制

“GPU 地形分析”不是与 FSGP、SEB 并列的另一种理论，而是上游地形前端的一种计算实现。上游已经包含 SEB 体系的定向几何分析；下游是本工程新增的预测接口、风险适配和保护逻辑，确实存在两层几何评估。

这不是论文唯一要求的模块拆分。理论上可以把“点云到概率高程、概率高程到车辆风险”集成为一体；当前拆分是工程接口和已有实现的选择。

### 2.1 三组开关决定 GP 是否真的参与导航

依据 [当前配置](../src/fsgp_bgk_se2/config/hunter_fastlio.yaml)：

| 参数 | 默认 | 实际含义 |
| --- | --- | --- |
| prediction_backend | none | 不计算 GP |
| prediction_backend=feature_sgp | 可选 | 产生 GP 高程和方差等诊断层 |
| gp_refine_observed | false | 不用 GP 改写已观测高程 |
| gp_complete_supported | false | 不用 GP 填入未知格 |
| use_observed_geometry | true | 足迹几何优先取输入高程，不取核平滑均值 |
| baseline_is_pose_risk | true | 上游已是姿态风险，不能再当点占据膨胀一次 |

只开启 feature_sgp、后两项仍关闭时，GP 是并行的预测诊断分支，不能声称它已经通过补全改善了规划。

另外，在线 GP 接收的是上游有效高程格组成的 XYZ，不是原始完整点云。离线实验中按来源标签提取的地面点，与线上经过高度筛选和栅格融合的输入并不完全等价。

## 3. 从点云到实测高程图

### 3.1 坐标、栅格与有效性

注册点已经位于地图坐标系，地形节点使用相应位姿处理高度筛选和测量误差；不能再把同一帧点云重复变换一次。

在物理世界顺序下，可用下面的公式理解格子索引：

$$
i=\left\lfloor\frac{x-x_{\min}}{\Delta}\right\rfloor,\qquad
j=\left\lfloor\frac{y-y_{\min}}{\Delta}\right\rfloor.
$$

默认局部窗口约 $20.8\times20.8$ 米，$\Delta=0.2$ 米，因此为 $104\times104$ 格。底层环形缓冲区索引还包含起始偏移，不能直接把原始消息矩阵当成世界顺序图像；节点已有对应转换。

每格保存高程 $H_i$、方差 $V_i$ 和是否有效 $O_i$。未知与零高程是两回事，未知不能通过补零自动变成可通行。

注意：这里称为“实测高程”，是为了区别 GP 预测。它仍是经过筛选、融合的测量估计，不等于无误差真值，也不保证已经完美分离地面与低矮障碍。

### 3.2 点的高程方差从哪里来？

设点经过位姿变换后，高程可写为

$$
z=e_3^\top(Rp+t),\qquad e_3=(0,0,1)^\top.
$$

一阶误差传播的基本形式为

$$
s_z^2\approx
J_p\Sigma_pJ_p^\top+
J_R\Sigma_RJ_R^\top+
\sigma_{t_z}^2,
\quad J_p=e_3^\top R.
$$

$J_R$ 来自姿态的小扰动对点高程的影响。它解释了为什么同样的雷达测量误差，在远距离点或车身倾斜时可能造成不同高程误差。

当前 CUDA 实现采用束噪声和位姿协方差的简化传播，见 [compute_var](../src/uneven_mapping/terrain_analyzer/src/TerrainBank.cu#L412)。上述式子说明其原理，不表示代码完整保留了所有误差交叉协方差。若输入位姿协方差缺失或全零，也不能指望输出方差自动反映全部 SLAM 漂移。

### 3.3 一格内多个测量怎样融合？

先假设旧高程和新测量独立、服从高斯分布：

$$
h\sim\mathcal N(\mu^-,P^-),\qquad
z=h+\epsilon,\quad \epsilon\sim\mathcal N(0,R).
$$

贝叶斯更新为

$$
p(h\mid z)\propto
\exp\left[-\frac12
\left(\frac{(h-\mu^-)^2}{P^-}+
\frac{(z-h)^2}{R}\right)\right].
$$

把关于 $h$ 的二次项配方，得到

$$
P^+=\left((P^-)^{-1}+R^{-1}\right)^{-1}
=\frac{P^-R}{P^-+R},
$$

$$
\mu^+=P^+\left(\frac{\mu^-}{P^-}+\frac zR\right)
=\frac{R\mu^-+P^-z}{P^-+R}.
$$

也可以写成卡尔曼增益形式：

$$
K=\frac{P^-}{P^-+R},\qquad
\mu^+=\mu^-+K(z-\mu^-).
$$

新测量越精确，即 $R$ 越小，权重越大。当前代码使用这类标量更新，并设置最小方差防止无限趋近零。

**但异常门控需要按实际实现理解：**当前 CUDA 代码计算的是 $|z-\mu^-|/\sqrt R$，超阈值时直接替换旧值；它不是标准的 $|z-\mu^-|/\sqrt{P^-+R}$ 创新门控，也不是“超阈值就丢弃测量”。见 [compute_elevation](../src/uneven_mapping/terrain_analyzer/src/TerrainBank.cu#L498)。

因此这是一种带替换规则的工程高程融合，不能把整段流程都称为严格最优卡尔曼滤波。连续点云之间也存在相关性，方差下界并不能完全解决重复计数。

### 3.4 上游已在做什么 SEB 分析？

上游 CUDA 节点在不同航向的椭圆支撑邻域中计算协方差，取最小特征值对应的法向，并计算曲率代理和定向倾角，再得到基线风险 $R_0(x,y,\theta)$。

当前基线曲率代理为 $3\lambda_{\min}/\sum\lambda$，与后述 Feature-SGP 使用的 $\lambda_{\min}/\sum\lambda$ 还差一个比例，不能混用阈值。其几何阈值也与下游矩形足迹阈值不同。

尤其是当前 GPU 配置的 max_curvature=1000，对理论上约为 0 至 1 的该代理非常宽松，不能误认为基线曲率支路提供了很强的几何保护。这是应继续核校的参数含义，不在本文中修改。

代码还生成最近邻式 inpainted 层用于基线几何分析，但原始 elevation 的未知 NaN 仍保留。下游默认依据 elevation 而不是 inpainted 判断是否观测，避免把这种补值当成真实测量。见 [上游 SE(2) 几何](../src/uneven_mapping/terrain_analyzer/src/TerrainBank.cu#L732)。

## 4. 可选 Feature-SGP：由观测预测高程

### 4.1 为什么用高斯过程？

只对高程插值能给出一个高度，但通常不能清楚表达“周围没有足够观测时我有多不确定”。GP 把未知地面看成随机函数：

$$
f(u)\sim\mathcal{GP}(m(u),k(u,u')).
$$

其定义是：任意有限位置集合上的函数值联合服从多元高斯。核函数规定两个输入处的地形变化应当多相似。

当前不是直接预测车辆风险，而是先预测地面高程。高程预测和通行风险之间还隔着车辆几何分析。

### 4.2 为什么叫“Feature-based”？

对观测点 $p_i=(x_i,y_i,z_i)$，当前代码按 XY 近邻形成邻域，计算

$$
\bar p_i=\frac1k\sum_{j\in\mathcal N_i}p_j,\qquad
C_i=\frac1{k-1}\sum_{j\in\mathcal N_i}
(p_j-\bar p_i)(p_j-\bar p_i)^\top.
$$

令 $0\le\lambda_1\le\lambda_2\le\lambda_3$，定义

$$
\kappa_i=\frac{\lambda_1}{\lambda_1+\lambda_2+\lambda_3+\varepsilon},
\qquad
g_i=\frac1k\sum_{j\in\mathcal N_i}|z_j-z_i|.
$$

平面上的法向变化很小，所以最小特征值接近零；点云离散或存在非平面结构时，该比例通常增加。

这里的 $\kappa_i$ 是表面变化代理，不是严格的微分几何曲率。$g_i$ 单位为米，没有除以水平距离，所以也不是严格的坡度梯度 $\partial h/\partial x$。

模型输入为四维特征

$$
u_i=\left[
x_i-\bar x,\ y_i-\bar y,\
\frac{\kappa_i-\bar\kappa}{s_\kappa},\
\frac{g_i-\bar g}{s_g}
\right].
$$

当前对特征做标准化，不是论文中完整的 PCA 去相关或特征优先保留策略。采样主要采用空间分层。见 [特征计算](../src/fsgp_bgk_se2/fsgp_bgk_se2/feature_sgp.py#L128)。

### 4.3 未知位置没有高度，如何构造查询特征？

查询只接收 $(x_*,y_*)$，不允许接收待预测的真实 $z_*$。

在已观测特征点中找近邻，以归一化反距离权重插值：

$$
w_j=\frac{1/\max(d_j,\varepsilon)}
{\sum_l1/\max(d_l,\varepsilon)},\qquad
\kappa_*=\sum_jw_j\kappa_j,\quad
g_*=\sum_jw_jg_j.
$$

然后组成 $u_*$。因此查询特征也不是真实已知值，而是由观测估计的特征。当前没有完整传播这一步特征插值的不确定性，这是模型边界之一。

### 4.4 先拟合大趋势，再预测细节

当前实现先从观测拟合一个平面：

$$
m_0(x,y)=a(x-\bar x)+b(y-\bar y)+c.
$$

令 $T_i=[x_i-\bar x,y_i-\bar y,1]$，正则最小二乘解为

$$
\hat\beta=(T^\top T+\lambda I)^{-1}T^\top z.
$$

再令

$$
r_i=z_i-m_0(x_i,y_i),\quad
s_z=\max(\operatorname{std}(r),0.05\text{ m}),\quad
y_i=r_i/s_z.
$$

GP 学习的是归一化残差 $y_i$，最终加回平面。这样不必让有限诱导点同时承担整体坡面和所有局部起伏。

平面完全从输入观测拟合，不是人为添加的车底平面或虚构地面点。

### 4.5 当前 RQ 核的含义

代码使用带四维 ARD 尺度的 Rational Quadratic 核：

$$
k(u,u')=\sigma_f^2
\left(1+\frac{d^2(u,u')}{2\alpha}\right)^{-\alpha},
\qquad
d^2=\sum_{d=1}^{4}\frac{(u_d-u'_d)^2}{\ell_d^2}.
$$

相近位置、相似几何特征的点更相关。ARD 表示四个维度各自有尺度，不要求“1 米位置差”和“1 单位标准化特征差”产生同等影响。

XY 尺度初值为 0.7 米，两个标准化特征尺度初值为 2；这些是可优化初值，不是训练后必然保持不变的参数。

### 4.6 从高斯联合分布推导后验

先看没有稀疏近似的标准公式。这里的 $\mathbf y$、$f$ 是归一化残差及其模型，物理高程 $h$ 要在之后加回趋势、恢复尺度。设训练值满足

$$
\mathbf y=\mathbf f_X+\epsilon,\quad
\epsilon\sim\mathcal N(0,\sigma_n^2I),\quad
A=K_{XX}+\sigma_n^2I.
$$

训练值与一个查询函数值联合为

$$
\begin{bmatrix}\mathbf y\\f_*\end{bmatrix}
\sim\mathcal N\left(
\begin{bmatrix}\mathbf m_X\\m_*\end{bmatrix},
\begin{bmatrix}A&k_{X*}\\k_{*X}&k_{**}\end{bmatrix}
\right).
$$

对这个联合高斯分布作条件化，或对其二次型配方，即得到

$$
\mu_*=m_*+k_{*X}A^{-1}(\mathbf y-\mathbf m_X),
\qquad
v_*=k_{**}-k_{*X}A^{-1}k_{X*}.
$$

第一式表示“先验均值加上观测修正”；第二式表示“先验方差减去观测解释掉的部分”。若预测一条新的带噪测量，再加 $\sigma_n^2$。

实现应通过线性求解而不是显式构造大矩阵逆来计算这些量。

### 4.7 当前稀疏实现不是任意 SVGP

源码是 GPyTorch 的 ExactGP、InducingPointKernel 和 ExactMarginalLogLikelihood 组合。检查本地 GPyTorch 1.14.3 实现可见，它含诱导核低秩近似、训练迹修正和预测对角修正；不能将其说成使用自由变分分布的普通 SVGP。

设训练输入 $X$ 有 $N$ 点，诱导输入 $U$ 有 $M$ 点：

$$
Q_{XX}=K_{XU}K_{UU}^{-1}K_{UX}.
$$

其作用是用少量代表点近似相关结构，不是把未知高度直接赋给诱导点作为“真值”。

忽略数值抖动、按样本数归一化等细节，其训练目标可理解为 SGPR 下界：

$$
\mathcal L=
\log\mathcal N(\mathbf y;\mathbf m_X,Q_{XX}+\sigma_n^2I)
-\frac{\operatorname{tr}(K_{XX}-Q_{XX})}{2\sigma_n^2}.
$$

迹项惩罚低秩近似没有解释的方差。代码用 Adam 最大化相应目标，而不是只拟合一个确定性的插值曲面。

Woodbury 恒等式给出

$$
(Q_{XX}+\sigma_n^2I)^{-1}
=\sigma_n^{-2}I-\sigma_n^{-4}K_{XU}
\left(K_{UU}+\sigma_n^{-2}K_{UX}K_{XU}\right)^{-1}K_{UX}.
$$

主要矩阵求解由 $N$ 维转向 $M$ 维。稀疏 GP 核心运算通常约为 $O(NM^2+M^3)$，但当前完整调用还包括近邻、特征、局部预测和 CPU/GPU 拷贝，不能用该复杂度替代实测耗时。

当前预算上限为：全局特征参考 1024、训练 384、诱导 96；训练默认 12 步。局部条件分支另可保留最多 16384 个观测参考，这几个数字不能混成一个“点云点数”。

预测时也不是简单地用 $K_{**}=Q_{**}$：本地库默认启用 SGPR 对角修正，补上非负的 $k_{**}-Q_{**}$，避免把低秩近似未表示的先验方差全部丢掉。

进一步核对本机预测缓存可见，默认对角修正也进入训练协方差缓存。因此，对单个不与训练输入整体重合的查询集，略去数值误差时，将实际核心预测写成

$$
D_X=\operatorname{diag}\bigl(\max(k(u_i,u_i)-Q_{ii},0)\bigr),\qquad
A_{\mathrm{pred}}=Q_{XX}+D_X+\sigma_n^2I,
$$

$$
Q_{*X}=K_{*U}K_{UU}^{-1}K_{UX},
$$

$$
\mu_{\mathrm{SGPR}}=m_*+Q_{*X}A_{\mathrm{pred}}^{-1}(\mathbf y-\mathbf m_X),
\quad
v_{\mathrm{SGPR}}=k_{**}-Q_{*X}A_{\mathrm{pred}}^{-1}Q_{X*}.
$$

若关闭该修正，则相应 $D_X=0$。训练目标中的低秩项、评估模式的修正以及查询先验方差是三个不同环节；不能拿一条无噪声诱导插值公式代替全部实现。带 $D_X$ 时仍可对对角矩阵 $D_X+\sigma_n^2I$ 使用 Woodbury 求解。

本机小规模矩阵核验中，显式条件均值、方差与库返回值的差约为 $10^{-6}$ 量级；这项核验用于核对公式，不是新增地形性能实验。

诱导点从训练输入中初始化，但四维位置随后可以参与优化，学习后不必仍与某个原始点重合。它们不是要求雷达每帧只留下 96 个回波。

### 4.8 最终高程方差还加了什么？

全局分支将归一化 GP 结果恢复到米：

$$
\hat h_g=m_0+s_z\mu_g.
$$

当前输出方差还包含趋势估计和未解析局部变化：

$$
v_g=\max\left[
s_z^2v_{\mathrm{SGPR}}
+t_*^\top C_\beta t_*
+(\gamma d_g s_{\mathrm{unresolved}})^2,\
\sigma_{\mathrm{ref}}^2
\right],
$$

$$
C_\beta=(T^\top T+\lambda I)^{-1}
\max\left(\frac{\sum_i r_i^2}{n-3},\sigma_{\mathrm{ref}}^2\right).
$$

这里 $d_g$ 是到全局特征参考的最近距离，$\gamma$ 默认 0.5。未解析变化尺度由观测残差近邻差异估计，不使用隐藏真值。

这些附加量使方差不至于因稀疏模型局部过于自信而过小，但它们包含工程启发式，不是对全部误差源的严格联合贝叶斯推导。

### 4.9 关于“噪声是 2 cm”的必要更正

noise_std_m=0.02 参与似然噪声的初始化、局部模型的固定噪声项和若干方差下界。**它不意味着全局 GP 似然噪声训练全程固定为 2 cm。**

当前 optimizer 使用 model.parameters()，包含 likelihood.noise_covar.raw_noise；核尺度、核幅值、RQ 形状参数、诱导位置和残差常数均值也参与优化。输出 observation_var 加的是学习后的全局似然噪声对应的物理方差。

因此，先前实验说明中若把全局 GP 噪声模型表述为“始终固定 2 cm”，应按此处更正。这里不回填或改写已经冻结的实验数组。

## 5. 局部条件分支：为何不只依赖全局稀疏模型？

诱导点预算有限时，全局模型可能抹平小范围起伏。当前实现额外使用局部观测邻域，并在几何合格时**替换**全局结果，而不是将两者作为独立证据再相乘。

### 5.1 二次趋势和局部残差 GP

以查询点为原点，邻域 XY 偏移除以局部尺度 $s$，得到 $(u_j,v_j)$。构造

$$
B_j=[1,u_j,v_j,u_j^2,u_jv_j,v_j^2],\qquad
w_j=(u_j^2+v_j^2+0.2)^{-1}.
$$

加权拟合：

$$
G=B^\top WB+\Lambda,\qquad
\hat\beta=G^{-1}B^\top Wz,\qquad
e=z-B\hat\beta.
$$

在残差上使用 RBF 核。令 $a$ 为局部残差幅值，$C=aK+\sigma_{\mathrm{ref}}^2I$，$k$ 为邻域到查询原点的核向量，则

$$
\hat h_\ell=\hat\beta_0+k^\top C^{-1}e,
\qquad
v_{\ell,0}=\max\{a-k^\top C^{-1}k+a(G^{-1})_{00},
\sigma_{\mathrm{ref}}^2\}.
$$

这里的 RBF 长度尺度 0.7 作用于归一化局部坐标，不可直接解释成全局固定的 0.7 米。

### 5.2 留一残差用于防止虚假自信

令加权拟合杠杆值为 $l_j=w_jB_jG^{-1}B_j^\top$，当前用

$$
e_j^{\mathrm{LOO}}\approx
\frac{e_j}{\max(1-l_j,0.15)},\qquad
v_\ell=v_{\ell,0}+
\max\left(\operatorname{mean}[(e^{\mathrm{LOO}})^2]-\sigma_{\mathrm{ref}}^2,0\right).
$$

这样即使局部拟合看起来很好，也不轻易把区间缩成零。由于有正则和截断，这是留一误差启发式，不应声称为精确无偏的泛化方差。

### 5.3 何时允许使用局部结果？

局部点需包围查询、邻域不能太大、最近观测不能太远、XY 几何不能近似退化为直线，且预测方差须通过门控。

实现利用邻域方位角最大空隙不超过 $\pi$ 来检查包围关系，并检查二维几何矩阵的特征值比例。否则保留全局结果。

因此，当前最终高程是“全局稀疏 GP 加条件局部替换及方差保护”，不是单一、全空间统一的纯 GP 后验。见 [局部模型](../src/fsgp_bgk_se2/fsgp_bgk_se2/feature_sgp.py#L228)。

## 6. 支撑、接受、补全：三个不同概念

### 6.1 有预测值不代表被接受

设全部输入观测的 XY 凸包为 $\mathcal H$，最终最近观测距离为 $d$。当前定义

$$
S(x)=\mathbf 1[x\in\mathcal H,\ d\le1.2\text{ m}],
\qquad
A(x)=S(x)\mathbf 1[\sqrt{v_h(x)}\le0.10\text{ m}].
$$

局部条件分支开启时，这里的最近距离可来自更密的局部参考，而不是上节 $d_g$ 的同一集合。

凸包判断使用 float64 物理查询，避免 float32 舍入将实际包外点移动到包内。当前实现可在包外给出诊断均值，但不接受这种包外外推。

### 6.2 可信区间怎样理解？

以高斯近似解释潜在高程时，名义 95% 区间为

$$
[\hat h-1.96\sqrt{v_h},\ \hat h+1.96\sqrt{v_h}].
$$

若标准差为 2 cm，总宽约为 7.84 cm。它是竖直高程范围，不是 XY 地图宽度。

名义 95% 不保证实际一定覆盖 95%：模型偏差、特征插值、混合分支和噪声假设都可能破坏校准。需要用独立参考检验实际覆盖和区间宽度，不能只看方差数值小。

### 6.3 两个可选导航接入方式

实测修正只在已观测且通过门控处使用

$$
\omega=\frac{V_{\mathrm{obs}}}{V_{\mathrm{obs}}+v_h},\qquad
H'=(1-\omega)H_{\mathrm{obs}}+\omega\hat h.
$$

形式类似精度加权，但 GP 也来自这些观测，二者相关。当前没有按独立证据相乘进一步压缩方差，仍不能称其为严格独立贝叶斯融合。

未知补全要求：原来未知、局部模型合格、预测已接受、标准差不超过 4 cm、最近观测不超过 0.35 m，并且不覆盖受保护的高风险区域。补入的方差不低于 $0.0004\text{ m}^2$。

这两个开关默认都关闭。它们的作用、输入掩码及保护规则见 [节点接入](../src/fsgp_bgk_se2/fsgp_bgk_se2/fsgp_bgk_se2_node.py#L328) 和 [有界补全](../src/fsgp_bgk_se2/fsgp_bgk_se2/supported_completion.py#L6)。

补全通过高程门控，并不能证明它不会抹平窄脊、台阶或漏掉遮挡障碍。本轮配对实验已经发现新增危险漏判，所以暂不能把默认关闭说成一个无关紧要的性能选项。

## 7. 当前 BGK 适配究竟算什么？

### 7.1 空间核均值

对邻域内有效格，用高斯权重

$$
w_{ij}=\exp\left(-\frac{\|x_i-x_j\|^2}{2\sigma_b^2}\right),
\quad
\alpha_{ij}=\frac{w_{ij}O_j}{\sum_lw_{il}O_l},
\quad
\bar H_i=\sum_j\alpha_{ij}H_j.
$$

默认核尺度约 0.45 m、截断半径约 1.2 m。实现按离散栅格偏移形成核，边界截断后重新归一化。

### 7.2 为什么还要计算一个局部方差？

代码使用

$$
V_{\mathrm{BGK},i}
=\underbrace{\sum_j\alpha_{ij}H_j^2-\bar H_i^2}_{\text{邻域高度离散}}
+\underbrace{\sum_j\alpha_{ij}V_j}_{\text{输入测量方差}}.
$$

可以用混合分布的全方差公式理解其矩：

$$
\operatorname{Var}(Z)
=E[\operatorname{Var}(Z\mid J)]
+\operatorname{Var}(E[Z\mid J]).
$$

这与代码两项对应，但不表示代码真的保存了一个完整混合概率分布。

**重要区别：它不是“同一格后验均值的纯估计方差”。**即使一个坡面被无噪声、密集观测，邻域高度仍有变化，因此第一项也可非零。不能把它与 GP latent_var 或 KF var 当成完全同一种量。

例如等权混合高度 0 m 和 0.2 m、各自方差 $0.0004\text{ m}^2$，得到均值 0.1 m、此处的方差 $0.0104\text{ m}^2$。它与把两个独立测量融合成同一高度后的后验方差不是同一道计算。

见 [核平滑实现](../src/fsgp_bgk_se2/fsgp_bgk_se2/bgk_seb.py#L574)。

### 7.3 为什么不能称为原论文完整时空 BGK？

参考论文第三节还定义了历史可通行性、时间衰减和历史不确定性权重。例如其历史风险更新结构包含

$$
\omega_t=e^{-\lambda(t-t_0)},\qquad
\omega_\sigma=(v_{t-1}+\varepsilon)^{-1},\qquad
M_t=\frac{\omega_t\omega_\sigma M_{t-1}+M_{\mathrm{pre}}}
{\omega_t\omega_\sigma+1}.
$$

当前下游适配器没有完整实现这套风险历史缓冲和时序后验更新。上游 KF、滚动地图和动态清理具有历史状态，但不等于已经实现了论文的时空 BGK。

因此更准确的描述是“BGK 风格的空间核平滑及离散程度估计”，不是原论文的全部算法复现。

### 7.4 二维 BGK 风险只是一个辅助输出

当前还从平滑高程差分得到坡度、邻格高差，并形成

$$
R_{\mathrm{BGK}}
=0.45s_{\mathrm{slope}}+
0.35s_{\mathrm{step}}+
0.20s_{\mathrm{var}}.
$$

bgk_traversability 为 $1-R_{\mathrm{BGK}}$。**这个二维分数没有作为上述权重形式的一个独立项直接加入最终方向风险**；最终方向融合使用的是上游姿态风险、下游 SEB 足迹风险及局部方差分数。

同样，RViz 可以展示 bgk_mean，但默认足迹几何使用未平滑的输入高程。显示面、预测面、实际判定用的面需要分别理解。

## 8. SEB：由地面变成“这辆车、这个方向”的风险

### 8.1 先旋转车辆足迹

当前下游使用车体系矩形足迹：车体前后各 0.42 m、半宽 0.313 m，再加 0.10 m 边距。对每个足迹采样偏移 $(\xi_j,\eta_j)$，航向为 $\theta$ 时：

$$
\begin{bmatrix}x_j\\y_j\end{bmatrix}
=\begin{bmatrix}x\\y\end{bmatrix}
+\begin{bmatrix}\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}\xi_j\\\eta_j\end{bmatrix}.
$$

实际查询还要取整到栅格。拟合时必须用取整后真正查到的位置反算车体系坐标，而不是仍用理想偏移；否则平面坡也可能产生伪残差。

默认 5×5 采样再加 9 个边界偏移，共 34 个记录，部分本来重复，取整后也可能重复。不能称为 34 个独立测量，更不是 34 个真实轮胎接触点。

### 8.2 足迹平面拟合的推导

在有效采样位置拟合

$$
z_j\approx a\xi_j+b\eta_j+c,\qquad
\min_{a,b,c}\sum_j(z_j-a\xi_j-b\eta_j-c)^2.
$$

对 $c$ 求导得 $c=\bar z-a\bar\xi-b\bar\eta$。代回并中心化，令

$$
A=\sum(\xi_j-\bar\xi)^2,\quad
B=\sum(\xi_j-\bar\xi)(\eta_j-\bar\eta),\quad
C=\sum(\eta_j-\bar\eta)^2,
$$

$$
d=\sum(\xi_j-\bar\xi)(z_j-\bar z),\qquad
e=\sum(\eta_j-\bar\eta)(z_j-\bar z).
$$

剩余正规方程为

$$
\begin{bmatrix}A&B\\B&C\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}
=\begin{bmatrix}d\\e\end{bmatrix}.
$$

若 $D=AC-B^2>0$，则

$$
a=\frac{Cd-Be}{D},\qquad b=\frac{Ae-Bd}{D}.
$$

这就是代码按每个位置、每个航向并行累加的基本计算；不足三次有效采样或几何矩阵退化时不能正常拟合。

### 8.3 从平面得到纵坡与横坡

车体系沿前后方向的切线为 $(1,0,a)$，其倾角为 $\arctan|a|$；左右方向同理：

$$
\hat\phi=\arctan|a|,\qquad \hat\rho=\arctan|b|.
$$

当前代码用它们作为纵坡、横坡角代理。二者不应被直接称作含悬架、轮胎接触的真实车身欧拉角；在一般复合坡上也不是完整刚体姿态求解。

对一般地面梯度 $\nabla h=(h_x,h_y)$，近似有

$$
a=h_x\cos\theta+h_y\sin\theta,\qquad
b=-h_x\sin\theta+h_y\cos\theta.
$$

所以同一位置只改变 $\theta$，就会改变纵横坡分量。

### 8.4 为什么不把普通坡当成台阶？

先去掉足迹平面的坡度：

$$
e_j=z_j-a\xi_j-b\eta_j,\qquad
\Delta_{\mathrm{res}}=\max_j e_j-\min_j e_j.
$$

这里计算的是残差极差，不是残差标准差。理想平面即使整体较陡，去平面后极差仍接近零；突起、坑、脊和台阶会留下非平面残差。

此外还保留平滑前的实测跳变保护。在一个方向上，若四个相邻高程都有效，使用

$$
J_{\mathrm{edge}}
=(z_1-z_0)-\tfrac12[(z_0-z_{-1})+(z_2-z_1)].
$$

线性坡面和理想二次曲面的连续增量会被抵消；孤立高度跳变则被保留。不具备四点支持时，退回用平滑面坡度补偿相邻实测高差。

这类保护减少“平滑把台阶抹掉”的风险，但有限分辨率和有限足迹采样仍可能漏掉很窄的障碍。

### 8.5 将物理量转换为工程风险

统一软评分函数为

$$
s(v;v_f,v_h)=
\operatorname{clip}\left(\frac{v-v_f}{v_h-v_f},0,1\right).
$$

$v_f$ 是低风险范围上界，$v_h$ 是该指标评分达到 1 的阈值。它是一种人为定义的归一化，不是概率推导。

当前默认主要阈值为：

| 指标 | 低风险阈值 | 评分到 1 的阈值 |
| --- | --- | --- |
| 纵坡角 | 0.08 rad | 0.55 rad |
| 横坡角 | 0.08 rad | 0.45 rad |
| 跳变/残差极差 | 0.04 m | 0.18 m |
| BGK 局部方差 | 0.01 m² | 0.20 m² |

方差的 0.01 m² 对应标准差 0.1 m，不是标准差 1 cm；单位必须区分。

设有效采样记录比例为 $\eta$，最低要求 $\eta_{\min}=0.55$，则

$$
r_{\mathrm{sup}}=
\operatorname{clip}\left(\frac{\eta_{\min}-\eta}{\eta_{\min}},0,1\right).
$$

这是采样记录有效比例，不是严格的足迹面积覆盖率，也不是支撑概率。

综合足迹风险为

$$
R_{\mathrm{SEB}}=
\max(r_{\mathrm{pitch}},r_{\mathrm{roll}},
r_{\mathrm{step}},r_{\mathrm{sup}}).
$$

使用最大值是为了不让某一项严重问题被其他低风险项平均稀释。

### 8.6 一个方向相关的具体例子

仍取 $h(x,y)=0.5x$，坡角约 $\arctan(0.5)=26.57^\circ$。

车头沿 X 时，纵坡约 0.464 rad、横坡约 0；车头沿 Y 时二者交换。纵坡上限 0.55 rad、横坡上限 0.45 rad，因此沿 X 和横着走不能给同一风险。

这些是连续方向的解释；实际 36 个航向格使用格中心角，通常与严格的 0°、90°相差半格，判断会存在离散误差。

还要注意：当前几何保护在单项归一化风险达到 0.90 时就将最终风险置为 1，早于上表的名义阈值。例如横坡约 $0.08+0.90(0.45-0.08)=0.413$ rad 就触发保护。

为隔离下游效应，以该平面、完整观测、每格 $0.0004\text{ m}^2$ 输入方差及 $R_0=0$ 做 CPU 小算例，当前代码给出：

| 实际航向格中心 | 纵坡评分 | 横坡评分 | 最终风险 |
| --- | --- | --- | --- |
| 5° | 约 0.813 | 0 | 约 0.385 |
| 95° | 0 | 1 | 1 |

这只是原理算例，不是线上完整链路成绩；加入实际基线、遮挡、回波噪声后数值还会改变。

见 [足迹和定向风险](../src/fsgp_bgk_se2/fsgp_bgk_se2/bgk_seb.py#L702)。

## 9. 最终融合：不能把硬障碍平均掉

### 9.1 软融合与基线下界

默认先计算

$$
\tilde R=
\operatorname{clip}(0.45R_0+0.45R_{\mathrm{SEB}}+0.10r_{\mathrm{var}},0,1),
\qquad R_1=\max(\tilde R,R_0).
$$

$R_0$ 是上游的姿态风险，不是上节的二维 $R_{\mathrm{BGK}}$。

随后还有按顺序执行的保护，而不是只用这条加权式：

1. 纵坡、横坡或台阶单项风险达到 0.90，置 1。
2. 足迹平面不可解或有效采样不足，至少提升到 unknown_risk。
3. 中心格无有效高程，按 unknown_risk=0.65 处理。
4. 当前点云检测出碰撞柱体，则最后强制置 1，并使该碰撞格可被地图消费者识别。

因此“风险不低于基线”需要注明有效域：代码的未知中心分支会直接写 0.65，并非在所有无效格上都保证数值不低于原基线。

### 9.2 柱子为什么需要另一条证据链？

地面高程是 2.5D 表面，单凭平滑的地面无法完整表达柱体。

当前 vertical_columns 单独读取与地图同坐标系、时间差不超过 0.35 s 的当前点云，考察格内竖向跨度和相对已观测地面的上部回波；有可信实测地面时还允许邻格回波互相佐证。

它不使用 GP 隐藏高程当作已测地面，也不能让邻格有回波就任意膨胀出一圈碰撞格。命中后用 collision_risk 保留硬占据，避免经过软权重后被“平均成可通行”。

这仍是有高度范围、点数和近邻门槛的几何检测，不是完备三维碰撞判定。细柱、缺乏回波或不合格时间戳都可能导致保护未触发。

见 [立体障碍保护](../src/fsgp_bgk_se2/fsgp_bgk_se2/obstacle_columns.py#L6)。

### 9.3 为什么不能把风险 0.8 说成“80% 概率撞车”？

因为该值由阈值归一化、最大值、权重、基线下界和硬保护共同形成，并没有训练为实际通行事件的条件概率。

严格的通行概率需要先明确随机变量和事件，例如

$$
P\bigl(|\phi|<\phi_{\max},
|\rho|<\rho_{\max},
\Delta_{\mathrm{res}}<d_{\max},
\text{无碰撞}\mid\text{观测}\bigr),
$$

再传播联合不确定性并检验概率校准。当前工程没有完整做这件事，所以应称“航向相关风险/代价地图”，而不是已经校准的通行概率地图。

## 10. 地图如何真正交给规划控制？

增强消息仍是 SE2Grid：

| 层 | 用途 |
| --- | --- |
| elevation / var | 上游测量估计；可选补全开启时另有写回 |
| valid | 数据是否可用；碰撞观测也需被消费者识别 |
| bgk_mean / bgk_var | 平滑表面和局部离散程度 |
| risk | 多航向的最终风险 |
| collision_risk | 当前立体障碍硬占据 |
| gp_mean / gp_latent_var 等 | 仅开启 GP 时产生的预测诊断 |

下游诊断开关默认关闭，所以部分 seb_* 分解层默认不发布。图像颜色不是算法本身，规划读取消息数值和有效性。

默认规划启动器读取增强地图，使用 risk 的方向查询，并把 collision_risk 作为二维硬障碍层。这样不应把任一航向危险都无条件铺成二维硬障碍，也不应在已经做过足迹判断的风险上重复膨胀。

### 10.1 必须区分三种不同阈值

| 位置 | 当前值 | 含义 |
| --- | --- | --- |
| 离线地形风险评价 | 0.58 | 报告中将分数二分类、统计 FNR/FPR 的阈值 |
| 规划二维硬代价 | risk_hard_threshold=0.58 | 默认增强启动器在 collision_risk 上使用，不能直接当成所有方向风险的统一门限 |
| 规划方向风险硬拒绝 | se2_hard_threshold=0.999 | 当前代码默认值，配合启动器 se2_risk_lethal=true |
| 下游几何保护 | geometry_guard_risk=0.90 | 对单项几何评分提前保护，先把 risk 置成 1 |

所以，离线用 0.58 算出的漏判率，不能不经核对就解释为“部署规划器以同一门限拒绝后的实际漏判率”。当前部署、评价阈值并不完全统一，这是解释实验时必须披露的差异。

未知格还受 valid、allow_unknown 和有限机器人周边保护逻辑管理，不是简单地把 0.65 与 0.999 比较后就允许驶入。

见 [增强规划启动器](../src/planner/terrain_nav/launch/fsgp_bgk_terrain_nav.launch.py) 和 [方向风险查询](../src/planner/terrain_nav/src/terrain_nav_node.cpp#L1215)。

## 11. 动态更新与“时空模型”不是同一概念

上游滚动高程持续更新，默认还有射线自由空间证据清理：当当前射线穿过旧高物体位置、该格没有当前回波，且附近存在足够低地面支持时，达到计数门槛后以邻域地面替换旧高度。

当前 CUDA 计数并非严格要求连续帧：有新点或未知格会重置，没有新自由证据时并不总是清零。因此更准确地说是达到一定次数的自由射线支持，而不是承诺连续三帧或固定 0.3 秒一定清除。

这不是语义物体跟踪，也不是完整原论文时序 BGK；它同样不等于 FAST-LIO 累计点云中的动态残影已经被清理。

GP 还存在拟合周期和查询缓存。缓存是避免重复计算，不是新的统计观测或后验时序融合。默认拟合周期约 1 秒，预测层可能相对当前扫描有年龄，不能假定是实时同步的完整世界。

## 12. 哪些部分在 GPU，哪些仍在 CPU？

| 部分 | 当前实现 |
| --- | --- |
| 上游高程、基线几何与方向风险 | 原生 CUDA TerrainBank |
| 下游空间核和方向足迹扫描 | CuPy 自定义 CUDA kernel；有 NumPy 托底 |
| 全局 GP 特征、核与线性代数 | PyTorch / GPyTorch，可选 CUDA |
| 局部邻域树、凸包检查 | SciPy / NumPy CPU |
| 局部批量线性求解 | PyTorch，可选 CUDA |
| ROS 消息、缓冲顺序转换与数据拷贝 | 主要 CPU，并有主机/显卡传输 |

这里的 GPU 地形前端和下游 CuPy 后端是两套实现/开关，不能只检查一个参数就断言所有步骤都在 GPU。

CPU 和 GPU 后端服务同一接口，但上游 CPU 版和原生 CUDA 版并非每一处统计更新都保证位级等价。下游风险组件已有 CPU/GPU 数值一致性测试，不能由此推断整个线上系统完全相同。

最近冻结批次的风险组件在 104×104×36 工作负载下 GPU P95 约 4.60 ms；而 GP 完整拟合与查询的多来源 P95 约 138–234 ms。计时边界不同，不能据前者声称打开 GP 也能逐帧 10 Hz 全量拟合。

## 13. 一次处理的伪代码

    收到注册点云与匹配位姿
    更新滚动窗口，传播测量高程方差
    更新有效高程格；必要时按门控替换
    用自由射线证据处理部分旧高度
    计算上游基线方向风险 R0

    保存原始高程、方差、有效掩码
    如果开启 GP 且观测数足够：
        到周期则从有效高程格重新拟合
        必要时计算/复用预测
        产生均值、方差、支撑、接受诊断层
        若明确开启：修正实测或有界补全

    计算空间核均值与局部离散程度
    对每个 XY 格和每个航向：
        旋转并离散化足迹
        计算采样支持，拟合足迹平面
        计算纵横坡、去平面残差与实测跳变
        融合基线和不确定性，应用几何/未知保护

    以当前同坐标、近时间点云检测立体碰撞柱
    对命中格保留最终硬风险
    发布增强 SE2Grid，规划器按位姿查询

## 14. 最容易混淆的六个概念

1. GP 高程方差、测量高程方差和 BGK 局部方差不是同一种量。
2. Feature-SGP 的高度变化特征不是严格的地面梯度；其曲率代理也不是曲率真值。
3. 支撑通过不等于预测正确，预测正确也不等于车辆可通行。
4. 有历史地图不等于实现完整时空 BGK；GPU 加速不等于全链路低时延。
5. SEB 的采样支持不是轮胎接触仿真，也不是精确面积覆盖率。
6. 当前最终风险分数不是已校准概率，离线评价门限也不能自动替代部署门限。

## 15. 当前方法的准确表述

可以表述为：

> 在滚动测量高程与上游 SE(2) 基线风险上，接入可选的特征诱导点 GP 及有条件局部地形预测，通过空间核矩估计、定向车辆足迹分析和独立立体占据保护，生成用于规划的航向相关地形风险图。

暂不能表述为：

> 已完整复现 FSGP-BGK 的时空贝叶斯模型，并在 GPU 上生成经过概率校准、全场景安全的未知区域通行概率。

本说明没有将尚未解决的窄结构漏判、GP 接入新增漏判或遇障后的导航恢复问题写成已解决。若后续将其作为新算法论文，应进一步统一统计模型、评价/部署契约，并用独立几何和真实物理闭环验证，而不是只调整风险权重。

## 16. 源码与参考

| 对象 | 入口 |
| --- | --- |
| 默认参数 | [hunter_fastlio.yaml](../src/fsgp_bgk_se2/config/hunter_fastlio.yaml) |
| 地形模块启动与默认开关 | [terrain_analysis.launch.py](../src/fsgp_bgk_se2/launch/terrain_analysis.launch.py) |
| 原生 CUDA 前端 | [TerrainBank.cu](../src/uneven_mapping/terrain_analyzer/src/TerrainBank.cu) |
| 特征、诱导 GP、局部预测与接受门控 | [feature_sgp.py](../src/fsgp_bgk_se2/fsgp_bgk_se2/feature_sgp.py) |
| 核平滑、足迹与融合 | [bgk_seb.py](../src/fsgp_bgk_se2/fsgp_bgk_se2/bgk_seb.py) |
| ROS 接入、预测缓存、原始保护与输出 | [fsgp_bgk_se2_node.py](../src/fsgp_bgk_se2/fsgp_bgk_se2/fsgp_bgk_se2_node.py) |
| 有界补全 | [supported_completion.py](../src/fsgp_bgk_se2/fsgp_bgk_se2/supported_completion.py) |
| 实测立体障碍 | [obstacle_columns.py](../src/fsgp_bgk_se2/fsgp_bgk_se2/obstacle_columns.py) |
| 规划接口与风险查询 | [terrain_nav_node.cpp](../src/planner/terrain_nav/src/terrain_nav_node.cpp) |

论文参照为用户提供的 Real-time Spatial-temporal Traversability Assessment via Feature-based Sparse Gaussian Process 第三节、SEB-Naver 的地形表示与可通行性部分。对当前工程的描述以源码为准；GPyTorch 近似细节还核对了本机安装的 InducingPointKernel、InducingPointKernelAddedLossTerm 实现。

实际启动若覆盖了默认参数，还应以运行节点的参数为准，不能仅由功能包名称或脚本中的 GPU 字样判断开启了哪个算法分支。
