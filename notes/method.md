# 3 方法：面向 2018 潜变量世界模型（World Models）的受控动作敏感性探针

## 3.1 预备：世界模型与可隔离的动作通道

本文研究 Ha 与 Schmidhuber（2018）的潜变量世界模型。它由三个相互配合的模块构成——视觉模块 $V$、记忆模块 $M$ 与控制器 $C$。我们沿用它原有的分工，仅对记号作最小程度的统一，以便把本文的度量直接定义在其输出之上。本小节给出全部形式化定义；模型的具体取值（维度、层数、超参数等）不影响本节结论，统一留待 §4 与附录。

**观测与潜变量。** 记第 $t$ 个观测（一帧图像）为 $x_t\in\mathbb{R}^{c\times H\times W}$，记忆模块的循环隐状态为 $h_t\in\mathbb{R}^{d_h}$，潜变量为 $z_t\in\mathbb{R}^{d_z}$，动作向量为 $a_t\in\mathbb{R}^{d_a}$。

**视觉模块 $V$。** $V$ 是一个变分自编码器，把每一帧压缩为一个潜变量。其编码器给出高斯后验：

$$
q_\phi(z_t \mid x_t) = \mathcal{N}\big(z_t;\ \mu_\phi(x_t),\ \operatorname{diag}(\sigma_\phi^2(x_t))\big) \qquad \text{(1)}
$$

采样经由重参数化完成：

$$
z_t = \mu_\phi(x_t) + \sigma_\phi(x_t)\odot\epsilon, \quad \epsilon\sim\mathcal{N}(0, I_{d_z}) \qquad \text{(2)}
$$

解码器再由 $z_t$ 重建观测。之所以采用高斯后验而非确定性的编码器输出，是为了让潜变量携带一个可显式控制其方差的随机源——这一随机源正是 §3.3 中“采样噪声地板”得以定义的先决条件。

**记忆模块 $M$。** $M$ 以动作与潜变量为输入，建模转移分布 $p(z_{t+1}\mid z_t,h_t,a_t)$。它在每个时间步把输入拼接为 $u_t=[\,z_t;\,a_t\,]\in\mathbb{R}^{d_z+d_a}$，循环隐状态按式 (3) 递推：

$$
h_t = f_\theta\big(h_{t-1},\ [\,z_t\,;\,a_t\,]\big) \qquad \text{(3)}
$$

隐状态再经一个线性头输出 $K$ 个高斯分量的混合参数（权重、均值与方差），由此构成混合密度：

$$
p_\theta(z_{t+1}\mid z_t,h_t,a_t) = \sum_{k=1}^{K}\pi_{t,k}\ \mathcal{N}\big(z_{t+1};\ \mu_{t,k},\ \operatorname{diag}(\sigma_{t,k}^2)\big) \qquad \text{(4)}
$$

其中每个分量取**对角**协方差，即 $\mu_{t,k},\sigma_{t,k}\in\mathbb{R}^{d_z}$ 逐维给出均值与标准差。之所以采用**混合**而非点估计，是为了显式表达转移的多模态不确定性——这使“关闭采样、直接取混合均值”这一确定性动作效应（§3.3 中的 $D$）在数学上被良好定义。

**动作通道。** 式 (3) 表明，动作 $a_t$ 只通过输入拼接 $u_t$ 的最后 $d_a$ 个坐标进入 $M$；在循环网络首层的输入权重 $W_{\text{in}}$ 上，这恰对应一段固定的列切片。我们把该切片记为

$$
W_a := W_{\text{in}}\big[\,:,\ d_z \!:\! d_z{+}d_a\,\big]\subset\mathbb{R}^{d_h\times d_a}, \qquad \text{(5)}
$$

并称之为 $M$ 的**动作通道**。$W_a$ 的定义是本文全部控制得以成立的结构性前提：正因存在这样一个可被单独指认的列切片，才可能在训练中令其“留空”（§3.4）并单独测量其效应——这一可分离性正是现代视频 / JEPA 世界模型所不具备的。

**控制器 $C$。** $C$ 是一个线性映射：读入 $[\,z_t;\,h_t\,]$、输出动作；为使动作落在环境允许的物理范围内，其线性输出先经 $\tanh$ 压缩，再线性映射到区间 $[\,\ell,u\,]$：

$$
a_t = \ell + \tfrac{1}{2}\big(\tanh(W_c[\,z_t;\,h_t\,]+b_c)+1\big)\odot(u-\ell), \qquad \text{(6)}
$$

其中 $\ell,u\in\mathbb{R}^{d_a}$ 为逐维动作下界与上界（$\ell<u$）。$C$ 的参数以**无梯度黑箱优化（CMA-ES）**在真实环境中训练，与 $V$、$M$ 的梯度训练相互独立。需要强调，**$C$ 不参与本文的探针**：探针只作用于 $M$ 的动作通道，$C$ 的作用仅在于复现原流程并提供 §4 中的任务性能读数。

**训练次序。** 三个模块分阶段训练：先训练 $V$，将其冻结后编码出潜变量序列；再训练 $M$；最后以训练好的 $V$ 与 $M$ 为固定前端训练 $C$。这一解耦是本文得以把 $M$ 的动作通道单独隔离出来的前提。

## 3.2 世界模型的学习目标

$V$ 与 $M$ 分别由两个损失训练，且先后独立——这一分离使 $M$ 的动作通道不经过 $V$ 的梯度，是 §3.4 能对它单独做控制的前提。本小节定义这两个目标，并引入贯穿后文的采样温度。

**视觉模块的目标。** $V$ 的训练目标是标准变分自编码器目标，由重建项与正则项相加而成：

$$
\mathcal{L}_V(\phi) = \mathbb{E}_{q_\phi(z_t\mid x_t)}\big[-\log p_\phi(x_t\mid z_t)\big] + \beta\,D_{\mathrm{KL}}\big(q_\phi(z_t\mid x_t)\,\|\,p(z_t)\big), \qquad \text{(7)}
$$

其中 $p_\phi(x_t\mid z_t)$ 是解码器给出的观测似然，$p(z_t)=\mathcal{N}(0,I_{d_z})$ 是标准高斯先验，$\beta$ 为平衡两项的权重。重建项要求 $z_t$ 保留足以还原 $x_t$ 的信息，而 KL 正则项把 $z_t$ 约束为一个良态的高斯潜变量；若只有重建项，$z_t$ 会退化为 $x_t$ 的确定性编码。该约束同时给出后文所需的两点：$z_t$ 可被当作连续量做差分（§3.3 的位移 $\Delta z$ 建立于此），且自带一个可显式控制其方差的随机源，后者即采样噪声地板 $B$ 的来源。

**记忆模块的目标。** $M$ 以最大似然训练，等价于最小化转移分布的负对数似然。由式 (4) 的混合密度，单步项为

$$
\ell_{t}(\theta) = -\log\sum_{k=1}^{K}\pi_{t,k}\ \mathcal{N}\big(z_{t+1};\ \mu_{t,k},\ \operatorname{diag}(\sigma_{t,k}^2)\big), \qquad \text{(8)}
$$

而整段序列上的目标为各步之和：

$$
\mathcal{L}_M(\theta) = \sum_{t}\ \ell_{t}(\theta). \qquad \text{(9)}
$$

这里不是用一个点估计或平方误差去拟合 $z_{t+1}$，而是去拟合 $z_{t+1}$ 的整个条件分布；混合形式允许该分布多峰，以容纳同一动作下的多种可能走向。这样得到的是一整个分布而非一个点，因而“从分布中采样”与“取分布均值”两个操作都被明确定义，它们的差正是 §3.3 要比较的 $B$ 与 $D$。

**采样温度。** 训练得到的转移分布是未加温的（$\tau=1$）。为在推理时调节其随机性，我们对式 (4) 中每个分量的标准差作

$$
\tilde\sigma_{t,k} = \sigma_{t,k}\,\sqrt{\tau}, \qquad \text{(10)}
$$

$\tau=1$ 时退化为原分布。温度只改变从分布抽样时的噪声量，不改变分布的位置（均值）：$\tau$ 越大，采样噪声臂 $B$ 越大，而取均值的确定性臂 $D$ 不变。$D$ 与 $B$ 由此在 $\tau$ 轴上被分离，这是 §3.3 定义 $x_{\text{cross}}$、§4 做温度轴验证的根据。

## 3.3 动作敏感性比值 $D/B$

本节定义探针的读数 $D/B$。$M$ 的转移同时受两个来源驱动：一是动作（式 (3) 中经 $W_a$ 输入），二是模型自身的采样噪声（式 (2) 的 $\epsilon$、式 (4) 的混合抽样）。**动作依赖性的度量，取二者的相对大小，而非任何一方的绝对响应。**

**rollout 协议。** 探针在一条**自回归 rollout** 上进行。给定一段**共享的 warm-up 前缀**，各支臂得到逐位相同的初始状态；此后固定一个动作序列与一个采样种子 $\xi$，按式 (3)、式 (4) 逐步推演。采用全局时间下标，设 warm-up 消费前 $W$ 步的输入 $z_{0:W-1},a_{0:W-1}$，其末态记为 $h_{W-1}$；rollout 自

$$
\big(\hat z_W,\ \hat h_{W-1}\big)=\big(z_W,\ h_{W-1}\big)
$$

起，对 $t\ge W$ 依次作

$$
\hat h_t = f_\theta\big(\hat h_{t-1},\ [\,\hat z_t\,;\,a_t\,]\big),\qquad
\hat z_{t+1}\sim p_\theta\big(z_{t+1}\mid \hat z_t,\hat h_t,a_t\big), \qquad \text{(11)}
$$

直至视界 $T$。共享 warm-up 使得各支臂起点完全相同，因而其后的任何差异都可归因于**单一受控因子**（动作，或种子）。把式 (11) 的抽样替换为混合均值，则得到一条**确定性** rollout：

$$
\bar z_{t+1}=\sum_{k=1}^{K}\pi_{t,k}\,\mu_{t,k}. \qquad \text{(12)}
$$

**散度。** 两支 rollout 在视界 $t$ 的差异，以归一化空间中的相对位移度量：

$$
d_t\big(\hat z,\hat z^{\mathrm{ref}}\big)=\frac{\big\lVert \hat z_{t}-\hat z^{\mathrm{ref}}_{t}\big\rVert_2}{\mathbb{E}\,|\Delta z|}, \qquad \text{(13)}
$$

其中分母 $\mathbb{E}\,|\Delta z|$ 是**归一化潜空间中相邻帧位移的典型尺度**，全篇恒定。以它为分母，使读数成为与量纲无关的“相当于几步典型变化”，跨环境、跨 checkpoint 可比。

**三支臂。** 在同一初始状态上，三支臂各自只改动一个因子：

$$
\begin{aligned}
A &:= d_t\big(\mathcal{R}(a^{\mathrm{perm}},\xi),\ \mathcal{R}(a,\xi)\big),\\
B &:= d_t\big(\mathcal{R}(a,\xi'),\ \mathcal{R}(a,\xi)\big),\\
D &:= d_t\big(\mathcal{R}_{\det}(a^{\mathrm{perm}}),\ \mathcal{R}_{\det}(a)\big),
\end{aligned} \qquad \text{(14)}
$$

其中 $a^{\mathrm{perm}}$ 为动作序列的全局置换，$\mathcal{R}$ 为式 (11) 的随机 rollout，$\mathcal{R}_{\det}$ 为式 (12) 的确定性 rollout。三者分别度量：$A$——“只改动作”（置换动作、种子固定）引起的改变；$B$——“只换种子、动作不动”引起的改变，即**采样噪声地板**；$D$——**完全关闭采样**后“只改动作”引起的改变。需强调，$D$ **独立测得**——它以自己的确定性 rollout 为参照，而**不是**由 $A$ 减去某个采样项得到；这一独立性是后文把 $D$ 与 $B$ 在 $\tau$ 轴上分离的前提。

**比值估计量。** 记第 $i$ 条 rollout 上、$R$ 次采样重复产生的两支臂读数为 $D_i^{(r)}(t),B_i^{(r)}(t)$。先在**每条 rollout 内**对重复取臂均值，

$$
D_i(t)=\frac{1}{R}\sum_{r=1}^{R}D_i^{(r)}(t),\qquad
B_i(t)=\frac{1}{R}\sum_{r=1}^{R}B_i^{(r)}(t), \qquad \text{(15)}
$$

再取该 rollout 的比值；报告量为**逐 rollout 比值 $\,D_i/B_i\,$ 在 $N$ 条 rollout 上的均值**，

$$
\widehat{D/B}(t)=\frac{1}{N}\sum_{i=1}^{N}\frac{D_i(t)}{B_i(t)}, \qquad \text{(16)}
$$

而非均值之比 $\bar D/\bar B$。该形式对每条 rollout 等权，而均值之比会以各 rollout 的 $B_i$ 为权重，使个别高 $B_i$ 的 rollout 主导分母与结果。$\widehat{D/B}=1$ 因此有直接读数——在该视界与温度下，**动作效应恰等于采样噪声**。（注：式 (15) 是 rollout 内的臂均值、式 (16) 才是跨 rollout 的比值均值，两层聚合不同。）

**动作剂量与交叉剂量。** 为使读数落在可比较的刻度上，我们沿一条**动作保留比例** $\rho$ 构造一族剂量：在一份固定的潜变量序列上，令比例为 $\rho$ 的动作行保留为真实动作，其余行替换为与之无关的动作（该族剂量的生成细节见 §4）。每个剂量再由一个标量刻画，即**消融动作通道前后 §3.2 目标的增量**：以作用在 $W_a$ 上的乘子 $\lambda$（动作切片乘子）缩放该通道，取

$$
x:=\ \mathcal{L}_M^{\mathrm{ablated}}-\mathcal{L}_M^{\mathrm{intact}}, \qquad \text{(17)}
$$

其中 $\mathcal{L}_M^{\mathrm{ablated}}$ 为 $\lambda=0$（$W_a$ 置零）时的目标值，$\mathcal{L}_M^{\mathrm{intact}}$ 为 $\lambda=1$（通道完好）时的目标值；$x$ 的单位为 nats，且单调于动作被保留的程度。最后定义**交叉剂量** $x_{\text{cross}}$ 为使 $\widehat{D/B}=1$ 的剂量，由 $\widehat{D/B}$ 对 $x$ 的线性插值给出：

$$
x_{\text{cross}}:=\ \{\,x\ :\ \widehat{D/B}(x)=1\,\}, \qquad \text{(18)}
$$

$x_{\text{cross}}$ 即“需要多少动作信息，动作效应才能追平采样噪声”的读数，是 §4 温度轴验证的主量。

## 3.4 受控单元：三族构造与范数钉住

要把一个读数归因于动作通道，需**独立地**改变该通道的两个方面：它**是否承载动作信息**，以及它的**尺度**。本小节定义这两个控制。

**三族：改变动作通道的信息。** 记训练数据为 $\mathcal{D}=\{(z_t,a_t,z_{t+1})\}$，其中潜变量与动作按 episode 排成序列。我们构造三个**仅在动作信号上不同**的训练分布：

$$
\begin{aligned}
\mathcal{D}_{\mathrm{real}}  &:=\ \{(z_t,\ a_t,\ z_{t+1})\},\\[2pt]
\mathcal{D}_{\mathrm{shuf}}  &:=\ \{(z_t,\ \sigma(a)_{t},\ z_{t+1})\},\\[2pt]
\mathcal{D}_{\mathrm{noact}} &:=\ \{(z_t,\ \mathbf{0},\ z_{t+1})\},
\end{aligned} \qquad \text{(19)}
$$

其中 $\mathbf{0}$ 为零向量，$\sigma$ 为作用于**整个动作数组**的一次固定行置换——$\sigma$ 保持动作取值点的多重集不变，却把每一行动作移到与之不相干的时刻，从而**打断动作与下一潜变量之间的对应**。相应地，$\mathcal{D}_{\mathrm{shuf}}$ 的潜变量序列与 $\mathcal{D}_{\mathrm{real}}$ 逐位相同（二者共享同一 $z$ 序列），仅动作一列被重排。三者对应三种信息内容：`real` 保留动作与其后果的真实对应；`shuf` 保留动作的边缘分布、但破坏其预测关系；`noact` 令动作恒为零。**`noact` 是关键控制**：动作输入恒为零时，式 (5) 中 $W_a$ 的每一列都在乘以零，故 $\partial \mathcal{L}_M/\partial W_a\equiv 0$——该通道**从不被更新**，始终停留在初始化值。这正是一个被单独留空的动作通道；而在视频 / JEPA 设定中不存在可被如此留空的对象。

**范数钉住：改变动作通道的尺度。** 由于读数可能同时取决于动作通道的**尺度** $\lVert W_a\rVert$ 与它**是否承载信息**，直接比较不同 checkpoint 会混淆这两个因素。为把尺度单独固定，我们在训练中通过一个前置钩子，在**每次前向**把 $W_a$ 投影到目标范数 $c$：

$$
W_a\ \leftarrow\ W_a\cdot\frac{c}{\lVert W_a\rVert+\varepsilon}, \qquad \text{(20)}
$$

其中 $\varepsilon$ 为防止除零的小量。该投影只作用于 $W_a$，模型其余部分照常训练；因此在钉住范数的同时，“通道是否承载信息”仍由式 (19) 的族决定。

**$2\times2$：分离信息与尺度。** 把两个因素交叉——信息 $\in\{+,-\}$ 与范数 $\in\{c_{\mathrm{lo}},c_{\mathrm{hi}}\}$——得到四个单元：

$$
\begin{aligned}
\texttt{real}     &:=\ (\text{info}+,\ c_{\mathrm{hi}}), &\qquad
\texttt{realdown} &:=\ (\text{info}+,\ c_{\mathrm{lo}}),\\
\texttt{shufup}   &:=\ (\text{info}-,\ c_{\mathrm{hi}}), &\qquad
\texttt{shuf}     &:=\ (\text{info}-,\ c_{\mathrm{lo}}),
\end{aligned} \qquad \text{(21)}
$$

其中 $c_{\mathrm{hi}}$ 与 $c_{\mathrm{lo}}$ 分别取 `real` 与 `shuf` 两个训练完成的动作通道的固有尺度，从而把“跨族钉住”与“不钉住”置于同一标尺（其数值见 §4）。由此，四个对比各自隔离一个因素在另一因素固定时的效应：

$$
\underbrace{\texttt{realdown}/\texttt{shuf}}_{\text{信息效应 @ }c_{\mathrm{lo}}},\quad
\underbrace{\texttt{real}/\texttt{shufup}}_{\text{信息效应 @ }c_{\mathrm{hi}}},\quad
\underbrace{\texttt{shufup}/\texttt{shuf}}_{\text{范数效应 @ info}-},\quad
\underbrace{\texttt{real}/\texttt{realdown}}_{\text{范数效应 @ info}+}. \qquad \text{(22)}
$$

这一设计把“该比值是否区分动作信息”从一个单一对照问题，变为一个可分别读出**信息**与**尺度**两路贡献的析因问题：若读数只在“有信息”的单元上高、而对“尺度”不敏感，它才是本文所主张的动作依赖度量。

## 3.5 完整训练与探针流程

前四节给出了全部数学对象。本节把它们组装为两个流程：**训练一个受控单元**（Algorithm 1）与**对单元实施探针**（Algorithm 2）。两个流程均只使用已定义的符号；其中出现的自由参数（学习率 $\eta$、批大小、隐维 $d_h$、视界 $T$、rollout 条数 $N$、采样重复数 $R$、温度取值 $\tau$、范数目标 $c$ 等）由 §4 的实验设定与附录的超参数表统一给出，本节不取具体值。

**Algorithm 1：训练一个受控单元**（对应 §3.2 的目标与 §3.4 的控制）

```
输入：训练三元组序列 D = {(z_t, a_t, z_{t+1})}，族类型 fam ∈ {real, shuf, noact}，
      可选范数目标 c（仅范数钉住单元）。
输出：M 的 checkpoint（含参数 θ*）。

1  θ ← 按正交初始化；若 fam = noact：令动作输入恒为 0（式 19 第三行），
   使得 ∂L_M/∂W_a ≡ 0，W_a 在整个训练中不被更新；
   若给出 c：为 M 安装 forward pre-hook，使每次前向满足式 (20) 的投影。
2  repeat
3     从 D 中采样一批长度为 L 的片段 (z_{1:L}, a_{1:L}, z_{2:L+1})；
4     按式 (3)(4) 计算混合参数 (π_t,k, μ_t,k, σ_t,k)；
5     按式 (8)(9) 计算负对数似然 L_M(θ)；
6     θ ← θ − η ∇_θ L_M(θ)（Adam，梯度裁剪）；
7  until 达到预定的训练步数
8  以验证集 NLL 选择最优 checkpoint，返回 θ*。
```

**Algorithm 2：探针评估一个单元**（对应 §3.3 的三臂与比值，可选剂量轴）

```
输入：M 的 checkpoint，一个 rollout 索引集 {1,…,N}，warm-up 长度 W，
      视界 T，采样温度 τ，条内重复数 R，可选剂量乘子 λ。
输出：读数 D/B(t)（对每个视界 t∈T）。

1  for i = 1..N：
2     取第 i 条 episode 的潜变量与动作序列 (z, a)；
3     用前 W 步做 warm-up，得共享起点 (ẑ_W, ĥ_{W-1}) = (z_W, h_{W-1})；
4     若给出 λ：将动作切片乘子置为 λ（λ=1 完好，λ=0 置零），构造 a^(λ)；
       否则 a^(λ) ← a。
5     for r = 1..R：
6        按式 (14) 生成五条 rollout：
          参考：R(a^(λ), ξ)、臂A：R(a^perm, ξ)、臂B：R(a^(λ), ξ′)、
          确定性臂：R_det(a^(λ)) 与 R_det(a^perm)（式 12）；
7        按式 (13) 计算各臂散度 d_t：A、B 相对参考，D 相对另一条确定性 rollout；
8     D_i(t) ← 1/R Σ_r D_i^(r)(t)，B_i(t) ← 1/R Σ_r B_i^(r)(t)（式 15）；
9     D_i/B_i(t) ← D_i(t)/B_i(t)。
10  D/B(t) ← 1/N Σ_i D_i(t)/B_i(t)（式 16）。
11  若需交叉剂量：对一组 λ 扫描值重复 1–10，得 D/B 对 x 的曲线，
    按式 (17)(18) 插值出 x_cross。
```
