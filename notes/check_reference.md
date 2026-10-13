# 引用复核记录（check_reference.md）

**范围**：`paper/introduction.md`（§1）与 `paper/related_work.md`（§2 全部，含 §2.1–§2.4）中出现的**全部引用**。
**复核日期**：2026-10-07 起；§2 逐条核证于 2026-10-08。
**复核方式**：逐条打开**权威记录**（arXiv 摘要页 / 原文 HTML / 博客原文 / GitHub API）取其元数据与内容；**未打开全文的条目明确标出**。

## 保障级别定义

| 级别 | 含义 |
|---|---|
| **A｜已读原文正文** | 本会话打开了**正文/全文**（或博客全文），所引内容有**内容级**保障 |
| **B｜仅读摘要/元数据页** | 本会话打开了**摘要页**（标题+作者+摘要），**未见正文**；正文级主张未证 |
| **C｜未读** | 本会话**未打开**任何权威页；仅凭训练知识或交接稿/项目笔记 |
| **D｜占位符** | 尚未成为引用 |

> ⚠ **任何"未证数字"在被引用前必须读原文确认。**

---

## A. 已读原文正文（本会话，内容级保障）

| # | 引用 | 我实际读到的 | 与之相关的未证项 |
|---|---|---|---|
| A1 | **Tallec, Blier & Kalainathan (2018)** *Reproducing "World Models"* | 博客**全文**：860±120 vs 870±120、"only serves as a recurrent reservoir"、作者三人；GitHub API `created_at = 2018-06-26` | 无（860/870 已逐字确认） |
| A2 | **ADGEfficiency (2019/2025)** *World Models (The Long Version)* | 博客**全文**：893 分（"a conveniently selected random seed"）、"Created Dec 2019 / Updated Oct 2025"、成本 \$3,648；GitHub API `created_at = 2020-04-25` | 作者署名是**网名**，非真名 |
| A3 | **Rigter et al., AVID** (arXiv:2410.12822v2, 24 Nov 2024) | **全文**（用户提供 PDF）；§4.2 评估指标共 6 个：Action Error Ratio、FVD、FID、SSIM、PSNR、LPIPS（Table 1–3） | ✅ **已证**：**Action Error Ratio = 动作预测误差之比（生成视频 ÷ 真实视频）**（§4.2）⇒ 属**动作可恢复性**口径 |

---

## B. 仅读摘要/元数据页（本会话，标题+作者+摘要级）

| # | 引用 | ID | 已确认 | 未证 |
|---|---|---|---|---|
| B1 | **MiraBench**（Yang 等，2026） | ✅ **已读原文全文**（用户提供 PDF，arXiv:2605.29360v1）；见文末记录 | 已核 |
| B2 | **WorldEcho**（Chen 等，2026） | ✅ **已读原文全文**（用户提供 PDF，arXiv:2608.24885v1）；见文末记录 | 已核 |
| B3 | **Yeom 等（2026）** | ✅ **已读原文全文**（用户提供 PDF，arXiv:2606.07687v1）；见文末记录 | 已核（venue 待定） |
| B4 | **Dueling World Models**（Li 等，2026） | arXiv:2608.06706 | 标题、**4 位作者：Jiazhuo Li, Yiming Fei, Zhiruo Zhou, Heikichi Hayashi** | ⚠ **$AS$ 由 $1.28\to0.002$ 摘要中无**（未证，见 E4）；归纳性描述"quietly go action-blind"与摘要相符 ✔ |
| B5 | **stable-worldmodel**（Maes 等，2026） | arXiv:2605.21800 | 标题、**12 位作者：Maes, Le Lidec, Facury, Massaudi, Chaurasia, Capuano, Gao, Gillin, Haramati, Scieur, LeCun, Balestriero** | 全文 |
| B6 | **Nunes 等（2020）** | arXiv:1910.02564 | 标题、4 位作者、摘要（action inference + "high perceptual scores can perform poorly"） | ⚠ **R² 0.82 / SVG-LP ≈0 未证**；**venue "ICRA 2020" 未在此页确认**（见 E8） |
| B7 | **Zhang 等（2025）** *What Do Latent Action Models Actually Learn?* | arXiv:2506.15691 | 标题、8 位作者、**NeurIPS-2025 已录用**（Comments 字段） | 全文 |
| B8 | **Delta-JEPA**（Zhang 等，2026） | arXiv:2606.31232 | 标题、14 位作者、LDAD、"displacement-level supervision" | 全文 |
| B9 | **Dreamer**（Hafner 等） | arXiv:1912.01603 | 标题 *Dream to Control*、4 位作者、提交 2019-12 | ⚠ **venue ICLR 2020 / 年份取 2019 还是 2020 未在此页确认**（见 E7） |
| B10 | **DreamerV3**（Hafner 等，2023） | arXiv:2301.04104 | 标题 *Mastering Diverse Domains through World Models*、4 位作者 | 全文 |
| B11 | **Moerland 等** *Model-based RL: A Survey* | arXiv:2006.16712 | 标题、4 位作者（Moerland, Broekens, Plaat, Jonker）、arXiv v4=2022 | ⚠ **期刊版本年份（2023?）与卷期未证**（见 E6） |
| B12 | **Thomas 等（2017）** *Independently Controllable Factors* | arXiv:1708.01289 | 标题、9 位作者（Thomas, Pondard, Bengio, Sarfati, Beaudoin, Meurs, Pineau, Precup, Bengio） | 全文 |
| B13 | **Iso-Dream**（Pan 等，2022） | **arXiv:2205.13817** | 标题、作者（Minting Pan, Xiangming Zhu, Yunbo Wang, Xiaokang Yang）、NeurIPS 2022（出自 `citation_trace_2026-10-03.csv:2450`） | 全文（见 E1 更正） |

---

## C. 本会话完全未读（C 级）

| # | 引用 | 现状 | 风险 |
|---|---|---|---|
| C1 | **Ha & Schmidhuber (2018)** *World Models* | ✅ **已读原文全文**（用户提供 PDF，arXiv:1803.10122v4）；逐句核对正文描述，**发现 2 处与原文不符**（见文末记录） | 低：**2 处均已修正（2026-10-07）**（控制器训练环境、τ=1.15；见文末 ✅ 已修正①/②） |
| C2 | **VidGenStein（Bierling & Gavves, 2026）** | ✅ **已读原文**（用户提供 PDF）；$S_a=1.170$、"约 8× AVID"、三条归因均属实 | 低：仅 venue 待定，正文用中性描述规避 |
| C3 | **Sutton & Barto (2018)** *Reinforcement Learning: An Introduction* | **书目已核**（官方页 `incompleteideas.net/book/the-book-2nd.html`：题名/作者/第二版/MIT Press/2018）；**正文未读** | 低：教科书级领域定义，无需正文级核证 |

---

## D. 占位符（尚未成为引用）

| 位置 | 占位 |
|---|---|
| §1 ¶1 | `[[CITE: 1–2 篇综述]]` |
| §1 ¶2 | `[[CITE: 直接复现]]`、`[[CITE: 后续引用]]` |
| §1 ¶6 | `[[检索日期]]`、附录 `[[X]]` |

---

## E. 本次复核发现的错误与更正（编号）

| # | 问题 | 证据 | 处置 |
|---|---|---|---|
| **E1** | **（更正我自己的记录）** Iso-Dream 的 arXiv ID：`related_work.md` **没有写任何 ID**；我此前"写了 2205.14951"是我的错误，该串**在全仓库不存在** | 正确 ID 出自 `notes/artifacts/milestone/citation_trace_2026-10-03.csv:2450` | **已在 §2.2(g) 补入 `arXiv:2205.13817`**（Pan, Zhu, Wang, Yang；NeurIPS 2022） |
| **E2** | **AVID 年份/venue**：写"2024, arXiv" | arXiv v1 2024-10-01；**已被 RLJ 2025 接收**（rlj.cs.umass.edu/2025/papers/Paper64.html） | 改为 **Rigter 等，2025（RLJ）**，note 保留 arXiv:2410.12822 |
| **E3** | **AVID 被误置为“比值路线老根”**——我此前称其“在精神上已采用动作敏感比值” | **撤回**。AVID **全文**（用户提供 PDF）§4.2 定义 **Action Error Ratio** = 动作预测误差之比（生成 ÷ 真实），属**动作可恢复性**；该结论**已由原文证实** | **删除该主张**；AVID 移至“动作可恢复性”一脉作前身/基线（§2.2(e)） |
| **E4** | **Dueling WM 的 $AS: 1.28\to0.002$** 未证 | 摘要中无此数字（来自交接稿 `[VERIFY-NUM]`） | 引用前须读原文；否则**不写数字** |
| **E5** | **VidGenStein 的 $S_a=1.170$、"约 8× AVID"** 未证 | ✅ **已核**：用户提供 PDF 原文；$S_a=1.170$（Table 2/3、§4.2）、"roughly 8×"（§4.2）、三条归因均属实 | 引用前须读原文 → **已完成**；只写"报告了"，non-archival 注不得省（已保留） |
| **E6** | **Moerland 综述的期刊版本年份** | arXiv v4=2022；IRIS 引作 2023 | 引文年份二选一并统一（建议按**期刊版**核） |
| **E7** | **Dreamer 年份** | arXiv 2019-12 / ICLR 2020 | §2.1 写"Hafner 等，2019；2023"——2019 对 arXiv 版；定稿前统一 |
| **E8** | **Nunes 的 venue "ICRA 2020"** | arXiv 摘要页未显示 venue | 引文前核 ICRA 2020 proceedings |

---

## F. 下一步复核清单（引用前必须完成）

1. ~~**C2 VidGenStein**：读原文，核 $S_a=1.170$、$8\times$、non-archival 状态（**硬约束**）。~~ ✅ **已完成（2026-10-07）**；venue 改用中性描述。
2. **C1 Ha & Schmidhuber**：以 arXiv:1803.10122 原文核对正文描述（三段式、$P(z_{t+1}\mid z_t,h_t,a_t)$）。
3. **B4 Dueling WM**：核 $1.28\to0.002$ 是否在正文。
4. **B6 Nunes**：核 R² 数字与 ICRA 2020 venue。
5. **B11 Moerland**：核期刊版年份/卷期。
6. **B9 Dreamer**：定年份 2019/2020。
7. **B1/B2/B3**：MiraBench / WorldEcho / Yeom 的"无未训练通道对照"须**自查全文**后再写入正文（现为二手结论）。
8. **B1/B2/B3/B4 发表状态**：三篇 2026 预印本提交前重查是否转正式 venue（GATE-3）。

---

## G. 结论

- **A 级（可放心引用）**：2 条 —— Tallec、ADGEfficiency。
- **B 级（可用作背景，数字须再核）**：13 条 —— 见 B 表。
- **C 级（未读，其中 1 条为硬约束项）**：3 条 —— Ha、VidGenStein、Sutton & Barto。（原 C4 WorldModelGym 已**删除**，非 arXiv 论文、无元数据。）
- **D 级（占位符）**：5 处。
- **本次更正**：AVID 年份/定位（E2/E3，已在 `related_work.md` 同步）；Iso-Dream 正确 ID 已补入 §2.2(g)（E1，见上更正）。

---

## 引用记录 · 2026-10-07 23:58

> **新增约定（自本次起）**：每写入一处引用，即在本文件末尾追加一段，记录 **标题 / 作者 / 发表时间 / 我方引用句 / 支持性原文**。

### 1. Iso-Dream

| 项 | 内容 |
|---|---|
| 标题 | Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models |
| 作者 | Minting Pan, Xiangming Zhu, Yunbo Wang, Xiaokang Yang |
| 发表时间 | 2022（NeurIPS 2022）；arXiv:2205.13817 |
| 我方引用句 | `related_work.md` §2.2(g)：「… and Iso-Dream (Pan et al., 2022, arXiv:2205.13817, which isolates non-controllable visual dynamics in world models).」 |
| 支持性原文 | 标题、作者、venue 出自 `notes/artifacts/milestone/citation_trace_2026-10-03.csv:2450`（S2 后向追溯）。⚠ **本会话未读正文**；「isolates non-controllable visual dynamics」依据标题与 S2 记录，非正文核证。 |

### 2. AVID

| 项 | 内容 |
|---|---|
| 标题 | AVID: Adapting Video Diffusion Models to World Models |
| 作者 | Marc Rigter, Tarun Gupta, Agrin Hilmkil, Chao Ma |
| 发表时间 | 2025（Reinforcement Learning Journal）；arXiv:2410.12822（v1 2024-10-01） |
| 我方引用句 | `related_work.md` §2.2(e)：「AVID (Rigter et al., 2025, arXiv:2410.12822; Reinforcement Learning Journal) adapts pretrained video diffusion models to action-conditioned world models …」 |
| 支持性原文 | 摘要（本会话已读）：「we propose to adapt pretrained video diffusion models to action-conditioned world models, without access to the parameters of the pretrained model … AVID trains an adapter on a small domain-specific dataset of action-labelled videos」✔ |

### 3. VidGenStein —— ✅ 已核（原文已取得，见文末核实段）

| 项 | 内容 |
|---|---|
| 标题 | Turning Video Foundation Models into Action-Conditioned World Models by Steering Denoising（旧名：Conditioned World Models by Steering Denoising） |
| 作者 | Lukas Bierling, Stratis Gavves |
| 发表时间 | 2026-10-02。**venue 待定**：交接简报作"NeurIPS 2026 RoboPAD workshop"，但与原文不符；用户已确认 OpenReview 两页均为**匿名投稿**（旧标题 *Conditioned World Models by Steering Denoising*）。现按中性描述 **"一篇同期的匿名投稿（非存档）"**，正文不写 workshop。OpenReview forum `c2Mc4pOnub`。 |
| 我方引用句 | §1 ¶6（已改）：「例如，**VIDGENSTEIN** 一篇同期的匿名投稿（非存档）报告了 $S_a=1.170$ 这一读数。」 |
| 支持性原文 | ✅ 已取得（用户提供 PDF 原文）。见文末「引用记录 · 2026-10-07 23:58（VidGenStein 原文核实）」。 |

---

## 引用记录 · 2026-10-07 23:58（VidGenStein 原文核实）

> 承接上一条约定：本段记录该引用的**标题 / 作者 / 发表时间 / 我方引用句 / 支持性原文**。

| 项 | 内容 |
|---|---|
| 标题 | Turning Video Foundation Models into Action-Conditioned World Models by Steering Denoising |
| 作者 | **Anonymous Author(s)**（双盲匿名稿；据交接简报为 Lukas Bierling, Stratis Gavves，**本轮未能由原文确认**） |
| 发表时间 / venue | 原文页脚：**"Submitted to 40th Conference on Neural Information Processing Systems (NeurIPS 2026). Do not distribute."**；文件日期 2026-10-02。**venue 待定**：交接简报作"RoboPAD workshop"，与原文不符；用户已确认 OpenReview 两页均为**匿名投稿**（旧标题 *Conditioned World Models by Steering Denoising*）。故正文按中性描述 **"一篇同期的匿名投稿（非存档）"**，不写 workshop。 |
| 我方引用句 | §1 ¶6（已改）：「例如，**VIDGENSTEIN** 一篇同期的匿名投稿（非存档）报告了 $S_a=1.170$ 这一读数。」 |
| 支持性原文①（$S_a=1.170$） | Table 2 末行："VIDGENSTEIN −0.180 **1.170** 40.3 546"；Table 3："VIDGENSTEIN (100M) 100.6M 40.3 546 **1.17**"；§4.2："raises action share to 1.170" |
| 支持性原文②（约 8× AVID） | §4.2："increases action share over AVID **roughly 8×**"；AVID $S_a=0.142$（Table 2），$1.170/0.142\approx8.2\times$ |
| 支持性原文③（三条归因） | Abstract / §3.2 / §3.3："learned mixture can suppress the adapter gradient"、"a competing full-prediction branch admits copying of the frozen prediction"、"the action is informative primarily at high noise"；Table 1（$\sigma\le0.5$ 无效应，$\sigma=0.99$ 达 0.908%） |
| 支持性原文④（$S_a$ 定义） | Eq. 5–6，$S_a=E_a/C$，"the fraction of the adapter's change that is action-sensitive"；§4.1："Sa can exceed one" |

### ⚠ 本轮新发现的矛盾（venue，需裁定）

| 来源 | 说法 |
|---|---|
| 交接简报 | NeurIPS 2026 **RoboPAD workshop Poster**（**non-archival**） |
| **原文 PDF 页脚** | **"Submitted to 40th Conference on NeurIPS 2026"**（= 主会投稿模板）；全文**不含** "workshop / RoboPAD / archival / poster" 任何字样 |

**含义**：PDF 表现得更像**主会匿名投稿**，而非 workshop 稿。对引用的影响：
- "**非存档**"这一性质**仍成立**（匿名投稿 + "Do not distribute"）；
- 但 "**workshop 报告**" 这一 venue 描述**与原文不符** → 建议改为中性写法（如"一篇同期的匿名投稿（非存档）"），或按 OpenReview 实际状态定稿。

---

## 引用记录 · 2026-10-07（Sutton & Barto 书目核证）

| 字段 | 内容 |
|---|---|
| 标题 | **Reinforcement Learning: An Introduction**（Second Edition） |
| 作者 | Richard S. Sutton, Andrew G. Barto |
| 发表时间 / 出版 | MIT Press, Cambridge, MA, **2018**（第二版，2018） |
| 我方引用句 | §2.1：「基于模型的强化学习（model-based RL）先为环境的动力学学得一个内部模型，再借助该模型——而非仅靠与环境的试错交互——来规划行为或改进策略（Sutton & Barto, 2018; Moerland 等，2023）。」 |
| 支持性原文 | 官方书目页 `incompleteideas.net/book/the-book-2nd.html`（access 2026-10-07）：题名 **Reinforcement Learning: An Introduction**、作者 **Richard S. Sutton and Andrew G. Barto**、**Second Edition**、**MIT Press, Cambridge, MA, 2018** —— 逐字确认。 |
| 保障级别 | **书目已核（元数据）；正文未读**。此为教科书级领域定义（MBRL 定义），仅需元数据归属核证，无需正文级核证。**不确定上传全书**。 |
| 备注 | `tallec2019worldmodels` 旧 key 问题不影响此处；引用 key 用 `sutton2018rl`。 |

---

## 引用记录 · 2026-10-07（MiraBench 原文核证）

| 字段 | 内容 |
|---|---|
| 标题 | **MIRABENCH: Evaluating Action-Conditioned Reliability in Robotic World Models** |
| 作者 | Tianzhuo Yang, Zihan Shen, Zirui Mi, Zhaoyi Zhang, Jiayi Zhou, Jiaming Ji, Juntao Dai, Jiawei Chen, Boyuan Chen, Yaodong Yang（PKU；†通讯：Boyuan Chen, Yaodong Yang） |
| 发表时间 / 出处 | **arXiv:2605.29360v1 [cs.AI], 28 May 2026**；标注 **"Preprint."**；ID 页脚一致 |
| 我方引用句 | §1 ¶6 / §2.2②：「面向机器人视频世界模型的诊断基准 MiraBench 报告"视觉保真度是动作保真度的坏代理"（Yang 等，2026）」 |
| 支持性原文①（核心断言） | Abstract / §1：「**visual fidelity is a poor proxy for action fidelity**」（逐字相符），列为三条 central findings 之首 |
| 支持性原文②（三层结构） | Abstract：「Physics Adherence…Action-Following Fidelity…Optimism Bias Detection」；三项与 `related_work.md` 的描述一致 |
| 支持性原文③（无未训练通道对照） | 全文检索："untrained"、"randomly initialized"、"action channel" **均无命中**；"ablation" 仅指 Cosmos-GR1 微调步数与聚合器基线（§5）与扰动归因（附录），**均非未训练通道对照** ⇒ 本文 §2.2 的"未提供对未训练通道伪影的控制"**成立** |
| 支持性原文④（与 2018 模型的关系） | 参考文献 **[14] David Ha and Jürgen Schmidhuber. World models. 2018**；仅在"World models for embodied AI"**谱系**中引用（"early model-based RL [36, 14]"），**未对 2018 潜变量模型做任何探针** ⇒ 与本文"尚无此类探针"的定位**不矛盾** |
| 附带发现（可用于 §2.2） | 该文 Related Work 自述：现有世界模型评测"**evaluate … on visual quality metrics (FVD, PSNR, SSIM, or human preference)**"，与本文 §2.2 "视觉保真 ≠ 动作保真" 叙事同调；并把 WorldArena [34] / CtrlWorld [13] 列为"最接近的动作跟随测量前驱"。 |
| 保障级别 | **A｜已读原文全文**（用户提供 PDF，含 Abstract / §1–§2 及全文，2,900+ 行提取文本）。 |

---

## 引用记录 · 2026-10-07（WorldEcho 原文核证）

| 字段 | 内容 |
|---|---|
| 标题 | **Do Robotic World Models Really Follow Actions? Diagnosing and Aligning Action-Conditioned Generation for Policy Learning** |
| 作者 | Sixiang Chen, Jiaming Liu, Jixian Wu, Yichen Guo, Tinghao Wang, Siyuan Qian, Hao Chen, Jiajun Cao, Jian Tang, Shanghang Zhang（PKU 等；通讯 Jian Tang / Shanghang Zhang） |
| 发表时间 / 出处 | **arXiv:2608.24885v1 [cs.RO], 25 Aug 2026** |
| 我方引用句（§2.2②） | 「WorldEcho（Chen 等，2026）以视觉完整性与 SE(3) 轨迹对齐探测动作跟随」 |
| 支持性原文①（视觉完整性） | §3.3「Visual Integrity Assessment… image quality … motion smoothness … end-effector visibility … arm integrity」；gate 式 (4) Gvis = Gquality∧Gmotion∧GEEF∧Garm |
| 支持性原文②（SE(3) 轨迹对齐） | §3.3「End-Effector Trajectory Alignment… recovers the per-frame position … orientation R_t ∈ **SO(3)** for each end effector」 |
| ⚠ **我方引言原句过强（已修正）** | 引言原写「WorldEcho **得到同样结论**（视觉保真度是动作保真度的坏代理）」——**WorldEcho 并无此结论**：全文 "proxy"/"orthogonal"/"correlat"/"decoupl" **零命中**；其结论是「两种失败模式：visual collapse 与 action mismatch」（§3.2、Fig.2）。已改为「WorldEcho 把动作跟随拆为视觉完整性与末端轨迹对齐两个维度，发现二者可分别失效」（§3.3「A visually valid rollout **may nevertheless ignore** the conditioned actions」直接支撑）。 |
| 支持性原文③（无未训练通道对照） | 全文 "untrained"/"initialized"/"action input" **零命中**；"ablation" 仅指 WorldSync 消融（Table 2），**非未训练通道对照** ⇒ §2.2"均未控制未训练通道伪影"**成立** |
| 支持性原文④（与 2018 模型的关系） | 参考文献检索 "Schmidhuber"/"Ha and" **零命中** ⇒ WorldEcho **未引用** 2018 World Models ⇒ 与本文定位**不矛盾** |
| 保障级别 | **A｜已读原文全文**（用户提供 PDF）。 |

---

## 引用记录 · 2026-10-07（Yeom 原文核证）

| 字段 | 内容 |
|---|---|
| 标题 | **What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction** |
| 作者 | Jewon Yeom\*, Hanseul Kim\*, Jeongjae Park, Sungmok Jung, Jaejin Lee, Taesup Kim†（首尔大学数据科学研究生院） |
| 发表时间 / 出处 | **arXiv:2606.07687v1 [cs.CV], 5 Jun 2026**；venue 待定 |
| 我方引用句①（引言 ¶6） | 「一项对视频世界模型潜变量的探针研究发现动作可恢复性与像素保真度**基本正交**（Yeom 等，2026）」 |
| 支持性原文① | 正文：「visual fidelity and action recoverability are **largely orthogonal**: the standard metrics (PSNR, FVD, LPIPS) do not [correlate with control]」（逐字相符） |
| 我方引用句②（§2.2③） | 「Yeom 等（2026）用逆动力学探针发现动作相关性由**预测**而非**重建**目标驱动」 |
| 支持性原文② | Abstract：「action-relevant structure is driven primarily by temporal video pretraining **rather than pixel reconstruction fidelity**」；「temporal predictive structure—**not reconstruction fidelity**—as the primary ingredient」 |
| 支持性原文③（无未训练通道对照） | 全文 "untrained" **零命中**；全文为 frozen-encoder 逆动力学探针（8 个 encoder 家族），**无"剩余部分照常训练、单独留空动作通道"的对照** ⇒ §2.2"均未控制未训练通道伪影"**成立** |
| 支持性原文④（与 2018 模型的关系） | "Schmidhuber" / "Ha and" **零命中** ⇒ **未引用** 2018 World Models ⇒ 与本文定位**不矛盾** |
| 保障级别 | **A｜已读原文全文**（用户提供 PDF）。 |

---

## 引用记录 · 2026-10-07（Ha & Schmidhuber 原文核证）

| 字段 | 内容 |
|---|---|
| 标题 | **World Models** |
| 作者 | David Ha, Jürgen Schmidhuber（Google Brain；NNAISENSE；Swiss AI Lab, IDSIA） |
| 发表时间 / 出处 | **arXiv:1803.10122v4 [cs.LG], 9 May 2018** |
| 我方引用句 | §1 ¶1：「Ha 与 Schmidhuber（2018）是该方向的早期工作之一：他们提出一个由三部分组成的架构——把每一帧压缩为隐变量 $z_t$ 的 VAE、对转移 $P(z_{t+1}\mid z_t,h_t,a_t)$ 建模的循环混合密度网络（MDN-RNN），以及完全在"梦"中优化的小控制器——使整个训练回路可以脱离真实环境。」 |
| 支持性原文①（VAE） | §2.1：「we use a simple Variational Autoencoder … as our V model to compress each image frame into a small latent vector z.」 |
| 支持性原文②（MDN-RNN 转移） | §2.2：「the RNN will model P (zt+1 | at, zt, ht)」；「we approximate p(z) as a mixture of Gaussian distribution」 |
| 支持性原文③（小控制器） | §2.3：「C is a simple single layer linear model」；参数量 867（CarRacing）/ 1,088（VizDoom） |
| 支持性原文④（可脱离真实环境） | Abstract：「We can even train our agent entirely inside of its own hallucinated dream generated by its world model, and transfer this policy back into the actual environment.」 |
| 支持性原文⑤（动作单通道） | §2.2：动作 $a_t$ 仅作为 M 的输入（$P(z_{t+1}\mid a_t,z_t,h_t)$）；C 输入 $[z_t,h_t]$、V 输入为帧——$a_t$ 仅被 M 消费。（**注**：当前引言 ¶1 已删去此句，此条供后续恢复时引用。） |
| ✅ **已修正①**（2026-10-07） | 原 ¶1「完全在"梦"中优化的小控制器」**只对 VizDoom 成立**，已改为「轻量小控制器（线性映射 $a_t=W_c[z_t,h_t]+b_c$）——使整个训练回路可以脱离真实环境：控制器的训练既可基于真实环境，也可完全在"梦"中完成」。原文依据（此处保留备查）：§3.4「We can put our **trained C** back into this hallucinated environment generated by M.」；§4.1「a policy **learned inside of the real environment** appears to somewhat function inside of the dream」。即 CarRacing 的 C 在**真实环境**训练（§3.4 为演示）；仅 VizDoom 在**虚拟环境**内训练（§4.1 步骤 5："maximizes the expected survival time **inside the virtual environment**"）。 |
| ✅ **已修正②**（2026-10-07） | ¶5 原「$\tau\approx1$」已改为 **$\tau=1.15$**。原文为 **$\tau=1.15$**。Table 2 测试 $\tau\in\{0.10,0.50,1.00,1.15,1.30\}$；最佳实际分 $1092\pm556$ 在 $\tau=1.15$；附录「we increased the temperature slightly and used **τ = 1.15**」。 |
| 附带发现（可引用） | 原文 Table 2 是一次 $\tau$ 扫描（$\tau$ 为行，给 virtual score 与 actual score）；**注：该表任务为 VizDoom 的 Take Cover，非 CarRacing**（CarRacing §3.4 仅 demo 可调 $\tau$、无取值；$\tau=1.15$ 见附录 A.5 的 DoomRNN）。$\tau=0.10$ 时虚拟分 $2086\pm140$、实际分仅 $193\pm58$（mode collapse，控制器利用模型缺陷）——原文据此上调 $\tau$ 以增加不确定性、防止控制器钻世界模型的空子。该机制任务无关，与本文 §5 的 $\tau$ 轴叙事**同调**。 |
| 保障级别 | **A｜已读原文全文**（用户提供 PDF，含 Abstract / §1–§7 / 附录）。 |

---

## 引用记录 · 2026-10-08（§2 相关工作 落盘 · 逐条核证）

> 本节记录 2026-10-08 写入 `paper/related_work.md`（§2 全文）时，其中**每一条引用**的 **标题 / 作者 / 发表时间 / 我方引用句 / 支持性原文出处**。凡本次**新读全文**者标 **A**；仅核书目或摘要者标 **B**。写作用语一律按措辞"报告 / 度量"，不写"证明"。

### 1. Sutton & Barto（2018）

| 项 | 内容 |
|---|---|
| 标题 | Reinforcement Learning: An Introduction（Second Edition） |
| 作者 | Richard S. Sutton, Andrew G. Barto |
| 发表时间 | MIT Press, Cambridge, MA, 2018 |
| 我方引用句 | §2.1：「……来规划行为或改进策略（Sutton & Barto, 2018; Moerland 等，2023）。」 |
| 支持性原文 | 官方书目页 `incompleteideas.net/book/the-book-2nd.html`：题名/作者/Second Edition/MIT Press/2018，逐字确认。 |
| 保障级别 | **B｜书目已核**（元数据）；正文未读（教科书级定义，无需正文级核证）。 |

### 2. Moerland 等（2023）

| 项 | 内容 |
|---|---|
| 标题 | Model-based Reinforcement Learning: A Survey |
| 作者 | Thomas M. Moerland, Joost Broekens, Aske Plaat, Catholijn M. Jonker |
| 发表时间 | Foundations and Trends in Machine Learning, 16(1):1–118, 2023（DOI 10.1561/2200000086）；arXiv:2006.16712（v4 2022） |
| 我方引用句 | §2.1：「……先为环境的动力学学得一个内部模型，再借助该模型……来规划行为或改进策略（Sutton & Barto, 2018; Moerland 等，2023）。」 |
| 支持性原文 | 本 PDF §1 定义：「we define model-based RL as: 'any MDP approach that i) uses a model (known or learned) and ii) uses learning to approximate a global value or policy function'」；首页脚注 `arXiv:2006.16712v4`。 |
| 保障级别 | **A｜已读原文全文**（本 PDF 首页+§1）；**刊行卷期/年份**经官网（Emerald/ACM DOI）确认 2023。 |

### 3. Ha & Schmidhuber（2018）

| 项 | 内容 |
|---|---|
| 标题 | World Models |
| 作者 | David Ha, Jürgen Schmidhuber |
| 发表时间 | arXiv:1803.10122v4 [cs.LG], 9 May 2018 |
| 我方引用句 | §2.1：「World Models（Ha & Schmidhuber, 2018）是一个影响深远的早期潜变量世界模型：它把视觉表征学习与时间动力学建模解耦——VAE 将每帧压缩为隐变量 zt，MDN-RNN 建模转移 P(zt+1∣zt,ht,at)，一个小控制器则依托学得的潜空间（"梦"）训练——既可基于真实环境，也可完全在"梦"中完成。」 |
| 支持性原文 | §2.1 VAE「compress each image frame into a small latent vector z」；§2.2「model P(zt+1 | at, zt, ht)」；§2.3「C is a simple single layer linear model」；Abstract「train our agent entirely inside of its own hallucinated dream … and transfer this policy back into the actual environment」。 |
| 备注 | §2.1 同时用"三部分解耦"与"控制器可真实/梦中训练"两处表述；后者即本文件前段「✅已修正①」的口径。 |
| 保障级别 | **A｜已读原文全文**（见前段本文件记录）。 |

### 4. Hafner 等（2019）Dreamer

| 项 | 内容 |
|---|---|
| 标题 | Dream to Control: Learning Behaviors by Latent Imagination |
| 作者 | Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi |
| 发表时间 | ICLR 2020；arXiv:1912.01603（v1 2019-12，v3 2020-03） |
| 我方引用句 | §2.1：「……Dreamer 以潜空间想象提升控制效率……（Hafner 等，2019；2023）。」 |
| 支持性原文 | 首页「Published as a conference paper at ICLR 2020」；Abstract「efficiently learn behaviors by propagating analytic gradients of learned state values back through trajectories imagined in the compact state space」；§1「exceeds existing approaches in data-efficiency, computation time, and final performance」。 |
| 备注 | **年份口径**：正文按 **arXiv 版**取 2019（用户 2026-10-08 决定 C4）；正式投用时应改 ICLR 2020 并与 bib 统一。 |
| 保障级别 | **A｜已读原文全文**。 |

### 5. Hafner 等（2023）DreamerV3

| 项 | 内容 |
|---|---|
| 标题 | Mastering Diverse Domains through World Models（预印本标题；正式版改题 Mastering diverse control tasks through world models） |
| 作者 | Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap |
| 发表时间 | arXiv:2301.04104（本 PDF v2 2024-04-17）；正式版 **Nature 640:647–653 (2025)**，DOI 10.1038/s41586-025-08744-2 |
| 我方引用句 | §2.1：「……DreamerV3 以固定超参跨域通用（Hafner 等，2019；2023）。」 |
| 支持性原文 | Abstract「a general algorithm that outperforms specialized methods across over 150 diverse tasks, with a single configuration」；「robustness techniques based on normalization, balancing, and transformations enable stable learning across domains」。 |
| 备注 | 正式版标题为 "Mastering **diverse control tasks**…"，与预印本不同；正文年份按 arXiv 版取 2023。Nature 2025 系本会话联网核证（官网），非本 PDF。 |
| 保障级别 | **A｜已读原文全文**（本 PDF）；Nature 版为 **B｜元数据已核（联网）**。 |

### 6. Tallec（2018）

| 项 | 内容 |
|---|---|
| 标题 | world-models: Reimplementation of "World Models" in PyTorch |
| 作者 | Corentin Tallec（含 Blier, Kalainathan，见前段 record） |
| 发表时间 | 2018（GitHub `created_at` 2018-06-26）；non-archival |
| 我方引用句 | §2.1：「Tallec（2018）给出使用最广的 PyTorch 实现并端到端复现了 CarRacing 设置。」 |
| 支持性原文 | 仓库/博客全文：860±120（trained）vs 870±120（untrained）、"only serves as a recurrent reservoir"（见前段 A1 记录）。 |
| 备注 | 年份取 **2018**（用户 2026-10-08 沿用 §3.1 决定）。 |
| 保障级别 | **A｜已读原文全文**（GitHub/博客）。 |

### 7. ADGEfficiency（2019；2025）

| 项 | 内容 |
|---|---|
| 标题 | World Models（TensorFlow 复现，博客） |
| 作者 | ADGEfficiency（网名署；真名未核） |
| 发表时间 | 2019（"Created Dec 2019"）；2025（"Updated Oct 2025"）；non-archival |
| 我方引用句 | §2.1：「ADGEfficiency（2019；2025）的 TensorFlow 复现在 CarRacing 上达到 893 分。」 |
| 支持性原文 | 博客全文：893 分（"a conveniently selected random seed"）。 |
| 保障级别 | **A｜已读原文全文**（博客）。 |

### 8. Maes 等（2026）stable-worldmodel

| 项 | 内容 |
|---|---|
| 标题 | stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation |
| 作者 | Lucas Maes, Quentin Le Lidec, Luiz Facury, Nassim Massaudi, Ayush Chaurasia, Francesco Capuano, Richard Gao, Taj Gillin, Dan Haramati, Damien Scieur, Yann LeCun, Randall Balestriero |
| 发表时间 | arXiv:2605.21800v1 [cs.LG], 20 May 2026；Preprint |
| 我方引用句 | §2.1：「面向可复现世界模型研究的平台（Maes 等，2026）则把数据、训练与评测统一起来。」 |
| 支持性原文 | Abstract「unifying the full pipeline under a single, scalable framework」；贡献列表 (1) Lance-based data layer、(2) baselines/planning solvers、(3) environments and tasks。 |
| ⚠ 注意 | 勿与同组另一篇 **LeWorldModel（arXiv:2603.19312）** 混引；本文引的是平台篇 2605.21800。 |
| 保障级别 | **A｜已读原文全文**。 |

### 9. Nunes 等（2019）

| 项 | 内容 |
|---|---|
| 标题 | Action-conditioned Benchmarking of Robotic Video Prediction Models: a Comparative Study |
| 作者 | Manuel Serra Nunes, Atabak Dehban, Plinio Moreno, José Santos-Victor |
| 发表时间 | arXiv:1910.02564v1 [cs.CV], 7 Oct 2019；**原文无 venue 字样**（无 ICRA） |
| 我方引用句 | §2.2：「这一提问可追溯到对动作条件化机器人视频预测的比较性基准，其中动作无关的预测器便无法从预测帧反推动作（Nunes 等，2019）。」 |
| 支持性原文 | 「the action-free SVG-LP model has an R² score of approximately 0 … the inference model is unable to identify the actions and limits itself to predicting a constant average」；Table 1：SVG R²=−0.0582、Oracle R²=0.8203。 |
| 备注 | 年份取 **2019**（依原文 arXiv）；`related_work.md` 旧稿的"ICRA 2020"**未见于本 PDF**，已删。 |
| 保障级别 | **A｜已读原文全文**。 |

### 10. Bierling & Gavves（2026）VidGenStein

| 项 | 内容 |
|---|---|
| 标题 | Turning Video Foundation Models into Action-Conditioned World Models by Steering Denoising |
| 作者 | Anonymous（双盲；据交接稿为 Lukas Bierling, Stratis Gavves） |
| 发表时间 | 文件日期 2026-10-02；页脚「Submitted to 40th Conference on NeurIPS 2026」；**非存档、匿名投稿** |
| 我方引用句 | §2.2：「动作份额 Sa 度量动作敏感变化相对适配器总变化的比例（Bierling & Gavves，2026）」 |
| 支持性原文 | Eq.5–6「Sa = Ea/C, the fraction of the adapter's change that is action-sensitive」；Table 2「1.170」；§4.2「raises action share to 1.170」。 |
| 备注 | 正文**不写数字 1.170、用定性**；venue 用中性描述。 |
| 保障级别 | **A｜已读原文全文**。 |

### 11. Li 等（2026）Dueling World Models

| 项 | 内容 |
|---|---|
| 标题 | Dueling World Models: Advantage-Style Action Channels for Common-Mode Distractor Rejection |
| 作者 | Jiazhuo Li, Yiming Fei, Zhiruo Zhou, Heikichi Hayashi |
| 发表时间 | arXiv:2608.06706v1 [cs.LG], 7 Aug 2026 |
| 我方引用句 | §2.2：「动作分离 AS 则度量不同动作下输出间的平均成对距离（Li 等，2026）。」 |
| 支持性原文 | §4.1「Action separation (AS) is the mean pairwise distance among per-action predictions from a common state」；引言/§4.2「AS collapses from ≈1.28 to ≈0.002」（FourRooms，随干扰单元 n 增至 30，"for every variant"）。 |
| 备注 | 正文**不写数字**（仅定性）；本稿曾写"分母是输出间距离"属**错误**（$AS$ 无分母），已改为"平均成对距离（不含分母）"。$1.28\to0.002$ 已核，保留在 bib/台账。 |
| 保障级别 | **A｜已读原文全文**。 |

### 12. Yang 等（2026）MiraBench

| 项 | 内容 |
|---|---|
| 标题 | MIRABENCH: Evaluating Action-Conditioned Reliability in Robotic World Models |
| 作者 | Tianzhuo Yang, Zihan Shen, Zirui Mi, Zhaoyi Zhang, Jiayi Zhou, Jiaming Ji, Juntao Dai, Jiawei Chen, Boyuan Chen, Yaodong Yang |
| 发表时间 | arXiv:2605.29360v1 [cs.AI], 28 May 2026；Preprint |
| 我方引用句 | §2.2：「报告视觉保真度是动作保真度的坏代理（Yang 等，2026）」 |
| 支持性原文 | Abstract/§1「visual fidelity is a poor proxy for action fidelity」（逐字）；三层结构 Physics Adherence / Action-Following Fidelity / Optimism Bias Detection。 |
| 备注 | 全文 "untrained"/"randomized"/"action channel" **零命中** ⇒ 无未训练通道对照。 |
| 保障级别 | **A｜已读原文全文**。 |

### 13. Chen 等（2026）WorldEcho

| 项 | 内容 |
|---|---|
| 标题 | Do Robotic World Models Really Follow Actions? Diagnosing and Aligning Action-Conditioned Generation for Policy Learning |
| 作者 | Sixiang Chen, Jiaming Liu, Jixian Wu, Yichen Guo, Tinghao Wang, Siyuan Qian, Hao Chen, Jiajun Cao, Jian Tang, Shanghang Zhang |
| 发表时间 | arXiv:2608.24885v1 [cs.RO], 25 Aug 2026 |
| 我方引用句 | §2.2：「或以视觉完整性与 SE(3) 轨迹对齐探测动作跟随（Chen 等，2026）」 |
| 支持性原文 | §3.3「Visual Integrity Assessment…」+「End-Effector Trajectory Alignment… orientation R_t ∈ SO(3)」。 |
| 备注 | 全文 "untrained"/"initialized" **零命中**；引言原句"得到同样结论"曾**过强**、已改（见前段记录）。 |
| 保障级别 | **A｜已读原文全文**。 |

### 14. Thomas 等（2017）

| 项 | 内容 |
|---|---|
| 标题 | Independently Controllable Factors |
| 作者 | Valentin Thomas, Jules Pondard, Emmanuel Bengio, Marc Sarfati, Philippe Beaudoin, Marie-Jean Meurs, Joelle Pineau, Doina Precup, Yoshua Bengio |
| 发表时间 | arXiv:1708.01289v2 [cs.LG], 25 Aug 2017 |
| 我方引用句 | §2.3：「……其做法或以归一化的选择性比值驱动策略与特征的学习（Thomas 等，2017；2018）……」 |
| 支持性原文 | Eq.(1) sel = E[|fk(s′)−fk(s)| / Σ_{k′}|fk′(s′)−fk′(s)|]；§2.2「we define the following objective … Lae − λ Σk E[Σa πk(a|s)sel(s,a,k)]」；§1「We propose a specific objective function」。 |
| 备注 | selectivity 是**归一化比值**（使用于**训练奖励**，非对外报告的诊断读数）；正文按"训练目标/驱动学习"表述。注：引文中的 λ 为**原文自身的损失权重**，与本文的 λ（动作切片乘子）不同义。 |
| 保障级别 | **A｜已读原文全文**。 |

### 15. Thomas 等（2018）

| 项 | 内容 |
|---|---|
| 标题 | Disentangling the Independently Controllable Factors of Variation by Interacting with the World |
| 作者 | Valentin Thomas, Emmanuel Bengio, William Fedus, Jules Pondard, Philippe Beaudoin, Hugo Larochelle, Joelle Pineau, Doina Precup, Yoshua Bengio |
| 发表时间 | NIPS 2017 Workshop on Learning Disentangled Representations（non-archival）；arXiv:1802.09484v1 [stat.ML], 26 Feb 2018 |
| 我方引用句 | §2.3：「……或以归一化的选择性比值驱动策略与特征的学习（Thomas 等，2017；2018）……」 |
| 支持性原文 | §3「a general quantity S which we refer to as selectivity and that is used as a reward signal for πφ」；Eq.(1) S(h,φ)=E[log A(h′,h,φ)/E_{p(ϕ|h)}[A(h′,h,ϕ)]]；§1「We build on a previously studied [Thomas et al., 2017] mechanism」。 |
| 备注 | 系 Thomas 2017 的**扩展**；仍以 selectivity 为训练奖励 ⇒ 归入"训练目标"一侧成立。 |
| 保障级别 | **A｜已读原文全文**。 |

### 16. Pan 等（2022）Iso-Dream

| 项 | 内容 |
|---|---|
| 标题 | Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models |
| 作者 | Minting Pan, Xiangming Zhu, Yunbo Wang, Xiaokang Yang |
| 发表时间 | NeurIPS 2022；arXiv:2205.13817v3 [cs.LG], 18 Dec 2022 |
| 我方引用句 | §2.3：「……或把两类动力学分置于独立的状态转移分支（Pan 等，2022）。」 |
| 支持性原文 | 首页脚注「36th Conference on Neural Information Processing Systems (NeurIPS 2022)」；§3.2「three-branch architecture to decouple … controllable dynamics state s_t, noncontrollable … z_t」；「transition models with separate parameters」；「we introduce the training objective of inverse dynamics」。 |
| 保障级别 | **A｜已读原文全文**。 |

### 17. Zhang 等（2025）

| 项 | 内容 |
|---|---|
| 标题 | What Do Latent Action Models Actually Learn? |
| 作者 | Chuheng Zhang, Tim Pearce, Pushi Zhang, Kaixin Wang, Xiaoyu Chen, Wei Shen, Li Zhao, Jiang Bian |
| 发表时间 | NeurIPS 2025；arXiv:2506.15691v3 [cs.LG], 12 Nov 2025 |
| 我方引用句 | §2.3：「后者从理论上刻画了潜变量何时学到可控、动作相关的变化，并指出数据采集策略的随机性与辅助预测任务的影响（Zhang 等，2025）。」 |
| 支持性原文 | 首页脚注「39th Conference on Neural Information Processing Systems (NeurIPS 2025)」；§1「insights into when inferred latent actions capture true control signals compared to noise」；§4.2「higher randomness in data-generating policies benefits LAM's learning」；§4.4「adding an action-prediction head encourages LAM to prioritize the learning of controllable changes」。 |
| 备注 | 正文已加限定"**在其线性潜在动作模型框架内**"（对象为 LAM，非转移模型）。 |
| 保障级别 | **A｜已读原文全文**。 |

### 18. Yeom 等（2026）

| 项 | 内容 |
|---|---|
| 标题 | What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction |
| 作者 | Jewon Yeom, Hanseul Kim, Jeongjae Park, Sungmok Jung, Jaejin Lee, Taesup Kim |
| 发表时间 | arXiv:2606.07687v1 [cs.CV], 5 Jun 2026；venue 待定（CoRL 2026 待核） |
| 我方引用句 | §2.3：「此外，逆动力学探针被用于检验表征的动作相关性，表明其由预测目标而非重建目标驱动（Yeom 等，2026）」 |
| 支持性原文 | Abstract「action-relevant structure is driven primarily by temporal video pretraining rather than pixel reconstruction fidelity」；「temporal predictive structure—not reconstruction fidelity—as the primary ingredient」。 |
| 备注 | 全文 "untrained" 零命中。 |
| 保障级别 | **A｜已读原文全文**。 |

### 19. Zhang 等（2026）Delta-JEPA

| 项 | 内容 |
|---|---|
| 标题 | Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference Decoding |
| 作者 | Zhenghao Zhang, Yuanxiang Wang, Zhenyu Guan, Yujia Yang, Bingkang Shi, Tianyu Zong, Hongzhu Yi, Guoqing Chao, Xingchen Chen, Tiankun Yang, Chenxi Bao, Tao Yu, Jingjing Zhou, Jungang Xu |
| 发表时间 | arXiv:2606.31232v1 [cs.AI], 30 Jun 2026 |
| 我方引用句 | §2.3：「Delta-JEPA 以潜位移动作解码提出方法层面的改进，并附一个固定历史、只变动作的响应探针（Zhang 等，2026）。」 |
| 支持性原文 | Abstract/Contributions「We propose Delta-JEPA … a Latent Difference Action Decoder (LDAD)」（方法主张）；§"Action-Sensitive Latent Dynamics"「we sample 512 Two-Room starting histories and keep each history representation fixed while varying only the final action input … measure its displacement relative to the zero-action prediction, ẑ_{t+1}(a) − ẑ_{t+1}(0)」（即固定历史、只变动作的响应探针）。 |
| 备注 | 正文由旧稿"proposes a fix, not a diagnostic"改为"提出方法层面改进 **+ 附只变动作探针**"（更准）；无未训练通道对照（λ=0 时"collapse"指表征退化，非伪影）。注：此处 λ 为 **Delta-JEPA 自身的损失权重**，非本文的动作切片乘子。 |
| 保障级别 | **A｜已读原文全文**。 |

---

## 引用记录 · 2026-10-11（§4 实验 落盘 · Ha & Schmidhuber 再引用）

| 项 | 内容 |
|---|---|
| 标题 | World Models |
| 作者 | David Ha, Jürgen Schmidhuber |
| 发表时间 / 出处 | arXiv:1803.10122v4 [cs.LG], 9 May 2018 |
| 我方引用句（§4.2） | 「原文在 CarRacing-v0 上报告 $906\pm21$（Ha 与 Schmidhuber，2018）。」 |
| 支持性原文 | 原文 CarRacing 实际分 **906 ± 21**（Table 1）；训练预算 10,000 rollouts / 1,800 代 / 种群 64×16——见本文件 Ha & Schmidhuber 条（A 级，已读全文）。 |
| 备注 | §1 已引该文；此为 §4.2 就“原文分数”这一具体数字的**再引用**登记。 |
---

## 引用记录 · 2026-10-12 00:23（§5 讨论 · Ha & Schmidhuber 再引用）

| 项 | 内容 |
|---|---|
| 标题 | World Models |
| 作者 | David Ha, Jürgen Schmidhuber |
| 发表时间 / 出处 | arXiv:1803.10122v4 [cs.LG], 9 May 2018 |
| 我方引用句（§5.1） | 「这正好解释原文为何把温度调至 $\tau=1.15$（Ha 与 Schmidhuber，2018）。」 |
| 我方引用句（§5.2） | 「……前提是 2018 的 MDN-RNN 存在一段可单独指认的动作列切片 $W_a$（式 (5)）……」（Ha 与 Schmidhuber，2018） |
| 支持性原文 | §4.3 + Table 2（标题逐字「Take Cover scores at various temperature settings」）：$\tau=0.10$ 行 virtual $2086\pm140$ / actual $193\pm58$；$\tau=1.15$ 行 actual $1092\pm556$；附录 A.5「In the virtual DoomRNN environment … used τ = 1.15」；§3.4 CarRacing「adjust τ to control the uncertainty of the environment generated by M」（demo 可调、无取值）；§4.3 mode collapse「the monsters … fail to shoot fireballs」。均见本文件 Ha & Schmidhuber 条（A 级，已读全文）。 |
| 备注 | §1、§4.2 已引该文；此为 §5.1/§5.2 的**再引用**登记。**关键澄清**：原文 $\tau$ 扫描表（Table 2）与 $\tau=1.15$ 均为 **VizDoom 的 Take Cover**；CarRacing 仅 §3.4 demo 可调 $\tau$、无取值。 |

---

## 引用记录 · 2026-10-12 00:23（§5 讨论 · AVID 再引用）

| 项 | 内容 |
|---|---|
| 标题 | AVID: Adapting Video Diffusion Models to World Models |
| 作者 | Marc Rigter, Tarun Gupta, Agrin Hilmkil, Chao Ma |
| 发表时间 / 出处 | Reinforcement Learning Journal, 2025；arXiv:2410.12822v2 [cs.CV], 24 Nov 2024 |
| 我方引用句（§5.4 更正表 #3） | 「AVID（Rigter 等，2025）用 Action Error Ratio（§4.2：动作预测误差之比，生成 ÷ 真实），属动作可恢复性一脉，非“动作敏感性比值”。」 |
| 支持性原文 | §4.2 逐字「**Action Error Ratio** – To assess the consistency between the videos and the action sequences, we train a model to predict actions from real videos. The Action Error Ratio is the ratio of errors obtained by using this model on generated videos, divided by the error obtained on real videos.」；Table 1–3 报告该指标（如 AVID (71M) 1.154）。 |
| 备注 | §2.2(e) 已引该文；此为 §5.4 的**再引用**登记。**更正仓内旧记录**：本文件 A3/E3 原标「正文未读/未证」，现由原文（用户提供 PDF）证实。 |
---

## 引用记录 · 2026-10-12 23:49（§5 讨论 · Ha & Schmidhuber 原文逐字核对）

| 项 | 内容 |
|---|---|
| 标题 | World Models |
| 作者 | David Ha, Jürgen Schmidhuber |
| 发表时间 / 出处 | Distill, 2018 / arXiv:1803.10122（用户提供 PDF「world model 2018.pdf」） |
| 我方引用句（§5.1） | 「原文这次 $\tau$ 扫描是在 VizDoom 的 Take Cover 上做的：$\tau=0.10$ 时虚拟分 $2086\pm140$、实际分仅 $193\pm58$……至 $\tau=1.15$ 实际分达 $1092\pm556$」（Ha 与 Schmidhuber，2018，Table 2） |
| 我方引用句（§5.3） | 「其 $V$ 以 L2 重建 + KL、$M$ 以 NLL 训练，均未给出 $R^2$ 或 $\Sigma\Delta\mathrm{NLL}$」 |
| 支持性原文 | **Table 2**（标题逐字「Take Cover scores at various temperature settings」）：τ=0.10 → virtual 2086±140 / actual 193±58；0.50 → 2060±277 / 196±50；1.00 → 1145±690 / 868±511；1.15 → 918±546 / **1092±556**；1.30 → 732±269 / 753±139；random policy 210±108；gym leader 820±58。**§4.5**：「if we set the temperature parameter to a very low value of τ = 0.1 …… the monsters inside this dream environment fail to shoot fireballs …… due to mode collapse」；「the best score obtained is 1092 ± 556 with τ = 1.15」。**附录 A.5**：「In the virtual DoomRNN environment we constructed, we increased the temperature slightly and used τ = 1.15」。**附录 A.1**：「using L2 distance between the input image and the reconstruction to quantify the reconstruction loss we optimize for, in addition to KL loss」。**附录 A.2**：「The MDN-RNNs were trained for 20 epochs …… 5 Gaussian mixtures」。全文报告 VAE 重建仅以图、MDN 预测仅以图；**未报告 $R^2$ 或 NLL 数值**。 |
| 备注 | 本次逐字核对**确认** §5.1/§5.3 关于 Ha 2018 的事实无误；并**纠正内部先前的疑问**：原文 VAE 损失为 **L2 + KL**（非对数似然）。 |

## 引用记录 · 2026-10-13 01:06（§1 ¶2 / §2 复现谱系 · Tallec 复现「使用最广」的旁证）

| 项 | 内容 |
|---|---|
| 标题 | Reproducing "World Models": Is training the recurrent network really needed? |
| 作者 | Corentin Tallec, Léonard Blier, Diviyan Kalainathan |
| 发表时间 / 出处 | 2018；博客 `https://ctallec.github.io/world-models/`；代码 `https://github.com/ctallec/world-models`（2026-10-12 读） |
| 我方引用句（§1 ¶2） | 「使用最广的公开 PyTorch 复现即是一例（Tallec 等，2018）」 |
| 我方引用句（§2.1） | 「Tallec（2018）给出使用最广的 PyTorch 实现并端到端复现了 CarRacing 设置」 |
| 支持性原文①（其范围仅 CarRacing） | 小节标题逐字「Reproducibility on the CarRacing environment」；结论逐字「We reproduced the paper "World Models" **on the CarRacing environment**」；结论内限定逐字「**On the CarRacing-v0 environment**, it seems that the recurrent network only serves as a recurrent reservoir」。 |
| 支持性原文②（其关键实验） | 表：`With a trained MDRNN (Ours)` $860\pm120$ / `With an untrained MDRNN (Ours)` $870\pm120$。 |
| 支持性原文③（「使用最广」的旁证：下游引用） | 见下方「旁证表」。 |
| 备注 | 「使用最广」是**工程流行度 + 下游引用**的合成判断，非原文或某单篇的断言；正文须保持「使用最广的公开 PyTorch 复现」表述，**不作绝对化**（不写"唯一/全部复现"）。 |
| 保障级别 | **A｜已读原文全文**（博客原文全读，2026-10-12）。 |

**旁证表 · 引用 Tallec 复现的下游文献**（均为**参考文献级 / 片段级**核验，**未读全文**）

| # | 引用方 | 年份 / venue | 引用证据（出处） | 保障级别 |
|---|---|---|---|---|
| 1 | Risi & Stanley, *Deep Neuroevolution of Recurrent and Discrete World Models* | GECCO 2019（arXiv:1906.08857） | 参考文献列「Reproducing "World Models" is training the recurrent network really needed ? https://ctallec.github.io/world-models/」（dl.acm.org 片段） | **B｜仅参考表片段** |
| 2 | Chang 等, *Reinforcement learning with convolutional reservoir computing* | Applied Intelligence 2020 | 同上逐字（dl.acm.org 片段） | **B｜仅参考表片段** |
| 3 | *Convolutional Reservoir Computing for World Models* | arXiv:1907.08040 | 参考文献列「Reproducing "world models" … https://ctallec.github.io/world-models/, 2018」 | **B｜仅参考表片段** |
| 4 | Wang, He, Jin, *Does Learning Recurrent Networks Matter in Partially Observable RL?* | arXiv:2610.13197（2026-10-09） | 参考文献项「Tallec, Blier, Kalainathan, Reproducing "World Models"…」（摘要页已读；引文项为 HTML 片段） | **B｜摘要级 + 片段** |
| 5 | IN-RIA 学位论文, *Recurrent Neural Networks and Reinforcement Learning* | HAL tel-02434367 | 正文「The code of the replication is available at https://github.com/ctallec/world-models」 | **B｜片段** |

**本节须补核（投稿前）**：① 各引用方的**年份 / venue** 系凭片段推得，**未开原文**，引用前须逐条核；② 引用**数量**与仓库**星数（约 706★）**均为**快照值**、来自镜像页，非稳定量，若正文要用**定量**措辞（如"被广泛引用"）须以 Google Scholar / Semantic Scholar 复核后再定；③ 凡本次**未读全文**者一律维持 **B 级**（依台账既定口径：标题正确 ≠ 引用内容正确）；④ ADGEfficiency、AdeelMufti 两复现在检索中**未发现学术引用**——检索未见 ≠ 零引用，不得据此下断言。
