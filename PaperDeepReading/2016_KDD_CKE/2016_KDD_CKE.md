# **精读笔记**

知识图谱增强推荐（Knowledge-aware Recommendation）的经典开山工作之一

第一次证明了，把知识图谱及其多模态信息与协同过滤进行联合学习，能够显著提升推荐性能

后续模型则围绕“如何更充分利用知识图谱”不断演进，进一步发展为 **兴趣传播（RippleNet）**、**图神经网络聚合（KGCN、[[2019_KDD_KGAT]]）**、**高阶关系建模（[[2021_WWW_KGIN]]）** 等方向

## 快速总结

论文认为，一个物品不仅包含用户交互行为，还蕴含丰富的知识库信息，包括：

- 结构知识（Knowledge Graph）
- 文本知识（Description）
- 图像知识（Poster）

传统协同过滤只能利用用户行为，而 CKE 希望学习知识库的语义表示，并与协同过滤联合训练，从而获得更加丰富的物品表示。

最终物品表示由四部分相加组成：

$$
e_j = \eta_j + v_j + X_{\frac{L_t}{2}, j *} + Z_{\frac{L_v}{2}, j*}
$$

其中包括：

- 协同过滤偏移向量（item offset vector）$\eta_j$：捕捉用户-物品交互中的协同信号
- 结构表示（structural representation）$v_j$：由 Bayesian TransR 从知识图谱结构中学得
- 文本表示（textual representation）$X_{\frac{L_t}{2}, j*}$：由 Bayesian SDAE 从物品文本描述中学得，即 SDAE 中间隐藏层输出
- 视觉表示（visual representation）$Z_{\frac{L_v}{2}, j*}$：由 Bayesian SCAE 从物品图像（海报/封面）中学得，即 SCAE 中间隐藏层输出

> **符号说明**：后续模型（如 [[2019_KDD_KGAT]]、[[2021_WWW_KGIN]]）习惯用 $e_j$ 统一表示物品嵌入，但 CKE 中 $e_j$ 是四部分之和——其中 $\eta_j$ 来自协同过滤（CF），$v_j$ 来自知识图谱结构嵌入（KG），$X_{\frac{L_t}{2}, j*}$ 来自文本自编码器的中间层，$Z_{\frac{L_v}{2}, j*}$ 来自视觉自编码器的中间层。为便于理解，也可简记为 $e_j = \eta_j + v_j + x_j + z_j$。

### 核心贡献

1. 首次统一利用 **结构、文本、图像** 三类知识进行推荐
2. 使用联合训练，使知识表示直接服务于推荐目标
3. 采用 Bayesian TransR 学习知识图谱结构表示，并将其与推荐任务联合优化
4. 在 `MovieLens-lM` 和 `IntentBooks` 上的实验结果表明，CKE 优于各种基准方法

### 局限性

1. 仅学习实体和关系的嵌入表示（**单跳** 三元组），未在图结构上进行邻居聚合或高阶多跳传播——后续 [[2019_KDD_KGAT]] 正是由此改进
2. 三种模态采用简单相加融合，没有学习不同模态的重要性
3. 用户表示仍然完全来自协同过滤，没有利用知识图谱建模用户兴趣传播
4. 文本和图像编码器使用 SDAE/SCAE，在今天已被 Transformer 和预训练视觉模型取代

## 研究背景

### 现有方法

| 对比维度         | 传统协同过滤（CF）                        | CKE                                                          |
| ---------------- | ----------------------------------------- | ------------------------------------------------------------ |
| **使用信息**     | 仅利用用户-物品交互（评分、点击、购买等） | 同时利用用户-物品交互和知识库中的结构、文本、图像等多源信息  |
| **物品表示**     | 每个物品只有一个协同过滤隐向量            | 物品表示由协同过滤向量、结构知识表示、文本表示和视觉表示共同组成 |
| **知识利用方式** | 不利用知识图谱或仅依赖人工特征工程        | 自动学习知识库中的语义表示，无需人工设计特征                 |
| **冷启动能力**   | 对新物品依赖历史交互，性能下降明显        | 即使交互较少，也可借助知识库中的属性、关系、文本和图片获得表示 |
| **训练方式**     | 仅优化推荐目标（如 BPR）                  | 推荐任务与知识表示学习进行联合优化（Joint Learning）         |
| **表示能力**     | 难以表达物品之间的语义关系                | 能够学习实体、关系及多模态信息的潜在语义表示                 |

## 方法论

CKE 框架由两大步骤组成：

1. **知识库嵌入（Knowledge Base Embedding）**：分别从结构知识、文本知识、视觉知识中提取物品的语义表示
2. **协同联合学习（Collaborative Joint Learning）**：将三类嵌入与协同过滤隐向量融合为最终物品表示，通过成对排序联合优化所有组件

### 符号表

#### 数据定义

| 符号 | 含义 |
| ---- | ---- |
| $U$ | 用户集合，$|U| = m$ |
| $I$ | 物品集合，$|I| = n$ |
| $R \in \mathbb{R}^{m \times n}$ | 用户隐式反馈矩阵，$R_{ij} = 1$ 表示用户 $i$ 与物品 $j$ 有过交互 |
| $G = (V, E)$ | 结构知识对应的异质无向图，$V$ 为实体集合，$E$ 为关系边集合 |
| $S$ | 结构嵌入训练四元组集合：$(v_h, r, v_t, v_t')$ 满足 $(v_h, r, v_t)$ 为正三元组、$(v_h, r, v_t')$ 为负三元组 |
| $D$ | 协同过滤训练三元组集合：$(i, j, j')$ 满足 $R_{ij} = 1$ 且 $R_{ij'} = 0$ |

#### 可学习参数

| 符号 | 维度 | 含义 |
| ---- | ---- | ---- |
| $u_i$ | $\mathbb{R}^d$ | 用户 $i$ 的隐向量 |
| $\eta_j$ | $\mathbb{R}^d$ | 物品 $j$ 的协同过滤偏移向量（item offset vector） |
| $e_j$ | $\mathbb{R}^d$ | 物品 $j$ 的最终表示（四部分之和） |
| $v_h, v_t$ | $\mathbb{R}^k$ | 头/尾实体的结构嵌入（实体空间） |
| $r$ | $\mathbb{R}^d$ | 关系嵌入（关系空间） |
| $M_r$ | $\mathbb{R}^{k \times d}$ | 关系 $r$ 的投影矩阵，将实体从实体空间映射到关系空间 |
| $W_l, b_l$ | — | SDAE 第 $l$ 层的权重与偏置 |
| $Q_l, c_l$ | — | SCAE 第 $l$ 层的权重（卷积核/全连接）与偏置 |

> **注意**：CKE 中实体嵌入维度为 $k$，关系嵌入/CF 隐向量维度为 $d$，两者可以不同。投影矩阵 $M_r \in \mathbb{R}^{k \times d}$ 起到维度桥接作用。

### 结构嵌入（Structural Embedding）— Bayesian TransR

结构知识被建模为一个异质图 $G = (V, E)$，其中节点为不同类型的实体，边为不同类型的关系。CKE 使用 **TransR** 学习实体和关系的向量表示，并将其扩展为贝叶斯版本。

#### TransR 基础

TransR 的核心思想是：实体和关系属于不同的语义空间，不应在同一空间中建模。对于每个三元组 $(v_h, r, v_t)$：

**投影**：通过关系特定投影矩阵 $M_r$，将实体从实体空间 $\mathbb{R}^k$ 映射到关系空间 $\mathbb{R}^d$：
$$
\begin{align}
v_h^r &= v_h M_r \\
v_t^r &= v_t M_r
\end{align}
$$

**评分函数**（能量函数）：在关系空间中，头实体投影 + 关系向量应接近尾实体投影：
$$
f_r(v_h, v_t) = \|v_h^r + r - v_t^r\|_2^2
$$

三元组越合理，评分越低。

#### Bayesian 扩展

CKE 对 TransR 引入贝叶斯建模，为所有参数赋予高斯先验。生成过程如下：

1. 对每个实体 $v$：$v \sim \mathcal{N}(0, \lambda_v^{-1} I)$
2. 对每个关系 $r$：$r \sim \mathcal{N}(0, \lambda_r^{-1} I)$，$M_r \sim \mathcal{N}(0, \lambda_M^{-1} I)$
3. 对每个四元组 $(v_h, r, v_t, v_t') \in S$：从概率 $\sigma\big(f_r(v_h, v_t) - f_r(v_h, v_t')\big)$ 中采样

其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 为 sigmoid 函数，$S$ 为训练四元组集合。负样本 $(v_h, r, v_t')$ 通过替换正确三元组的尾实体为同类型另一实体构造。

> **直觉**：$\sigma(f_r(v_h, v_t) - f_r(v_h, v_t'))$ 意味着——正三元组的评分 $f_r(v_h, v_t)$ 相比负三元组的评分 $f_r(v_h, v_t')$ 越小（差距越大），该四元组被采样的概率越高。这等价于要求正样本的能量远低于负样本。

对每个物品实体 $j$，其结构表示即为 Bayesian TransR 学到的实体嵌入 $v_j$。

### 文本嵌入（Textual Embedding）— Bayesian SDAE

文本知识来自物品的摘要/描述文本。CKE 使用 **堆叠降噪自编码器（Stacked Denoising Auto-Encoder, SDAE）** 学习文本的分布式表示。

#### SDAE 结构

SDAE 共 $L_t$ 层，前 $L_t/2$ 层为编码器，后 $L_t/2$ 层为解码器：

- **输入 $X_0$**：对原始文本 bag-of-words $X_{L_t}$ 随机遮盖部分词条（masking noise）得到的损坏输入
- **中间层 $X_{L_t/2}$**：潜在紧凑表示（latent representation），即文本嵌入
- **输出 $X_{L_t}$**：目标是重建干净的原始 bag-of-words

$X_l$ 的每一行 $X_{l, j*}$ 对应物品 $j$ 在第 $l$ 层的表示。

#### 生成过程

对 SDAE 每一层 $l$：

1. 权重：$W_l \sim \mathcal{N}(0, \lambda_W^{-1} I)$
2. 偏置：$b_l \sim \mathcal{N}(0, \lambda_b^{-1} I)$
3. 层输出：$X_l \sim \mathcal{N}\big(\sigma(X_{l-1} W_l + b_l), \lambda_X^{-1} I\big)$

其中 $\sigma(\cdot)$ 为激活函数（如 sigmoid/tanh）。给定损坏输入 $X_0$ 和干净目标 $X_{L_t}$ 均被观测，模型通过逐层重建学习文本的语义表示。

物品 $j$ 的文本表示取 SDAE 中间隐藏层的输出：

$$
x_j = X_{\frac{L_t}{2},\, j*}
$$

### 视觉嵌入（Visual Embedding）— Bayesian SCAE

视觉知识来自物品的图像（电影海报、书籍封面）。CKE 使用 **堆叠卷积自编码器（Stacked Convolutional Auto-Encoder, SCAE）** 学习视觉表示。

#### SCAE 结构

SCAE 共 $L_v$ 层，与 SDAE 类似分为编码器和解码器。区别在于：

- **卷积层**（除第 $L_v/2$ 和 $L_v/2 + 1$ 层外）：使用卷积运算替代全连接，保留图像的邻域关系和空间局部性
- **全连接层**（第 $L_v/2$ 和 $L_v/2 + 1$ 层）：位于编码器-解码器交界处，作为维度转换的瓶颈

输入 $Z_0$ 为添加高斯噪声的损坏图像，目标 $Z_{L_v}$ 为原始 RGB 图像（$3 \times 64 \times 64$ 张量）。$Z_{L_v, j*}$ 为物品 $j$ 的原始像素表示。

#### 生成过程

对 SCAE 每一层 $l$：

1. 权重：$Q_l \sim \mathcal{N}(0, \lambda_Q^{-1} I)$
2. 偏置：$c_l \sim \mathcal{N}(0, \lambda_c^{-1} I)$
3. 层输出：
   - 若 $l$ 为全连接层：$Z_l \sim \mathcal{N}\big(\sigma(Z_{l-1} Q_l + c_l), \lambda_Z^{-1} I\big)$
   - 若 $l$ 为卷积层：$Z_l \sim \mathcal{N}\big(\sigma(Z_{l-1} * Q_l + c_l), \lambda_Z^{-1} I\big)$

其中 $*$ 表示卷积运算，$\sigma(\cdot)$ 为激活函数。

物品 $j$ 的视觉表示取 SCAE 中间全连接层的输出：

$$
z_j = Z_{\frac{L_v}{2},\, j*}
$$

### 协同联合学习（Collaborative Joint Learning）

获得三类知识嵌入后，CKE 将其与协同过滤统一到一个联合学习框架中。

#### 物品最终表示

物品 $j$ 的最终隐向量由四部分直接相加：

$$
e_j = \eta_j + v_j + X_{\frac{L_t}{2},\, j *} + Z_{\frac{L_v}{2},\, j*}
$$

其中：
- $\eta_j$：来自协同过滤的偏移向量，学习交互信号
- $v_j$：来自 Bayesian TransR 的结构嵌入，学习 KG 结构语义
- $X_{\frac{L_t}{2}, j*}$：来自 Bayesian SDAE 的文本嵌入，学习文本语义
- $Z_{\frac{L_v}{2}, j*}$：来自 Bayesian SCAE 的视觉嵌入，学习视觉语义

> 后三项充当 "桥梁"——将结构、文本、视觉知识与隐式反馈偏好连接起来，使得推荐目标函数的梯度能够反向传播到各个嵌入组件。

#### 成对偏好建模

沿用 BPR 的思路，对于用户 $i$，若 $R_{ij} = 1$（交互过）且 $R_{ij'} = 0$（未交互），则假设 $i$ 偏好 $j$ 胜于 $j'$。定义成对偏好概率：

$$
p(j > j'; i \mid \theta) = \sigma(u_i^T e_j - u_i^T e_{j'})
$$

其中 $\theta$ 为所有模型参数，$\sigma$ 为 sigmoid 函数。分数差 $\hat{y}(i, j) - \hat{y}(i, j') = u_i^T e_j - u_i^T e_{j'}$ 越大，$i$ 越可能偏好 $j$。

#### 完整生成过程

整合上述所有组件，CKE 的完整贝叶斯生成过程为（与论文 Section 5 完全对应）：

1. **结构知识**：
   - (a) 实体先验：$v \sim \mathcal{N}(0, \lambda_v^{-1} I)$
   - (b) 关系与投影矩阵先验：$r \sim \mathcal{N}(0, \lambda_r^{-1} I)$，$M_r \sim \mathcal{N}(0, \lambda_M^{-1} I)$
   - (c) 四元组建模：从 $\sigma\big(f_r(v_h, v_t) - f_r(v_h, v_t')\big)$ 采样

2. **文本知识**（SDAE 每层 $l$）：
   - (a) $W_l \sim \mathcal{N}(0, \lambda_W^{-1} I)$
   - (b) $b_l \sim \mathcal{N}(0, \lambda_b^{-1} I)$
   - (c) $X_l \sim \mathcal{N}\big(\sigma(X_{l-1} W_l + b_l), \lambda_X^{-1} I\big)$

3. **视觉知识**（SCAE 每层 $l$）：
   - (a) $Q_l \sim \mathcal{N}(0, \lambda_Q^{-1} I)$
   - (b) $c_l \sim \mathcal{N}(0, \lambda_c^{-1} I)$
   - (c) 全连接层：$Z_l \sim \mathcal{N}\big(\sigma(Z_{l-1} Q_l + c_l), \lambda_Z^{-1} I\big)$；卷积层：$Z_l \sim \mathcal{N}\big(\sigma(Z_{l-1} * Q_l + c_l), \lambda_Z^{-1} I\big)$

4. **物品偏移向量**：$\eta_j \sim \mathcal{N}(0, \lambda_I^{-1} I)$，然后构造 $e_j = \eta_j + v_j + X_{\frac{L_t}{2}, j*} + Z_{\frac{L_v}{2}, j*}$

5. **用户隐向量**：$u_i \sim \mathcal{N}(0, \lambda_U^{-1} I)$

6. **成对偏好**：对每个 $(i, j, j') \in D$，从 $\sigma(u_i^T e_j - u_i^T e_{j'})$ 采样

#### 优化目标

计算完整后验不可行，转而最大化后验概率（MAP 估计）。最大化后验等价于最大化以下对数似然目标函数：

$$
\begin{aligned}
\mathcal{L} = &\sum_{(i, j, j') \in D} \ln \sigma(u_i^T e_j - u_i^T e_{j'}) \\
&+ \sum_{(v_h, r, v_t, v_t') \in S} \ln \sigma\big(\|v_h M_r + r - v_t M_r\|_2^2 - \|v_h M_r + r - v_{t'} M_r\|_2^2\big) \\
&- \frac{\lambda_X}{2} \sum_{l} \|\sigma(X_{l-1} W_l + b_l) - X_l\|_2^2 \\
&- \frac{\lambda_Z}{2} \sum_{l \notin \{\frac{L_v}{2}, \frac{L_v}{2}+1\}} \|\sigma(Z_{l-1} * Q_l + c_l) - Z_l\|_2^2 \\
&- \frac{\lambda_Z}{2} \sum_{l \in \{\frac{L_v}{2}, \frac{L_v}{2}+1\}} \|\sigma(Z_{l-1} Q_l + c_l) - Z_l\|_2^2 \\
&- \frac{\lambda_U}{2} \sum_i \|u_i\|_2^2 - \frac{\lambda_I}{2} \sum_j \|e_j - v_j - X_{\frac{L_t}{2}, j *} - Z_{\frac{L_v}{2}, j*}\|_2^2 \\
&- \frac{\lambda_v}{2} \sum_v \|v\|_2^2 - \frac{\lambda_r}{2} \sum_r \|r\|_2^2 - \frac{\lambda_M}{2} \sum_r \|M_r\|_2^2 \\
&- \frac{1}{2} \sum_l \big(\lambda_W \|W_l\|_2^2 + \lambda_b \|b_l\|_2^2\big) \\
&- \frac{1}{2} \sum_l \big(\lambda_Q \|Q_l\|_2^2 + \lambda_c \|c_l\|_2^2\big)
\end{aligned}
$$

各项含义：

| 项 | 含义 |
| --- | --- |
| 第一项 | BPR 成对排序损失：最大化正样本与负样本的得分差 |
| 第二项 | 结构嵌入损失：最大化正三元组与负三元组的能量差 |
| 第三项 | SDAE 重建损失：逐层最小化重建误差 |
| 第四项 | SCAE 卷积层重建损失 |
| 第五项 | SCAE 全连接层重建损失 |
| 第六项起 | L2 正则化项：分别约束用户向量、物品偏移向量、实体嵌入、关系嵌入、投影矩阵以及各层权重与偏置 |

其中 **第六项正则化尤为关键**：$\|e_j - v_j - X_{\frac{L_t}{2}, j*} - Z_{\frac{L_v}{2}, j*}\|_2^2$ 显式约束物品最终表示 $e_j$ 不能偏离 "CF 偏移 + 三类知识嵌入之和" 太远——这等价于要求 $e_j$ 必须在 CF 信号和知识嵌入的共同约束下学习，是 "联合" 学习的核心机制。

#### 训练过程

采用随机梯度下降（SGD）优化。每次迭代：

1. 随机采样一个 BPR 三元组 $(i, j, j') \in D$
2. 找到包含物品 $j$ 或 $j'$ 的结构嵌入四元组子集 $S_{j, j'} \subset S$
3. 对目标函数关于各参数求梯度并更新

> CKE 的联合学习不是交替优化——每个 SGD step 同时更新 CF 参数和三类知识嵌入组件的参数。这与 [[2019_KDD_KGAT]] 的交替优化策略不同。

#### 预测

训练完成后，用户 $i$ 对物品 $j$ 的偏好分数为：

$$
\hat{y}(i, j) = u_i^T e_j
$$

按分数降序排列所有未交互物品，取 Top-K 作为推荐结果：

$$
i : j_1 > j_2 > \cdots > j_n \;\Longleftrightarrow\; u_i^T e_{j_1} > u_i^T e_{j_2} > \cdots > u_i^T e_{j_n}
$$

### 小结

CKE 的方法论核心可以总结为三层架构：

1. **嵌入层**：三个并行的嵌入组件——Bayesian TransR（结构）、Bayesian SDAE（文本）、Bayesian SCAE（视觉）——各自从异构知识中提取语义表示
2. **融合层**：通过最直接的 **向量加法** 将 CF 隐向量与三类知识嵌入融合为统一的物品表示 $e_j$
3. **优化层**：在统一的贝叶斯框架下，通过 MAP 估计将推荐排序损失（BPR）与各嵌入组件的重建损失/排序损失联合优化

这一架构的优点是简洁直观——后续工作正是在这三个层面上做文章：
- [[2019_KDD_KGAT]] 将结构嵌入从单跳 TransR 替换为多跳 GNN 传播，并引入注意力机制学习融合权重
- [[2021_WWW_KGIN]] 将融合层的简单加法替换为意图感知的细粒度聚合，并将关系嵌入显式融入消息构造
