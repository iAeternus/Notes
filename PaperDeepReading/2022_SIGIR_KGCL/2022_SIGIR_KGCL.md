# **精读笔记**

## 快速总结

真实世界的知识图谱存在长尾实体分布和话题无关连接两类噪声，现有 GNN-based KG 推荐模型（如 KGAT、KGCN）在聚合多跳邻居时不加区分，噪声随层数放大。

KGCL 的核心思路是：对 KG 做两次随机三元组 mask 生成两个增广视图，用同一物品在两个视图下表示的一致性高低来量化其受 KG 噪声影响的程度——一致性高的物品 KG 信息可靠，一致性低的则被噪声污染。这个一致性分数反过来指导 U-I 交互图的边 dropout：可靠物品的交互边保留，噪声物品的交互边丢弃。最后两个视图在表示空间中做 InfoNCE 对比学习，整个过程端到端联合训练。

### 核心贡献

1. 首次将知识图谱学习与用户-物品交互建模纳入联合 **自监督学习** 范式，提升推荐鲁棒性，缓解数据噪声和稀疏问题
2. 提出通用框架 KGCL，核心机制是知识图谱引导的拓扑去噪，通过跨视图自判别信号做对比学习，并附理论分析论证其区分 true/false hard negative 的能力
3. 在 Yelp2018、Amazon-Book、MIND 三个数据集上全面验证了整体性能、冷启动用户、长尾物品、KG 注入噪声四种场景下的一致优越性

## 局限性

1. KG 增广用固定概率随机 mask，所有三元组被 mask 的概率相同，未区分不同关系类型的重要性差异（如 "主演" 对电影推荐远比 "出生地" 重要）
2. 结构一致性 $c_i$ 依赖两次随机 mask 的具体结果，同一物品跑两次会得到不同 $c_i$，方差未讨论
3. 对比学习的负样本来自同一 batch 内其他节点，batch size 不够大时负样本覆盖面和硬度受限
4. 用户表示完全来自 U-I 交互图的 LightGCN 传播，未像 KGIN 那样直接从知识图谱建模用户兴趣——去噪后的 KG 信息是否可增强用户侧表示未探索

## 研究背景

推荐系统从矩阵分解演进到 GNN 协同过滤（如 LightGCN），表达能力提升但数据稀疏问题未变。知识图谱在物品之间建立属性、关系层面的语义连接，弥补协同信号的稀疏性。但真实 KG 并非高质量——大量实体只被极少三元组覆盖（长尾分布），且物品与实体的连接经常与主题无关。

按信息利用方式，已有 KG 增强推荐模型分三类：

| 名称 | 解释 | 缺陷 | 参考文献 |
| ---- | ---- | ---- | -------- |
| 基于嵌入（Embedding-based） | 用 TransE/TransR 等方法预训练实体和关系向量，注入到推荐模型的物品表示中 | 只建模一跳三元组，无高阶传播；对 KG 噪声无防护 | [[2016_KDD_CKE]] |
| 基于路径（Path-based） | 在 KG 上提取元路径，沿路径传播用户偏好或聚合物品语义 | 元路径定义依赖领域专家和大量人工，路径质量难保证 | RippleNet、MCRec |
| 基于 GNN（GNN-based） | 在 KG 上递归执行消息传播，聚合多跳邻居信息，捕获高阶关系依赖 | 假设 KG 高质量——长尾实体和话题无关连接会随层数增加被放大，误导物品表示 | [[2019_KDD_KGAT]]、KGCN、KGIN |

另外需提 SGL——基于图对比学习的推荐模型，直接在 U-I 交互图上做随机 node/edge dropout 构造增广视图。SGL 的问题在于完全随机 drop 交互边，不知道哪些边更该保留。KGCL 与 SGL 的关键区别：KGCL 的 dropout 不是随机的，而是由 KG 结构一致性指导的。

**KGCL 的定位：** 属于 GNN-based KG 推荐这条线，但不假设 KG 高质量，本质是在 GNN 传播之前先给 KG 和交互图 "去噪"。

## 方法论

KGCL 整体结构分四部分：

1. 在原始 KG 上用异质图注意力做物品和实体的消息聚合，同时用 TransE 平移假设辅助训练增强关系语义空间
2. 用随机 mask 给 KG 做两次增广，通过两次增广下物品表示的一致性量化噪声影响
3. 用一致性分数指导 U-I 交互图的边 dropout
4. 两个增广视图下做跨视图对比学习，BPR 推荐损失和对比损失联合优化

```text
                原始知识图谱
                      │
           随机Mask两次（三元组）
             ┌──────────────┐
             │              │
          KG View1      KG View2
             │              │
      GNN聚合得到物品表示  GNN聚合得到物品表示
             │              │
             └───计算一致性──┘
                     │
           一致性高 → KG可靠
           一致性低 → KG噪声多
                     │
        ↓ 指导用户-物品图 Edge Dropout
             ┌──────────────┐
             │              │
        UI View1        UI View2
             │              │
          LightGCN      LightGCN
             │              │
             └────InfoNCE───┘
                     │
               BPR + Contrast
                     │
                 联合优化
```

### 符号表

#### 数据定义

| 符号 | 含义 |
| ---- | ---- |
| $U$ | 用户集合 |
| $I$ | 物品集合 |
| $Y \in \{0, 1\}^{|U| \times |I|}$ | 用户-物品交互矩阵，$y_{u,i}=1$ 表示有交互 |
| $G_u = \{V, E\}$ | 用户-物品交互二分图，$V = U \cup I$，边 $(u,i)$ 在 $y_{u,i}=1$ 时存在 |
| $G_k = \{(h, r, t)\}$ | 知识图谱，$(h, r, t)$ 表示头实体 $h$ 经关系 $r$ 连接到尾实体 $t$ |
| $N_i$ | 物品 $i$ 在 $G_k$ 中的邻居实体集合 |
| $N_u$, $N_i$ | LightGCN 传播中用户 $u$ 的交互物品集合 / 物品 $i$ 的交互用户集合 |

#### 可学习参数

| 符号 | 维度 | 含义 |
| ---- | ---- | ---- |
| $x_u$ | $\mathbb{R}^d$ | 用户 $u$ 的嵌入向量 |
| $x_i$ | $\mathbb{R}^d$ | 物品 $i$ 的嵌入向量 |
| $x_e$ | $\mathbb{R}^d$ | 实体 $e$ 的嵌入向量 |
| $r_{e,i}$ | $\mathbb{R}^d$ | 物品 $i$ 与实体 $e$ 之间关系 $r(e,i)$ 的嵌入向量 |
| $W$ | $\mathbb{R}^{d \times 2d}$ | 异质注意力权重矩阵 |
| $x_h, x_r, x_t$ | $\mathbb{R}^d$ | TransE 语义增强中头实体/关系/尾实体的嵌入向量 |

#### 超参数

| 符号 | 含义 |
| ---- | ---- |
| $p_k$ | KG 增广的随机 mask 概率 |
| $p_\tau$ | 交互图 dropout 概率的截断阈值 |
| $p_a$ | 均值影响力控制系数 |
| $\tau$ | InfoNCE 损失的温度参数 |
| $\lambda_1$ | 对比损失权重 |
| $\lambda_2$ | L2 正则化系数 |

### 关系感知知识聚合

知识图谱中不同类型关系承载不同语义，不应统一处理。对于物品 $i$ 和邻接实体 $e$，注意力权重由关系向量 $r_{e,i}$、实体表示 $x_e$、物品表示 $x_i$ 三者共同决定：

$$
\begin{align}
x_i &= x_i + \sum_{e \in N_i} \alpha(e, r_{e, i}, i) \cdot x_e \\[4pt]
\alpha(e, r_{e, i}, i) &= \frac{\exp\Big(\text{LeakyReLU}\big(r_{e, i}^\top W [x_e \| x_i]\big)\Big)}{\sum_{e' \in N_i} \exp\Big(\text{LeakyReLU}\big(r_{e', i}^\top W [x_{e'} \| x_i]\big)\Big)}
\end{align}
$$

残差连接保留物品本身的原始信息。$\alpha$ 把关系向量 $r_{e,i}$ 放在外面跟线性投影后的拼接向量做内积，同一条边在不同关系类型下会得到不同的注意力分数。相比 KGAT 用 $(W_r e_t)^T \tanh(W_r e_h + e_r)$ 计算注意力，KGCL 直接用拼接映射加关系内积，省去了为每个关系学一个投影矩阵的参数开销。

GNN 聚合学到的是 "给定关系下邻居有多重要"，但没有显式建模 "头实体 + 关系 ≈ 尾实体" 的平移结构。因此 KGCL 额外引入 TransE 做辅助训练，交替优化以下损失：

$$
\mathcal{L}_{TE} = \sum_{(h, r, t, t') \in G_k} -\ln \sigma\Big(f_d(x_h, x_r, x_{t'}) - f_d(x_h, x_r, x_t)\Big)
$$

其中 $f_d = \|x_h + x_r - x_t\|_1$ 是 L1 距离，$t'$ 是随机替换尾实体生成的负样本。使得关系空间嵌入保持平移语义——Head + Relation ≈ Tail 的几何约束让同一关系类型的实体对在向量空间中有一致的偏移模式。

### 知识图谱增强

核心问题：某个物品的 KG 信息到底可不可靠？看物品表示对 KG 结构扰动敏不敏感。

对 KG 做两次相互独立的随机三元组 mask，每次按概率 $p_k$ 决定每条三元组 $(e, r, i)$ 是否保留，两套独立的二值 mask 向量为 $M_k^1$ 和 $M_k^2$：

$$
\eta_1(G_k) = \{(e, r, i) \odot M_k^1\}, \quad \eta_2(G_k) = \{(e, r, i) \odot M_k^2\}
$$

在两个增广子图上分别跑关系感知聚合（Eq 1），得到物品 $i$ 的两个增广表示 $f_k(x_i, \eta_1(G_k))$ 和 $f_k(x_i, \eta_2(G_k))$。两个表示接近说明物品对 KG 结构变化不敏感，噪声影响小；差异大说明去掉某些三元组后表示剧烈变化，很可能依附于噪声实体。

量化为 KG 结构一致性分数 $c_i$：

$$
c_i = s\Big(f_k(x_i, \eta_1(G_k)),\, f_k(x_i, \eta_2(G_k))\Big)
$$

$s(\cdot)$ 是余弦相似度。$c_i$ 越大，物品越 "稳定"。这个分数是 KGCL 的枢纽——把 KG 侧的噪声估计传递到 U-I 交互侧，决定下一阶段交互图增强的方向。

### 知识引导的对比学习

#### 交互图增强

用指数函数把 $c_i$ 映射为影响力权重 $w_{u,i} = \exp(c_i)$，做 min-max 归一化并引入截断概率 $p_\tau$：

$$
p'_{u, i} = \max\left(\frac{w_{u, i} - w_{min}}{w_{max} - w_{min}},\; p_\tau\right)
$$

引入均值影响力 $p_a$ 调节整体 dropout 强度：

$$
p_{u, i} = p_a \cdot \frac{\mu_{p'}}{p'_{u, i}}
$$

$p'_{u,i}$ 在分母上——$c_i$ 越高的物品，$p_{u,i}$ 越低，交互边不容易被 drop；$c_i$ 越低的物品，$p_{u,i}$ 越高，交互边大概率被丢弃。用 $p_{u,i}$ 作为 Bernoulli 分布参数生成两套 mask $M_u^1, M_u^2$，得到两个增广交互图：

$$
\varphi_1(G_u) = (V, M_u^1 \odot E), \quad \varphi_2(G_u) = (V, M_u^2 \odot E)
$$

与 SGL 的关键区别：dropout 不是随机的，每个交互边的丢留由物品的 KG 质量决定。

#### LightGCN 协同编码

物品的初始嵌入来自 Eq (1) 的关系感知 KG 聚合结果，用户的初始嵌入随机初始化。第 $l+1$ 层传播：

$$
x_u^{(l+1)} = \sum_{i \in N_u} \frac{x_i^{(l)}}{\sqrt{|N_u| |N_i|}}, \quad x_i^{(l+1)} = \sum_{u \in N_i} \frac{x_u^{(l)}}{\sqrt{|N_i| |N_u|}}
$$

没有可学习的权重矩阵和特征变换，只保留对称归一化求和——LightGCN 的精简设计，去掉特征变换和非线性激活后对协同过滤更有效。堆叠 $L$ 层后取最后一层：

$$
x_u = x_u^{(L)}, \quad x_i = x_i^{(L)}
$$

两个增广视图各自过一遍编码器，产生两套表示 $(x_u^1, x_i^1)$ 和 $(x_u^2, x_i^2)$。同一节点在两个视图下的表示不同，因为交互图结构被不同的 dropout mask 修改过，物品侧的初始 KG 聚合也在不同的 KG 增广子图上做。

#### 跨视图协同对比学习

对于每个用户 $u$（或物品 $i$），正样本是它在另一个视图下的表示，负样本是所有其他节点在另一个视图下的表示。InfoNCE 损失：

$$
\mathcal{L}_c = \sum_{n \in V} -\ln \frac{\exp(s(x_n^1, x_n^2) / \tau)}{\sum_{n' \in V, n' \neq n} \exp(s(x_n^1, x_{n'}^2) / \tau)}
$$

$s(\cdot)$ 是余弦相似度，$\tau$ 是温度参数。负样本是 in-batch 的。

#### 联合训练

主任务 BPR 成对排序损失：

$$
\mathcal{L}_b = \sum_{u \in U} \sum_{i \in N_u} \sum_{i' \notin N_u} -\ln \sigma(\hat{y}_{u, i} - \hat{y}_{u, i'})
$$

其中 $\hat{y}_{u,i} = x_u^\top x_i$。总损失：

$$
\mathcal{L} = \mathcal{L}_b + \lambda_1 \mathcal{L}_c + \lambda_2 \|\Theta\|_2^2
$$

所有参数共享同一个嵌入空间，BPR 梯度、对比学习梯度、TransE 梯度三股力量同时作用于同一套用户/物品/实体/关系向量。

### 理论分析

核心落在 "false hard negative" 概念上。给定某节点与负样本的相似度 $s$，对比学习梯度函数：

$$
g(s) = \sqrt{1 - s^2} \cdot \exp\left(\frac{s}{\tau}\right)
$$

当 $s$ 在 0.7-0.9 之间时，$g(s)$ 可冲到 40 左右；$s$ 小于 0.5 时梯度几乎为零。训练主要被 hard negative 驱动，easy negative 贡献极少。

KG 噪声的影响：噪声实体让两个本来不相似的物品学出相似的表示，在对比学习中成为 "false hard negative"——看起来是难区分的负样本，实际因噪声被错误推到一起。无去噪时，这些 false hard negative 的相似度 $s$ 落在高梯度区，产生极大误导性梯度：

$$
\|\theta_r(U_f, v) - s_0\| \gg t_0
$$

其中 $U_f$ 是 false hard negative 集合，$s_0$ 是梯度峰值点，$\theta_r$ 是无去噪时估计的相似度。

KGCL 去噪后，false hard negative 的相似度远离梯度峰值：

$$
\|\theta_r(U_f, v) - s_0\| \geq \|\theta_a(U_f, v) - s_0\| \gg \|\theta_f(U_f, v) - s_0\|
$$

对于 true hard negative，去噪反而让相似度被更准确估计，保持在梯度峰值附近：

$$
\|\theta_f(U_t, v) - s_0\| \gg \|\theta_a(U_t, v) - s_0\| \geq \|\theta_r(U_t, v) - s_0\|
$$

**KGCL 通过去噪，把不该相似的物品推远（false hard negative 不再误导训练），把真正相似的物品拉近（true hard negative 继续提供高质量训练信号）**

时间复杂度：KG 聚合 $O(|E_k| \times d)$，KG 增广 $O(|E_k| + |V| \times d)$，LightGCN 传播 $O(|E| \times d)$，InfoNCE $O(B_{id} \times (|U| + |I|) \times d)$，与 KGAT 等同属一个量级。

## 实验

### 数据与设置

三个数据集覆盖不同场景：Yelp2018（餐饮/商户）、Amazon-Book（图书）、MIND（新闻）。KG 构建方式：Yelp 和 Amazon 映射到 Freebase 实体，MIND 用 spacy-entity-linker 链接到 Wikidata。评估协议沿用 all-ranking 策略，指标 Recall@20 和 NDCG@20，嵌入维度统一 64。

### 整体结果

KGCL 在三个数据集、两个指标上全部最优。关键观察：

1. 比 KGIN 提升约 3-6%，比 SGL 提升约 5-8%
2. KGIN 在 KG 增强方法里排第二，说明细粒度意图建模有效，但 KGIN 同样未处理 KG 噪声
3. SGL 作为纯自监督方法已超过大部分 KG 增强方法，说明 U-I 图上做对比学习本身就强——但完全随机 dropout 是其短板

### 消融实验

两个变体：
- **w/o KGA**：去掉 KG 引导的交互图增强，改回 SGL 同款随机 edge dropout
- **w/o KGC**：去掉整个 KG 对比学习模块，直接用关系感知聚合后的表示做 CF 对比

两个变体均比完整 KGCL 差。去掉 KG 对比学习的影响比去掉交互图增强更明显——KG 侧的对比学习（Eq 4 + Eq 8）对去噪贡献最大。

### 鲁棒性分析

1. **长尾物品**：按交互密度分五组，KGCL 在最稀疏组优势最大，提升超过 10%
2. **冷启动用户**：Yelp/Amazon 交互 < 20、MIND 交互 < 5 的用户，KGCL 同样优势明显
3. **KG 噪声注入**：往 KG 随机注入 10% 噪声三元组，KGAT 平均掉 13.57%，KGIN 掉 3.37%，KGCL 只掉 0.58%——几乎不受影响。KG 越脏，KGCL 优势越大

### 案例分析

MIND 数据集中两条新闻案例：

- **Kevin Spacey 法律纠纷**：KG 连出 Democratic Party（政治立场）和 Juilliard School（毕业院校），与新闻主题无关。不开 KGC 时推荐的全是政治/国家主题新闻；开 KGC 去噪后推荐的是 Cuba Gooding Jr. 法律纠纷和 The Hunchback of Notre Dame 舞台剧——全是娱乐圈/电影相关
- **阿里巴巴新闻**：去噪前推荐偏中国政治/国家新闻，去噪后回归 Big Tech 公司新闻

说明 KGCL 的去噪不是无差别削弱所有 KG 信息，而是有针对性地压制话题不相关的实体连接，留下真正语义相关的部分。

## 结论

KGCL 反过来琢磨怎么有选择地不用 KG 里的信息。两层增强配合跨视图对比学习给出自然的去噪机制：KG 结构一致性发现噪声，交互图 dropout 抑制噪声传播，对比学习提供自监督信号让降噪后表示更鲁棒。方法与具体 CF backbone 无关，论文用 LightGCN，换成其他 GNN 结构也能插进去。

