# CLIP
Compact Layout Index for Pgvector (HNSW)

全国大学生计算机系统能力大赛-PolarDB数据库创新设计赛二等奖作品，决赛第一阶段排名第九。

原代码仓库地址：https://gitee.com/hin629/polardb_competition_2025.git

---

## 核心优化思路

### 目标问题与总体策略

我们围绕 [HNSW](#hnsw) 向量索引的查询吞吐与构建效率进行优化，重点针对以下三类瓶颈建立“算法压缩 + 内存布局 + 计算加速”的组合方案：

| 主要瓶颈                        | 对应策略                                                     |
| ------------------------------- | ------------------------------------------------------------ |
| **内存占用偏高**                | 引入 **[SQ8](#sq8) 标量量化**，将 `float32` 压缩为 `uint8`，在可控精度损失下实现约 **4×** 存储压缩 |
| **搜索阶段缓存/页表未命中偏高** | 基于 [DSM](#dsm) 构建共享内存 **Slab**，以紧凑 **[AOS](#aos)** 布局存放节点热数据，并在建图后进行 **BFS 重排**以提升局部性 |
| **距离计算开销**                | 批量 [SIMD](#simd) 距离计算，适配 **[AVX2/AVX-512](#avx2-avx512)/[VNNI](#vnni)** 指令集以提升计算吞吐 |

### 优化策略

#### 1. 算法优化

##### （1）SQ8 标量量化

将 `float32` 向量按线性映射量化为 `uint8`，在显著降低向量存储与带宽压力的同时尽量保持召回率。量化参数在索引构建阶段通过采样估计得到，并采用**全局共享参数**（而非为每个向量单独存储参数），从而避免额外元数据带来的空间开销。
 在我们的测试中，量化带来的存储缩减不仅降低了常驻内存压力，也改善了建图与查询阶段的有效吞吐（建图阶段收益更明显）。

##### （2）面向 [L2](#l2) 距离的等价变形

利用恒等式：
$$
|A - B|^2 = \sum_i a_i^2 + \sum_i b_i^2 - 2 \sum_i a_i b_i
$$
查询阶段可将计算重心转移到内积项，并在 SQ8 路径中以整数内积为主，避免反量化回 `float` 的开销；同时配合 VNNI 指令提升内积吞吐。该变形在数学上等价，不改变最近邻的排序语义。

#### 2. 工程优化

##### （1）共享内存 Slab

基于 PostgreSQL DSM 构建 Slab，将索引运行时所需的图结构与节点热数据紧凑存放在**单一连续 DSM 段**中，通过**偏移量寻址**替代大量指针解引用。即使原版索引数据能够被 OS 缓存驻留，碎片化分配与频繁指针追逐仍会带来显著的缓存/页表未命中与访存等待；Slab 的连续布局能够降低这一类开销，从而减少 cache miss 对吞吐的影响。
 Slab 不改变原 HNSW 的建图与查询逻辑，主要差异在于数据获取方式：邻接表存储连续编号的 `internalId`，查询时可通过 `internalId` 直接定位到节点的 `[adj | vector | meta]`，更有利于顺序访问与硬件预取。

##### （2）建图后 BFS 重排

在索引构建完成后对节点进行 BFS-based 物理重排。该过程仅改变节点在内存中的排列与 `internalId` 映射关系，不改变每个节点的邻接集合；重排后同步修正邻接表中的 `internalId`，保证图拓扑与搜索语义保持一致。实验表明，该重排能够改善页级局部性（[DTLB](#dtlb) 的 miss 下降），从而进一步提升查询效率。

##### （3）显式预取

在建图/查询的距离计算与邻居扩展路径中加入显式预取，通过 `_mm_prefetch` 对后续将访问的节点数据进行预取，以减少 CPU 对于数据的等待时间。预取距离采用固定步长（如 `i+4`），在我们的测试中相较 `i+1` 更稳定，`i+2` 与 `i+4` 差异不明显（具体收益与硬件与工作负载相关）。

##### （4）AOS 布局选择

节点采用 AOS（结构体数组）布局 `[adj0][vector][meta]`，将一次扩展所需的热字段尽量聚集在相邻 cacheline/页内，从而提升缓存命中与访问效率。我们也评估过 [SOA](#soa)（数组结构体），但在该访问模式下 SOA 更容易导致一次扩展触发多次离散访问；相对而言 AOS 更有利于将邻接与向量访问合并为更少的内存触碰，从而获得更稳定的吞吐表现。

#### 3. 其他轻量优化

| 优化手段     | 位置          | 说明                                    |
| ------------ | ------------- | --------------------------------------- |
| 批量距离计算 | `hnswutils.c` | 同时计算 4 个目标，复用解包/转换结果    |
| 标签式访问集 | `hnswutils.c` | O(1) 重置 visited，降低热路径初始化开销 |
| 搜索候选池   | `hnswutils.c` | 预分配避免热路径 `palloc`               |
| 64 字节对齐  | `hnswbuild.c` | 提升 SIMD 友好性与对齐访问概率          |

以上属于“边角成本削减”，整体带来的 [QPS](#qps) 增益约 **5%**。

### 预期效果与实测概览

| 指标     | 预期影响                                                | 实测概览                                           |
| -------- | ------------------------------------------------------- | -------------------------------------------------- |
| 内存占用 | 约 **4×** 压缩（理论 75% 减少）                         | 数据压缩约 **72%**，召回率损失约 **1%**            |
| 查询开销 | 降低 [LLC](#llc)/[TLB](#tlb) 未命中，并提升 SIMD 利用率 | cache miss 导致的等待明显降低（体现在 QPS 提升上） |
| QPS      | 受数据维度、局部性与 SIMD 利用率影响，存在显著提升空间  | 本地 128 维约 **+100%**；比赛 200 维约 **+300%**   |



---

## 核心设计与关键技术

### 1. SQ8 标量量化系统

在向量索引的整体开销中，向量数据本身往往是内存与带宽的主要来源。对于 HNSW 这类图结构检索，向量越大、访问越随机，缓存与 TLB 的压力就越明显：不仅常驻内存占用上升，查询过程中“把向量搬到 CPU”的成本也会成为吞吐瓶颈。基于这一点，我们在不改变索引接口与基本流程的前提下，引入 SQ8 标量量化，将 `float32` 向量压缩为 `uint8` 表示，以降低存储体积与内存带宽占用；同时结合 L2 距离的等价变形，把查询阶段的计算重心转为内积，并利用 VNNI 指令进一步提升吞吐。

#### 核心策略：

##### （1）全局量化参数策略：

我们采用**全局共享**的量化参数 `(lower_bound, diff)`，并将其存放在 HNSW 的 `metapage` 中，使得所有向量共用同一套映射关系，从而避免为每个向量额外存储量化参数（否则每个向量需额外保存 `lower_bound` 与 `diff` 两个 `float`，即 8 bytes/vector）。该设计保证 SQ8 仍能保持紧凑、连续的存储布局优势。

> 权衡说明：若采用每向量独立量化参数，召回率在我们的实验中可接近 100%，但索引中除了 `uint8` 编码外还需额外存储两份 `float` 数据；在大规模数据下，该额外开销会显著削弱压缩收益，且会增加访问与布局复杂度，因此最终选择全局参数方案。

量化与反量化关系如下（实现中对 `code` 会进行饱和截断以保证落在 `[0,255]`）：

```
量化公式：code = round((value - lower_bound) / diff * 255)
解码公式：value ≈ lower_bound + code * diff / 255
```

##### （2）L2 等价变形与 VNNI 内积加速

为降低查询阶段的浮点计算开销，我们将 L2 距离计算转化为内积主导的形式。在索引元组中预存每个向量的 `sum2 = ||B||^2`（以及实现中需要的辅助项），查询时仅需计算内积项并组合得到距离（其中 `||A||^2` 对同一条 query 为常数项，不影响排序）。在 SQ8 路径中，内积采用整数累加，并通过 **VNNI** 指令集（相较普通 AVX2/AVX-512 内积实现）获得约 **3–5%** 的额外吞吐提升。

**QPS 对比：**

| 场景     | 优化前     | 优化后     | 提升       |
| -------- | ---------- | ---------- | ---------- |
| 比赛测试 | 11987.7860 | 12655.9500 | **+5.57%** |
| 本地测试 | 18541.4990 | 19130.8140 | +3.18%     |

**数据规模对比**

| 向量类型           | 数据 Table 大小 | 索引大小 |
| ------------------ | --------------- | -------- |
| `vector` (float32) | 8.6 GB          | 11.0 GB  |
| `sq8vec` (uint8)   | 2.4 GB          | 4.4 GB   |

相比 `vector(float32)`，`sq8vec(uint8)` 在数据表侧约 **3.6×** 压缩、在索引侧约 **2.5×** 压缩（不同部分受元数据与图结构占比影响）。



### 2. 共享内存 Slab 架构

HNSW 的性能瓶颈在多数场景下并不来自“距离计算本身”，而来自“将下一步要访问的向量、邻接表与元信息及时送到 CPU”。在原版实现中，即使系统内存充足、索引相关页面大多可驻留在页缓存中，搜索过程仍会因为节点对象、邻接表、元信息与向量数据**分散存放**而触发大量指针追逐与随机访存。由于图搜索的访问序列由邻接关系动态决定，下一跳地址高度不规则，硬件预取器难以有效预测，最终表现为 **LLC miss** 与 **TLB miss** 主导的长延迟停顿：吞吐受到内存访问延迟约束，尾延迟也容易被随机 miss 拉长。

为此，我们将“查询热路径必经的数据”（邻接表、向量数据与必要元数据）从离散对象重组为紧凑的连续布局，把访问模式从“指针解引用 + 任意地址跳转”转化为“数组下标 + 可预测偏移”的形式，使 CPU 更容易形成顺序访问与预取，从而降低昂贵的缓存与页表未命中开销。

考虑到 PostgreSQL 的多进程架构，如果每个后端进程各自缓存一份索引数据，不仅会带来重复内存占用，而且缓存生命周期与命中行为难以控制，容易出现抖动。我们因此基于 PostgreSQL DSM 为每个索引构建一个**单段连续**的共享内存区域，实现“构建一次，多后端 attach 复用”。同时，由于不同进程 attach 后得到的虚拟地址并不固定，Slab 内部不能保存普通指针；我们将所有访问统一改为 `internalId/offset` 寻址，使其具备可重定位性，保证任意后端 attach 后都能正确访问。

Slab 的邻接表仅保存邻居的 `internalId(uint32)`。访问邻居时，通过 `base + internalId * stride` 直接定位到邻居的数据块，再以固定偏移读取该节点的邻接、向量与元信息，从而显著减少查询路径上的指针追逐与不可预测跳转。对 CPU 而言，这种“连续数组 + 固定步长”的访问形态更利于预取与批量计算，也更稳定地控制跨页触碰。

#### 核心策略

（1）**单段连续内存**：使用 PostgreSQL DSM 分配单一连续段，将热路径数据集中存放，降低随机访存与指针追逐成本。
（2）**偏移量寻址**：使用 `internalId/offset` 替代进程相关指针，确保 Slab 可重定位并减少解引用开销。

Slab 主要由以下部分组成：全局头部（元数据与偏移表）、第 0 层数据池（AOS）、上层邻接表池（紧凑 chunks）、节点锁池、标签查找表、删除元素池。

<img src="C:\Users\littl\AppData\Roaming\Typora\typora-user-images\image-20260107171556684.png" alt="image-20260107171556684" style="zoom:33%;" />

为保证结构紧凑并兼顾动态写入，我们对 Slab 内存按功能分区分配：

- **Level 0 池**：按 `numElements` 分配固定大小元素。由于所有节点均包含 level0 邻接表，因此将 level0 邻接与向量数据放在同一元素块内分配，以减少访问时的离散跳转。

<img src="C:\Users\littl\AppData\Roaming\Typora\typora-user-images\image-20260107173146763.png" alt="image-20260107173146763" style="zoom:25%;" />

- **Upper Layers 池**：上层邻接结构因节点 level 不同而不定长。为避免在元素内内嵌可变长结构造成空间浪费，我们采用等大小 chunk 的上层邻接池，并在 meta 中记录 `adjUpperOffset` 作为起点。访问时通过“chunk 偏移 + layerIndex”完成纯偏移寻址，既节省内存，又保持连续可预取的访问形态。
- **预留缓冲**：额外预留 10000 个元素空间用于动态插入（包含 Level0 池与 Upper Layers 池的预留）。
- **删除元素池**：采用环形缓冲区复用已删除节点的 `internalId`。
- **其余固定结构**：锁、header 等为轻量固定数据，直接按需分配。

在元素 meta 中我们还维护 `deleted` 标记，用于 DELETE/VACUUM 相关流程，以及后续插入时对已删除槽位的复用，从而提升内存利用效率。

Slab 的存储可适配多种索引向量类型；每个索引分配的共享内存大小由“单元素大小 × 元素数量 + 额外池开销”决定，而不依赖于原始磁盘结构的指针组织方式。

在实现上，Slab 可根据磁盘索引重建：数据库重启后，查询路径会优先检测并 attach 已存在的 Slab；若 Slab 尚未构建且内存资源允许，则会在首次访问时扫描索引并重建 Slab（该过程会牺牲首条查询性能，但重建完成后可恢复高吞吐查询）。
 **注**：本文档对应的提交为了缩短建图时间，省略了建图后的写盘过程；如需验证“落盘后重建 Slab”的能力，可参考仓库最终提交（eb52184b3f6221290bf96f7166ebb4eba1cefc7e）进行测试。

#### 效果对比

**Slab 显著降低 LLC/DTLB 压力**

我们在相同 workload 下，对“无 Slab”与“有 Slab”两种实现采集硬件性能计数器（每组 n=10），并比较关键指标的**中位数**：

| 指标                  | 无 Slab       | 有 Slab       | 变化       |
| --------------------- | ------------- | ------------- | ---------- |
| cache_miss_rate       | 21.513%       | 16.757%       | -22.1%     |
| llc_miss_rate         | 51.724%       | 44.324%       | -14.3%     |
| **llc_misses_per_s**  | **4.139e6/s** | **2.067e6/s** | **-50.1%** |
| dtlb_miss_rate        | 0.713%        | 0.638%        | -10.5%     |
| **dtlb_misses_per_s** | **2.470e6/s** | **1.106e6/s** | **-55.2%** |
| ipc                   | 0.709         | 0.701         | -1.2%      |
| l1d_miss_rate         | 4.908%        | 7.854%        | +60.0%     |
| branch_miss_rate      | 2.235%        | 4.112%        | +84.0%     |

其中我们最关注 **`llc_misses_per_s`** 与 **`dtlb_misses_per_s`**：它们更直接反映图搜索在“节点 → 邻居 → 邻居向量 → 继续扩展”的不规则访问链路中所产生的昂贵停顿。原版结构更容易出现跨对象/跨页跳转与指针追逐，CPU 难以预取，最终以 **LLC miss（下探到更慢层级取数）** 与 **DTLB miss（触发 [page-walk](#page-walk)）** 的形式放大为主要成本。Slab 将图结构与向量数据组织为连续 DSM 段，并采用 `internalId + stride` 的数组式寻址，访问集中度更高、跨页跳转更少，因此这两类“最贵的 miss”近似腰斩：LLC miss 每秒减少约 2.07e6（**-50.1%**），DTLB miss 每秒减少约 1.36e6（**-55.2%**）。因此 Slab 的收益主要来自**降低随机访存与指针追逐引发的内存与页表开销**。

我们也观察到 `l1d_miss_rate` 与 `branch_miss_rate` 在 Slab 版本中上升。这与整体吞吐提升并不矛盾：L1D miss 的代价远低于 LLC miss 与 page-walk，且在访问更紧凑、预取更积极时，L1 发生替换的概率会上升；分支失误率上升则主要来自 Slab 路径中新增的边界校验与状态判断（例如 `internalId` 校验、`deleted` 判断、紧凑邻接表 `count` 约束等）。在 HNSW 这类内存敏感负载下，减少昂贵的 **LLC/DTLB** 停顿比 L1 与分支的波动更能决定最终 QPS。

**QPS：**

| 场景     | 无 Slab    | 有 Slab    | 提升         |
| -------- | ---------- | ---------- | ------------ |
| 比赛测试 | 3117.8300  | 9264.3070  | **+197.14%** |
| 本地测试 | 10311.7610 | 18665.1150 | +81.00%      |



### 3. 索引数据 AOS 布局

在 Slab 中，我们将每个元素按 `[Adj0 | Vector | Meta]` 的顺序紧凑排列，使查询热路径中经常共同访问的字段在物理内存上尽量相邻。对于 HNSW 的交错访问模式，AOS 的优势在于：当访问某个节点时，邻接信息、向量数据与必要元数据更可能落在相邻 cacheline/相邻页内，被一次或少数几次缓存填充带入；同时对向量进行显式预取时，也更容易顺带预热同一元素附近的邻接或 meta，从而降低下一步扩展时发生随机 miss 的概率。

#### 效果对比

**AOS 的页级局部性更好（DTLB 更低），整体执行效率更高（[IPC](#ipc)/QPS 更优），但 LLC 指标不占优**

我们对比了两种 Slab 内存布局：AOS 与 SOA。在相同 workload 下采集硬件性能计数器（每组 n=10），并对关键指标的中位数进行对比：

| 指标                  | SOA             | AOS             | 变化（AOS vs SOA） |
| --------------------- | --------------- | --------------- | ------------------ |
| **ipc**               | **0.698952**    | **0.713793**    | **+2.12%**         |
| cache_miss_rate       | 16.004%         | 16.464%         | +2.87%             |
| l1d_miss_rate         | 8.003%          | 7.857%          | -1.82%             |
| llc_miss_rate         | 41.841%         | 44.490%         | +6.33%             |
| **llc_misses_per_s**  | **1.82646e6/s** | **2.02237e6/s** | **+10.73%**        |
| dtlb_miss_rate        | 0.710%          | 0.640%          | -9.86%             |
| **dtlb_misses_per_s** | **1.19255e6/s** | **1.08276e6/s** | **-9.21%**         |
| branch_miss_rate      | 3.979%          | 3.892%          | -2.19%             |

从结果看，AOS 相对 SOA 的主要收益集中在 **DTLB 层面**：`dtlb_miss_rate` 下降约 **9.86%**，`dtlb_misses_per_s` 下降约 **9.21%**。这与布局机制一致：SOA 将邻接、向量与元信息拆分到不同区域后，热路径会更频繁地在多块内存区域之间跳转，更容易触碰到更多不同页面，从而增加 DTLB 抖动与 page-walk。AOS 将这些热字段聚集在同一元素块附近，使得访问邻居时更可能在更少页面内完成“取邻接 + 取向量 + 读元信息”，因此页级局部性更好、DTLB 压力更低。

与此同时，AOS 在 **LLC 维度并不占优**：`llc_miss_rate` 与 `llc_misses_per_s` 均上升。这意味着 SOA 在“连续读取向量、批量计算距离”的阶段更容易形成缓存复用（向量集中存放的优势）；而 AOS 将邻接/元信息与向量共同放置，会把更多结构性数据一并带入缓存，带来一定的缓存容量与带宽占用，从而使 LLC 指标偏弱。

综合来看，在该 workload 下 AOS 的 **IPC 提升约 2.12%**，并在端到端吞吐上也体现为稳定收益，说明“更少跨页/更少 page-walk”带来的停顿缓解能够抵消 LLC 指标上的劣势。对于当前的比赛环境，**页级局部性（DTLB/page-walk）比纯向量流式读取的 LLC 复用更能决定整体吞吐与尾延迟**，因此我们最终选择 AOS 作为默认布局。

**QPS：**

| 场景     | SOA        | AOS        | 提升       |
| -------- | ---------- | ---------- | ---------- |
| 比赛测试 | 12663.5110 | 13159.8320 | **+3.92%** |
| 本地测试 | 17733.1220 | 18616.3267 | +4.98%     |



### 4. BFS 图重排序

索引构建完成后，我们在 Slab 构建前执行 `HnswRelayoutGraph()`：以入口点（entry point）为起点，沿 **layer0 邻接关系**进行 BFS 遍历，并按 BFS 的出队顺序对 `graph->head` 链表进行重排。随后在 Slab 构建阶段顺链表分配 `internalId` 时，BFS 中相邻访问的节点更可能获得更接近的 `internalId`，从而在 DSM 的连续数组布局中落到更接近的物理位置，提升页级/缓存局部性。
 需要强调的是，该重排仅改变节点的物理排列与编号映射，不改变节点的邻接集合与图拓扑语义；重排后通过同步更新邻接表中的 `internalId` 引用，保证搜索逻辑与结果语义保持一致。

#### 效果对比

**BFS 重排序主要降低 DTLB 压力**

我们在相同 workload 下，对“无 BFS 重排序”和“启用 BFS 重排序”两种实现采集硬件性能计数器（每组 n=10），并对关键指标的中位数进行对比：

| 指标                  | 无 BFS          | 有 BFS          | 变化        |
| --------------------- | --------------- | --------------- | ----------- |
| ipc                   | 0.691971        | 0.708302        | +2.36%      |
| cache_miss_rate       | 16.418%         | 16.327%         | -0.55%      |
| l1d_miss_rate         | 8.349%          | 7.866%          | -5.79%      |
| llc_miss_rate         | 40.985%         | 44.035%         | +7.44%      |
| llc_misses_per_s      | 2.03205e6/s     | 2.03918e6/s     | +0.35%      |
| **dtlb_miss_rate**    | **0.890%**      | **0.638%**      | **-28.31%** |
| **dtlb_misses_per_s** | **1.50940e6/s** | **1.09526e6/s** | **-27.44%** |
| branch_miss_rate      | 3.782%          | 3.847%          | +1.72%      |

可以看到，BFS 重排序最显著的收益集中在 **DTLB** 维度：`dtlb_misses_per_s` 从 1.50940e6/s 降至 1.09526e6/s（**-27.44%**），`dtlb_miss_rate` 从 0.890% 降至 0.638%（**-28.31%**）。这说明 BFS 重排主要通过“让更可能被连续访问的一批节点在内存中更接近”来减少跨页跳转：在 Slab 的数组式寻址下，`internalId` 更接近意味着更可能落在更少的页面内完成一轮“取邻接/取向量/继续扩展”，从而降低 DTLB 抖动并减少 page-walk 次数。

与此同时，`llc_miss_rate` 出现上升，但 `llc_misses_per_s` 基本不变（+0.35%），表明 BFS 重排并未显著改变 LLC miss 的绝对规模；更可能的情况是布局变化带来了访问分布变化，使 miss rate 指标出现波动。但总体来说，BFS 带来的提升主要由 **DTLB_miss 的显著下降**以及 **IPC 的提升**所主导。

**QPS：**

| 场景     | 无 BFS     | 有 BFS     | 提升       |
| -------- | ---------- | ---------- | ---------- |
| 比赛测试 | 12070.1410 | 13159.8320 | **+9.03%** |
| 本地测试 | 17538.1570 | 18184.2210 | +3.68%     |

**建图开销（BFS 额外耗时）：**

| 场景                   | 额外开销 |
| ---------------------- | -------- |
| 比赛测试（10M 数据集） | ~10s     |
| 本地测试（1M 数据集）  | ~4s      |



---

## 整体代码架构

在不改变 pgvector 原版 HNSW 的索引语义（图拓扑、邻居选择逻辑、对外接口与操作行为）的前提下，我们新增了两条面向性能的运行时路径：

**（1）SQ8 全局量化 + SIMD/VNNI 距离内核：** 将 `float32` 向量编码为 `uint8` codes，并结合 `sum/sum2` 的 L2 等价变形，把距离计算主路径转为整数内积累加；在支持的 CPU 上使用 AVX2/AVX-512/VNNI 提升吞吐。

**（2）DSM Slab：**将原本在磁盘格式中分散存放的索引热数据（邻接、向量、必要元信息）在运行时投影到一个连续 DSM 段内，通过 `internalId + stride` 的数组式寻址替代多层指针解引用，并配合 BFS 重排 / AOS / 预取降低 LLC/DTLB 压力。



### 1. 模块分层与依赖关系

整体按三层组织：

- **Index [AM](#am) 层（`hnsw.c/.h`）**：负责与 PostgreSQL 的 Index AM 接口对接，管理构建、扫描、插入、vacuum 等流程。
- **Slab 层（`hnswslab.c/.h`）**：提供 Slab 的构建、attach、访问、并发控制，以及运行时搜索路径。
- **SQ8 层（`sq8.c/.h`）**：提供全局量化参数计算、向量编码/重编码，以及 SIMD/VNNI 距离计算内核。

相对原版的关键变化是：查询、插入以及部分 vacuum 更新会优先走 Slab 路径。



### 2. 数据面：磁盘格式与 Slab

#### （1）磁盘侧

磁盘组织保持原版 HNSW：`metapage + element tuple + neighbor tuple`。本方案仅在 `HnswMetaPageData` 中新增 SQ8 相关字段（`sq8LowerBound / sq8Diff / isSq8Index`）。

#### （2）内存侧

Slab 是磁盘索引的运行时投影，放在单个 DSM 段中：`HnswSlabHeader + pools`。
 结构定义主要位于 `hnswslab.h`，构建与访问逻辑位于 `hnswslab.c`。



### 3. 控制面：Slab 生命周期与并发策略

Slab 由一个全局目录表管理：`(dbid, relid) -> dsm_handle + state`。

**（1）初始化：**`_PG_init()` 调用 `HnswInit()`；其中 `HnswInitLockTranche()` 进一步调用 `HnswSlabShmemInit()`，初始化 Slab。

**（2）构建与发布：**`BuildGraph()` 在图结构仍驻留内存时调用 `BuildSlabFromMemory()`，通过 `HnswSlabBuildBegin / AddVector / SetAdjacency / Finish` 写入 DSM；完成后将目录状态置为 `READY`。

**（3）运行期 attach 与本地缓存：**`HnswSlabAttach()` 映射 DSM 段，并在后端维护轻量的本地 LRU（`HnswSlabLocalCache`），避免每次 scan 反复 attach/detach，降低系统调用与锁开销。



### 4. 关键路径调用链

#### （1）建图

1. `BuildGraph()`：若为 SQ8 索引，先 `Sq8SampleVectors()` 采样并计算全局量化参数，写入 `metapage`。
2. 建图流程保持原版（内存图结构、邻居选择与连接），新增两步：
   - `HnswRelayoutGraph()`：建图后执行 BFS 重排（影响后续 `internalId` 分配顺序）。
   - `BuildSlabFromMemory()`：将完整图写入 DSM Slab（一次性生成运行时紧凑布局）。

#### （2）查询

1. `GetScanItemsSlab()`：调用 `HnswSlabAttachOrBuildFromDisk()` 获取可用 Slab，并读取/缓存 entry point（atomic）。
2. `GetScanValue()`：若启用 SQ8 且输入为 `Sq8Vector`，使用 `Sq8ReencodeToGlobalParams()` 将查询向量按全局参数重编码为 codes-only，并缓存 `sq8EncodedQuery + sum/sum2` 供批量距离计算复用。
3. `HnswSlabSearchUpperLayers()` 与 `HnswSlabSearchBaseLayer()`：在 Slab 的 AOS 与偏移寻址结构上执行 HNSW 搜索；热路径包含显式预取与批量距离计算。

#### （3）插入

1. `HnswInsertTupleOnDisk()`：按原版流程先写入磁盘索引。
2. 非 build 场景下同步更新 Slab：`HnswSlabAddPoint()`将新增点写入 Slab。

#### （4）删除

`hnswvacuum.c` 在更新 `heaptids[]` 后调用 `HnswSlabSyncUpdateHeaptidsFromVacuum()`，用于保持 Slab 视图与磁盘结果一致：

- 主 heaptid 发生变化：更新 `label -> internalId` 映射；
- `heaptids` 为空：标记节点为 deleted，并将 `internalId` 回收至 deleted pool；
- 其他情况：按需更新 `heaptidsLength` 等字段，保证后续查询可见性一致。



---

## 主要修改

#### 新增文件

| 文件名 | 功能描述 |
|--------|----------|
| **hnswslab.c / hnswslab.h** | 共享内存 Slab 架构实现，包含 DSM 段管理、AOS 内存布局、搜索算法 |
| **sq8utils.c / sq8utils.h** | SQ8 量化 SIMD 加速距离函数，支持 AVX2/AVX512/VNNI 多种实现路径 |
| **sq8vec.c / sq8vec.h** | SQ8 向量 PostgreSQL 类型实现，包含量化/解码函数 |

#### 修改文件

| 文件名 | 主要修改内容 |
|--------|-------------|
| **hnsw.h** | 新增 SQ8 支持字段、优化的 visited/search pool 数据结构、Slab 状态 |
| **hnswscan.c** | 使用 Slab 架构替代磁盘 I/O，增加 SQ8 查询向量重编码逻辑 |
| **hnswbuild.c** | 增加 Slab 构建流程、SQ8 全局量化参数采样 |
| **hnswutils.c** | 增加批量距离计算、优化的 visited/search pool 实现、预取优化 |
| **hnswinsert.c** | 增加 Slab 插入支持 |
| **hnswvacuum.c** | 增加 Slab 删除标记支持 |



---

## 实现细节

### 技术栈

- **编程语言**：C
- **SIMD 指令集**：
  - AVX2 / AVX512 / AVX-VNNI / AVX512-VNNI
- **内存管理**：PostgreSQL DSM API
- **并发控制**：LWLock（轻量级锁）、Spinlock
- **开发工具**：GCC/Clang

### 关键数据结构





#### 1. Sq8IndexTuple（SQ8 索引元组）
```c
typedef struct Sq8IndexTuple
{
	int32		vl_len_;		/* varlena header */
	int16		dim;			/* 向量维度 */
	int16		unused;			/* 内存对齐填充 */
	int32		sum;			/* 预计算的和：计算 VNNI 距离时复用 */
	int32		sum2;			/* 预计算的平方和：计算 Cosine/L2 距离时复用 */
	uint8		codes[FLEXIBLE_ARRAY_MEMBER];	/* 量化后的向量数据（纯 uint8 数组） */
}			Sq8IndexTuple;
```
**说明**：全局共享一套量化参数，这里只存必要的数据。`sum` 和 `sum2` 是为了加速 SIMD 计算而预先算好的。

#### 2. HnswSlabHeader（Slab 头部）
```c
typedef struct HnswSlabHeader
{
	uint32		magic;          /* 魔数，校验内存段合法性 */
	uint32		version;        /* 版本控制 */
	
	/* 索引核心参数 */
	int			m;              /* M 参数 */
	int			maxM0;			/* Layer 0 的最大连接数 (2*M) */
	int			dimensions;     /* 向量维度 */
	int			maxLevel;       /* 当前最大层级 */
	
	/* 元素计数（旧字段，兼容用） */
	uint32		numElements;
	
	/* 动态计数器 - 使用原子变量支持并发 */
	pg_atomic_uint32 curElementCount;	/* 当前总元素数（用于 OOB 检查） */
	pg_atomic_uint32 numDeleted;		/* 已删除元素计数 */
	uint32		maxElements;			/* 预分配的最大容量 */
	
	/* 入口点信息（旧字段） */
	uint32		entryPointId;
	int			entryPointLevel;
	
	/* 入口点 - 原子变量支持并发更新 */
	pg_atomic_uint32 entryPointIdAtomic;
	pg_atomic_uint32 entryPointLevelAtomic;	/* 原子存储的层级（-1 存为 MAX_UINT） */
	
	/* 节点锁池偏移量（细粒度锁） */
	Size		nodeLockOffset;
	uint32		nodeLockCount;
	
	/* 随机层级生成因子 (1/log(M)) */
	double		mult;
	
	/* 全局锁（保护入口点变更等写操作） */
	slock_t		globalLock;
	
	
	/* 删除元素池（环形缓冲区） */
	Size		deletedPoolOffset;		/* 偏移量 */
	pg_atomic_uint32 deletedPoolHead;	/* 队头（读位置） */
	pg_atomic_uint32 deletedPoolTail;	/* 队尾（写位置） */
	uint32		deletedPoolCapacity;	/* 池容量 */
	slock_t		deletedPoolLock;		/* 保护并发操作的自旋锁 */
	
	/* Label 查找表（HeapTID -> InternalID 映射） */
	Size		labelLookupOffset;		/* 哈希表偏移量 */
	uint32		labelLookupCapacity;	/* 哈希桶数量 */
	slock_t		labelLookupLock;		/* 哈希表自旋锁 */
	

	/* AOS 布局偏移量配置 */
	Size		dataLevel0Offset;		/* Level 0 数据起始偏移 */
	Size		dataLevel0PerElement;	/* Level 0 每个元素的总字节数 */
	Size		offsetDataInElement;	/* 向量数据在元素内的相对偏移 */
	Size		offsetMetaInElement;	/* 元数据在元素内的相对偏移 */
	Size		vectorStride;			/* 向量部分占用的字节 */
	Size		adj0Stride;				/* Layer 0 邻接表部分占用的字节 */
	Size		nodeMetaStride;			/* 元数据部分占用的字节 */
	
	/* 上层邻接表池（紧凑存储） */
	Size		adjUpperOffset;		/* 池起始偏移 */
	Size		adjUpperStride;		/* 单个 chunk（一层连接）的大小 */
	
	/* 紧凑分配器状态 */
	uint64		totalUpperChunks;		/* 总共预分配的 chunk 数 */
	pg_atomic_uint64 usedUpperChunks;	/* 当前已使用的 chunk 数（原子递增） */
	
	/* 总体大小 */
	Size		totalSize;
	
	/* SQ8 全局量化参数 */
	bool		isSq8Index;
	float		sq8LowerBound;      /* 全局下界 min */
	float		sq8Diff;            /* 全局范围 (max - min) */
} HnswSlabHeader;
```
**说明**：这是驻留在共享内存起始位置的“超级块”。它记录了所有内存池的偏移量（Offset），使得其它进程 attach 上来后能直接通过 `base + offset` 访问数据。









#### 3. HnswSlabNodeMeta（节点元数据）

```c
typedef struct HnswSlabNodeMeta
{
	ItemPointerData heaptids[HNSW_HEAPTIDS]; /* 物理元组指针 */
	uint8		heaptidsLength;              /* 有效元组指针数量 */
	uint8		level;                       /* 节点的最大层级 */
	uint8		deleted;                     /* 软删除标记 */
	uint8		version;                     /* 版本号 */
	uint32		adjUpperOffset;		         /* 指向 Upper Layers 池的偏移索引 */
} HnswSlabNodeMeta;
```
**说明**：每个节点的小型元数据，紧跟在向量数据之后（AOS 布局）。`adjUpperOffset` 指向紧凑内存池的一个整数索引。

#### 4. HnswSupport（扩展后的支持结构）
```c
typedef struct HnswSupport
{
	FmgrInfo   *procinfo;           /* 距离函数指针 */
	FmgrInfo   *normprocinfo;       /* 归一化函数 */
	Oid			collation;
	HnswNormKind normKind;
	HnswDistanceKind distanceKind;
	int			dimensions;         /* 向量维度 */
	
	/* 全局 SQ8 量化参数（所有向量共享） */
	bool		useSq8GlobalQuant;  /* 是否启用全局量化 */
	float		sq8LowerBound;      /* 全局最小值 */
	float		sq8Diff;            /* 全局范围 */
	
	/* 预计算的性能加速参数 */
	float		sq8Scale;           /* diff / 255 */
	float		sq8Scale2;          /* scale * scale */
	
	/* 查询向量缓存 */
	uint8	   *sq8EncodedQuery;    /* 编码后的查询向量 */
	int			sq8EncodedQueryDim; 
	int32		sq8QuerySum;        /* 查询向量的和（预计算） */
	int32		sq8QuerySum2;       /* 查询向量的平方和（预计算） */

	/* 
	 * 搜索候选池（减少 palloc 开销）
	 * 在 HnswFindElementNeighbors/HnswSearchLayer 中复用
	 */
	struct HnswSearchPool *searchPool;
	struct HnswBuildVisited *buildVisited;
}			HnswSupport;
```
**说明**：这个结构体挂载在 Postgres 的函数上下文中，我们在其中缓存了 SQ8 的全局参数和预计算好的查询向量信息 (`query_sum` / `query_sum2`)，从而在内层循环计算距离时，无需重复计算这些常量，直接传入 SIMD 函数。



### 核心方法/函数



#### Slab 架构相关函数（hnswslab.c）

| 函数名                        | 功能说明                   | 输入/输出                                         |
| ----------------------------- | -------------------------- | ------------------------------------------------- |
| `HnswSlabShmemSize()`         | 估算共享内存目录大小       | 输出：Size                                        |
| `HnswSlabBuildBegin()`        | 开始构建 Slab，分配 DSM 段 | 输入：numElements, totalUpperChunks, 索引参数     |
| `HnswSlabBuildAddVector()`    | 添加向量到 Slab (AOS 布局) | 输入：internalId, vector data                     |
| `HnswSlabSearchBaseLayer()`   | 基于 Slab 的基层搜索       | 输入：query, k, ef；输出：候选列表                |
| `HnswSlabSearchUpperLayers()` | 基于 Slab 的上层贪婪搜索   | 输入：query, level；输出：最近点                  |
| `HnswSlabGetVector()`         | 获取向量数据指针           | 输入：handle, internalId；输出：char*             |
| `HnswSlabGetAdj0()`           | 获取 Layer 0 邻接表        | 输入：handle, internalId；输出：uint32*           |
| `HnswSlabGetNodeMeta()`       | 获取节点元数据             | 输入：handle, internalId；输出：HnswSlabNodeMeta* |
| `HnswSlabAddPoint()`          | 动态插入点                 | 输入：handle, vector, heaptid                     |





















#### SQ8 距离计算函数（sq8utils.c）

| 函数名                                              | 功能说明                       | 输入/输出                                               |
| --------------------------------------------------- | ------------------------------ | ------------------------------------------------------- |
| `Sq8L2SquaredDistanceGeneric()`                     | 通用 L2² 距离                  | 输入：两个向量；输出：float                             |
| `Sq8L2SquaredDistanceWithSum2Batch()`               | 使用预计算 sum² 的批量 L2²     | 输入：query, targets[], sum2s[], count；输出：results[] |
| `Sq8L2SquaredDistanceWithSum2BatchVNNIImpl()`       | AVX-VNNI 加速的批量 L2²        | 输入：query, targets[], sum2s[], count；输出：results[] |
| `Sq8L2SquaredDistanceWithSum2BatchAVX512VNNIImpl()` | AVX-512 VNNI 加速版本          | 输入：query, targets[], sum2s[], count；输出：results[] |
| `Sq8ReencodeToGlobalParams()`                       | 将局部量化重编码为全局量化参数 | 输入：原编码, 原参数, 新参数；输出：新编码              |
| `Sq8ComputeGlobalQuantParams()`                     | 从采样向量计算全局量化参数     | 输入：样本向量；输出：lb, diff                          |
| `Sq8EncodeWithGlobalParams()`                       | 使用全局参数编码 float 向量    | 输入：float[], lb, diff；输出：uint8[]                  |

#### 索引构建相关函数（hnswbuild.c 新增/修改）

| 函数名                  | 功能说明                            | 输入/输出                                    |
| ----------------------- | ----------------------------------- | -------------------------------------------- |
| `Sq8SampleVectors()`    | 采样向量计算全局 SQ8 参数           | 输入：HnswBuildState；输出：lb, diff         |
| `BuildSlabFromMemory()` | 从内存图构建 Slab                   | 输入：HnswBuildState                         |
| `HnswRelayoutGraph()`   | BFS 图重排序 / 获取 Slab 需要的信息 | 输入：HnswBuildState；输出：totalUpperChunks |

#### 搜索优化相关函数（hnswutils.c 新增）

| 函数名                            | 功能说明                | 输入/输出                                          |
| --------------------------------- | ----------------------- | -------------------------------------------------- |
| `HnswComputeDistance()`           | 通用距离计算包装        | 输入：query, target, support；输出：float          |
| `HnswComputeNeighborDistances()`  | 批量邻居距离计算        | 输入：query, neighbors[], count；输出：distances[] |
| `HnswSq8BatchNeighborDistances()` | SQ8 批量距离计算 (VNNI) | 输入：query, neighbors[], count；输出：distances[] |
| `HnswPrefetchElementValue()`      | 预取元素值              | 输入：element                                      |
| `HnswPrefetchElementNeighbors()`  | 预取邻接表              | 输入：element, layer                               |



---

## 关键代码片段

### 1. Slab AOS 布局访问
```c
// hnswslab.h - 紧凑内存布局访问
static inline char *
HnswSlabGetElementBase(HnswSlabHandle *handle, uint32 internalId)
{
	/* 计算元素的基地址：起始地址 + ID * 元素大小 */
	return handle->dataLevel0 + (Size)internalId * handle->dataLevel0PerElement;
}

static inline char *
HnswSlabGetVector(HnswSlabHandle *handle, uint32 internalId)
{
	/* 获取向量数据：元素基地址 + 向量偏移 */
	return HnswSlabGetElementBase(handle, internalId) + handle->offsetDataInElement;
}

static inline uint32 *
HnswSlabGetAdj0(HnswSlabHandle *handle, uint32 internalId)
{
	/* 获取第0层邻接表：它位于元素的起始位置（偏移为0） */
	return (uint32 *)HnswSlabGetElementBase(handle, internalId);
}

static inline HnswSlabNodeMeta *
HnswSlabGetNodeMeta(HnswSlabHandle *handle, uint32 internalId)
{
	/* 获取节点元数据：元素基地址 + 元数据偏移 */
	return (HnswSlabNodeMeta *)(HnswSlabGetElementBase(handle, internalId) + handle->offsetMetaInElement);
}
```
**说明**：这些内联函数展示了 AOS 布局的优势。通过一次基地址计算，加上编译期确定的偏移量，我们就能访问到该节点的向量数据、第0层邻居表和元数据。这种紧凑布显著地提高了 CPU 缓存命中率。



### 2. SQ8 量化处理

#### （1）编码
```c
/*
 * 从 float 转换成 sq8
 * 量化公式：code = round((value - lower_bound) / diff * 255)
 */
static void
Sq8EncodeVector(const float *values, int dim, float lower_bound, float diff, uint8 *codes)
{
    float scale = 255.0f / diff;
    for (int i = 0; i < dim; i++)
    {
        float codef = (values[i] - lower_bound) * scale;
        if (codef < 0.0f)
            codef = 0.0f;
        else if (codef > 255.0f)
            codef = 255.0f;
        codes[i] = (uint8) (codef + 0.5f);
    }
}
```

**算法说明**：

- 将[lower_bound, lower_bound+diff]映射到[0, 255]。
- 使用四舍五入保证精度。
- 边界检查确保编码值在有效范围内。

#### （2）解码
```c
/*
 * Decode SQ8 codes to float values
 * 解码公式：value = code * diff / 255 + lower_bound
 */
static void
Sq8DecodeVector(const uint8 *codes, int dim, float lower_bound, float diff, float *values)
{
    float scale = diff / 255.0f;
    for (int i = 0; i < dim; i++)
    {
        values[i] = (float) codes[i] * scale + lower_bound;
    }
}
```



### 3.BFS 图重排序
```c
    if (entryPoint != NULL)
        queue[tail++] = entryPoint;
    iter = graph->head;
    for(;;){
        while (head < tail)
        {
            HnswElement element = queue[head++];
            /* ... 判断 visited ... */
            ordered[orderCount++] = element;
            neighbors = HnswGetNeighbors(base, element, 0);
            for (int i = 0; neighbors != NULL && i < neighbors->length; i++)
            {
                HnswElement neighborElement = HnswPtrAccess(base, neighbors->items[i].element);
                /* ... */
                queue[tail++] = neighborElement;
            }
        }

        if (orderCount >= total)
            break;
        /* ... */
    }
    // 按新顺序重建链表
    for (int i = 0; i < orderCount; i++)
    {
        HnswElement next = (i + 1 < orderCount) ? ordered[i + 1] : NULL;
        HnswPtrStore(base, ordered[i]->next, next);
    }
```

**说明：**在建索引后建 Slab 前执行，按 Level 0 的邻接表 BFS。



### 4. 搜索热路径：显式预取 + SQ8  批量距离计算
```c
if (support->distanceKind == HNSW_DISTANCE_SQ8_L2 &&
    support->sq8EncodedQuery != NULL)
{
    int batch_idx = 0;
    uint32 batch_ids[4];                 /* 本批的 internalId */
    const uint8 *batch_vectors[4];       /* 本批的 codes 指针（只传 codes，避免解码回 float） */
    int32 batch_sum2s[4];                /* 本批每个目标向量预计算的 sum2 */
    int64 batch_results[4];              /* 本批输出的 L2²（q_sum2 + t_sum2 - 2*dot） */

    for (uint32 j = 0; j < size; j++)
    {
        uint32 candidate_id = HnswSlabGetAdjNeighbor(adj, j);

#if HNSW_SLAB_PREFETCH_DISTANCE > 0
        /* 显式预取：提前把 (j + D) 的邻居相关数据拉进缓存，D 默认取 4 */
        if (j + HNSW_SLAB_PREFETCH_DISTANCE < size)
        {
            uint32 next_id = HnswSlabGetAdjNeighbor(adj, (int)(j + HNSW_SLAB_PREFETCH_DISTANCE));
            /* 预取 visited 标记（减少标签数组的 cache miss） */
            if (next_id < visited->capacity)
                HNSW_SLAB_PREFETCH_R(visited->tags + next_id);
            /* 预取向量数据（AOS 布局下能顺带预热同一块里的 meta/adj 数据） */
            if (next_id < handle->maxElements)
                HNSW_SLAB_PREFETCH_R(HnswSlabGetVector(handle, next_id));
        }
#endif
        /* visited 判重：O(1) tag 技巧，避免重复入堆与重复算距离 */
        if (HnswSlabVisitedCheckAndMark(visited, candidate_id))
            continue;
        /* 越界/未发布元素防御（防止脏邻接导致越界读） */
        if (candidate_id >= maxElements || candidate_id >= curElementCount)
            continue;
        /* 收集 4 个一批：只抓 codes 与 sum2，避免热路径反量化 */
        batch_ids[batch_idx] = candidate_id;
        {
            char *vectorData = HnswSlabGetVector(handle, candidate_id);
            Sq8IndexTuple *qt = (Sq8IndexTuple *) vectorData;
            batch_vectors[batch_idx] = qt->codes;
            batch_sum2s[batch_idx] = qt->sum2;
        }
        batch_idx++;
        /* 满 4 个就批量计算一次距离 */
        if (batch_idx == 4)
        {
            Sq8L2SquaredDistanceWithSum2Batch(
                support->sq8EncodedQueryDim,
                support->sq8EncodedQuery,
                support->sq8QuerySum,
                support->sq8QuerySum2,
                batch_vectors,
                batch_sum2s,
                4,
                batch_results);
            /* 将本批结果回灌到 candidate_set/top_candidates（保持算法语义不变，只优化计算与访存） */
            for (int k = 0; k < 4; k++)
            {
                double dist = (double) batch_results[k];
                uint32 cid = batch_ids[k];
                /* ... 与原版一致：比较 lowerBound/ef，决定入堆/丢弃 ... */
            }
            batch_idx = 0;
        }
    }
    /* 处理不足 4 个的尾批 */
    if (batch_idx > 0)
    {
        Sq8L2SquaredDistanceWithSum2Batch(
            support->sq8EncodedQueryDim,
            support->sq8EncodedQuery,
            support->sq8QuerySum,
            support->sq8QuerySum2,
            batch_vectors,
            batch_sum2s,
            batch_idx,
            batch_results);
        for (int k = 0; k < batch_idx; k++)
        {
            double dist = (double) batch_results[k];
            uint32 cid = batch_ids[k];
            /* ... 同上：入堆/剪枝/丢弃 ... */
        }
    }
}

```

**说明：**以 j + 4 为距离预取后续邻居的 `visited->tags` 与 `HnswSlabGetVector()`，把“将来必然要访问的数据”更早带入 L1/L2，降低 LLC miss 带来的停顿。把 4 个候选的 `codes + sum2` 组成一个 batch，一次调用 `Sq8L2SquaredDistanceWithSum2Batch()`，减少函数调用/循环开销，并为 VNNI/AVX 路径提供更好的吞吐条件。

### 5. Slab 动态插入
```c
/* 1) 复用 deletedId：从环形池弹出一个已删除 internalId，原地覆盖并 updatePoint */
while (HnswSlabPopDeletedId(handle, &internalId))
{
    LWLock *reuseLock = HnswSlabGetNodeLock(handle, internalId);
    if (reuseLock != NULL)
        LWLockAcquire(reuseLock, LW_EXCLUSIVE);
    meta = HnswSlabGetNodeMeta(handle, internalId);
    /* 可能弹出的是“已被取消删除”的节点：跳过继续找 */
    if (meta->deleted == 0)
    {
        if (reuseLock != NULL)
            LWLockRelease(reuseLock);
        continue;
    }
    /* 清理旧 label */
    if (meta->heaptidsLength > 0)
        HnswSlabLabelRemove(handle, HnswSlabEncodeHeaptid(&meta->heaptids[0]));
    /* 写入新 heaptids */
    for (int i = 0; i < HNSW_HEAPTIDS && i < heaptidsLength; i++)
        meta->heaptids[i] = heaptids[i];
    for (int i = heaptidsLength; i < HNSW_HEAPTIDS; i++)
        ItemPointerSetInvalid(&meta->heaptids[i]);
    pg_write_barrier();
    meta->heaptidsLength = heaptidsLength;
    /* 写入新 label */
    if (label != 0)
        HnswSlabLabelInsert(handle, label, internalId);
    /* 取消删除标记 + 全局 deleted 计数回退 */
    meta->deleted = 0;
    if (reuseLock != NULL)
        LWLockRelease(reuseLock);
    pg_atomic_fetch_sub_u32(&header->numDeleted, 1);
    /* 复用节点走 updatePoint：逻辑更简单，避免重复分配路径 */
    HnswSlabUpdatePoint(handle, internalId, vectorData, vectorSize,
                        queryValue, support, efConstruction);
    return internalId;
}

/* 2) 无可复用 deletedId：从 curElementCount 原子拿一个新槽位 */
curMax = pg_atomic_read_u32(&header->curElementCount);
for (;;)
{
    uint32 desired;
    if (curMax >= header->maxElements)
        elog(ERROR, "HNSW slab 已满 (cur=%u, max=%u)", curMax, header->maxElements);
    desired = curMax + 1;
    if (pg_atomic_compare_exchange_u32(&header->curElementCount, &curMax, desired))
        break;
}
internalId = curMax;

/* ... 写 vector、初始化 meta/adj0 等 ... */

/* 3) upperChunks 原子分配：每个 level>0 的点需要 newLevel 个 chunk（对应 layer1..layerN） */
if (newLevel > 0)
{
    uint64 needed = (uint64) newLevel;
    uint64 oldUsed = pg_atomic_fetch_add_u64(&header->usedUpperChunks, needed);
    /* 池不足则回滚（并降级到 level0），保证不越界写 upper pool */
    if (oldUsed + needed > header->totalUpperChunks)
    {
        pg_atomic_fetch_sub_u64(&header->usedUpperChunks, needed);
        meta->adjUpperOffset = 0;
        meta->level = 0;
        newLevel = 0;  /* 降级：只保留 layer0 */
    }
    else
    {
        meta->adjUpperOffset = (uint32) oldUsed; /* 记录该点 upper chunks 的起始 chunk 下标 */
    }
}
else
{
    meta->adjUpperOffset = 0;
}
/* 4) entry point 更新*/
if (curMax == 1 || curEpLevel < 0)
{
    SpinLockAcquire(&header->globalLock);
    header->entryPointId = internalId;
    header->entryPointLevel = newLevel;
    /* 查询侧用 atomic 读取：保证并发可见性 */
    pg_atomic_write_u32(&header->entryPointIdAtomic, internalId);
    pg_atomic_write_u32(&header->entryPointLevelAtomic, (uint32) newLevel);
    SpinLockRelease(&header->globalLock);
    /* 同步 handle 内的缓存字段（本 backend 直接用） */
    handle->entryPointId = internalId;
    handle->entryPointLevel = newLevel;
    if (newLock != NULL)
        LWLockRelease(newLock);
    return internalId;
}
```

**说明：**删除节点进入 `ring buffer`，插入优先复用旧槽位，再去找预先分配的空间插入节点。


### **性能优化点**

#### 1. **内存占用与布局**
| 优化点        | 实现方式                                        | 影响 |
|---------|---------|------|
| SQ8 量化 | `float32 → uint8` 存储 | 向量数据约 **4×** 压缩；召回率损失控制在 **≈1%** 左右 |
| 全局量化参数 | 全量共享 `lower_bound/diff`（存于 `metapage`） | 避免 per-vector 参数开销，节省 **8 bytes/vector** |
| 紧凑 AOS 布局 | 底层`[Adj0 | Vec | Meta]`，上层根据实际数量分配 | 尽可能让 Slab 布局紧凑，避免 OOM |

#### 2. **计算优化**
| 优化点            | 实现方式                         | 影响                                    |
|---------|---------|------|
| SIMD 批量距离计算 | 一次处理 4 个候选                | 提高计算吞吐，减少循环/解包开销         |
| 显式预取          | 预取（`_mm_prefetch(i+4)`）      | 预取将要计算的数据，改善等待时间        |
| 64B 对齐          | 对齐分配（`posix_memalign(64)`） | 改善 SIMD 访问对齐与 cacheline 访问效率 |

#### 3. **Cache 优化**
| 优化点               | 实现方式                                                     | 影响                                                         |
|---------|---------|------|
| Slab（DSM 连续投影） | 将热路径数据投影到单段 DSM，使用 `internalId + stride` 数组寻址 | `llc_misses_per_s` 与 `dtlb_misses_per_s` 显著下降，对应 QPS 大幅提升 |
| BFS 重排序           | 基于 layer0 邻接的 BFS 遍历顺序重排                          | 主要降低 DTLB/page-walk 压力                                 |

### 特殊处理技巧

#### **1. 边界条件处理**
| 场景              | 处理方式                                             |
|---------|---------|
| 空索引（n=0）     | `numElements == 0` 时不创建 Slab，查询直接返回空结果 |
| 单元素索引（n=1） | entry point 设为唯一元素，查询路径直接返回该元素     |

#### **2. 参数选择与调优依据**
| 参数       | 取值   | 选择依据                                                     |
|---------|------|---------|
| 预取距离   | i + 4  | 在维度为 200 的测试中更稳定；`i+2` 与 `i+4`差异不大，`i+1`收益偏弱 |
| 批处理大小 | 4      | 在寄存器压力与循环开销之间折中；批量到 8 在我们的测试中反而更慢 |
| 采样数量   | 20,000 | 用于全局量化参数估计，在精度与构建成本之间折中               |

#### **3. 对原功能的影响**

在保持原有 HNSW 语义与使用方式不变的前提下，我们的实现对外完整兼容原生索引操作流程，包括 `INSERT / DELETE / VACUUM` 等常规维护行为。整体执行路径仍沿用原 HNSW 的构建与查询框架，仅在内存侧额外引入 Slab 的紧凑布局作为运行时加速结构，用于改善访问局部性与降低指针追逐开销。
同时，该方案兼容 `vector / halfvec / sq8` 等多种向量表示与索引类型，可在不同精度与性能配置下复用同一套 Slab 框架。



---

## 测试与验证

我们也针对部分优化策略，完成了基于 [perf](#perf) 硬件计数器的性能对比测试：

指令如下
```bash
#!/usr/bin/env bash
set -euo pipefail

HOST=127.0.0.1
PORT=5432
DB=db1
USER=test
PASSWORD=123456
TABLE_NAME=vector_table
HDF5_FILE="sift-128-euclidean.hdf5"
K=10
METRIC=l2
HNSW_EF_SEARCH=30
WORKERS=1
QUERY_NUM=10000
PERF_WAIT_MS=1000
DURATION=7

EVENTS="cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,\
dTLB-loads,dTLB-load-misses,\
branches,branch-misses,\
cycles,instructions"

OUT_DIR="$(pwd)"
PID_FILE="${OUT_DIR}/pg_backend_pids.txt"
PERF_OUT="${OUT_DIR}/perf_stat.txt"
QUERY_LOG="${OUT_DIR}/run_query.log"

rm -f "$PID_FILE" "$PERF_OUT" "$QUERY_LOG"
# 运行的测试函数
python3 query.py \
  --host "$HOST" --port "$PORT" --database "$DB" --user "$USER" --password "$PASSWORD" \
  --table_name "$TABLE_NAME" --hdf5_file "$HDF5_FILE" \
  --k "$K" --metric "$METRIC" --workers "$WORKERS" --query_num "$QUERY_NUM" \
  --pgvector_hnsw_ef_search "$HNSW_EF_SEARCH" \
  --perf_wait_ms "$PERF_WAIT_MS" \
  > "$QUERY_LOG" 2>&1 &

QPID=$!
echo "[INFO] query.py pid: $QPID"

# 获取当前运行的 pid
PIDS=""
for i in {1..1000}; do  # up to ~10s (1000 * 0.01)
  if [[ -s "$PID_FILE" ]]; then
    # Read numeric lines, de-dup, space-separate
    PIDS=$(grep -E '^[0-9]+$' "$PID_FILE" | sort -u | tr '\n' ' ' | xargs || true)
    if [[ -n "$PIDS" ]]; then
      CNT=$(echo "$PIDS" | wc -w | tr -d ' ')
      if [[ "$CNT" -ge "$WORKERS" ]]; then
        ok=1
        for p in $PIDS; do
          kill -0 "$p" 2>/dev/null || ok=0
        done
        if [[ "$ok" -eq 1 ]]; then
          break
        fi
      fi
    fi
  fi
  sleep 0.01
done

if [[ -z "$PIDS" ]]; then
  echo "[ERROR] failed to capture backend pids from $PID_FILE"
  echo "[ERROR] last 120 lines of query log:"
  tail -n 120 "$QUERY_LOG" || true
  kill "$QPID" 2>/dev/null || true
  exit 1
fi

echo "[INFO] backend pids (space-separated): $PIDS"

PARGS=""
for p in $PIDS; do
  PARGS="$PARGS -p $p"
done
echo "[INFO] perf args:$PARGS"

PSLIST=$(echo "$PIDS" | tr ' ' ',')
ps -p "$PSLIST" -o pid,comm,args | head -n 20 || true

# 运行 perf stat
sudo perf stat -e "$EVENTS" $PARGS -- sleep "$DURATION" 2> "$PERF_OUT" || true
wait "$QPID" || true

echo "[DONE] perf: $PERF_OUT  querylog: $QUERY_LOG"
```







（1）批处理主要提升 IPC、降低 cycles

我们对比了“无批处理”和“有批处理”两种实现，在相同 workload 下采集硬件性能计数器（10次），并对比关键指标的中位数。对比结果如下：

| 指标              | 无批处理      | 有批处理      | 变化（有批处理 vs 无批处理） |
| :---------------- | :------------ | :------------ | :--------------------------- |
| **ipc**           | **0.689915**  | **0.709746**  | **+2.87%**                   |
| **cycles**        | **6.33686e9** | **6.05933e9** | **-4.38%**                   |
| instructions      | 4.34263e9     | 4.31117e9     | -0.72%                       |
| cache_miss_rate   | 17.306%       | 16.474%       | -4.81%                       |
| llc_miss_rate     | 45.464%       | 44.230%       | -2.71%                       |
| llc_misses_per_s  | 2.01982e6/s   | 2.04788e6/s   | +1.39%                       |
| dtlb_miss_rate    | 0.609%        | 0.637%        | +4.60%                       |
| dtlb_misses_per_s | 1.04266e6/s   | 1.08194e6/s   | +3.77%                       |
| l1d_miss_rate     | 7.646%        | 7.860%        | +2.80%                       |
| branch_miss_rate  | 3.350%        | 3.924%        | +17.13%                      |

从结果看，批处理带来的核心变化是 **IPC 上升（+2.87%）** 与 **cycles 下降（-4.38%）**，同时 instructions 略降（-0.72%）。这说明批处理并不是单纯“减少 cache miss”，而是让 CPU 在单位时间内更有效地执行指令：一次处理多个目标可以复用查询向量的预处理/解包结果、减少函数调用与循环开销、提升指令级并行度与流水线利用率，最终体现为更高 IPC 与更少 cycles。

需要注意的是，一些内存侧指标并没有同步变好：`llc_misses_per_s`、`dtlb_misses_per_s` 反而小幅上升，`branch_miss_rate` 也上升。这是可解释的：批处理通常会让循环结构更复杂、分支路径更多（例如批内边界、尾处理），同时因为整体算得更快，单位时间内可能触发更多访问/更多候选处理，导致“每秒 miss 数”不一定下降。但综合看，批处理仍然把瓶颈更明显地从“**低效控制流/低并行”转向“更高吞吐的计算路径**”，因此 IPC 与 cycles 的改善更能代表其收益来源。

（2）显式预取显著降低 LLC miss

我们对比了“无预取”和“有预取”两种实现，在相同 workload 下采集硬件性能计数器（10次），并对比关键指标的中位数。对比结果如下：

| 指标                 | 无预取          | 有预取          | 变化（有预取 vs 无预取） |
| :------------------- | :-------------- | :-------------- | :----------------------- |
| **llc_misses_per_s** | **2.79502e6/s** | **2.03242e6/s** | **-27.28%**              |
| llc_miss_rate        | 51.551%         | 44.426%         | -13.82%                  |
| l1d_miss_rate        | 8.077%          | 7.831%          | -3.05%                   |
| dtlb_miss_rate       | 0.661%          | 0.643%          | -2.72%                   |
| ipc                  | 0.690629        | 0.699008        | +1.21%                   |
| cache_miss_rate      | 16.497%         | 16.778%         | +1.70%                   |
| dtlb_misses_per_s    | 1.05137e6/s     | 1.09294e6/s     | +3.95%                   |
| branch_miss_rate     | 3.607%          | 3.881%          | +7.60%                   |
| cycles               | 5.98294e9       | 6.16612e9       | +3.06%                   |
| instructions         | 4.12440e9       | 4.29892e9       | +4.23%                   |

从结果看，显式预取的最主要收益体现在 **LLC 层面**：**llc_misses_per_s** 下降约 **27.28%**，**llc_miss_rate** 下降约 **13.82%**，同时 **l1d_miss_rate** 也下降约 **3.05%**。这与预取机制一致：在遍历邻接表、即将访问后续邻居向量/邻接数据之前提前 `_mm_prefetch`，能够把未来即将访问的数据更早带入缓存，减少“真正访问时才发现未命中”的概率，从而显著减少下探到 LLC/更低层级的等待。

同时我们也看到一些指标出现上升（例如 `branch_miss_rate`、`instructions`、`cycles`、以及 `dtlb_misses_per_s` 小幅上升）。这通常可以理解为：预取本身会引入额外指令与控制流，同时更激进的访问/预取节奏可能改变页级触碰模式，使得“每秒 DTLB miss 数”不一定同步下降。但从性能机理上，预取的价值主要在于降低 **昂贵的缓存未命中（尤其是 LLC miss）**，每秒 LLC miss 直接减少约 **0.76e6** 次，因此显式预取有效**降低内存等待、提升热路径吞吐**。



### 性能测试

注：下面的测试在本地部署的 docker 里进行，受限于硬件配置，测试的为小数据集，和提交到比赛上测出的提升有所差异。

#### 测试环境

- **硬件**：
  - CPU：13th Gen Intel(R) Core(TM) i7-13620H（支持 avx2 / avx_vnni）
  - docker 容器配置
    ```bash
    Memory=0 MemorySwap=0 OOMKillDisable=<no value> CpuQuota=0 CpuPeriod=0 Cpuset= CpuShares=0 ShmSize=2147483648
    ```

  - 数据库配置
    ```bash
    shared_buffers = 2GB
    polar_xlog_queue_buffers = 256MB
    maintenance_work_mem = 1536MB
    max_parallel_maintenance_workers = 8
    max_parallel_workers = 8
    ```

- **数据集**：
  - sift-128-euclidean（[erikbern/ann-benchmarks: Benchmarks of approximate nearest neighbor libraries in Python](https://github.com/erikbern/ann-benchmarks)）

#### 测试结果

##### 索引构建性能

编译完后第一次建索引预热，然后重复进行**重启数据库 -> 删除索引 -> 建索引**五次，去除最高最低取平均值。
建索引配置如下：
```sql
// 优化后配置
CREATE INDEX ON vector_table
USING hnsw (embedding sq8vec_l2_ops)
WITH (m = 10, ef_construction = 64);
// 原版配置
CREATE INDEX ON vector_table
USING hnsw (embedding vec_l2_ops)
WITH (m = 10, ef_construction = 64);
```

| 原版构建时间 | 优化后构建时间 | 提升   |
| ------------ | -------------- | ------ |
| 51.645 s     | 29.265 s       | 43.33% |



##### 查询性能（QPS）

后续的 QPS 测试均为重复十次操作，每次操作重复 10 遍查询，取相对最稳定的那次操作去最高最低取平均值。
QPS 查询代码如下：
```bash
for i in 1 2 3 4 5 6 7 8 9 10; do echo '=== 最终Baseline测试 \$i ===' && python3   query.py  --host   127.0.0.1   --port 5432 --database db1 --user test --password 123456 --table_name vector_table --hdf5_file "sift-128-euclidean.hdf5" --k 10 --metric l2 --query_num 10000 --pgvector_hnsw_ef_search 30 && sleep 1 2>&1 | grep -E 'QPS|recall'; done
```



取第一次 recall >= 0.85 的数据

| 原版 QPS (ef = 29) | 优化后 QPS (ef = 31) | 提升    |
| ------------------ | -------------------- | ------- |
| 8790.47            | 18108.59             | 106.00% |



##### QPS 测试

| ef_search | 原版 QPS | 优化后 QPS | 提升    |
| --------- | -------- | ---------- | ------- |
| 30        | 8743.69  | 18098.99   | 106.99% |
| 50        | 6979.98  | 17173.27   | 146.04% |
| 100       | 4557.87  | 15233.77   | 234.23% |



##### 召回率测试

| ef_search | 原版召回率 | SQ8 召回率 | 差距     |
| --------- | ---------- | ---------- | -------- |
| 30        | 0.856630   | 0.849080   | 0.007550 |
| 50        | 0.913820   | 0.905480   | 0.008340 |
| 100       | 0.963050   | 0.950480   | 0.012570 |



##### 内存占用对比

| 组件      | 原版   | 优化后 | 差异    |
| --------- | ------ | ------ | ------- |
| 原始数据  | 427 MB | 176 MB | -251 MB |
| 索引数据  | 709 MB | 346 MB | -363 MB |
| Slab 内存 | /      | 325 MB | +325 MB |



### 各项优化效果量化

| 优化项           | 基准平均QPS | 优化后平均QPS | QPS 提升 | 备注               |
| ---------------- | ----------- | ------------- | -------- | ------------------ |
| SQ8 量化         | 7940.454    | 10311.761     | +29.86%  | 4x 压缩            |
| 共享内存 Slab    | 10311.761   | 18665.115     | +81.01%  | 消除指针追逐       |
| AOS 布局         | 17733.122   | 18616.327     | +4.98%   | 对比 SOA 布局      |
| 显式预取         | 18344.760   | 19067.166     | +3.94%   | 预取 i+4 位置      |
| BFS 图重排序     | 17538.157   | 18184.221     | +3.68%   | 额外 ~10s 建图开销 |
| 批量距离计算     | 18230.886   | 18819.430     | +3.23%   | 一次处理 4 个目标  |
| VNNI 加速        | 18541.499   | 19130.814     | +3.18%   | 需要 CPU 支持      |
| Visited 数组优化 | 18268.316   | 18513.445     | +1.34%   | O(1) 重置          |
| 搜索候选池       | 18548.202   | 18716.476     | +0.91%   | 避免 palloc        |

**注：**本文档的分项对比测试均采用“在最终版本基础上仅关闭某一项优化”的方式进行（除 SQ8 量化对比时均为无 Slab 的版本），因此各项提升不可简单相加；线上评测的 CPU 特性、内存配置与数据规模不同，绝对数值会有偏移，但趋势一致。

**所有原始测试数据在仓库（最新的 commit）内的 test 文件夹**

---

## 边界情况说明



### 1. 正确性与安全防护

**（1）事务一致性与回滚清理：**Slab 构建阶段注册了事务回调 `HnswSlabXactCallback()`。一旦发生 `XACT_EVENT_ABORT`/`PARALLEL_ABORT`，会主动 `dsm_detach()` 并从 Slab 目录移除 `(dbid, relid)`，同时执行 `dsm_unpin_segment()`。这样可以确保不会留下处于 BUILDING 状态的残段，避免后续 attach 误读到“半成品”内存。

**（2）Slab 头部一致性校验：**`HnswSlabAttach()` 在 attach DSM 后首先校验 `HNSW_SLAB_MAGIC` 和 `HNSW_SLAB_VERSION`。若不匹配则直接 `elog(ERROR)` 并提示重建，防止跨版本结构变更或脏内存导致字段解析错位、进而引发不可预期的行为。

**（3）越界与脏邻接的防御性处理：**

- `HnswSlabIsValidInternalId()` 同时检查 `internalId < maxElements`（DSM 容量上限）与 `internalId < curElementCount`（已发布元素上限），把“未分配/未发布”的 ID 直接挡在入口。
- 邻接 `count` 统一通过 `HnswSlabClampNeighborCount()` 限制在 `maxM0/m` 的合法范围内；即使 `count` 被破坏，也不会导致遍历时越界访问。
- 距离计算入口 `HnswSlabComputeDistance()` 对无效 `internalId` 直接返回 `DBL_MAX`，等价于把它视作“最差候选”，从而避免对向量区域的越界读。
- visited 判定如果出现 `internalId >= visited->capacity`，按“已访问”处理并跳过，宁可保守丢弃，也不允许越界写 visited 数组。

**（4）并发写入的处理：**对于邻接表、`heaptidsLength` 等需要被读路径“看到”的发布字段，写入顺序采用“先写 payload、再发布”的策略，并用 `pg_write_barrier()` 做内存序保证。以 `HnswSlabMutuallyConnectNewElement()` 为例：先写 `adj[i+1]` 等内容，barrier 之后再写 `adj[0]=count`。读路径以 `count` 作为发布点，从而避免读到半更新的数据结构。

**（5）预取的越界保护：**邻接数组额外预留 1 个 padding 槽并置 0（如 `adj[neighborCount + 1] = 0`），使得启用 `HNSW_SLAB_PREFETCH_DISTANCE` 时，即便出现“向前多预取一步”，也不会落到未初始化地址或无效指针区域。



### 2. 动态插入/删除的边界处理

| 边界情况 | 代码函数 | 当前处理方式 | 潜在影响 |
|----------|----------|--------------|----------|
| Slab 容量耗尽 | `HnswSlabAddPoint()` | `curElementCount >= maxElements` 直接 `elog(ERROR)` | 插入失败，需要扩容/重建 |
| 上层邻接池耗尽 | `HnswSlabAddPoint()` | `usedUpperChunks` 超出 `totalUpperChunks` 时回滚分配并 `WARNING`，将新点 降级为 level0（`meta->level=0`） | 上层结构稀疏化，可能降低 recall/加大 base layer 压力 |



### 3. SQ8 与 SIMD/数值边界

| 边界情况 | 代码位置 | 当前处理方式 |
|----------|----------|--------------|
| `diff≈0`（向量全相等/采样范围极窄） | `Sq8ComputeQuantizationParams()` / `Sq8ComputeGlobalQuantParams()` | `diff < 1e-10f` 强制钳到 `1e-10f`，避免除零 |
| `diff_global` 非法（≤0 或非有限） | `Sq8ReencodeToGlobalParams()` | 直接输出全 0 codes + `sum/sum2=0`，保证行为可定义（但会显著劣化距离区分度） |
| 查询向量非 `sq8vec` | `hnswscan.c:GetScanValue()` | 仅当 `useSq8GlobalQuant && IsSq8Vector()` 才重编码为 `Sq8IndexTuple`；否则保持原值走原始距离函数 |
| 指令集不可用/OS 不支持 ZMM state | `Sq8utilsInit()` | 运行期 CPUID 探测，按 `AVX512(VNNI)/AVX2(VNNI)/Generic` 选择函数指针，不满足则自动回退 |

### 4. 适用场景与规模限制

| 场景 | 说明 |
|------|------|
| 大规模只读查询 | 最佳场景，充分利用 Slab（共享 DSM + AOS + 预取） |
| 频繁插入/删除 | 需要同步更新 Slab（node lock + label/deleted pool），并发写多时可能有锁竞争 |
| 内存受限环境 | Slab DSM 段为“单段连续内存”，分配失败会导致 Slab 不可用 |
| 超大规模数据 | 记录偏移量的数据为 `uint32`，大规模数据不足以支持构建 |



---

## 存在的问题与改进方向

| 问题 | 改进方向 |
|------|---------|
| 冷启动性能 | 1. 后台预热线程，数据库启动时异步重建 Slab<br>2. 持久化 Slab 到文件，避免每次重建需要扫描收集信息 |
| SQ8 召回率损失 | 1. 实现自适应量化：高频查询区域用更高精度<br>2. 可以根据不同数据的实际情况制定量化目标，提高精度<br>3.当前只采样前 20000 个数据，可以改成在全体数据内部随机采样 |
| 内存占用 | 1. 实现分层 Slab：热数据在内存，冷数据在磁盘<br>2. LRU 淘汰策略，支持超大索引 |
| 动态插入上层池耗尽 | 1. 支持扩容：新建更大 DSM 段并双写迁移（或分段 DSM）<br>2. 将 `numElements+10000` 预留改成可配置（不同 workload 预留不同） |
| 可观测性 | 1. 添加 Slab 统计信息视图（cache 命中率、内存使用等）<br>2. 提供 EXPLAIN 分析，展示查询路径 |
| 超大规模数据 | 1. 将 uint32 ID 扩展为 uint64<br>2. 支持分片 Slab |



### 详细改进计划

#### 1. 冷启动优化

- **现状与问题：**数据库重启后，Slab 需要在首次查询时重建，导致首条查询往往出现 10–30s 的冷启动延迟。
- **改进方向：**
  - **后台预热：**在实例启动阶段拉起后台预热线程，异步扫描目标索引并提前完成 Slab 重建，将耗时从“首条查询”转移到“后台准备阶段”。
  - **持久化辅助重建：**将 Slab 的关键结构持久化到独立文件，启动时通过 [mmap](#mmap) 快速载入作为重建的依据，而不直接在 `mmap` 上承载查询路径。我们曾评估过直接基于 `mmap` 索引文件进行查询，但由于 OS 页缓存下的访问模式仍可能呈现碎片化、物理不连续等特征，整体 cache 友好性不足，QPS 反而可能劣于原版布局；因此更倾向于把 `mmap` 用在加速恢复/重建，而非直接查询介质。

#### 2. 精度优化

- **现状与问题：**SQ8 量化在显著降低计算与内存开销的同时，会带来约 1% 的召回率下降。
- **改进方向：**
  - **两阶段检索：**召回阶段继续使用 SQ8 进行粗排与候选生成；在候选集规模可控的前提下，引入精排阶段，用原始向量或更高精度表示重新计算距离并排序，以在保持吞吐的同时尽量弥补量化带来的精度损失。
  - **采样逻辑优化：**当前全局量化采样只采样了前 20000 个数据，如果为极端数据可能会带来很大的误差，可以优化成全局随机采样/根据具体数据集确定量化方式。

#### 3. 内存优化

- **现状与问题：**Slab 作为额外的常驻结构会引入额外内存占用；在更大规模数据集或内存受限环境下，可能出现内存压力甚至触发资源上限。
- **改进方向：**
  - **冷热分层存储：**将访问频率高的热点节点常驻 Slab；低频冷节点保留在磁盘或原始结构中，仅在需要时加载/访问，从而把有限内存优先用于收益最高的部分。
  - **基于访问频率的淘汰策略：**结合访问统计实现 LRU（或带时间衰减的 LFU）淘汰，将 Slab 的驻留集合控制在可预测范围内，并尽可能与真实查询分布对齐。

#### 4. 动态插入稳定性（容量与上层池扩展）

- **现状与问题：**当前上层邻接池 `totalUpperChunks` 在构建期一次性预估（`totalUpperChunks + 10000`）。当在线写入峰值超出预留容量时，会进入“新点降级为 level0”的退化路径，导致图层级结构偏离预期，进而影响查询性能与结果稳定性。
- **改进方向：**
  - **预留容量参数化：**将预留量由固定常数改为可配置项，允许根据具体情况进行针对性调优，降低退化触发概率。
  - **在线扩容机制：**支持上层池扩容，例如采用“新 DSM 段 + 迁移/双写切换”的方式平滑扩展，或引入分段 DSM将容量管理从单段模型拓展为可追加的段式模型，以提升高写入场景下的稳定性。



整体上看，我们的布局与访问路径优化相比原版 HNSW 已显著缓解了 cache miss 带来的性能损失，但硬件计数器仍显示查询阶段的瓶颈主要来自**随机访存导致的长延迟等待**以及**分支错判后的流水线恢复开销**。这在以指针追逐为主的图索引检索中具有一定的结构性特征。值得注意的是，我们的实验表明，基于 BFS 的节点重排能够有效降低 `dtlb_misses`（并伴随 QPS 的正向变化），说明**图节点在内存中的物理布局**会显著影响页级局部性与缓存可命中性，从而直接作用于最终查询效率。因此，我们进一步提出一种面向真实负载的自适应优化思路：在运行过程中对查询访问进行轻量统计，学习并识别热点节点及常见访问路径；在低峰期或按阈值触发对图结构进行周期性重排（例如按访问频率加权的 BFS/聚类式局部性重排），将高频访问子图在内存中尽量聚集，以提升 TLB/LLC 命中并降低访存等待，从而针对实际业务分布持续提高检索吞吐与稳定性。







---

## 参考资料

### 论文

[Graph Reordering for Cache-Efficient Near Neighbor Search](https://arxiv.org/pdf/2104.03221).（BFS 重排序）

### 开源项目
- **pgvector**：https://github.com/pgvector/pgvector（基础框架）
- **vsag**：https://github.com/alipay/vsag（SQ8 量化参考）
- **hnswlib**：https://github.com/nmslib/hnswlib（Slab 的布局参考）



## 术语表

**AOS**

<a id="aos"></a> AOS Array of Structures，结构体数组。 内存布局方式，将每个对象的所有字段紧凑存储在连续内存中。适合同时访问一个对象的多个字段。

**AVX2-AVX512**

<a id="avx2-avx512"></a> AVX2 / AVX-512 Intel 推出的 SIMD 指令集扩展。 AVX2 支持 256 位宽度的向量运算；AVX-512 支持 512 位宽度的向量运算。两者均可显著提升浮点/整数计算效率，是向量检索的关键加速技术。

**DTLB**

<a id="dtlb"></a> DTLB Data Translation Lookaside Buffer，数据转换后备缓冲区。 CPU 用于缓存虚拟地址到物理地址映射关系的硬件结构，可加速地址转换、减少页表遍历开销。

**HNSW**

<a id="hnsw"></a> HNSW Hierarchical Navigable Small Worlds，分层可导航小世界。 一种基于多层图结构的高效近邻搜索索引算法，兼顾索引构建效率与查询性能，是向量检索的主流算法之一。

**IPC**

<a id="ipc"></a> IPC Instructions Per Cycle，每时钟周期执行的指令数。 衡量 CPU 执行效率的核心指标，数值越高表示 CPU 每个时钟周期能执行更多指令。

**LLC**

<a id="llc"></a> LLC Last Level Cache，最后一级缓存。 通常指 CPU 的 L3 缓存，容量最大但访问延迟高于 L1、L2。是 CPU 与内存之间的最后一道缓存层级。

**mmap**

<a id="mmap"></a> mmap Memory-Mapped File，内存映射文件。 Linux 系统调用，将文件内容映射到进程虚拟地址空间，实现文件与内存的高效交互，避免频繁 read/write 系统调用。

**page-walk**

<a id="page-walk"></a> page-walk 页表遍历。 当 DTLB 未命中时，CPU 需要遍历多级页表完成虚拟地址到物理地址转换，耗时较长，是内存访问延迟的重要来源之一。

**perf**

<a id="perf"></a> perf Linux 下的性能分析工具。 可采集 CPU 硬件计数器（缓存命中/未命中、指令数、分支预测错误等）、函数调用栈、系统调用等性能数据，是性能调优常用工具。

**SIMD**

<a id="simd"></a> SIMD Single Instruction Multiple Data，单指令多数据。 并行计算技术，一条指令可同时处理多个数据元素，可显著提升向量运算与批量数据处理效率。

**SOA**

<a id="soa"></a> SOA Structure of Arrays，数组结构体。 内存布局方式，将不同字段拆分为独立数组，适合批量访问单一字段；与 AOS 相对。本文中评估后选择 AOS 作为 Slab 的布局方式。

**VNNI**

<a id="vnni"></a> VNNI Vector Neural Network Instructions，向量神经网络指令集。 Intel 面向神经网络优化的 SIMD 指令集，可显著加速整数向量内积运算，是低精度向量检索的重要加速技术。

**DSM**

<a id="dsm"></a> DSM Dynamic Shared Memory，动态共享内存。 PostgreSQL 提供的共享内存分配与管理机制，可创建/attach/detach 可重定位的共享内存段。

**QPS**

<a id="qps"></a> QPS Queries Per Second，每秒查询数。 衡量检索吞吐的核心指标，数值越高代表单位时间能处理的查询越多。本文用 QPS 评估优化收益。

**L2**

<a id="l2"></a> L2 / L2² Euclidean Distance / Squared Euclidean Distance，欧氏距离/欧氏距离平方。 向量相似度度量方式。

**TLB**

<a id="tlb"></a> TLB Translation Lookaside Buffer，地址转换缓存。 缓存虚拟地址到物理地址的映射。TLB miss 会触发 page-walk，导致明显的内存访问延迟；DTLB 是 TLB 的数据侧部分。

**SQ8**

<a id="sq8"></a> SQ8 Scalar Quantization (8-bit)，8 位标量量化。 将 float32 向量按线性映射量化为 uint8，显著降低内存与带宽压力，用于提升向量检索吞吐。

**AM**

<a id="am"></a> AM Access Method，访问方法接口。 PostgreSQL 的索引接口层（Index AM），负责 build/scan/insert/vacuum 等流程对接。
