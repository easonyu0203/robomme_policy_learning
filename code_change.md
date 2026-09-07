# D&R 代码改动说明（分支 `dnr-fixes`，基于 `origin/prompt-vla` 7d01bc3）

日期：2026-09-07。主 commit `605963e`，消融配置 `652e88b`。

## 动机：置换实验

用官方 `perceptual-framesamp-modul/79999` 做推理时诊断：只把 32 个帧块在 slot 之间随机打乱，token 内容（SigLIP、3D 正弦位置、state）和 mask 一个字不动，唯一变化的是每个记忆 key 的 RoPE 相位。

| | 16 任务均值 | StopCube | SwingXtimes | PickXtimes | VideoUnmask |
|---|---|---|---|---|---|
| 原样 | 38.3 | 44 | 96 | 92 | 36 |
| 打乱 slot | 16.0 | 4 | 16 | 34 | 28 |

结论：actor 从 slot 相位读"什么时候"，从内容读"是什么"。framesamp 的 slot k 永远对应相对时刻 k/31，所以它有一个免费的时钟；任何选择式记忆（TokenDrop、A2R、hiersel 的 gather）slot 到时间的映射随内容变，这个时钟就没了。TokenDrop 在 StopCube 上 5.3 对 framesamp 42.0 是同一个现象。

**Refer to:**
> "pruning causes the accuracy of LLaVA on the RefCOCO validation set to drop from **56.14% to 15.34%**." "we identify **misaligned position IDs after pruning as the primary cause of this degradation**, as both the **order and value** of these IDs are crucial for maintaining performance in grounding tasks." GAP is "a simple yet effective adjustment to position IDs that recovers REC accuracy back to **51.42%**, which is 90% of the original performance" — no training, no extra memory/compute. from "Grounding-Aware Token Pruning (GAP)"

They decompose the failure into two distinct misalignments:
- **Permuted**: pruners reorder tokens to `{v₁,v₄,v₃,v₀,v₂}` but "their position IDs are assigned solely based on the order in which they are input to the LLM, remaining as `{p₁,p₂,p₃,p₄,p₅}`." (what i did)
- **Shifted**: "if tokens v₁ and v₂ are removed, the remaining tokens `{v₃,v₄,v₅}` are assigned position IDs `{p₁,p₂,p₃}`" — i.e. compact re-indexing. (TODO; well no need to do maybe)

Critically, Fig 2(b): "**even without losing any visual tokens**, the presence of both types of misalignment alone results in performance degradation on RefCOCO." That isolates position IDs from information loss. GAP's fix = "reconstruct the position IDs as they were prior to pruning." MiniGPTv2: 88.69% → 2.73% (pruned) → 68.91% (GAP). Tested on PruMerge, TRIM, CLS-similarity, text-visual similarity, random, spatial — **not** FastV/SparseVLM/VisionZip.


## 三处改动

### 1. Gumbel-Top-K 替换逐 token 伯努利

- 配置：`perceptual_memory.selector.sampling: topk`（默认 `bernoulli` = 旧行为），`score_norm: zscore`，`tau`，`noise_scale`
- 代码：`selector.py` 新增 `zscore_margin`、`gumbel_topk`
- 旧：每个 token 独立做两类 Gumbel-softmax，保留数随机，靠 ratio / z / load-balance 三个辅助损失兜底；部署却是严格 top-K
- 新：keep−drop 的 margin 先 z-score，训练时加 Gumbel 噪声，取恰好 K 个；反向用以第 K 名与第 K+1 名中点为阈值的 sigmoid 做直通。三个辅助损失全部删除；推理走同一函数、不加噪声
- 效果：训练与部署数量一致；决策从绝对阈值变成组内排名，selector 不需要校准尺度；"全留 / 全不留"两种塌缩在构造上不存在。load-balance 损失在等间隔布局下等于"禁止总是保留 frame 0"，删除后 selector 可以保留锚点帧

### 2. 整棵树端到端

- 配置：`selector.e2e_tree: true`（默认 false），与 `multilevel`、`ema_reducer` 互斥
- 代码：`percep_mem.py` 的 `_pick`、`_reduce_one_round_ext`、`_hierarchical_reduce_ext`
- 因为开销基本不变甚至更低（前向不变（旧代码本来每个节点都跑 selector），多出各节点 selector 的反向），而且考虑会更稳定、以及尽量避免distribution shift p(memory|history)（同时也尽量避免 p(action|obs, memory)的distribution shift）, 就换成整棵树了。


### 3. 时间编位 RoPE 加物理 gather

- 配置：`mem_rope: time`（history config 根级，默认 `slot`），`selector.root_gather: true`
- 代码：`history_gemma.py` MemoryAttention 第 84 到 91 行附近；`mem_pos` 穿过 HistoryBlock 的 remat/scan（`static_argnums` 7→8）和 `history_pi0.py` 全部调用点（`embed_memory` 多返回一个 `mem_pos`）
- 旧：key 的 RoPE 位置 = `arange(mem_len)`，query 从 S 往后数。相位差 = slot 距离。后果：推理不能物理 gather（commit d738e40），归约后必须按索引重排（commit 0464533），选择后 slot 相位不再表示时间
- 新：key 位置 = 该 token 的 steps-ago（从 recency 嵌入 `expm1` 反推），query 位置 = 0，相位差 = recency，与 slot 顺序、选择、gather 无关。root 可以把保留的 K 个 token 物理取出，policy 只看 K 个
- 这个是主要变化，因为原先的position ID只适用于framesamp。

## 结果（bud64 / pool128，40k 步，同一评测协议）

| 配置 | 均值 | actor 看到的记忆 |
|---|---|---|
| hiersel_pool128_multilevel_emareducer（旧） | 20.25 | 64 slot 原位 mask |
| framesamp-modul_bud64（无 selector） | 27.12 | 64 token = 4 帧 |
| **dnr_bud64_pool128（三处修复，ckpt 39999，3 seed）** | **28.63 ± 0.25** | **32 token 物理 gather** |

逐任务（ckpt 39999，每任务 50 集；单任务噪声约 ±7，只看大的模式）：

| 任务组 | 任务 | seed 0 | seed 7 | seed 42 | 均值 |
|---|---|---|---|---|---|
| Counting | BinFill | 36 | 28 | 26 | 30.0 |
| | StopCube | 20 | 22 | 30 | 24.0 |
| | PickXtimes | 62 | 60 | 50 | 57.3 |
| | SwingXtimes | 72 | 92 | 88 | 84.0 |
| Permanence | ButtonUnmask | 10 | 16 | 28 | 18.0 |
| | VideoUnmask | 28 | 38 | 28 | 31.3 |
| | VideoUnmaskSwap | 20 | 18 | 22 | 20.0 |
| | ButtonUnmaskSwap | 10 | 16 | 18 | 14.7 |
| Reference | PickHighlight | 20 | 10 | 18 | 16.0 |
| | VideoRepick | 20 | 18 | 8 | 15.3 |
| | VideoPlaceButton | 38 | 34 | 26 | 32.7 |
| | VideoPlaceOrder | 22 | 24 | 28 | 24.7 |
| Imitation | MoveCube | 70 | 54 | 60 | 61.3 |
| | InsertPeg | 2 | 0 | 4 | 2.0 |
| | PatternLock | 6 | 12 | 12 | 10.0 |
| | RouteStick | 18 | 16 | 16 | 16.7 |
| | **Overall** | **28.375** | **28.625** | **28.875** | **28.625** |

## 消融配置（每组只还原一项，其余保持完整版）

| 配置 | 还原的项 | 具体设置 |
|---|---|---|
| `perceptual-dnr-modul_bud64_pool128_abl1_bernoulli.yaml` | Gumbel-Top-K （**感觉没太有必要做**） | root 用逐 token 伯努利，恢复 ratio 1e-3 / z 1e-4 / lb 0.1，原位 mask；内部节点仍 e2e + top-k |
| `perceptual-dnr-modul_bud64_pool128_abl2_noe2e.yaml` | 端到端树 | 内部节点回到 stop_gradient，只有 root 训 selector |
| `perceptual-dnr-modul_bud64_pool128_abl3_slotrope.yaml` | 时间 RoPE | 回到 slot 编位，原位 mask |

为支持 abl1，`652e88b` 把 root 的采样方式和内部节点解耦：内部节点固定 e2e + gumbel_topk（物理缩小需要恰好 K），root 可独立选 topk / bernoulli。时间编位下原位 mask 与 gather 等价，所以 abl1 是干净的对照。


## 其他文件

- `scripts/launch_dnr_devbox.sh`：4×A800 启动脚本，自动识别 bin/npy、缺 norm_stats 时从 a2r 仓库复制、预检 GPU 与重复启动
- `examples/robomme/subgoal_predictor.py`：Gemini / Qwen SDK 改惰性导入（感知记忆评测不需要）
- `tests/test_dnr_fixes.py`：6 项检查，`JAX_PLATFORMS=cpu` 两分钟跑完；旧套件 `tests/test_hierarchical_reduction.py` 19/19 通过
- 新增 stats：`keep_frac`（topk 下恒为 0.5）、`reduce_keep_frac`、`first_frame_keep_frac`（frame 0 的保留率）

## 未做与待办

- frame-0 锚点？
- 消融
