# D&R 代码改动说明（分支 `dnr-fixes`，基于 `origin/prompt-vla` 7d01bc3）

日期：2026-09-07。主 commit `605963e`，消融配置 `652e88b`；2026-09-09 移除时间编位 RoPE / root_gather（消融显示无收益），memory 侧接口回到原版 3 元组。

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


## 配置基线：与 hiersel_bud64_pool128_multilevel_emareducer 的对比

`perceptual-dnr-modul_bud64_pool128.yaml` 从 `perceptual-hiersel-modul_bud64_pool128_multilevel_emareducer.yaml` 改出。

不变的部分：budget 64、pool_budget 128、type hierarchical_selection、pool_sampling even、keep_ratio 0.5、selector depth 2 / 8 heads / 4 register、memory_feature、modulation 集成、use_time_emb true。也就是同一棵 3 节点树（8 帧 128 token，一轮 128→64，root 64→32），同样的 selector 结构。

改动的行（每行对应下面的两处修复）：

| 项 | hiersel_emareducer（旧） | dnr（新） | 对应修复 |
|---|---|---|---|
| `sampling` / `score_norm` / `tau` / `noise_scale` | 无（默认逐 token 伯努利） | topk / zscore / 1.0 / 1.0 | 1 Gumbel-Top-K |
| `ratio` / `z` / `load_balance` 损失权重 | 1e-3 / 1e-4 / 0.1 | 全 0 | 1 |
| `multilevel` / `ema_reducer` | true / true | false / false | 2 |
| `e2e_tree` | 无 | true | 2 端到端树 |

## 两处改动

### 1. Gumbel-Top-K 替换逐 token 伯努利

- 配置：`perceptual_memory.selector.sampling: topk`（默认 `bernoulli` = 旧行为），`score_norm: zscore`，`tau`，`noise_scale`
- 代码：`selector.py` 新增 `zscore_margin`、`gumbel_topk`
- 旧：每个 token 独立做两类 Gumbel-softmax，保留数随机，靠 ratio / z / load-balance 三个辅助损失兜底；部署却是严格 top-K
- 新：keep−drop 的 margin 先 z-score，训练时加 Gumbel 噪声，取恰好 K 个；反向用以第 K 名与第 K+1 名中点为阈值的 sigmoid 做直通。三个辅助损失全部删除；推理走同一函数、不加噪声
- 效果：训练与部署数量一致；决策从绝对阈值变成组内排名，selector 不需要校准尺度；"全留 / 全不留"两种塌缩在构造上不存在。load-balance 损失在等间隔布局下等于"禁止总是保留 frame 0"，删除后 selector 可以保留锚点帧

### 2. 整棵树端到端

- 配置：`selector.e2e_tree: true`（默认 false），与 `multilevel`、`ema_reducer` 互斥
- 代码：`percep_mem.py` 的 `_pick`、`_reduce_one_round`、`_hierarchical_reduce`（多了可选 `rng`）
- 因为开销基本不变甚至更低（前向不变（旧代码本来每个节点都跑 selector），多出各节点 selector 的反向），而且考虑会更稳定、以及尽量避免distribution shift p(memory|history)（同时也尽量避免 p(action|obs, memory)的distribution shift）, 就换成整棵树了。


#### 与原版的逐行对照（`_reduce_one_round`，percep_mem.py）

无标记的行是原版原样保留；`-` 是原版被替换掉的行；`+` 是新增行。

```diff
-    def _reduce_one_round(self, hidden, valid, scorer=None):
+    def _reduce_one_round(self, hidden, valid, scorer=None, rng=None):
+        # [改动2] rng: 本节点 Gumbel 噪声
         chunk, keep = self.reduce_chunk_size, self.reduce_chunk_keep
         b, n = hidden.shape[0], hidden.shape[1]
         n_chunks = -(-n // chunk)  # ceil
         pad = n_chunks * chunk - n
         if pad:
             hidden = jnp.pad(hidden, ((0, 0), (0, pad), (0, 0)))
             valid = jnp.pad(valid, ((0, 0), (0, pad)))
         dim = hidden.shape[-1]
         hc = hidden.reshape(b * n_chunks, chunk, dim)
         vc = valid.reshape(b * n_chunks, chunk)

-        # 原版：打分 stop_gradient，确定性 top-k，按位置排序，没有权重
-        logits = jax.lax.stop_gradient(
-            (scorer if scorer is not None else self.selector)(hc, vc)
-        )
-        idx = select_topk(logits, vc, keep)
-        idx = jnp.sort(idx, axis=-1)
+        # [改动2] 抽成 _pick：e2e_tree 关 → 执行的就是上面被删的 4 行，weight=None
+        #                    e2e_tree 开 → 活 selector + gumbel_topk(rng)，返回 STE 权重
+        idx, weight = self._pick(hc, vc, keep, scorer, rng)

         hc = batch_gather(hc, idx)
+        if weight is not None:                                # [改动2] 端到端的核心
+            # 前向 ×1（保留 token 的权重恰为 1），反向把 ⟨dL/d(out_i), hc_i⟩ 送进本节点 logits
+            hc = hc * batch_gather(weight, idx)[..., None].astype(hc.dtype)
         vc = batch_gather(vc[..., None], idx)[..., 0]

         return hc.reshape(b, n_chunks * keep, dim), vc.reshape(b, n_chunks * keep)
+        # rng=None（legacy 或 eval）→ _pick 走原版分支，行为与原版逐 bit 一致
```

配套新增的 `_pick`：

```diff
+    def _pick(self, hc, vc, keep, scorer, rng):
+        if self.e2e_tree:                                     # [改动2] 新分支
+            logits = self.selector(hc, vc)                    #   活 selector，不 stop_gradient
+            weight, idx = gumbel_topk(logits, vc, keep, rng,  #   恰好 keep 个，带 STE 权重
+                                      tau=self.tau, noise_scale=self.noise_scale,
+                                      score_norm=self.round_score_norm)
+            return idx, weight
+        # legacy 分支 = 原版被删的 4 行原样搬过来
+        logits = jax.lax.stop_gradient((scorer if scorer is not None else self.selector)(hc, vc))
+        idx = select_topk(logits, vc, keep)
+        return jnp.sort(idx, axis=-1), None
```

梯度链（`gumbel_topk` 定义，改动 1；内部节点复用，改动 2）：

```
dL/dw_i  = ⟨dL/d(out_i), hc_i⟩          乘法的导数：输出梯度与 token 内容的内积
dL/ds_i  = dL/dw_i · y_i(1−y_i)/τ       sigmoid 松弛的导数，thr 取第 K 名与第 K+1 名中点
dL/dlogit_keep_i = +dL/ds_i · (z-score 雅可比)，dL/dlogit_drop_i = −同值
→ selector 的 head / blocks / register tokens（三个节点共享，梯度相加）
```

被丢掉的 token 的直接 dL/dw 为零，它们的信号来自 z-score 把同节点 margin 耦合起来，以及每轮独立的 Gumbel 噪声让边界 token 偶尔被选中。

## 结果（bud64 / pool128，ckpt 39999，每任务 150 集 = 50 集 × 3 seed）

| 配置 | 均值 |
|---|---|
| hiersel_pool128_multilevel_emareducer（旧） | 20.25 |
| framesamp-modul_bud64（无 selector） | 27.12 |
| e2e 树 + 伯努利 root（`abl_only_tree`） | 28.33 |
| e2e 树 + Gumbel-Top-K + 时间 RoPE（原 full DNR，已删除） | 28.63 |
| **e2e 树 + Gumbel-Top-K（`abl3_slotrope` = 现在的主配置）** | **30.04** |

逐任务（单任务噪声约 ±7，只看大的模式）：

| 任务组 | 任务 | full DNR | 主配置 | only_tree | 主配置 − full | only_tree − 主配置 |
|---|---|---|---|---|---|---|
| Counting | BinFill | 30.0 | **40.7** | 32.7 | +10.7 | −8.0 |
| | StopCube | **24.0** | 11.3 | 9.3 | −12.7 | −2.0 |
| | PickXtimes | 57.3 | **64.7** | 62.0 | +7.3 | −2.7 |
| | SwingXtimes | 84.0 | 86.0 | **89.3** | +2.0 | +3.3 |
| Permanence | ButtonUnmask | **18.0** | 13.3 | 8.0 | −4.7 | −5.3 |
| | VideoUnmask | **31.3** | 30.7 | 28.0 | −0.7 | −2.7 |
| | VideoUnmaskSwap | 20.0 | **20.7** | 15.3 | +0.7 | −5.3 |
| | ButtonUnmaskSwap | 14.7 | 4.7 | **18.0** | −10.0 | +13.3 |
| Reference | PickHighlight | 16.0 | **18.0** | 16.7 | +2.0 | −1.3 |
| | VideoRepick | **15.3** | 12.0 | 14.7 | −3.3 | +2.7 |
| | VideoPlaceButton | 32.7 | **34.7** | 28.0 | +2.0 | −6.7 |
| | VideoPlaceOrder | 24.7 | 19.3 | **26.0** | −5.3 | +6.7 |
| Imitation | MoveCube | 61.3 | **62.0** | 57.3 | +0.7 | −4.7 |
| | InsertPeg | 2.0 | **4.7** | **4.7** | +2.7 | 0.0 |
| | PatternLock | 10.0 | **24.0** | 16.7 | +14.0 | −7.3 |
| | RouteStick | 16.7 | **34.0** | 26.7 | +17.3 | −7.3 |
| | **Overall** | **28.625** | **30.042** | **28.333** | **+1.417** | **−1.708** |

读数：

- 去掉时间编位 RoPE 涨 1.42 pp。时间 RoPE 这条路（改动 3）已从代码里删除，主配置回到 slot 编位 + 原位 mask。
- 再把 Gumbel-Top-K 换回伯努利跌 1.71 pp，两处改动的方向都得到确认。
- Gumbel-Top-K 的收益集中在 BinFill、PatternLock、RouteStick、VideoPlaceButton；ButtonUnmaskSwap 和 VideoPlaceOrder 反而是伯努利更好，量级都在单任务噪声附近。
- StopCube 是三个配置里唯一大幅退步的任务（24.0 → 11.3），值得单独看。

## 消融配置（每组只还原一项，其余保持完整版）

| 配置 | 还原的项 | 具体设置 |
|---|---|---|
| `perceptual-dnr-modul_bud64_pool128_abl1_bernoulli.yaml` | Gumbel-Top-K | root 用逐 token 伯努利，恢复 ratio 1e-3 / z 1e-4 / lb 0.1；内部节点仍 e2e + top-k |
| `perceptual-dnr-modul_bud64_pool128_abl2_noe2e.yaml` | 端到端树 | 内部节点回到 stop_gradient，只有 root 训 selector |

为支持 abl1，`652e88b` 把 root 的采样方式和内部节点解耦：内部节点固定 e2e + gumbel_topk（物理缩小需要恰好 K），root 可独立选 topk / bernoulli。

## 叠加：随机内部节点路由（`aux_node_prob`）

- 配置：`perceptual-dnr-modul_bud64_pool128_auxnode02.yaml`，即主配置加一行 `selector.aux_node_prob: 0.2`
- 代码：`percep_mem.py` 的 `_pick_node_input`、`_hierarchical_reduce(collect=...)`、`__call__` 的路由分支
- 动机：e2e 树里内部节点丢掉的 token 直接离开计算图，唯一的梯度来源是 Gumbel 噪声偶尔把它捞回来，离边界远就基本拿不到。而 root 的 keep-weight 是乘在 MemoryAttention 的 `exp(score)` 上再归一化的，被丢的 token 满足 `∂p/∂g_i ≠ 0`，拿得到"把它放进记忆动作会不会变好"的反事实梯度
- 做法：训练时每个样本以 0.2 的概率把 root 那一刀改喂给一个均匀随机的内部节点输入（`reduce_chunk_size` 宽的连续块），另外 0.8 走正常的树。评测完全不受影响
- 开销：零。归约树本来就对整个 batch 跑完，被路由的样本只是用 `where` 丢弃它的输出；内部节点输入是已有张量的切片。相比"再过一遍 policy 算辅助损失"的做法，后者要 2 倍前向，而且辅助损失同样会把 policy 训练在窄记忆上，并不能避免分布偏移
- 边界情况：整块都是 padding 的节点（短 episode + 右填充）会回落到 root，保证记忆里至少有一个有效 slot
- 新增 stat：`aux_node_frac`，实际被路由的样本比例，应当稳定在 0.2 附近
- 与 `multilevel` 的区别：multilevel 是 100% 概率、且内部节点全程 `stop_gradient`，policy 大部分步数训练在窄记忆上；这里是 20% 概率叠加在完整的 e2e 树之上


## 其他文件

- `scripts/launch_dnr_devbox.sh`：4×A800 启动脚本，自动识别 bin/npy、缺 norm_stats 时从 a2r 仓库复制、预检 GPU 与重复启动
- `examples/robomme/subgoal_predictor.py`：Gemini / Qwen SDK 改惰性导入（感知记忆评测不需要）
- `tests/test_dnr_fixes.py`：5 项检查（gumbel_topk、e2e 梯度、完整前向、内部节点路由、全部 pool128 配置），`JAX_PLATFORMS=cpu` 两分钟跑完；旧套件 `tests/test_hierarchical_reduction.py` 19/19 通过
- 新增 stats：`keep_frac`（topk 下恒为 0.5）、`reduce_keep_frac`、`aux_node_frac`

## 未做与待办

- frame-0 锚点？
- `aux_node_prob: 0.2` 的 40k 训练与评测
- StopCube 从 24.0 掉到 11.3 的原因
