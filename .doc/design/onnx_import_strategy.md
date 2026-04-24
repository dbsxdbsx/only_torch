# ONNX 导入/互通策略设计

> 跟第三方（PyTorch / onnxruntime / Netron 等）通过 ONNX 协作时的**定位、决策树、UX 契约、语义漂移对策**
>
> 📅 **状态**：规划文档（2026-04-23）。当前 `onnx_import.rs` 已实现基础装配 + Conv/Gemm 拆分；后续扩展按本文档的决策树推进。
>
> 🎯 **适用场景**：下次需要支持新的 PyTorch 模型、遇到未支持的 ONNX 算子、或讨论是否要扩展 shape 操纵类算子时，优先阅读本文档。
>
> 📖 **相关设计**：
>
> - [Graph 序列化与可视化设计](graph_serialization_design.md)（`.otm` 格式、GraphDescriptor、DOT 可视化基础）
> - [计算图可视化指南](visualization_guide.md)（节点样式、重复次数、provenance 的渲染规则）

## 1. 根本定位：ONNX 是协议，不是源

only_torch 与外部生态交互的事实路径：

```
PyTorch (训练) ──export──▶ ONNX ──import──▶ only_torch ──save──▶ .otm
                                                ▲                    │
                                                └──── load ──────────┘
                                                    （权威源格式）
```

结论：

- **`.otm` 才是 only_torch 的源**（可 round-trip、保真、带完整语义）
- **ONNX 只是跟 PyTorch 握手的协议**（single-shot import，不承诺 round-trip）
- **定位对标**：TensorRT / TVM 一侧（import 后即 lowering），而不是 tract / onnxruntime 一侧（严格保真）

这个定位决定了 import 阶段可以做**激进的 lowering**（常量折叠、模式重写、子图合并）—— 这不是越权，而是"编译器"的本职工作。

## 2. 算子分类决策树

遇到新的 ONNX 算子缺口时，按此决策树判断处理路径：

### Step 1：数值计算类 vs shape 元信息类

| 类别 | 特征 | 典型算子 | 处理方式 |
|------|------|---------|---------|
| **数值计算类** | 有梯度、有真实 tensor 数据、参与前向 | Conv, MatMul, GELU, LayerNorm, BatchNorm, Softmax | 直接做 runtime node |
| **shape 元信息类** | 无梯度或仅透传、输入/输出常为 shape 向量 | Shape, Gather, Slice, Reshape, Squeeze, Unsqueeze, Concat(维度维)，Cast, Where, Constant, ConstantOfShape | 进 Step 2 |

### Step 2：shape 类 —— 表达语义 vs 真动态计算

| 场景 | 特征 | 处理方式 |
|------|------|---------|
| **表达语义**（tracer 副产物） | 所有输入可静态推导，本质是"拼"一个已知 shape | Step 3-A：模式匹配 → 替换为内置算子 |
| **真动态计算** | 存在依赖运行时值的输入（如 transformer 按 seq_len slice） | Step 3-B：评估必要性 |

### Step 3-A：模式匹配（优先）

识别 ONNX tracer 常见的"一坨 shape 子图"，替换成 only_torch 已有的内置算子：

| PyTorch 源码意图 | ONNX 子图 | 目标内置算子 |
|------------------|-----------|-------------|
| `x.view(x.shape[0], -1)` | `Shape → Gather(0) → Unsqueeze → Concat([., -1]) → Reshape` | `Flatten { keep_first_dim: true }` |
| `x.view(B, -1)`（B 静态） | `Shape → … → Reshape` | `Flatten` 或 `Reshape { target_shape }` |
| `x[:, :, h1:h2, w1:w2]`（常量） | `Slice(static starts/ends)` | 内置 `Slice`（规划中） |
| `x.squeeze(dim)` / `x.unsqueeze(dim)` | `Squeeze` / `Unsqueeze` | 内置 `Squeeze` / `Unsqueeze`（规划中） |
| `torch.cat([a, b], dim=C)` | `Concat(axis=C)` | 已有 `NodeTypeDescriptor::Concat` |

### Step 3-B：真动态计算

| 目标场景 | 必要性 | 决策 |
|----------|--------|------|
| CV CNN（ResNet / MobileNet / YOLO 系列） | 基本用不到真动态 shape | 不做，ImportReport 提示用 `onnxsim` |
| Transformer / RNN 动态 seq_len | 必要 | 实现为 runtime shape op（路线 A） |

## 3. 三条实现路线

### 路线 A：逐个算子做成 runtime node

- ✅ 直接、统一、与 Conv2d/MatMul 对称
- ❌ 维护面随 ONNX 算子数爆炸
- ❌ shape 元信息塞进 Var/autograd 架构会引入概念污染（是否有 backward？输出类型与 tensor 节点是否一致？）
- ❌ 对 PyTorch tracer 的 shape 子图是浪费

**适用**：Step 3-B 的真动态场景

### 路线 B：import 期做常量折叠 + 模式匹配

- ✅ 一次投入吃掉 PyTorch tracer 产生的 90% shape 子图
- ✅ 不污染运行时
- ✅ 与业界 `onnxsim` / `onnxoptimizer` 同思路，可借鉴
- ❌ 对真正动态的 shape 无能为力
- ❌ 需要轻量 ONNX shape/value inference 实现

**适用**：Step 2 / Step 3-A —— **主推路线**

### 路线 C：约束 PyTorch 导出侧

- ✅ 零框架改动，借力上游生态
- ✅ 工具成熟（`onnxsim`, `torch.onnx.export(dynamo=True)`, 手工改 `torch.flatten` 替代 `x.view`）
- ❌ 依赖用户纪律
- ❌ 第三方（HuggingFace）下载的模型不一定能改

**适用**：文档 + example 的 `requirements.txt`，**B 的前置补充**

### 推荐组合：B + C 为主，A 按需补充

先让 `onnxsim` 吃掉 80% 的 shape 噪声，内部 import pass 处理剩下的 10%，真动态的 10% 按需走路线 A。

## 4. Import UX 契约

### 4.1 ImportReport：import 过程透明化

`OnnxImportResult` / `RebuildResult` 应带 `import_report` 字段：

```rust
pub struct ImportReport {
    /// ONNX 原始节点数
    pub original_node_count: usize,
    /// 最终 GraphDescriptor 节点数
    pub final_node_count: usize,
    /// 被折叠为常量的节点清单
    pub folded: Vec<FoldedRecord>,
    /// 被模式匹配重写的子图
    pub rewritten: Vec<RewriteRecord>,
    /// 告警（如动态子图被折成常量、算子升级、bias 升维等）
    pub warnings: Vec<ImportWarning>,
}

pub struct FoldedRecord {
    pub rule: &'static str,              // 规则名，如 "constant_folding"
    pub consumed_onnx_nodes: Vec<String>,
    pub produced_descriptor_node: u64,
}

pub struct RewriteRecord {
    pub pattern: &'static str,           // 如 "view_to_flatten"
    pub consumed_onnx_nodes: Vec<String>,
    pub produced_descriptor_nodes: Vec<u64>,
}
```

### 4.2 ImportOptions：显式控制入口

```rust
pub struct ImportOptions {
    /// 激进常量折叠（默认 true）
    pub fold_constants: bool,
    /// 启用模式匹配重写（默认 true）
    pub enable_pattern_rewrite: bool,
    /// 严格模式：保留 ONNX 结构，禁用所有 lowering（默认 false）
    pub strict: bool,
}
```

### 4.3 错误消息必须 actionable

遇到不支持算子时，错误至少包含三段：

1. **是什么**：算子名 + 出现位置（ONNX 节点名）
2. **为什么**：分类（shape 类？真动态？opset 不兼容？）
3. **怎么办**：
   - 可折叠的：建议 `pip install onnxsim && onnxsim model.onnx model-sim.onnx`
   - 导出时可避免的：建议改 PyTorch 代码（如 `torch.flatten` 替代 `x.view(x.shape[0], -1)`）
   - 真不支持的：明确告知"当前目标场景不支持此动态算子"

### 4.4 to_onnx：暂不提供

理由：

- Round-trip 不是 only_torch 的定位目标
- 用户要部署，`.otm` + Rust 原生推理是更直接的路径
- 维护双向算子映射表成本高
- 将来若提供，应作为**功能子集**（能映射的导出，不能的报错），而非 round-trip 保真

## 5. 可视化兼容性：折叠是利好

### 5.1 折叠后可视化更清晰，不是更差

**原则**：用户打开可视化时想看的是"模型意图"，不是"tracer 副产物"。

以 `x.view(x.shape[0], -1)` 为例：

| 状态 | 节点数 | 可视化观感 |
|------|--------|----------|
| ONNX 原始 | 5（Shape → Gather → Unsqueeze → Concat → Reshape） | 噪声，跟 PyTorch 源码对不上 |
| 折叠 + 模式匹配后 | 1（Flatten） | 清晰，与源码意图一致 |

业界共识：打开 Netron 前通常先跑 `onnxsim` —— 人类读图需要的抽象层级，与 ONNX 序列化层级**本来就不同**。

### 5.2 Provenance（来源追溯）

折叠/重写时给合成节点挂来源字段：

```rust
pub struct NodeDescriptor {
    // ... 现有字段 ...
    /// 此节点由哪些 ONNX 节点合并/重写而来（import 时填充）
    pub origin_onnx_nodes: Vec<String>,
}
```

可视化集成：

- DOT 节点 tooltip 显示 `origin: Shape_23, Gather_24, Reshape_25`
- `summary()` 可选新增 provenance 列
- Markdown summary 同样支持

**价值**：调试时可以精确追溯"这个 Flatten 是从 ONNX 的哪几个节点合成的"。信息不真丢，只是默认折叠。

### 5.3 `.otm` 序列化同步

`origin_onnx_nodes` 要写进 `.otm` 的 JSON：`.otm` 重载后，可视化仍能展示来源。这是 "`.otm` 是权威源" 定位的自然延伸。

## 6. 语义漂移：三层视角与对策

| 层次 | 是否漂移 | 严重程度 | 对策 |
|------|---------|---------|------|
| **数学/功能语义** | 不漂移（浮点误差内 bit-level 一致） | ✅ 安全 | 回归测试覆盖 |
| **图结构语义** | 漂移（节点数/名/连接变化） | ⚠️ 业界可接受 | 文档声明；`strict` 模式作为逃生舱 |
| **动态性语义** | **可能漂移** | ⚠️⚠️ 最需小心 | 模式匹配优先 + ImportReport 告警 |

### 6.1 动态性漂移典型案例

```python
x = x.view(x.shape[0], -1)   # PyTorch 源码
```

- 导出时用 `dynamic_axes={'input': {0: 'batch_size'}}`：子图**无法完全折叠**（Shape 的输出依赖运行时 batch 维），保真
- 未用 `dynamic_axes`：batch 维在 export 时被冻结成常量 → 子图**可完全折叠**成 `Reshape(shape=[N, rest])`
  - **陷阱**：模型只能吃 `batch=N` 的输入，换 batch 就挂

### 6.2 对策（按优先级）

1. **模式匹配优先**：能识别成 `Flatten`（支持动态 batch，已处理 `parent_shape[0] == 0`）就用 Flatten，**不要** fallback 到带静态 shape 的 Reshape
   - 对应 `src/nn/nodes/raw_node/ops/flatten.rs:68` 已有的动态修复
2. **ImportReport 告警**：检测到"此子图被折成了静态常量、但原本可能是动态的"时，在 warnings 里明确记录
3. **文档约束**：README / example 明确要求"要动态就在 PyTorch 导出时开 `dynamic_axes`"

## 7. 代码结构提议

实施时建议的模块组织：

```
src/nn/graph/
├── onnx_import.rs              # 现有：主流水线入口
├── onnx_ops.rs                 # 现有：算子映射表
├── onnx_error.rs               # 现有：错误类型
├── onnx_import/                # 新增：lowering 子模块
│   ├── mod.rs
│   ├── folding.rs              # 常量折叠 pass
│   ├── patterns.rs             # 模式匹配规则库
│   ├── report.rs               # ImportReport / ImportWarning
│   └── shape_inference.rs      # 轻量 shape/value inference
└── ...
```

### 7.1 Pass 顺序

```
load_onnx()
 ├─ 1. 解析 onnx bytes
 ├─ 2. 验证 opset
 ├─ 3. 构建 SymbolTable
 ├─ 4. 第一遍装配（initializer + 原始节点）
 ├─ 5. ⭐ shape_inference 填充中间值形状
 ├─ 6. ⭐ pattern rewrite pass（view→flatten 等）
 ├─ 7. ⭐ constant folding pass
 └─ 8. 返回 OnnxImportResult + ImportReport
```

Pass 5 / 6 / 7 是新增的 lowering 阶段，每个 pass 向 `ImportReport` 追加条目。

### 7.2 模式库可扩展性

每个模式是独立模块，实现统一 trait：

```rust
pub trait PatternRewrite {
    fn name(&self) -> &'static str;
    fn try_match(&self, desc: &GraphDescriptor, anchor: u64) -> Option<MatchResult>;
    fn rewrite(&self, desc: &mut GraphDescriptor, m: MatchResult) -> RewriteRecord;
}
```

新加一个模式 = 加一个文件 + 注册到 `patterns::all()`，不改主流程。

## 8. 回归保障

### 8.1 模型级回归

`tests/onnx_models/` 目录，每个目标模型一个最小 reproducer：

```
tests/onnx_models/
├── chess_cnn/          # 已验证：97.1% → 97.8%（2026-04 里程碑）
│   ├── export.py       # 生成 model.onnx 的脚本（避免 git-lfs）
│   └── test.rs         # import + forward + 数值比对
├── resnet18/           # 下一个目标（规划中）
│   └── ...
└── common/             # 共用测试工具
    └── numerical_check.rs
```

### 8.2 每加一个模型的标准动作

1. 用脚本导出 ONNX，`cargo test --test onnx_models` 跑一次
2. 失败 → 按决策树（第 2 节）定位缺口 → 优先走路线 B + C → 必要时走路线 A
3. 把最小失败 reproducer 提交到 `tests/onnx_models/<model>/` 作为回归保障
4. 通过后在 README 的 "supported models matrix" 登记

### 8.3 supported models matrix（在 README 维护）

| 模型 | 来源 | 核心算子 | forward 数值误差 | 状态 |
|------|------|----------|----------------|------|
| chess CNN | `examples/traditional/chinese_chess/` | Conv / Gemm / Flatten / ReLU / Softmax | 1e-6 | ✅ |
| ResNet-18 | torchvision | Conv / BatchNorm / ReLU / GlobalAvgPool / Gemm | ? | ⏳ |
| MobileNetV2 | torchvision | DepthwiseConv / BatchNorm / ReLU6 / Gemm | ? | ⏳ |

这个表同时是**能力声明**和**优先级路线图**。

## 9. Backlog（按需求驱动，本表为权威入口）

> 任何新 plan 立项前先看本表；立项时把 plan 文件名加进对应行的 "立项 plan" 列；
> plan 完成后把对应行**从表中移除**，改动记录到 [`CHANGELOG.md`](../../CHANGELOG.md)
> （避免 backlog 表与 CHANGELOG 重复维护）。
> "永远不做"的项保留在表中并标 ❌，作为决策依据存档。

> 本表上一次大修订：2026-04-24（YOLO followup three commits 完成后整合，
> 来源于旧 plan §9、新 plan §6、设计文档原 §9 三处散落的 backlog 项）。

### 9.1 业务层（4 项）—— 跟 ImportReport / R4 无关，按业务需求驱动

| 项 | 触发条件 | 预期产出 | 来源 | 立项 plan | 风险 |
|---|---|---|---|---|---|
| 迁移到 meng_ru_ling_shi 连线器 | yolo example 业务跑通后立即 | `meng_ru_ling_shi/` Rust 后端集成 + FRB 桥接 | 旧 plan §9.1 | 待立 | FRB 类型边界需小心，ImportReport 跨语言传输需序列化 |
| fine-tune 通道（R3 兜底） | VinXiangQi 在 QQ 象棋等其他软件精度 < 30/32 | README 写 fine-tune 指引：Roboflow/LabelImg 标 ~30 张 + 训 10-30 epoch | 旧 plan §9.3 | - | 文档级，无代码风险 |
| ROI CLI 参数化 / GUI 标定 | meng_ru_ling_shi 集成阶段 | example main.rs 加 CLI 参数 + meng_ru_ling_shi 端 GUI 标定工具 | 旧 plan §9.7 | - | 低 |
| 多输入图片批处理 | 用户提批量推理需求 | example 加 batch 模式 + Conv2d 等算子 batch>1 路径验证 | 旧 plan §9.8 | - | 中（动态 batch 路径已有但未充分覆盖） |

### 9.2 算子层（3 项）—— 按模型驱动

| 项 | 触发条件 | 预期产出 | 来源 | 立项 plan | 风险 |
|---|---|---|---|---|---|
| Upsample2d 非 nearest 模式（bilinear/bicubic） | 出现用 bilinear/bicubic 上采样的模型 | `Upsample2d::mode` 字段 + 前向/反向数学推导 + PyTorch 数值对照单测 | 旧 plan §9.2 | - | 中（反向梯度公式不平凡） |
| 首个非 YOLO 第三方模型验证（ResNet-18 / MobileNetV2） | 决定扩展模型支持范围时 | `tests/onnx_models/<model>/` + supported models matrix 登记 | 设计文档原 §9.4 | - | 中（可能撞到新算子缺口） |
| Conv 非对称四角 padding 支持 | 实际遇到 `pads=[1,2,3,4]` 这种非对称模型（YOLOv5 全对称，未触发） | 在 ONNX 导入端用 Pad + Conv(p=0) 组合表达；或 NodeTypeDescriptor::Conv2d 字段升 4 维 | YOLO followup Commit 1 留 backlog | - | 中（涉及 evolution 模块 ~14 处 Conv padding 字面量） |

### 9.3 ImportReport 扩充（3 项）—— R4 风险显式守住，YAGNI

> R4 立项原则：每个字段的入场券是"现在就有人填、现在就有人读"。
> 缺一项就不做。下面这 3 项被 R4 明确"暂缓"，等真实需求出现再开 plan。

| 项 | 触发条件 | 预期产出 | 来源 | 立项 plan | 风险 |
|---|---|---|---|---|---|
| `folded` 字段 + 独立 folding pass `onnx_import/folding.rs` | 缺口中出现 Shape→Gather→…→Reshape 这种 tracer 副产物子图 | `ImportReport.folded: Vec<FoldedRecord>` + 折叠 pass 实现 + ImportReport 联动 | 旧 plan §9.5 + 设计文档原 §9.5 | - | 中（设计折叠 pass 与 const_table 路径协调） |
| `ImportOptions { fold_constants, strict, ... }` | 用户提"想关掉某个 lowering 调试" 或 strict 模式 | 新 struct + load_onnx 重载入参 + 调用链每层透传 | 旧 plan §9.5 | - | 中（调用链透传影响面广） |
| `ImportWarning` 从 String 升级为结构化 enum | warnings 数量 ≥ 5 种且看出共性时（当前 4 种） | `ImportWarning { kind, location, ... }` + 现有 push 点全改 | 新 plan §6 | - | 低 |

### 9.4 架构层（1 项）

| 项 | 触发条件 | 预期产出 | 来源 | 立项 plan | 风险 |
|---|---|---|---|---|---|
| 真动态 shape op runtime | 接 transformer / RNN 必须动态 seq_len 场景 | 路线 A 的实际实施（详见本文档 §3） | 设计文档原 §9.7 | - | 高（架构性变更） |

### 9.5 永远不做（决策存档）

| 项 | 理由 | 来源 | 状态 |
|---|---|---|---|
| `to_onnx` 反向导出 round-trip | only_torch 定位是"小型推理框架/编译器"（TVM/TensorRT 一侧），不是"严格协议保真"工具（onnxruntime 一侧）。维护双向算子映射成本太高，且用户要部署直接用 `.otm` + Rust 原生推理更合适 | 设计文档 §1 / §4.4 | ❌ |

## 10. 关键参考

- **onnxsim**：<https://github.com/daquexian/onnx-simplifier> —— 业界事实标准的 ONNX 化简工具，折叠 + 模式重写的主要参考实现
- **onnxoptimizer**：<https://github.com/onnx/optimizer> —— ONNX 官方优化 pass 集合
- **torch.onnx.export dynamo 模式**：<https://pytorch.org/docs/stable/onnx_dynamo.html>
- 本项目已有基础（供实施时对齐）：
  - `src/nn/graph/onnx_import.rs`：主流水线 + Conv+bias 拆分（`is_conv_with_bias` 分支，2026-04）
  - `src/nn/graph/onnx_ops.rs`：ONNX ↔ NodeTypeDescriptor 双向映射
  - `src/nn/nodes/raw_node/ops/flatten.rs:68`：动态 batch 处理（`parent_shape[0] == 0` 分支）
  - `src/nn/graph/descriptor_rebuild.rs`：`RebuildResult.parameters` 字段
  - `examples/traditional/chinese_chess/`：端到端验证的第一个模型（PyTorch→ONNX→continue-train→`.otm`）
