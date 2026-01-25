# 计算图可视化指南

> 如何使用和解读 only_torch 的计算图可视化功能
>
> 📌 **前置阅读**：[Graph 序列化与可视化设计](graph_serialization_design.md)（基础架构）

## 1. 快速开始

### 1.1 基本用法

```rust
// 训练完成后保存可视化
graph.save_visualization("model", None)?;
// 生成：model.dot + model.png（需安装 Graphviz）

// 带模型分组的可视化（推荐）
graph.save_visualization("model", None)?;
```

### 1.2 安装 Graphviz

可视化依赖 [Graphviz](https://graphviz.org/)：

- **Windows**: `winget install graphviz` 或下载安装包
- **macOS**: `brew install graphviz`
- **Linux**: `apt install graphviz` / `yum install graphviz`

## 2. 可视化时机

### 2.1 训练后可视化（推荐）

**建议在训练完成后调用可视化**，此时可获得最完整的信息：

```rust
// 训练循环
for epoch in 0..max_epochs {
    // ... 训练逻辑 ...
}

// 训练完成后
graph.save_visualization("model", None)?;
```

### 2.2 信息完整度对比

| 时机 | 可获得的信息 | 缺失的信息 |
|------|-------------|-----------|
| **训练前** | 模型结构、参数数量 | 重复次数 `(×N)`、batch 维度 `?`、变长范围 |
| **训练中** | 部分重复次数（已遇到的形状） | 尚未遇到的 seq_len/batch_size |
| **训练后** | **完整信息** | 无 |

**原因**：only_torch 采用"惰性构建 + 缓存复用"策略：
- 首次遇到某个输入形状时创建节点
- 之后复用缓存，不重复创建
- 只有遍历完所有可能的形状后，可视化才完整

## 3. 节点样式说明

### 3.1 节点类型与颜色

| 类型 | 形状 | 颜色 | 说明 |
|------|------|------|------|
| **输入节点** | 椭圆 | 浅蓝 #E3F2FD | `BasicInput`、`TargetInput`、`SmartInput` |
| **参数节点** | 矩形 | 浅绿 #E8F5E9 | 可训练参数（权重、偏置） |
| **运算节点** | 圆角矩形 | 浅黄 #FFFDE7 | `MatMul`、`Add`、`Select` 等 |
| **激活函数** | 菱形 | 浅橙 #FFF3E0 | `Sigmoid`、`Tanh`、`ReLU` 等 |
| **损失函数** | 双八边形 | 浅红 #FFEBEE | `SoftmaxCrossEntropy`、`MSELoss` |

### 3.2 特殊节点样式

| 节点 | 样式 | 含义 |
|------|------|------|
| **ZerosLike** | 虚线边框 | 占位符节点：只在 t=0 时生效，之后被隐藏状态替代 |
| **折叠节点** | 双层边框 (`peripheries=2`) | 该节点在多个子图中出现（变长序列） |

### 3.3 节点标签格式

```
节点名 <重复次数>
**节点类型**
[输出形状]
(参数量)  ← 仅参数节点显示
```

示例：

```
select ×8
**Select**
[?, 1]
```

- `select`：节点名称
- `×8`：该节点被创建/使用了 8 次（橙色，表示时间步）
- `Select`：节点类型（加粗）
- `[?, 1]`：输出形状，`?` 表示动态 batch 维度

### 3.4 重复次数标注

| 标注 | 颜色 | 含义 |
|------|------|------|
| `×8` | 橙色 | 时间步重复（RNN 展开次数） |
| `×4-12` | 橙色 | 变长序列的时间步范围 |
| `(×9)` | 蓝色 | 子图重复（不同 seq_len 创建的独立子图数） |

示例：`select ×4-12 (×9)` 表示：
- 每个子图内有 4~12 个时间步（取决于序列长度）
- 共有 9 个独立子图（遇到了 9 种不同的 seq_len）

## 4. 循环层可视化

### 4.1 边的样式

| 边类型 | 样式 | 颜色 | 说明 |
|--------|------|------|------|
| **普通数据流** | 实线 | 黑色 | 标准前向传播 |
| **初始化边** | 虚线 | 橙色 | `SmartInput → ZerosLike`，标注 `t=0` |
| **回流边** | 虚线 | 橙色 | `Tanh → ZerosLike`，标注 `t=0~6` |
| **最终输出边** | 实线 | 橙色 | `Tanh → fc.MatMul`，标注 `t=7` |

### 4.2 时间步标签

以固定长度序列（seq_len=8）为例：

```
┌─────────────┐
│ SmartInput  │
└──────┬──────┘
       │ t=0（虚线）
       ▼
┌─────────────┐  ◄── 虚线边框（占位符）
│ ZerosLike   │
└──────┬──────┘
       │
       ▼
   ... RNN 计算 ...
       │
       ▼
┌─────────────┐
│    Tanh     │───────────────┐
└──────┬──────┘               │
       │ t=7（实线橙色）      │ t=0~6（虚线橙色，回流）
       ▼                      │
┌─────────────┐               │
│ fc.MatMul   │               │
└─────────────┘               │
       ▲                      │
       └──────────────────────┘
```

**标签含义**：
- `t=0`：初始化时刻，使用 ZerosLike 生成的零张量
- `t=0~6`：这些时刻（共 7 次）产生的输出会回流给下一时刻
- `t=7`：最后时刻的输出流向下游全连接层

### 4.3 变长序列标签

以变长序列（seq_len=4~12）为例：

| 边 | 标签 | 含义 |
|----|------|------|
| 初始化边 | `t=0` | 所有序列的初始化时刻 |
| 回流边 | `t=0~(2~10)` | 回流时间步范围（最短序列到最长序列） |
| 最终输出边 | `t=3~11` | 最后时刻的范围（最短到最长） |

**解读 `t=0~(2~10)`**：
- 最短序列 (len=4)：t=0,1,2 回流（t=3 是最终输出）
- 最长序列 (len=12)：t=0~10 回流（t=11 是最终输出）
- 所以回流边标注为 `t=0~(2~10)`

## 5. 模型分组

### 5.1 使用 ModelState 自动分组

```rust
impl MyModel {
    pub fn new(graph: &Graph) -> Self {
        Self {
            rnn: Rnn::new(graph, 1, 16, "rnn")?,
            fc: Linear::new(graph, 16, 2, "fc")?,
            // 自动使用类型名 "MyModel" 作为分组名
            state: ModelState::new_for::<Self>(graph),
        }
    }
}
```

### 5.2 手动指定分组名

```rust
state: ModelState::new(graph).named("Generator"),
```

### 5.3 可视化效果

使用 `save_visualization` 时，同一模型的节点会被框在一起：

```
┌─────────────────────────────────┐
│          MyModel                │
│  ┌─────────────────────────┐    │
│  │      RNN Layer          │    │
│  │  ┌───┐  ┌───┐  ┌───┐   │    │
│  │  │...│→ │...│→ │...│   │    │
│  │  └───┘  └───┘  └───┘   │    │
│  └─────────────────────────┘    │
│            ↓                    │
│  ┌─────────────────────────┐    │
│  │      FC Layer           │    │
│  │  ┌───────┐  ┌───────┐  │    │
│  │  │MatMul │→ │  Add  │  │    │
│  │  └───────┘  └───────┘  │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
```

## 6. API 参考

### 6.1 可视化方法

```rust
impl Graph {
    /// 保存可视化（DOT + PNG）
    /// - path: 输出路径（不含扩展名）
    /// - options: 可视化选项（None 使用默认）
    pub fn save_visualization(&self, path: &str, options: Option<VisOptions>) 
        -> Result<VisResult, GraphError>;

    /// 保存带分组的可视化（推荐）
    pub fn save_visualization(&self, path: &str, options: Option<VisOptions>) 
        -> Result<VisResult, GraphError>;

    /// 生成 DOT 格式字符串
    pub fn to_dot(&self) -> String;

    /// 生成带分组的 DOT 格式
    pub fn to_dot_grouped(&self) -> String;
}
```

### 6.2 返回值

```rust
pub struct VisResult {
    /// DOT 文件路径
    pub dot_path: PathBuf,
    /// PNG 图片路径（如果 Graphviz 可用）
    pub image_path: Option<PathBuf>,
}
```

## 7. 最佳实践

### 7.1 调试时

```rust
// 快速查看当前图结构
println!("{}", graph.to_dot());
```

### 7.2 生产环境

```rust
// 训练完成后保存完整可视化
let result = graph.save_visualization(
    &format!("outputs/{}_model", model_name),
    None
)?;
println!("图保存至: {}", result.dot_path.display());
```

### 7.3 文档生成

```rust
// 同时生成 summary
graph.save_summary("outputs/model_summary.md")?;
graph.save_visualization("outputs/model", None)?;
```

## 8. 常见问题

### Q1: 为什么 batch 维度显示为 `?`？

**A**: only_torch 支持动态 batch，同一个图结构可以处理不同 batch_size 的输入。`?` 表示该维度在运行时确定。

### Q2: 为什么变长 RNN 的全连接层标注 `(×9)`？

**A**: 表示遇到了 9 种不同的 seq_len，每种都创建了独立的子图。这是"惰性构建"机制的结果，不影响推理效率（相同形状会复用缓存）。

### Q3: ZerosLike 为什么是虚线框？

**A**: ZerosLike 是占位符节点：
- 只在 t=0 时产生实际输出（全零初始状态）
- t>0 时被上一时刻的隐藏状态替代
- 虚线边框表示其"虚拟"性质

### Q4: 如何只生成 DOT 文件不生成图片？

**A**: 直接使用 `to_dot()` 方法：

```rust
std::fs::write("model.dot", graph.to_dot())?;
// 手动渲染：dot -Tpng model.dot -o model.png
```

## 9. 相关文档

- [Graph 序列化与可视化设计](graph_serialization_design.md) - 基础架构设计
- [记忆/循环机制设计](memory_mechanism_design.md) - RNN/LSTM/GRU 实现细节
- [梯度流控制机制](gradient_flow_control_design.md) - `detach`、`no_grad` 等机制
