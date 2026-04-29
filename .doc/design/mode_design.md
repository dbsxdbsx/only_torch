# Mode 设计：训练 vs 推理的统一契约

## 1. 背景与动机

old `ExecutionContext { training, grad_enabled }` 把"层行为切换"和"是否保存反向缓存"拆成两个正交字段，再叠加 `no_grad_scope` / `inference_scope`，使整个图执行的语义出现 4 个组合状态：

| `training` | `grad_enabled` | 旧含义 |
|---|---|---|
| true | true | 训练 |
| true | false | `no_grad` 包住训练（少见、含义模糊） |
| false | true | `eval` 模式但仍可 backward（罕见、易误用） |
| false | false | 推理 / `eval + no_grad`（常见） |

这套语义带来三个具体问题：

1. **层行为与缓存策略在当前目标场景里不必独立组合**：冻结 BN fine-tune、saliency / adversarial 等研究场景确实可能需要"eval 行为 + 仍记录 backward"，但 only_torch 当前聚焦训练 / 验证 / 推理 / 演化评估，暂按 YAGNI 合并为二态。
2. **`no_grad_scope` 内 `backward()` 只是 warning**：用户可能拿不到任何梯度还以为在训练；调试期对静态图也没有"动态图缺 grad_fn 自动报错"的兜底。
3. **节点 `set_execution_ctx(&ExecutionContext)` 接口要 12 个新缓存型节点都各自手写两个字段**，重复代码多。

## 2. 新模型：`Mode { Train, Inference }`

```rust
pub enum Mode {
    Train,
    Inference,
}

impl Mode {
    pub const fn is_training(self) -> bool      { matches!(self, Mode::Train) }
    pub const fn caches_for_backward(self) -> bool { matches!(self, Mode::Train) }
    pub const fn allows_backward(self) -> bool   { matches!(self, Mode::Train) }
}
```

一个枚举同时承载三件事：

| 关注点 | `Mode::Train` | `Mode::Inference` |
|---|---|---|
| 层行为（Dropout/BN）| 训练分支（采样、累积统计） | 推理分支（确定性、用 running stats） |
| backward 缓存 | 节点保存中间张量 | 节点跳过缓存，省内存 |
| `backward()` | 允许 | 直接返回 `GraphError::InvalidOperation` |

设计原则：

- **Occam's razor**：不再保留任何"两个维度可以独立组合"的假设。
- **静态图友好**：`Inference` 模式下 backward 立刻报错，而非沉默吞掉。
- **节点零样板**：节点只需要一个字段 `should_cache_for_backward: bool` + 一行 `set_mode`。

## 3. 用户侧 API

```rust
let graph = Graph::new();           // 默认 Mode::Train

graph.train();                      // 切到训练模式（默认即如此）
graph.inference();                  // 切到推理模式
graph.is_training();                // bool
graph.mode();                       // Mode
graph.set_mode(Mode::Inference);    // 显式赋值

// 闭包式临时进入推理模式，闭包退出后回滚到进入前的模式；
// 即使闭包 panic 后被上层 catch_unwind 捕获，也会先恢复原 mode 再继续传播 panic。
let pred = graph.inference_scope(|g| {
    x.set_value(&sample)?;
    g.forward(&logits)?;
    Ok::<_, GraphError>(logits.value()?.unwrap())
})?;
```

`Graph::load_model()` 默认进入 `Mode::Inference`：加载即可推理，要继续训练时显式调用 `loaded.graph.train()`。

## 4. 节点侧契约

`TraitNode` 暴露：

```rust
fn set_mode(&mut self, _mode: Mode) {} // 默认 no-op
```

需要响应模式切换的节点（约 15 个）按下列模板实现：

```rust
struct MyOp {
    // ...
    cache: Option<Tensor>,
    should_cache_for_backward: bool,  // 默认 true
}

impl TraitNode for MyOp {
    fn calc_value_by_parents(&mut self, parents: &[&Tensor]) -> Result<(), GraphError> {
        let y = compute(parents);
        if self.should_cache_for_backward {
            self.cache = Some(prepare_cache(parents, &y));
        } else {
            self.cache = None; // 立刻释放上一次的缓存
        }
        self.value = Some(y);
        Ok(())
    }

    fn set_mode(&mut self, mode: Mode) {
        self.should_cache_for_backward = mode.caches_for_backward();
    }
}
```

`GraphInner::forward_via_node_inner` 在每次进入节点的 `forward_recursive` 时把当前 `Mode` 传下去；`NodeInner::calc_value_from_parents` 会在调用 `calc_value_by_parents` 之前同步一次 `set_mode(mode)`。公开 `backward()` 入口会在 ensure-forward 之前拒绝 `Mode::Inference`，因此误调 backward 不会先触发一次无缓存 forward。

## 5. 已接入 mode 的节点清单

| 类型 | 节点 | 缓存内容 |
|---|---|---|
| 层行为 | `Dropout` | mask（按训练 / 推理分支决定是否 dropout） |
| 层行为 | `BatchNorm` | x_hat、std、`is_training` 切 running vs batch 统计 |
| 重缓存 | `Conv2d` | im2col / padded input |
| 重缓存 | `Softmax`、`LogSoftmax` | output / softmax(x) |
| 重缓存 | `LayerNorm`、`RMSNorm` | x_hat + std/rms |
| 重缓存 | `Abs`、`Square`、`Reciprocal`、`Pow`、`Clip` | parent 输入 |
| 重缓存 | `Ln`、`Log2`、`Log10` | parent 输入 |

剩下的节点要么不缓存张量（`Add`、`Negate`、`Subtract` …），要么只缓存指针 / 形状元数据（`Concat`、`Reshape`、`Permute`、`Pad` …），不需要响应 mode 切换。

## 6. Evolution 的入口策略

`evolution::Trainer::predict_*` 和 `EvolutionTask::evaluate` 在所有"前向取值 / fitness 评估"路径上都先调用 `build.graph.inference()`：保证候选个体评估期间的内存占用与推理期保持一致，不会因为 backward 缓存把 fitness 评估拖慢或拖大内存。

## 7. 与旧 API 的对应

| 旧 API | 新等价物 | 备注 |
|---|---|---|
| `ExecutionContext { training: true,  grad_enabled: true }` | `Mode::Train` | 默认 |
| `ExecutionContext { training: false, grad_enabled: false }` | `Mode::Inference` | `load_model` 默认 |
| `ExecutionContext { training: true,  grad_enabled: false }` | 无对应 | 已删除 |
| `ExecutionContext { training: false, grad_enabled: true }` | 无对应 | 已删除 |
| `Graph::eval()` / `Graph::training()` | `Graph::inference()` / `Graph::is_training()` | |
| `Graph::set_eval_mode()` / `Graph::set_train_mode()` | `Graph::inference()` / `Graph::train()` | |
| `Graph::no_grad_scope` / `GraphInner::no_grad_scope` | `Graph::inference_scope` / `GraphInner::inference_scope` | 闭包内 backward 直接 `Err` |
| `Graph::is_grad_enabled` | `Graph::is_training()` | 因为 grad 开关已并入 mode |
| `TraitNode::set_execution_ctx(&ExecutionContext)` | `TraitNode::set_mode(Mode)` | 默认 no-op |

## 8. 与 `detach` 的关系

`Var::detach()` / `node.set_detached(true)` 仍然存在，仍然是节点级别的"局部梯度截断"，与 mode 完全正交：

- 训练模式下 `detach` 阻止上游收到梯度（GAN / Actor-Critic / Target Network 标准用法）。
- 推理模式下 `detach` 是 no-op，因为根本没人会做 backward。

具体测试见 `src/nn/tests/gradient_flow_control.rs`。

## 9. 测试入口

- `tests/test_mode_invariants.rs`：`Mode` 契约（默认值、互转、`inference_scope` 回滚和 panic-safe 恢复、推理模式 backward 报错、`load_model` 默认 inference）。
- `src/nn/tests/gradient_flow_control.rs`：`detach` + 多次 backward 行为，已删除全部 `no_grad_*` 段。
- `src/nn/tests/graph_handle.rs::test_graph_handle_inference_scope_*`：handle 层 smoke。
- `src/nn/tests/mode_cache.rs`：重缓存节点在 `Mode::Inference` forward 后跳过 backward cache，并在直接调用 VJP 时返回明确错误。
- 各重缓存节点的 `tests/node_*.rs`：默认仍跑 `Mode::Train`，验证缓存路径数值正确。

## 10. 历史与归档

旧文档 `gradient_flow_control_design.md` 被移到 [`.doc/_archive/gradient_flow_control_design.md`](../_archive/gradient_flow_control_design.md)，记录原 `ExecutionContext` 双字段时代的设计权衡，仅供考古使用，不再随接口同步更新。
