---
status: suspended
created: 2026-04-28
updated: 2026-04-28
---

# 模型参数收集样板能否更优雅

## 背景

当前手写模型需要显式实现 `Module::parameters()`，例如：

```rust
impl Module for ToySegmentationNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.conv1.parameters(),
            self.conv2.parameters(),
            self.head.parameters(),
        ]
        .concat()
    }
}
```

`src/nn/module.rs` 里的既有设计原则是：`forward()` 和 `new()` 不进入 `Module` trait，只有签名统一的 `parameters()` 放入 trait。

## 现象 / 影响

这种写法清晰但样板较多。层数变多后容易重复写 `parameters()` 拼接逻辑，也可能漏掉新增子模块参数。

## 已尝试

尚未做系统方案对比。当前只是记录问题，后续再评估宏方案与非宏方案。

## 当前卡点

需要在以下目标之间取舍：

- 保持 `Module` trait 简单，只负责 `parameters()`。
- 降低手写模型收集参数的重复代码。
- 避免为了少写几行样板引入过重的宏或隐藏行为。

## 暂缓原因

这不是当前主线阻塞项，且需要先看更多已有模型写法后再决定是否值得引入新 API。

## 下次恢复条件

当继续整理传统模型示例、或发现多处 `parameters()` 漏参数 / 重复维护成本变高时，再恢复评估。

## 下一步建议

- 先统计现有 `impl Module for ...` 的常见模式，确认重复是否足够值得抽象。
- 优先考虑轻量 helper，例如 `collect_parameters([a.parameters(), b.parameters(), ...])` 一类显式函数。
- 如果宏方案确实更清晰，再考虑 `module_parameters!(self, conv1, conv2, head)` 这种只覆盖参数收集的小宏，避免把 `forward()` 也卷进去。
---
status: suspended
created: 2026-04-28
updated: 2026-04-28
---

# 模型参数收集样板能否更优雅

## 背景

当前手写模型需要显式实现 `Module::parameters()`，例如：

```rust
impl Module for ToySegmentationNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.conv1.parameters(),
            self.conv2.parameters(),
            self.head.parameters(),
        ]
        .concat()
    }
}
```

`src/nn/module.rs` 里的既有设计原则是：`forward()` 和 `new()` 不进入 `Module` trait，只有签名统一的 `parameters()` 放入 trait。

## 现象 / 影响

这种写法清晰但样板较多。层数变多后容易重复写 `parameters()` 拼接逻辑，也可能漏掉新增子模块参数。

## 已尝试

尚未做系统方案对比。当前只是记录问题，后续再评估宏方案与非宏方案。

## 当前卡点

需要在以下目标之间取舍：

- 保持 `Module` trait 简单，只负责 `parameters()`。
- 降低手写模型收集参数的重复代码。
- 避免为了少写几行样板引入过重的宏或隐藏行为。

## 暂缓原因

这不是当前主线阻塞项，且需要先看更多已有模型写法后再决定是否值得引入新 API。

## 下次恢复条件

当继续整理传统模型示例、或发现多处 `parameters()` 漏参数 / 重复维护成本变高时，再恢复评估。

## 下一步建议

- 先统计现有 `impl Module for ...` 的常见模式，确认重复是否足够值得抽象。
- 优先考虑轻量 helper，例如 `collect_parameters([a.parameters(), b.parameters(), ...])` 一类显式函数。
- 如果宏方案确实更清晰，再考虑 `module_parameters!(self, conv1, conv2, head)` 这种只覆盖参数收集的小宏，避免把 `forward()` 也卷进去。