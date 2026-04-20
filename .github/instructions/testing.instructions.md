---
applyTo: "tests/**/*"
description: "Use when adding or updating tests, numerical validation, or PyO3/Gym integration in only_torch."
---

# Testing Instructions

- 能做 TDD 时优先先写失败用例，再做最小修复。
- 源码单元测试放在对应模块内的 `mod tests`；`tests/` 下的集成测试文件尽量只包含一个 `#[test]`。
- 复杂数值逻辑优先参考 `tests/*.py` 中的 PyTorch 对照脚本，再把结果写回 Rust 断言。
- RL / pyo3 测试要警惕 Python 导入竞态；优先使用 `serial_test` 或单线程运行。
- 迭代时先用 `just test-filter <pattern>` 或 `just test-serial`，最后再跑 `just test`。
- 不要写只证明 mock 存在的测试；要验证真实的 `Tensor`、`Var`、`Graph` 行为。
