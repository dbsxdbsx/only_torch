# 开发环境配置指南

> 最后更新: 2025-01-31
> 状态: **已验证**
> 适用范围: Only Torch 开发者

---

## rust-analyzer 与 cargo 编译冲突问题

### 问题描述

引入 PyO3 等带有 `build.rs` 构建脚本的依赖后，发现每次修改代码（即使与 PyO3 无关）运行 `cargo build` 或 `cargo test` 时，PyO3 及其相关依赖都会被重新编译，严重影响开发效率。

### 根本原因

**rust-analyzer（IDE 的 Rust 语言服务器）与手动 cargo 命令共享同一个 `target/` 目录**，导致编译冲突：

1. rust-analyzer 默认使用 `cargo check` 分析代码
2. 开发者手动运行 `cargo build` / `cargo test`
3. 两者的编译配置可能有细微差异（feature flags、环境变量等）
4. 当它们交替运行时，PyO3 的 `build.rs` 检测到环境变化，触发重新编译

PyO3 的 `build.rs` 特别敏感，因为它需要检测 Python 环境、设置链接参数等。

### 诊断方法

```bash
# 连续两次构建，观察是否重复编译 pyo3
cargo build
cargo build  # 如果这次还在编译 pyo3，说明存在问题

# 检查是否有 rust-analyzer 进程
tasklist | grep -i "rust"

# 查看详细编译日志
cargo build -vv 2>&1 | grep -E "(pyo3|Running|Dirty|Fresh)"
```

### 解决方案：分离 target 目录

让 rust-analyzer 使用独立的 target 目录，避免与手动 cargo 命令冲突。

#### 配置步骤

在 VS Code / Cursor 的 `settings.json` 中添加（项目级或用户级均可）：

```json
{
  "rust-analyzer.cargo.targetDir": "target/ra"
}
```

配置后：
- rust-analyzer 的编译产物 → `target/ra/`
- 手动 cargo 命令 → `target/`
- 两者互不干扰

#### 注意事项

- 首次配置后，rust-analyzer 会重新编译一次所有依赖到 `target/ra/`
- `target/ra/` 目录可以添加到 `.gitignore`（默认已忽略 `target/`）
- 磁盘空间会增加（两套编译产物），但通常可接受

### 其他可选方案

| 方案 | 优点 | 缺点 |
| :--- | :--- | :--- |
| **分离 target 目录**（推荐） | 保留完整 IDE 功能 | 磁盘空间翻倍 |
| 禁用 checkOnSave | 零额外磁盘 | 失去保存时错误检查 |
| 使用 sccache | 加速重复编译 | 不能完全避免重编译 |

### 验证方法

配置完成后：

```bash
# 确保 rust-analyzer 重新加载配置（重启 IDE 或重载窗口）

# 修改任意 Rust 文件
echo "// test" >> src/lib.rs

# 构建，应该只编译 only_torch，不编译 pyo3
cargo build
```

---

## 相关配置文件

### .cargo/config.toml

项目级 Cargo 配置（如需要）：

```toml
# 示例：固定 Python 路径（可选）
[env]
PYO3_PYTHON = "python3"

# 示例：使用 sccache（可选）
[build]
rustc-wrapper = "sccache"
```

---

## 参考资料

- [rust-analyzer Manual: Configuration](https://rust-analyzer.github.io/manual.html#configuration)
- [PyO3 Build Configuration](https://pyo3.rs/main/building-and-distribution)
