---
status: suspended
created: 2026-07-12
updated: 2026-07-12
---

# proc-macro-error2 2.0.1 future-incompat 警告

## 背景

项目升级到 Rust 1.97.0 后，启用 `blas-mkl` 的 `cargo check` / `test` / `clippy`
都会在成功结束后报告 future-incompat warning。

## 现象 / 影响

`proc-macro-error2 2.0.1` 触发 E0365：私有 `extern crate proc_macro` 被公开
re-export。Rust 1.97 当前仍接受该代码，项目构建、3421 主测试、lint 与全部 RL smoke
均通过；未来 Rust 版本可能把它提升为硬错误。

反向依赖链：

```text
proc-macro-error2 2.0.1
└── getset 0.1.6
    └── oci-spec 0.6.7
        └── ocipkg 0.2.9 (build dependency)
            └── intel-mkl-src 0.8.1
                └── only_torch
```

仅启用 `blas-mkl` 时进入该链，项目不直接依赖 `proc-macro-error2`。

## 已尝试

- `cargo report future-incompatibilities --id 1` 确认具体 lint 与上游仓库。
- `cargo tree --features blas-mkl -i proc-macro-error2` 确认反向依赖链。
- Rust 1.97 全量门禁确认当前不构成编译或运行阻断。

## 暂缓原因

问题位于 MKL 下载/解包工具的间接 build dependency。当前没有必要为非阻断 warning
引入本地 `[patch]` 或 fork，等待上游依赖自然升级成本更低。

## 下次恢复条件

- `intel-mkl-src` / `ocipkg` / `oci-spec` / `getset` 发布可消除该警告的版本；或
- 后续 Rust beta/stable 将该 lint 提升为硬错误。

## 下一步建议

依赖升级时复跑反向依赖命令与 `just check`。若上游长期无修复且新 Rust 已阻断，
再评估最窄范围的依赖升级或临时 patch，避免直接屏蔽 future-incompat warning。
