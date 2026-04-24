# yolov5_xiangqi 回归 fixture

VinXiangQi YOLOv5 中国象棋检测模型的 only_torch 端导入回归测试。

按 [`.doc/design/onnx_import_strategy.md`](../../../.doc/design/onnx_import_strategy.md) §8.1 的目录约定布局：

```
tests/onnx_models/yolov5_xiangqi/
├── README.md          # 本文档
├── .gitignore         # 忽略 .onnx + *.npy 等本地产物
├── export.py          # 重定向到 example 的 download_model.py（避免重复实现）
└── numeric_check.py   # 用 onnxruntime 跑同一输入存参考输出，供 Rust 端 forward 对照
```

对应 Rust 端集成测试在 [`tests/yolov5_xiangqi_import.rs`](../../yolov5_xiangqi_import.rs)，
默认 `#[ignore]`，本地按需 `cargo test --test yolov5_xiangqi_import -- --ignored` 跑。

## 当前覆盖范围

| 阶段 | 验证内容 | 状态 |
|------|---------|------|
| import | descriptor 节点数 + 4 种 rewrite 模式（Conv+bias / Constant 折叠×2 / Split 重写）出现次数 | ✅ |
| rebuild | `Graph::from_descriptor` 不报 shape mismatch | ⏭️ 已知 limitation，当前 ignored |
| forward | 与 onnxruntime 数值对照（前 100 元素，相对误差 < 1e-3） | ⏭️ 待 rebuild 通过后启用 |

## 准备步骤

```bash
# 1. 拉取模型（与 example 共用同一个脚本和落地路径）
uv run --with onnx python examples/traditional/chess_yolo_onnx_detect/download_model.py

# 2. （可选）生成 onnxruntime 参考输出供 Rust 数值对照
uv run --with onnxruntime --with numpy python tests/onnx_models/yolov5_xiangqi/numeric_check.py

# 3. 跑 Rust 集成测试
cargo test --test yolov5_xiangqi_import -- --ignored --nocapture
```

## 与 example 的关系

| 文件 | 用途 | 范围 |
|------|------|------|
| `examples/traditional/chess_yolo_onnx_detect/download_model.py` | 拉取 + 算子审计 | 唯一权威下载脚本 |
| 本目录 `export.py` | 转发到 example 脚本 | 满足 §8.1 目录约定 |
| `examples/traditional/chess_yolo_onnx_detect/main.rs` | 端到端 demo（截图 → FEN） | 演示用途 |
| `tests/yolov5_xiangqi_import.rs` | 自动化回归（CI 跳过、本地按需） | 防止 import 路径退化 |
