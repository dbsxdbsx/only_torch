#!/usr/bin/env python3
"""
yolov5_xiangqi fixture: 用 onnxruntime 跑一个固定输入，存输出 .npy 供 Rust 端 forward 数值对照

按 .doc/design/onnx_import_strategy.md §8.1 标准做法：
- 固定 seed 生成 [1, 3, 640, 640] 输入张量（避免依赖外部图像）
- onnxruntime 跑 forward，存 input.npy 和 output.npy 到本目录
- Rust 端集成测试加载这两个 .npy，跑 only_torch forward 后做 element-wise 对比

# 当前状态
only_torch 的 from_descriptor 在 YOLOv5 PAN/FPN 处出现 shape mismatch
(known framework limitation)，因此 Rust 端 forward 数值对照测试目前是 ignored。
本脚本仍然实现完整的"参考输出生成"功能，等下游 plan 修复 framework limitation
后即可启用 Rust forward 数值对照。

用法：
    uv run --with onnxruntime --with numpy --with onnx \\
        python tests/onnx_models/yolov5_xiangqi/numeric_check.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "vinxiangqi.onnx"
FIXTURE_DIR = Path(__file__).resolve().parent
INPUT_NPY = FIXTURE_DIR / "fixture_input.npy"
OUTPUT_NPY = FIXTURE_DIR / "fixture_output.npy"

INPUT_SHAPE = (1, 3, 640, 640)
SEED = 42


def main() -> None:
    if not MODEL_PATH.exists():
        print(
            f"[ERROR] ONNX 模型不存在: {MODEL_PATH}\n"
            f"  请先运行: python tests/onnx_models/yolov5_xiangqi/export.py",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "[ERROR] 缺少 onnxruntime；请用 'uv run --with onnxruntime' 跑本脚本",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 70)
    print("yolov5_xiangqi fixture：生成 forward 参考输出")
    print("=" * 70)
    print(f"  model:   {MODEL_PATH}")
    print(f"  input:   shape={INPUT_SHAPE} (固定 seed={SEED})")

    rng = np.random.default_rng(SEED)
    x = rng.random(INPUT_SHAPE, dtype=np.float32)
    np.save(INPUT_NPY, x)
    print(f"  saved input → {INPUT_NPY}")

    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"  ort input:  {input_name} {sess.get_inputs()[0].shape}")
    print(f"  ort output: {output_name} {sess.get_outputs()[0].shape}")

    out = sess.run([output_name], {input_name: x})[0]
    np.save(OUTPUT_NPY, out)
    print(f"  saved output → {OUTPUT_NPY} shape={out.shape} dtype={out.dtype}")

    print()
    print("  完成。Rust 端 forward 数值对照启用步骤：")
    print(
        "    1. 等 only_torch 的 from_descriptor 修复 PAN/FPN shape 传播 bug"
    )
    print(
        "    2. 移除 tests/yolov5_xiangqi_import.rs 中 forward 测试的 #[ignore]"
    )
    print(
        "    3. cargo test --test yolov5_xiangqi_import -- --ignored --nocapture"
    )


if __name__ == "__main__":
    main()
