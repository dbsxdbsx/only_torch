#!/usr/bin/env python3
"""
yolov5_xiangqi fixture 的模型获取脚本（转发到 example 的 download_model.py）

按 .doc/design/onnx_import_strategy.md §8.1 的目录约定，每个 fixture 都需要
一个 export.py 入口。VinXiangQi 模型的实际拉取逻辑放在 example 里以便
example 也能直接用，本文件只是转发，避免代码重复。

用法：
    uv run --with onnx python tests/onnx_models/yolov5_xiangqi/export.py
"""

import runpy
from pathlib import Path

EXAMPLE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "traditional"
    / "chinese_chess_yolo"
    / "download_model.py"
)

if __name__ == "__main__":
    if not EXAMPLE_SCRIPT.exists():
        raise SystemExit(f"未找到上游脚本: {EXAMPLE_SCRIPT}")
    runpy.run_path(str(EXAMPLE_SCRIPT), run_name="__main__")
