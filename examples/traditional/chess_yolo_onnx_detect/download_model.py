#!/usr/bin/env python3
"""
VinXiangQi 预训练模型获取脚本（Chinese Chess YOLO Example 配套）

VinXiangQi (https://github.com/Vincentzyx/VinXiangQi) 是一个开源的中国象棋
桌面识别工具，基于 YOLOv5 训练，能识别 14 类棋子（红/黑各 7 种）。

本脚本做四件事：
1. 从 GitHub Release 下载 VinXiangQi.v1.4.0.zip（约 93 MB）
2. 解压 Models/*.onnx 三个模型（小/中/万能带旋转）
3. 把"小模型.onnx"重命名为 vinxiangqi.onnx 拷到 models/ 目录
   （小模型 7.2 MB，推理最快；如要换中模型/带旋转模型，改 SELECTED_MODEL）
4. 用 onnx 库列算子清单 + 与 only_torch ONNX import 支持范围对比

用法（项目根目录运行）：
    uv run --with onnx python examples/traditional/chess_yolo_onnx_detect/download_model.py

下载与解压的中间产物放在跨平台 cache 目录(默认 ~/.cache/only_torch_yolo_cache/，
可用环境变量 XIANGQI_CACHE_DIR 覆盖)，最终模型文件放在 only_torch/models/vinxiangqi.onnx
（已被 .gitignore 排除）。

注意：VinXiangQi 是 C# + onnxruntime 应用，release zip 里**直接含 ONNX**，
所以本脚本不需要 ultralytics / torch 这些重依赖。
"""

import os
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

# Windows 控制台默认 GBK，强制 UTF-8 输出避免中文乱码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==================== 常量配置 ====================

VINXIANGQI_VERSION = "v1.4.0"
DOWNLOAD_URL = (
    f"https://github.com/Vincentzyx/VinXiangQi/releases/download/"
    f"{VINXIANGQI_VERSION}/VinXiangQi.{VINXIANGQI_VERSION}.zip"
)

# 中间下载/解压区:跨平台 cache 目录(第三方资料不放项目内)
# - 默认 ~/.cache/only_torch_yolo_cache/(Windows 落到 %USERPROFILE%/.cache/...)
# - 可用环境变量 XIANGQI_CACHE_DIR 覆盖,例如 D:/某个固定目录/(适合多项目共享下载)
CACHE_DIR = Path(
    os.environ.get(
        "XIANGQI_CACHE_DIR",
        str(Path.home() / ".cache" / "only_torch_yolo_cache"),
    )
)
ZIP_PATH = CACHE_DIR / f"VinXiangQi.{VINXIANGQI_VERSION}.zip"
EXTRACT_DIR = CACHE_DIR / f"VinXiangQi.{VINXIANGQI_VERSION}_extracted"

# zip 里三个 ONNX 模型（中文文件名）→ 选哪个
# - 小模型.onnx     7.2 MB  推理最快，足够桌面识别用（首选）
# - 中模型.onnx     27 MB   精度更高
# - 万能带旋转.onnx 27 MB   带旋转鲁棒性
MODELS_IN_ZIP = {
    "small": "小模型.onnx",
    "medium": "中模型.onnx",
    "rotation": "万能带旋转.onnx",
}
SELECTED_MODEL = "small"  # 默认用小模型；想换中模型/带旋转改这里

# 项目内最终落地路径（已被 .gitignore 排除）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_PATH = MODELS_DIR / "vinxiangqi.onnx"

# only_torch 当前 ONNX import 已支持的算子（基于 src/nn/graph/onnx_ops.rs 实际代码）
# 用于审计 VinXiangQi 模型的算子缺口
ONLY_TORCH_SUPPORTED_OPS = {
    # 激活
    "Relu", "Sigmoid", "Tanh", "Softmax", "LogSoftmax", "Gelu", "Selu",
    "Mish", "HardSwish", "HardSigmoid", "Softplus", "Elu", "LeakyRelu",
    # 算术
    "Add", "Sub", "Mul", "Div", "Neg", "MatMul",
    # 数学
    "Abs", "Exp", "Sqrt", "Log", "Sign", "Reciprocal", "Pow",
    # 形状
    "Reshape", "Flatten", "Concat",
    # 卷积/池化
    "Conv", "ConvTranspose", "MaxPool", "AveragePool", "Gemm",
    # 归一化
    "BatchNormalization",
    # 上采样（本 plan 已新增 Upsample2d 节点 + 双向桥接，装配层占位）
    "Resize", "Upsample",
    # 杂项
    "Clip", "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin",
    "Identity", "Max", "Min",
}


# ==================== 工具函数 ====================

def header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"== {title}")
    print("=" * 70)


def step(idx: int, title: str) -> None:
    print(f"\n[{idx}/4] {title}")


# ==================== 主流程 ====================

def step1_download() -> None:
    step(1, f"下载 VinXiangQi {VINXIANGQI_VERSION} release zip")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        size_mb = ZIP_PATH.stat().st_size / 1024 / 1024
        print(f"  已存在: {ZIP_PATH} ({size_mb:.1f} MB)，跳过下载")
        return
    print(f"  目标: {ZIP_PATH}")
    print(f"  来源: {DOWNLOAD_URL}")
    print(f"  下载中...（约 93 MB）")
    urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_PATH)
    size_mb = ZIP_PATH.stat().st_size / 1024 / 1024
    print(f"  完成: {size_mb:.1f} MB")


def step2_extract() -> None:
    step(2, "解压 Models/*.onnx 到中间目录")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    expected = [EXTRACT_DIR / name for name in MODELS_IN_ZIP.values()]
    if all(p.exists() for p in expected):
        print(f"  3 个 onnx 已存在于 {EXTRACT_DIR}/，跳过解压")
        return
    with zipfile.ZipFile(ZIP_PATH) as z:
        for member in z.namelist():
            if member.startswith("Models/") and member.endswith(".onnx"):
                # 去掉 Models/ 前缀，flatten 到 EXTRACT_DIR
                target_name = Path(member).name
                target = EXTRACT_DIR / target_name
                with z.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  解压: {member} -> {target}")


def step3_copy_selected() -> None:
    step(3, f"拷贝 {MODELS_IN_ZIP[SELECTED_MODEL]} -> {OUTPUT_PATH}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    src = EXTRACT_DIR / MODELS_IN_ZIP[SELECTED_MODEL]
    if not src.exists():
        print(f"  [ERROR] 源文件不存在: {src}", file=sys.stderr)
        sys.exit(1)
    shutil.copy2(src, OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"  完成: {OUTPUT_PATH} ({size_mb:.1f} MB)")


def step4_audit() -> None:
    step(4, "审计算子清单 + 对比 only_torch 支持范围")
    try:
        import onnx
    except ImportError:
        print("  [ERROR] 缺少 onnx 库；请用 'uv run --with onnx' 运行本脚本", file=sys.stderr)
        sys.exit(1)

    m = onnx.load(str(OUTPUT_PATH))
    g = m.graph

    print(f"  IR version: {m.ir_version}")
    print(f"  opset: {[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}")

    print(f"  inputs:")
    for inp in g.input:
        shape = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: shape={shape}")
    print(f"  outputs:")
    for out in g.output:
        shape = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: shape={shape}")

    op_counter = Counter(node.op_type for node in g.node)
    print(f"  total nodes: {sum(op_counter.values())}")
    print()
    print("  算子审计 ([OK]=已支持, [MISSING]=未支持，需补 import 分支):")
    print(f"  {'op_type':<25s} {'count':>8s}  status")
    print(f"  {'-' * 25} {'-' * 8}  {'-' * 30}")

    missing_ops = []
    for op, cnt in op_counter.most_common():
        is_supported = op in ONLY_TORCH_SUPPORTED_OPS
        status = "[OK]      已支持" if is_supported else "[MISSING] 未支持"
        print(f"  {op:<25s} {cnt:>8d}  {status}")
        if not is_supported:
            missing_ops.append((op, cnt))

    print()
    if missing_ops:
        print(f"  [缺口汇总] {len(missing_ops)} 种算子需要补 ONNX import 分支:")
        for op, cnt in missing_ops:
            print(f"    - {op} (出现 {cnt} 次)")
        print()
        print("  -> 按 plan 第 7 节 R1 决策树处理: 补 onnx_op_to_descriptors 分支")
    else:
        print("  [OK] 所有算子均已支持，可直接进入 import 阶段")


def main() -> None:
    header("VinXiangQi 预训练模型获取（Chinese Chess YOLO Example）")
    print(f"  Release: {VINXIANGQI_VERSION}")
    print(f"  选择模型: {SELECTED_MODEL} = {MODELS_IN_ZIP[SELECTED_MODEL]}")
    print(f"  最终输出: {OUTPUT_PATH}")

    step1_download()
    step2_extract()
    step3_copy_selected()
    step4_audit()

    header("完成")
    print(f"  模型已就位: {OUTPUT_PATH}")
    print(f"  下一步: 在 only_torch 端按 plan §3 补缺失算子 → 编写 example 主程序")


if __name__ == "__main__":
    main()
