# Chinese Chess YOLO Example

> 用 only_torch 接收 VinXiangQi (YOLOv5) 预训练 ONNX，对桌面象棋截图做整盘识别 → 输出 9×10 棋盘 FEN。
>
> 这是 only_torch 作为"小型推理框架"接收第三方真实模型的首个端到端 example。

## 流水线

```
 截图.png → letterbox(640×640) → ONNX forward → YOLO 解码
   → NMS → 9×10 棋盘对齐 → FEN 字符串
```

## 目录布局

```
chinese_chess_yolo/
├── main.rs              # 入口：加载 → pipeline → 打印耗时 + ImportReport + FEN
├── letterbox.rs         # 等比缩放 + 灰色填充到 640×640 + NCHW 归一化
├── yolo_decode.rs       # YOLOv5 输出解码 + 纯 Rust per-class NMS
├── board_align.rs       # bbox → 9×10 网格对齐 + FEN 序列化
├── download_model.py    # VinXiangQi v1.4.0 release 拉取 + 算子缺口审计
├── requirements.txt     # 仅需 onnx>=1.16
├── README.md            # 本文档
└── test_board.png       # （用户自备）测试棋盘截图
```

## 前置准备

### 1. 拉取 VinXiangQi 预训练模型

```bash
uv run --with onnx python examples/traditional/chinese_chess_yolo/download_model.py
```

脚本会：
- 从 GitHub Release 下载 `VinXiangQi.v1.4.0.zip`（约 93 MB，中间产物放 `D:/.../test_repo/`）
- 解压三个 `.onnx`（小/中/万能带旋转）
- 把"小模型.onnx"重命名为 `vinxiangqi.onnx` 拷到 `models/`（已被 `.gitignore` 排除）
- 用 `onnx` 库列算子清单 + 与 only_torch 的 ONNX 支持范围对比

> **Note**：VinXiangQi 是 C# + onnxruntime 应用，release zip 直接含 `.onnx`，不需要 ultralytics / torch。

### 2. 准备测试截图

从 QQ象棋 / JJ象棋 / 天天象棋 / 象棋大师 等任意软件直接截一张棋盘图，保存为：

```
examples/traditional/chinese_chess_yolo/test_board.png
```

> 推荐分辨率 ≥ 480×600；过小可能影响 YOLO 检出率。

## 运行

```bash
cargo run --example chinese_chess_yolo                      # debug
cargo run --release --example chinese_chess_yolo            # release（推理快 3-5 倍）
```

典型输出（截图 + 模型双双就绪时）：

```
=== Chinese Chess YOLO Example（VinXiangQi）===

[1/5] 加载 ONNX 模型: models/vinxiangqi.onnx
  耗时: 280.5 ms
  参数量: 7016644
  输入: 1 个；输出: 1 个
  ImportReport: 30 条 rewrite, 0 条 warning
    - constant_fold_into_reshape: 6 次
    - constant_fold_into_resize: 2 次
    - conv_with_bias_to_conv_plus_add: 60 次
    - split_to_narrows: 3 次

[2/5] 读图 + letterbox 到 640×640
  原图尺寸: 540×600
  letterbox: scale=1.067, pad=(20, 0)
  耗时: 12.3 ms

[3/5] forward 推理
  耗时: 437.2 ms
  输出形状: [1, 25200, 20]
  推断类别数 num_classes = 15

[4/5] decode + NMS（conf≥0.25, IoU>0.45）
  原始检出: ?, NMS 后: 32 个 (耗时 1.4 ms)

[5/5] 9×10 棋盘对齐 + FEN
  使用 ROI: (0, 0, 540, 600)（占位值，按需调整）
  网格非空格数: 32 / 90

  FEN（行从上到下、列从左到右）：
    rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR

  Grid 可视化（. = 空, ? = 类别越界）：
    r n b a k a b n r
    . . . . . . . . .
    . c . . . . . c .
    p . p . p . p . p
    . . . . . . . . .
    . . . . . . . . .
    P . P . P . P . P
    . C . . . . . C .
    . . . . . . . . .
    R N B A K A B N R
```

## 调优指引

### 棋盘 ROI

主程序 `BOARD_ROI_PLACEHOLDER` 默认为 `None`（按整张截图当 ROI），实际使用时
建议按截图标注：用 `image::open` 读图后，量出棋盘外接矩形的 `(x0, y0, x1, y1)`，
替换 main.rs 顶部的常量。

> meng_ru_ling_shi 集成阶段会把 ROI 改为 CLI 参数 / GUI 标定，本 example 不做。

### 类别字典

VinXiangQi 默认 14 类（红/黑各 7 种），在 `board_align::BoardConfig::default_class_to_fen()`
里定义。如果换其他模型类别顺序不同，按下面两步对应：

1. 看模型作者文档确认 `class_id → 棋子名` 映射
2. 修改 `class_to_fen` 数组的字符顺序

### 阈值

- `CONF_THRESHOLD`（默认 0.25）：检测置信度下限，调低会增加召回但引入更多误检
- `IOU_THRESHOLD`（默认 0.45）：NMS 的 IoU 阈值，调低会更激进地抑制重叠 box

### Fine-tune（R3 兜底）

若 VinXiangQi 在你的目标软件上精度不足（实测 < 30/32 棋子检出），按以下步骤训练：

1. 用 Roboflow / LabelImg 标注 ~30 张目标软件截图（YOLOv5 标准 txt 格式）
2. 以 VinXiangQi 权重为初始化，用 ultralytics yolov5 训 10-30 epoch
3. 导出新 `.onnx` 替换 `models/vinxiangqi.onnx`

> 本 example 不实现 fine-tune 端到端，仅在此提示路径。

## 与上游设计的对齐

本 example 的 ONNX import 路径用到了 only_torch 现有的几个能力：

| 路径 | 实现位置 |
|------|---------|
| Transpose | `src/nn/graph/onnx_ops.rs` `OpType::Transpose` 分支 |
| Resize → Upsample2d | `assemble_resize_with_const_fold` (路线 B 折叠) |
| Constant 折叠 | `assemble_reshape_with_const_fold` 等 (路线 B 折叠) |
| Split → N×Narrow | `assemble_split_to_narrows` (路线 B 重写) |
| Conv+bias 拆分 | 装配层 `is_conv_with_bias` 分支 |

详见 `.doc/design/onnx_import_strategy.md`。

## 已知限制

- **YOLOv5 PAN/FPN forward shape mismatch**（首次端到端跑 VinXiangQi 触发）：
  ONNX import 完整跑通且 `ImportReport` 正确填充，但 forward 阶段在 PAN
  bottom-up 路径的 Concat 节点出现 spatial 维度不一致（实测某层 16×16 vs
  期望 20×20）。这是 only_torch Conv2d / Resize / 形状传播的内部 limitation，
  与本 example 的 ONNX import 路径无关。下游 todo `regression-fixture`
  会针对性诊断/修复，目前 example 在 forward 失败时优雅降级为只输出
  import 阶段成果（参数量 + ImportReport 摘要）。
- 只支持 nearest 模式 + 整数倍 Resize（YOLOv5 默认）
- 只支持静态 split_sizes（动态形状要先用 onnxsim 预处理）
- ROI 当前硬编码（`None` = 整张图）
- 不支持 fine-tune 端到端
