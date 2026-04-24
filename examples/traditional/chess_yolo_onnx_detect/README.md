# Chinese Chess YOLO Example

> 用 only_torch 接收 VinXiangQi (YOLOv5) 预训练 ONNX,对桌面象棋截图做整盘识别 → 输出
> [视觉朝向] + [标准 FEN] 两份信息。
>
> 这是 only_torch 作为"小型推理框架"接收第三方真实模型的首个端到端 example。
>
> **状态**:内置两张 sample 截图(分别覆盖"红方在下"和"红方在上"两种视觉朝向),
> 实测 FEN 位级匹配人类标注。

## 流水线

```
 截图.png → letterbox(640×640) → only_torch ONNX forward → YOLO 解码 → NMS
   → ROI 自动锁定 → 视觉朝向自动检测 → 9×10 棋盘对齐 → 标准 FEN
```

## 关键概念:视觉朝向 vs 标准 FEN

中国象棋的 FEN 是**逻辑棋局的标准化表示**——约定永远红方在 row 9 底部、黑方在 row 0
顶部,**字符串本身无法表达原图视觉上红方在哪一侧**。

但作为视觉识别系统,我们能从图像里"看出"红方在上还是在下。本 example 把这一点拆成
两份独立输出:

| 输出 | 内容 | 跟视觉朝向 |
| --- | --- | --- |
| **视觉朝向** | `红方在棋盘上方 / 下方` | 是 |
| **标准 FEN** | `rnbakabnr/9/...` | 否(永远红方在底) |

下游系统(如把 FEN 重新画回图)可根据视觉朝向决定是否要再翻一次。

## 目录布局

```
chess_yolo_onnx_detect/
├── main.rs                  # 入口:加载 → pipeline → 打印耗时 + ImportReport + FEN 对比
├── letterbox.rs             # 等比缩放 + 灰色填充到 640×640 + NCHW 归一化
├── yolo_decode.rs           # YOLOv5 输出解码 + 纯 Rust per-class NMS
├── board_align.rs           # ROI 自动锁定 + 视觉朝向检测 + 9×10 网格对齐 + FEN 序列化
├── download_model.py        # VinXiangQi v1.4.0 release 拉取 + 算子缺口审计
├── requirements.txt         # 仅需 onnx>=1.16
├── README.md                # 本文档
└── samples/                 # 内置测试截图(开箱即跑)
    ├── sample_red_bottom.png  # 中盘残局,红方在原图下方(标准方向)
    ├── sample_red_top.png     # 初始局面,红方在原图上方(自动旋转 180°)
    └── example_answer.txt     # 人类标注的期望 FEN(行格式 `<图名>: <fen>`)
```

## 前置准备

### 1. 拉取 VinXiangQi 预训练模型

```bash
uv run --with onnx python examples/traditional/chess_yolo_onnx_detect/download_model.py
```

脚本会:
- 从 GitHub Release 下载 `VinXiangQi.v1.4.0.zip`(约 93 MB),中间产物落到本地 cache
  目录(默认 `~/.cache/only_torch_yolo_cache/`,可用 `XIANGQI_CACHE_DIR` 环境变量覆盖)
- 解压三个 `.onnx`(小/中/万能带旋转)
- 把"小模型.onnx"重命名为 `vinxiangqi.onnx` 拷到 `models/`(已被 `.gitignore` 排除)
- 用 `onnx` 库列算子清单 + 与 only_torch 的 ONNX 支持范围对比

> **Note**:VinXiangQi 是 C# + onnxruntime 应用,release zip 直接含 `.onnx`,不需要
> ultralytics / torch。

### 2. 准备截图(可选——内置 sample 已开箱即跑)

如果你想跑自己的截图,从任意中国象棋桌面软件直接截一张棋盘图,通过 CLI 参数指定路径
即可(分辨率 ≥ 480×600 体验更好)。

不论原图红方在上还是在下都能识别,本 example 已实现视觉朝向自动检测。

## 运行

```bash
# 默认:跑 sample 1(中盘残局,红方在下)
cargo run --example chess_yolo_onnx_detect

# 跑 sample 2(初始局面,红方在上 → 自动旋转回标准方向)
cargo run --example chess_yolo_onnx_detect -- \
  examples/traditional/chess_yolo_onnx_detect/samples/sample_red_top.png

# 跑用户自备截图
cargo run --example chess_yolo_onnx_detect -- <路径>.png

# release 模式(推理快 3-5 倍)
cargo run --release --example chess_yolo_onnx_detect
```

跑 `samples/` 下的图时,会自动从 `samples/example_answer.txt` 找对应答案做位级对比,
输出 `✓ 匹配` 或 `✗ 不匹配 期望=... 实际=...`。

### 实测输出(sample 1:中盘残局,红方在下)

```
[5/5] decode + NMS + 9×10 对齐 + FEN
  原始检出: 626, NMS 后: 77 个
  ROI 自动锁定: (128, 131, 890, 986)
  网格非空格数: 29 / 90

  视觉朝向（原图里红方在哪一侧）：
    红方在棋盘下方(标准方向)→ 不旋转

  标准 FEN(红方永远在 row 9 底,与视觉朝向解耦)：
    rnbakab1r/9/1c4P2/p1p6/c7P/2P1p4/P3P4/1C4C1N/9/RNBAKAB1R

  Grid 可视化:
    r n b a k a b . r
    . . . . . . . . .
    . c . . . . P . .
    p . p . . . . . .
    c . . . . . . . P
    . . P . p . . . .
    P . . . P . . . .
    . C . . . . C . N
    . . . . . . . . .
    R N B A K A B . R

  [自动对比] samples/example_answer.txt 期望值校验:
    ✓ 匹配 (FEN 位级一致)
```

### 实测输出(sample 2:初始局面,红方在上 → 自动旋转)

```
[5/5] decode + NMS + 9×10 对齐 + FEN
  原始检出: 721, NMS 后: 76 个
  ROI 自动锁定: (138, 137, 938, 1038)
  网格非空格数: 32 / 90

  视觉朝向（原图里红方在哪一侧）：
    红方在棋盘上方(黑方在下)→ 已旋转 180° 让 grid 回到标准方向

  标准 FEN(红方永远在 row 9 底,与视觉朝向解耦)：
    rnbakabnr/9/1c5c1/p3p1p1p/2p6/6P2/P1P1P3P/1C5C1/9/RNBAKABNR

  [自动对比] samples/example_answer.txt 期望值校验:
    ✓ 匹配 (FEN 位级一致)
```

注意 sample 1 / sample 2 是**不同棋局**(sample 1 是中盘残局,sample 2 是初始局面),
所以 FEN 字符串本来就不同。但即便是同一棋局,红方在上和红方在下两张图,**输出的标准
FEN 也会一致**——这是 FEN 标准的属性,跟视觉朝向解耦。

## 关键设计

### 类别字典(对齐 VinXiangQi 官方源码)

| class_id | 标签     | FEN 字符 | 说明      |
| -------- | -------- | -------- | --------- |
| 0        | b_ma     | n        | 黑馬      |
| 1        | b_xiang  | b        | 黑象      |
| 2        | b_shi    | a        | 黑士      |
| 3        | b_jiang  | k        | 黑將      |
| 4        | b_che    | r        | 黑車      |
| 5        | b_pao    | c        | 黑炮      |
| 6        | b_bing   | p        | 黑卒      |
| 7        | r_che    | R        | 红車      |
| 8        | r_ma     | N        | 红馬      |
| 9        | r_shi    | A        | 红仕      |
| 10       | r_jiang  | K        | 红帥      |
| 11       | r_xiang  | B        | 红相      |
| 12       | r_pao    | C        | 红炮      |
| 13       | r_bing   | P        | 红兵      |
| 14       | board    | -        | 整个棋盘 ROI(不进 FEN) |

来源:`VinXiangQi v1.4.0/VinXiangQi/YoloXiangQiModel.cs`。class 14 "board" 是 VinXiangQi
独有的设计,用于自动锁定棋盘外接矩形,本 example 也消费这个类作 ROI fallback。

### ROI 自动锁定

`auto_detect_board_roi()` 两条路径(按优先级):

1. **棋子检测中心包络**(优先,最鲁棒)
   - 取所有 class_id < 14 的 detection 中心点,算 (min/max cx, min/max cy) 包络矩形
   - 直接当 9×10 格点矩形(因为格点 (0,0) 和 (9,8) 的中心就在棋子中心范围里)
   - 要求至少 4 个棋子检出

2. **board 类(class 14)的 bbox 内缩 5%**(fallback)
   - 内缩是因为 VinXiangQi 训练时给整个棋盘外接矩形(含装饰边)标了 board 类,
     bbox 比格点矩形稍大
   - only_torch 推理路径下 board 类 bbox 数值比 ORT 小一些(框架内部数值漂移),
     所以只作 fallback

### 视觉朝向自动检测

`detect_red_on_top()` 看红帅(r_jiang, class 10)在棋盘 ROI 上半还是下半:

- 在上半 → 截图红方在棋盘上方,返回 true
- 在下半 → 标准方向(红方在棋盘下方),返回 false

返回 `true` 时 main.rs 会调 `rotate_grid_180` 把整盘转回标准方向再序列化为 FEN,
**FEN 字符串永远是红方在 row 9 底**。"红方在上"这个事实作为独立元信息单独输出。

## 调优指引

### 阈值

- `CONF_THRESHOLD`(默认 0.25):检测置信度下限,调低会增加召回但引入更多误检
- `IOU_THRESHOLD`(默认 0.45):NMS 的 IoU 阈值,调低会更激进地抑制重叠 box

### Fine-tune(R3 兜底)

若 VinXiangQi 在你的目标软件上精度不足(实测 < 30/32 棋子检出),按以下步骤训练:

1. 用 Roboflow / LabelImg 标注 ~30 张目标软件截图(YOLOv5 标准 txt 格式)
2. 以 VinXiangQi 权重为初始化,用 ultralytics yolov5 训 10-30 epoch
3. 导出新 `.onnx` 替换 `models/vinxiangqi.onnx`

> 本 example 不实现 fine-tune 端到端,仅在此提示路径。

## 与 only_torch 框架的对齐

本 example 的 ONNX import 路径用到了 only_torch 现有的几个能力:

| 路径               | 实现位置                                                        |
| ------------------ | --------------------------------------------------------------- |
| Transpose          | `src/nn/graph/onnx_ops.rs::OpType::Transpose`                   |
| Resize → Upsample2d | `assemble_resize_with_const_fold`(路线 B 折叠)                |
| Constant 折叠      | `assemble_reshape_with_const_fold` 等(路线 B 折叠)            |
| Split → N×Narrow   | `assemble_split_to_narrows`(路线 B 重写)                      |
| Conv+bias 拆分     | 装配层 `is_conv_with_bias` 分支                                 |
| **显式输出节点**   | `GraphDescriptor.explicit_output_ids` + `descriptor_rebuild` 优先用此 |

其中"显式输出节点"是为本 example 新加的——ONNX 模型 `graph.output` 显式声明了
唯一的 `output` 节点(shape `[1, 25200, 20]`),但常量折叠 + Split 重写后会留下若干
无后继的中间节点,如果走"无后继 = 输出"的拓扑推断会把它们都误当作输出。
现在 onnx_import 把 `graph.output` 的 id 列表填到 `descriptor.explicit_output_ids`,
`descriptor_rebuild` 优先用这个列表,精确还原作者原意。

详见 `.doc/design/onnx_import_strategy.md`。

## 在其它项目里复用

本 example 的代码分两层,可按需取用:

1. **框架层**(`only_torch::nn`):ONNX 加载 + 前向推理是 only_torch 公开 API,
   直接 `Cargo.toml` 加 `only_torch = { path = "..." }` 就能用
2. **业务层**(本目录下 4 个 `.rs`):letterbox / decode / NMS / ROI / 视觉朝向 / FEN
   全是中国象棋特有逻辑,目前在 example 里。最简单的复用方式是把
   `letterbox.rs` / `yolo_decode.rs` / `board_align.rs` 三个文件复制到目标项目改下命名空间

## 已知限制

- **NMS 后检出数偏多**:only_torch 推理路径下 NMS 后约 76-77 个检出,ORT 约 30 个。
  差异来自框架内部数值漂移(可能在 BatchNorm/Sigmoid 实现上),但不影响最高 conf 的
  piece-class 检测,因此 ROI + FEN 仍能位级匹配。**未来若要诊断**:可以加一个
  `numeric_check.py` 逐层对比 only_torch vs ORT 中间张量。
- 只支持 nearest 模式 + 整数倍 Resize(YOLOv5 默认)
- 只支持静态 split_sizes(动态形状要先用 onnxsim 预处理)
- ROI 容错:棋子检出 < 4 时退回 board 类 bbox(精度稍差,但仍能工作)
- `detect_red_on_top` 没检出红帅时默认按标准方向处理(不旋转),实测内置 sample 红帅
  都能稳定检出,fallback 暂未实现
- 不支持 fine-tune 端到端
