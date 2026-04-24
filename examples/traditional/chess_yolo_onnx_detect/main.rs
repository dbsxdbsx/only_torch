//! # Chinese Chess YOLO Example（VinXiangQi YOLOv5 模型 → 9×10 FEN）
//!
//! 端到端演示 only_torch 接收第三方真实 YOLOv5 模型(VinXiangQi)做象棋识别。
//! 内置两张测试 sample(分别是"红方在下"和"红方在上"两种视觉朝向),实测 FEN 位级
//! 匹配人类标注答案。
//!
//! ## 流水线
//!
//! ```
//!  截图(.png) → letterbox(640×640) → ONNX forward(only_torch) → YOLO 解码
//!     → NMS → ROI 自动锁定 → 视觉朝向自动检测 → 9×10 棋盘对齐 → 标准 FEN
//! ```
//!
//! ## 前置准备
//!
//! ```bash
//! # 拉取 VinXiangQi 预训练模型(首次运行需要约 93 MB 下载)
//! uv run --with onnx python examples/traditional/chess_yolo_onnx_detect/download_model.py
//! ```
//!
//! ## 运行
//!
//! ```bash
//! # 默认跑 sample 1(中盘残局,红方在下)
//! cargo run --example chess_yolo_onnx_detect
//!
//! # 跑 sample 2(初始局面,红方在上 → 自动旋转回标准方向)
//! cargo run --example chess_yolo_onnx_detect -- \
//!   examples/traditional/chess_yolo_onnx_detect/samples/sample_red_top.png
//!
//! # 跑用户自备截图
//! cargo run --example chess_yolo_onnx_detect -- <路径>.png
//! ```
//!
//! 跑 samples/ 下的图时会自动从 `samples/example_answer.txt` 找对应答案做位级对比。
//!
//! ## 关键概念:视觉朝向 vs 标准 FEN
//!
//! - **标准 FEN**:逻辑棋局表示,约定红方永远在 row 9 底部,跟原图视觉无关
//! - **视觉朝向**:原图里红方在上 / 在下,FEN 字符串本身**无法表达**这个信息,
//!   作为单独的元信息输出
//! - 实现:`detect_red_on_top` 看红帅(r_jiang)在棋盘上半还是下半;在上时
//!   `rotate_grid_180` 把整盘转回标准方向再序列化为 FEN

mod board_align;
mod letterbox;
mod yolo_decode;

use board_align::{
    auto_detect_board_roi, detect_red_on_top, rotate_grid_180, BoardConfig, BOARD_COLS, BOARD_ROWS,
    NUM_CLASSES,
};
use letterbox::{image_to_nchw_normalized, letterbox};
use only_torch::nn::{load_onnx, Graph, GraphError, ImportReport};
use only_torch::tensor::Tensor;
use std::path::Path;
use std::time::Instant;
use yolo_decode::{decode, nms};

const MODEL_PATH: &str = "models/vinxiangqi.onnx";
const SAMPLES_DIR: &str = "examples/traditional/chess_yolo_onnx_detect/samples";
const DEFAULT_TEST_IMAGE_PATH: &str =
    "examples/traditional/chess_yolo_onnx_detect/samples/sample_red_bottom.png";

const TARGET_SIZE: u32 = 640;
const CONF_THRESHOLD: f32 = 0.25;
const IOU_THRESHOLD: f32 = 0.45;

fn main() -> Result<(), GraphError> {
    println!("=== Chinese Chess YOLO Example（VinXiangQi）===\n");
    let total_start = Instant::now();

    // 命令行参数:第 1 个参数是测试图路径(可选,默认 test_board.png)
    let args: Vec<String> = std::env::args().collect();
    let test_image_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| DEFAULT_TEST_IMAGE_PATH.to_string());

    // ────────────────────────────────────────────
    // 1. 前置文件校验
    // ────────────────────────────────────────────
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("[预检失败] ONNX 模型不存在: {MODEL_PATH}");
        eprintln!();
        eprintln!("  请先拉取 VinXiangQi 预训练模型：");
        eprintln!(
            "    uv run --with onnx python \
            examples/traditional/chess_yolo_onnx_detect/download_model.py"
        );
        return Ok(()); // 优雅退出（不算 example 失败）
    }
    if !Path::new(&test_image_path).exists() {
        eprintln!("[预检失败] 测试棋盘截图不存在: {test_image_path}");
        eprintln!();
        eprintln!("  用法:");
        eprintln!(
            "    cargo run --example chess_yolo_onnx_detect                  \
             # 默认跑 sample 1 (红方在下)"
        );
        eprintln!(
            "    cargo run --example chess_yolo_onnx_detect -- {SAMPLES_DIR}/sample_red_top.png"
        );
        eprintln!(
            "    cargo run --example chess_yolo_onnx_detect -- <路径>.png    \
             # 用户自备截图"
        );
        return Ok(());
    }
    println!("使用测试图: {test_image_path}\n");

    // ────────────────────────────────────────────
    // 2. 加载 ONNX 模型（分两步，便于在 rebuild 失败时仍能展示 ImportReport）
    // ────────────────────────────────────────────
    println!("[1/5] 加载 ONNX 模型: {MODEL_PATH}");
    let load_start = Instant::now();

    // Step 2a: 仅 import（解析 + 折叠/重写）
    let import_result = load_onnx(MODEL_PATH).map_err(|e| {
        GraphError::ComputationError(format!("ONNX import 失败: {e}"))
    })?;
    let import_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    println!("  import 耗时: {import_ms:.1} ms");
    println!("  descriptor 节点数: {}", import_result.descriptor.nodes.len());
    print_import_report(&import_result.import_report);

    // Step 2b: rebuild（构图 + 形状传播校验）—— 失败时优雅降级
    println!("\n[2/5] 重建计算图（from_descriptor）");
    let rebuild_start = Instant::now();
    let result = match Graph::from_descriptor(&import_result.descriptor) {
        Ok(mut r) => {
            // 注入权重（仿 from_onnx_result 内部行为）
            for n in &import_result.descriptor.nodes {
                if let Some(t) = import_result.weights.get(&n.id) {
                    if let Some(p) = r.graph.inner().get_parameter(&n.name) {
                        // 直接 set value 到 Parameter 节点
                        let _ = p.set_value(Some(t));
                    }
                }
            }
            r.graph.eval();
            r.import_report = Some(import_result.import_report.clone());
            r
        }
        Err(e) => {
            let rebuild_ms = rebuild_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!();
            eprintln!("  [意外失败] rebuild 失败 ({rebuild_ms:.1} ms): {e:?}");
            eprintln!("  ONNX import 阶段已跑通,但 from_descriptor 重建出错。");
            eprintln!("  本路径在 VinXiangQi 模型上已验证可工作,若失败请保留 ImportReport 上报。");
            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            println!("\n=== 完成(rebuild 失败,跳过 forward + FEN 阶段)===");
            println!("  总耗时: {total_ms:.1} ms");
            return Ok(());
        }
    };
    let rebuild_ms = rebuild_start.elapsed().as_secs_f64() * 1000.0;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    println!("  rebuild 耗时: {rebuild_ms:.1} ms");
    println!("  total load 耗时: {load_ms:.1} ms");
    println!("  参数量: {}", result.graph.parameter_count());
    println!(
        "  输入: {} 个；输出: {} 个",
        result.inputs.len(),
        result.outputs.len()
    );

    // ────────────────────────────────────────────
    // 3. 读测试图 + letterbox 预处理
    // ────────────────────────────────────────────
    println!("\n[3/5] 读图 + letterbox 到 {TARGET_SIZE}×{TARGET_SIZE}");
    let pre_start = Instant::now();
    let raw_img = image::open(&test_image_path).map_err(|e| {
        GraphError::ComputationError(format!("读图失败 {test_image_path}: {e}"))
    })?;
    println!(
        "  原图尺寸: {}×{}",
        image::GenericImageView::dimensions(&raw_img).0,
        image::GenericImageView::dimensions(&raw_img).1
    );
    let lb = letterbox(&raw_img, TARGET_SIZE);
    let nchw_data = image_to_nchw_normalized(&lb.image, TARGET_SIZE);
    let pre_ms = pre_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  letterbox: scale={:.4}, pad=({},{})",
        lb.scale, lb.pad.0, lb.pad.1
    );
    println!("  耗时: {pre_ms:.1} ms");

    // ────────────────────────────────────────────
    // 4. forward
    // ────────────────────────────────────────────
    println!("\n[4/5] forward 推理");
    let input_var = &result.inputs[0].1;
    let output_var = &result.outputs[0];

    let input_tensor = Tensor::new(
        &nchw_data,
        &[1, 3, TARGET_SIZE as usize, TARGET_SIZE as usize],
    );
    input_var.set_value(&input_tensor)?;

    let infer_start = Instant::now();
    let forward_res = result.graph.forward(output_var);
    let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    println!("  耗时: {infer_ms:.1} ms");

    if let Err(e) = forward_res {
        eprintln!();
        eprintln!("  [意外失败] forward 报错: {e:?}");
        eprintln!("  本路径在 VinXiangQi 模型 + 内置两张 sample 截图上已验证可工作,");
        eprintln!("  若你的截图触发新失败请保留 ImportReport 上报。");
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        println!("\n=== 完成(forward 失败,跳过 FEN 阶段)===");
        println!("  总耗时: {total_ms:.1} ms(其中 forward 失败前 {infer_ms:.1} ms)");
        return Ok(());
    }

    let out_tensor_opt = output_var
        .value()
        .map_err(|e| GraphError::ComputationError(format!("拿输出 value 失败: {e:?}")))?;
    let out_tensor = out_tensor_opt
        .ok_or_else(|| GraphError::ComputationError("输出 value 为 None".to_string()))?;
    let out_shape = out_tensor.shape().to_vec();
    println!("  输出形状: {:?}", out_shape);

    // YOLOv5 输出布局：[1, num_anchors, 5+nc]，nc 由 shape 反推
    let last_dim = *out_shape
        .last()
        .ok_or_else(|| GraphError::ComputationError("输出张量无维度".to_string()))?;
    if last_dim < 5 {
        return Err(GraphError::ComputationError(format!(
            "YOLO 输出最后一维 {last_dim} < 5，无法解析"
        )));
    }
    let num_classes = last_dim - 5;
    println!("  推断类别数 num_classes = {num_classes}");

    let out_data = out_tensor.flatten_view().to_vec();

    // ────────────────────────────────────────────
    // 5. 解码 + NMS + 9×10 对齐
    // ────────────────────────────────────────────
    println!("\n[5/5] decode + NMS + 9×10 对齐 + FEN（conf≥{CONF_THRESHOLD}, IoU>{IOU_THRESHOLD}）");
    let post_start = Instant::now();
    let raw_dets = decode(&out_data, num_classes, CONF_THRESHOLD);
    let raw_count = raw_dets.len();
    let nms_dets = nms(raw_dets, IOU_THRESHOLD);
    let post_ms = post_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  原始检出: {raw_count}, NMS 后: {} 个 (耗时 {post_ms:.1} ms)",
        nms_dets.len()
    );

    // ── ROI 自动检测 ──
    let (orig_w, orig_h) = image::GenericImageView::dimensions(&raw_img);
    let roi = match auto_detect_board_roi(&nms_dets, &lb) {
        Some(r) => {
            println!("  ROI 自动锁定: {r:?}(优先棋子包络,fallback 到 board 类)");
            r
        }
        None => {
            let r = (0, 0, orig_w, orig_h);
            println!("  ROI 退化为整图: {r:?}(无棋子且无 board 类检出)");
            r
        }
    };

    let cfg = BoardConfig {
        roi,
        class_to_fen: BoardConfig::default_class_to_fen(),
    };
    let raw_grid = board_align::align_to_grid(&nms_dets, &lb, &cfg);

    // ── 视觉朝向自动检测(独立元信息,FEN 字符串本身无法表达) ──
    let red_on_top = detect_red_on_top(&nms_dets, &lb, roi);
    let grid = if red_on_top {
        rotate_grid_180(&raw_grid)
    } else {
        raw_grid
    };

    let piece_count = board_align::count_pieces(&grid);
    let fen = board_align::to_fen(&grid, &cfg);

    println!("  网格非空格数: {piece_count} / {}", BOARD_COLS * BOARD_ROWS);

    // ── 输出 1:视觉朝向(原图视觉信息,跟 FEN 解耦) ──
    println!("\n  视觉朝向（原图里红方在哪一侧）：");
    if red_on_top {
        println!("    红方在棋盘上方(黑方在下)→ 已旋转 180° 让 grid 回到标准方向");
    } else {
        println!("    红方在棋盘下方(标准方向)→ 不旋转");
    }

    // ── 输出 2:标准 FEN(逻辑棋局,永远红方在 row 9 底) ──
    println!("\n  标准 FEN(红方永远在 row 9 底,与视觉朝向解耦)：");
    println!("    {fen}");

    // 简单可视化 9×10 grid
    println!("\n  Grid 可视化（. = 空, ? = 类别越界）：");
    for row in grid.iter() {
        print!("    ");
        for cell in row.iter() {
            match cell {
                Some(class_id) if *class_id < NUM_CLASSES => {
                    print!("{} ", cfg.class_to_fen[*class_id]);
                }
                Some(_) => print!("? "),
                None => print!(". "),
            }
        }
        println!();
    }

    // ── 输出 3:跑 samples/ 下的图时,跟 example_answer.txt 自动对比 ──
    if let Some(expected) = lookup_expected_fen(&test_image_path) {
        println!("\n  [自动对比] samples/example_answer.txt 期望值校验:");
        if expected == fen {
            println!("    ✓ 匹配 (FEN 位级一致)");
        } else {
            println!("    ✗ 不匹配!");
            println!("    期望: {expected}");
            println!("    实际: {fen}");
        }
    }

    // ────────────────────────────────────────────
    // 总结
    // ────────────────────────────────────────────
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    println!("\n=== 完成 ===");
    println!("  总耗时: {total_ms:.1} ms");
    println!(
        "  各阶段: load={load_ms:.1} pre={pre_ms:.1} infer={infer_ms:.1} \
         post={post_ms:.1} (ms)"
    );

    Ok(())
}

/// 跑 samples/ 下的图时,从 `samples/example_answer.txt` 找对应期望 FEN
///
/// `example_answer.txt` 一行一个图,格式 `<basename>.png: <fen>`(`:` 后允许空格)。
/// 不在 samples/ 下、或文件不存在、或没找到对应图名,统一返回 None(静默跳过对比)。
fn lookup_expected_fen(image_path: &str) -> Option<String> {
    let path = Path::new(image_path);
    let basename = path.file_name().and_then(|s| s.to_str())?;
    // 仅在 samples/ 下时才查表(避免误匹配用户自备图)
    let parent = path.parent().and_then(|p| p.to_str()).unwrap_or("");
    if !parent.replace('\\', "/").ends_with("samples") {
        return None;
    }
    let answer_path = format!("{SAMPLES_DIR}/example_answer.txt");
    let content = std::fs::read_to_string(&answer_path).ok()?;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((name, fen)) = line.split_once(':') {
            if name.trim() == basename {
                return Some(fen.trim().to_string());
            }
        }
    }
    None
}

/// 打印 ImportReport 摘要
///
/// 验证 plan §3.4-3.6 的 ONNX rewrite/折叠路径是否都生效（按 pattern 分组计数）
fn print_import_report(report: &ImportReport) {
    println!(
        "  ImportReport: {} 条 rewrite, {} 条 warning",
        report.rewritten.len(),
        report.warnings.len()
    );
    let mut counter: std::collections::BTreeMap<&str, usize> =
        std::collections::BTreeMap::new();
    for r in &report.rewritten {
        *counter.entry(r.pattern).or_insert(0) += 1;
    }
    for (pat, cnt) in counter {
        println!("    - {pat}: {cnt} 次");
    }
    for w in &report.warnings {
        println!("    [warning] {w}");
    }
}
