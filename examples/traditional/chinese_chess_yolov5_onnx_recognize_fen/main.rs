//! # Chinese Chess YOLOv5 ONNX Recognize FEN Example（VinXiangQi YOLOv5 → 9×10 FEN）
//!
//! 端到端演示：only_torch 加载第三方真实 YOLOv5 ONNX → 识别中国象棋棋局 → 标准 FEN。
//! 内置两张测试 sample（红方在下 / 红方在上），实测 FEN 位级匹配人类标注答案。
//!
//! ## 用法
//!
//! ```bash
//! # 首次需先拉取 VinXiangQi 预训练模型（约 93 MB）
//! uv run --with onnx python examples/traditional/chinese_chess_yolov5_onnx_recognize_fen/download_model.py
//!
//! # 默认跑 sample 1（红方在下）
//! cargo run --example chinese_chess_yolov5_onnx_recognize_fen
//!
//! # 跑 sample 2（红方在上 → 自动旋转回标准方向）或自备截图
//! cargo run --example chinese_chess_yolov5_onnx_recognize_fen -- <路径>.png
//! ```
//!
//! ## 关键概念：视觉朝向 vs 标准 FEN
//!
//! - **标准 FEN**：逻辑棋局表示，约定红方永远在 row 9 底，跟原图视觉无关。
//! - **视觉朝向**：原图里红方在哪侧，FEN 字符串本身**无法表达**，作为独立元信息输出。
//!
//! 类别字典、ROI 检测、朝向归一化的实现细节见 `board_align.rs`；YOLOv5 输出
//! 解码 + per-class NMS 由库 `only_torch::vision::detection::adapter::yolo::v5`
//! 提供。本文件只关心"怎么用 only_torch"。

mod board_align;

use board_align::{BoardOutput, recognize};
use only_torch::nn::{Graph, GraphError, RebuildResult};
use only_torch::vision::detection::adapter::yolo::v5;
use only_torch::vision::preprocess::{image_to_nchw_normalized, letterbox};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MODEL_PATH: &str = "models/vinxiangqi.onnx";
const SAMPLES_DIR: &str = "examples/traditional/chinese_chess_yolov5_onnx_recognize_fen/samples";
const DEFAULT_IMAGE: &str =
    "examples/traditional/chinese_chess_yolov5_onnx_recognize_fen/samples/sample_red_bottom.png";

const TARGET_SIZE: u32 = 640;
const CONF_THRESHOLD: f32 = 0.25;
const IOU_THRESHOLD: f32 = 0.45;

fn main() -> Result<(), GraphError> {
    let image_path: PathBuf = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from(DEFAULT_IMAGE), PathBuf::from);
    if let Err(msg) = preflight(&image_path) {
        eprintln!("{msg}");
        return Ok(());
    }

    println!("=== Chinese Chess YOLO（VinXiangQi）===");
    println!("模型 : {MODEL_PATH}");
    println!("测试图: {}", image_path.display());

    let started = Instant::now();
    let model = Graph::from_onnx(MODEL_PATH)?;
    let raw_img = image::open(&image_path)
        .map_err(|e| GraphError::ComputationError(format!("读图失败: {e}")))?;
    let lb = letterbox(&raw_img, TARGET_SIZE);
    let raw_output = model.predict(&image_to_nchw_normalized(&lb.image, TARGET_SIZE))?;
    let detections = v5::detect(&raw_output, CONF_THRESHOLD, IOU_THRESHOLD)?;
    let board = recognize(&detections, &lb, lb.original_size);

    print_summary(&model, &board);
    print_grid(&board);
    let answer_check = compare_with_sample_answer(&image_path, &board);

    println!(
        "\n总耗时 {:.1} ms",
        started.elapsed().as_secs_f64() * 1000.0
    );
    answer_check
}

fn print_summary(model: &RebuildResult, board: &BoardOutput) {
    println!(
        "\n参数量 {} ｜ 检出 {} 个，落入网格 {} 个",
        model.graph.parameter_count(),
        board.detection_count,
        board.piece_count,
    );
    println!(
        "ROI: {:?}{}",
        board.roi,
        if board.roi_was_fallback {
            "（无棋子且无 board 类，退化为整图）"
        } else {
            ""
        }
    );
    println!(
        "视觉朝向: {}",
        if board.red_on_top {
            "红方在上 → 已旋转 180° 归一化为标准方向"
        } else {
            "红方在下（标准方向，未旋转）"
        }
    );
    println!("\nFEN: {}", board.fen);
}

fn print_grid(board: &BoardOutput) {
    println!("\nGrid（. = 空, ? = 类别越界）：");
    for line in board.grid_visualization().lines() {
        println!("  {line}");
    }
}

/// 跑 samples/ 下的图时与 `example_answer.txt` 自动对比；其它路径静默跳过。
fn compare_with_sample_answer(image_path: &Path, board: &BoardOutput) -> Result<(), GraphError> {
    let Some(expected) = lookup_expected_fen(image_path) else {
        return Ok(());
    };
    if expected == board.fen {
        println!("\n[OK] FEN 匹配 sample 答案");
        Ok(())
    } else {
        println!("\n[FAIL] FEN 不匹配 sample 答案");
        println!("  期望: {expected}");
        println!("  实际: {}", board.fen);
        Err(GraphError::ComputationError(format!(
            "sample FEN 不匹配: expected={expected}, actual={}",
            board.fen
        )))
    }
}

fn preflight(image_path: &Path) -> Result<(), String> {
    if !Path::new(MODEL_PATH).exists() {
        return Err(format!(
            "[预检失败] ONNX 模型不存在: {MODEL_PATH}\n\
             请先拉取: uv run --with onnx python {SAMPLES_DIR}/../download_model.py"
        ));
    }
    if !image_path.exists() {
        return Err(format!(
            "[预检失败] 测试图不存在: {}\n\
             用法: cargo run --example chinese_chess_yolov5_onnx_recognize_fen [<路径>.png]",
            image_path.display()
        ));
    }
    Ok(())
}

/// `example_answer.txt` 一行一个图，格式 `<basename>.png: <fen>`，`#` 开头为注释。
fn lookup_expected_fen(image_path: &Path) -> Option<String> {
    let basename = image_path.file_name()?.to_str()?;
    let parent = image_path.parent()?.to_str()?.replace('\\', "/");
    if !parent.ends_with("samples") {
        return None;
    }
    let answer_path = format!("{SAMPLES_DIR}/example_answer.txt");
    let content = std::fs::read_to_string(answer_path).ok()?;
    content.lines().find_map(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let (name, fen) = line.split_once(':')?;
        (name.trim() == basename).then(|| fen.trim().to_string())
    })
}
