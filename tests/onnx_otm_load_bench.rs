//! 决策性 benchmark：对比 `Graph::from_onnx` vs `Graph::load_model` 的冷启动、
//! 文件大小、参数量保真度、多次推理耗时。
//!
//! ## 定位（不属于回归 baseline 体系）
//!
//! - 与 `benches/*.rs`（Criterion，算子级回归）和 `bench-macro`（hyperfine，example 级
//!   wall-clock）都不同：本文件只在评估"某个新模型该走 ONNX 直载还是预处理为 OTM"时跑一次。
//! - 关心的指标含**非 wall-clock 维度**（文件大小、参数量丢失），Criterion / hyperfine
//!   都不擅长。
//! - 因此走 `#[ignore]` test + `just bench-onnx-vs-otm` 入口，不进 `bench-save/compare`
//!   回归节奏，不污染 PR 前快速 bench 时长预算。
//!
//! ## 跑法
//!
//! ```bash
//! just bench-onnx-vs-otm     # 推荐入口（release 模式）
//! ```
//!
//! 或直接：
//! ```bash
//! cargo test --release --test onnx_otm_load_bench -- --ignored --nocapture
//! ```
//!
//! ## 当前已沉淀结论（VinXiangQi YOLOv5，参考）
//!
//! 冷启动差距 < 5ms，OTM 文件甚至大 1.9%，OTM round-trip 还会丢 2 个参数（疑似 bug）。
//! 因此现行项目（如 chinese_chess/meng_ru_ling_shi）已统一走 `.onnx` 直载，不再维护
//! "预处理为 OTM" 的中间产物链路。换其他模型时可以再跑本 bench 复核结论。

use only_torch::nn::Graph;
use only_torch::tensor::Tensor;
use std::time::Instant;

const ONNX_PATH: &str = "models/vinxiangqi.onnx";
const OTM_PATH: &str = "models/vinxiangqi_bench.otm";
const TARGET: usize = 640;
const WARMUP: usize = 1;
const REPEAT: usize = 5;

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn dummy_input() -> Tensor {
    Tensor::new(&vec![0.5f32; 3 * TARGET * TARGET], &[1, 3, TARGET, TARGET])
}

fn measure_predict(model: &only_torch::nn::RebuildResult, input: &Tensor) -> (f64, f64) {
    for _ in 0..WARMUP {
        let _ = model.predict(input).expect("warmup predict 失败");
    }
    let mut total = 0.0;
    let mut min_ms = f64::INFINITY;
    for _ in 0..REPEAT {
        let s = Instant::now();
        let _ = model.predict(input).expect("predict 失败");
        let ms = elapsed_ms(s);
        total += ms;
        if ms < min_ms {
            min_ms = ms;
        }
    }
    (total / REPEAT as f64, min_ms)
}

#[test]
#[ignore]
fn bench_onnx_vs_otm_load_and_predict() {
    if !std::path::Path::new(ONNX_PATH).exists() {
        eprintln!(
            "[skip] {ONNX_PATH} 不存在，请先跑 download_model.py 拉取 VinXiangQi 模型再 bench。"
        );
        return;
    }

    let input = dummy_input();

    println!("==== 冷启动对比（VinXiangQi YOLOv5 ONNX 7.5MB）====\n");

    let s = Instant::now();
    let model_onnx = Graph::from_onnx(ONNX_PATH).expect("from_onnx 失败");
    let onnx_load_ms = elapsed_ms(s);
    println!("Graph::from_onnx           : {onnx_load_ms:>8.1} ms");
    println!(
        "  inputs={}, outputs={}, params={}",
        model_onnx.inputs.len(),
        model_onnx.outputs.len(),
        model_onnx.graph.parameter_count()
    );

    let outputs: Vec<&only_torch::nn::Var> = model_onnx.outputs.iter().collect();
    let s = Instant::now();
    model_onnx
        .graph
        .save_model(OTM_PATH, &outputs)
        .expect("save_model 失败");
    let save_ms = elapsed_ms(s);
    let otm_path = std::path::Path::new(OTM_PATH).with_extension("otm");
    let otm_size = std::fs::metadata(&otm_path).expect("OTM 不存在").len();
    let onnx_size = std::fs::metadata(ONNX_PATH).expect("ONNX 不存在").len();
    println!(
        "Graph::save_model -> .otm   : {save_ms:>8.1} ms （文件 {} bytes）",
        otm_size
    );
    println!(
        "  对比 ONNX {} bytes ({:+.1}% size)",
        onnx_size,
        (otm_size as f64 - onnx_size as f64) / onnx_size as f64 * 100.0
    );

    let s = Instant::now();
    let model_otm = Graph::load_model(&otm_path).expect("load_model 失败");
    let otm_load_ms = elapsed_ms(s);
    println!("Graph::load_model (.otm)   : {otm_load_ms:>8.1} ms");
    println!(
        "  inputs={}, outputs={}, params={}",
        model_otm.inputs.len(),
        model_otm.outputs.len(),
        model_otm.graph.parameter_count()
    );

    println!(
        "\n==== 推理速度对比（{} 次 warmup + {} 次取均值/最小） ====\n",
        WARMUP, REPEAT
    );
    let (avg_onnx, min_onnx) = measure_predict(&model_onnx, &input);
    let (avg_otm, min_otm) = measure_predict(&model_otm, &input);
    println!("ONNX path predict avg/min  : {avg_onnx:>8.1} ms / {min_onnx:>8.1} ms");
    println!("OTM  path predict avg/min  : {avg_otm:>8.1} ms / {min_otm:>8.1} ms");
    let diff_pct = (avg_otm - avg_onnx).abs() / avg_onnx * 100.0;
    println!(
        "两条路径推理差异: {diff_pct:.1}%（应当接近 0%，因为内存中都是同一套 only_torch graph）"
    );

    println!("\n==== 冷启动收益概览 ====");
    let speedup = onnx_load_ms / otm_load_ms;
    println!(
        "OTM 比 ONNX 启动快 {speedup:.1}x ({}ms 省到 {}ms)",
        onnx_load_ms.round() as i64,
        otm_load_ms.round() as i64
    );

    let _ = std::fs::remove_file(&otm_path);
}
