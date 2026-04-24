//! yolov5_xiangqi fixture: ONNX 导入路径回归测试
//!
//! 对应 fixture: `tests/onnx_models/yolov5_xiangqi/`
//!
//! ## 测试覆盖
//!
//! 1. **import 阶段**（默认 `#[ignore]`，本地按需跑）：
//!    - descriptor 节点数 ≈ 423
//!    - ImportReport 包含 4 种预期 rewrite 模式（Conv+bias / Constant 折叠 ×2 / Split 重写）
//!
//! 2. **rebuild 阶段**：当前因 only_torch framework limitation 跳过，
//!    `Graph::from_descriptor` 会在 PAN/FPN spatial shape 传播报错
//!    （known issue，下游 plan 修复后启用）
//!
//! 3. **forward 数值对照**：依赖 rebuild 通过，当前一并跳过；
//!    参考输出由 `numeric_check.py` 用 onnxruntime 生成到本目录的 .npy
//!
//! ## 运行方式
//!
//! ```bash
//! # 1. 拉模型
//! uv run --with onnx python tests/onnx_models/yolov5_xiangqi/export.py
//!
//! # 2. 跑回归（带 --ignored 才会跑）
//! cargo test --test yolov5_xiangqi_import -- --ignored --nocapture
//! ```
//!
//! CI 默认不跑（无 #[ignore] 标记的测试只覆盖 fixture 元信息层）

use only_torch::nn::load_onnx;
use std::path::Path;

const MODEL_PATH: &str = "models/vinxiangqi.onnx";

/// 跳过条件：模型文件不存在时跳过（CI 不会拉模型）
fn skip_if_no_model() -> bool {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "[skip] 模型文件不存在: {MODEL_PATH}\n  \
             先运行: uv run --with onnx python tests/onnx_models/yolov5_xiangqi/export.py"
        );
        return true;
    }
    false
}

#[test]
fn fixture_directory_exists() {
    // 元信息测试：fixture 目录骨架在 git 里（README + .gitignore + 脚本）
    let dir = Path::new("tests/onnx_models/yolov5_xiangqi");
    assert!(dir.exists(), "fixture 目录应存在");
    assert!(dir.join("README.md").exists(), "README.md 必填");
    assert!(dir.join("export.py").exists(), "export.py 必填");
    assert!(dir.join("numeric_check.py").exists(), "numeric_check.py 必填");
    assert!(dir.join(".gitignore").exists(), ".gitignore 必填（防止 .onnx 入 git）");
}

#[test]
#[ignore = "需要本地拉取 vinxiangqi.onnx，CI 跳过"]
fn import_descriptor_topology() {
    if skip_if_no_model() {
        return;
    }
    let result = load_onnx(MODEL_PATH).expect("ONNX 导入失败");

    // descriptor 节点数（VinXiangQi 小模型 ~420，含 Conv+bias 拆分增加节点）
    let node_count = result.descriptor.nodes.len();
    println!("descriptor 节点数: {node_count}");
    assert!(
        node_count > 300 && node_count < 600,
        "descriptor 节点数 {node_count} 超出预期范围 (300, 600)"
    );

    // 输入 / 输出节点
    use only_torch::nn::NodeTypeDescriptor;
    let inputs: Vec<_> = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::BasicInput))
        .collect();
    assert_eq!(inputs.len(), 1, "VinXiangQi 模型有且仅有一个 image 输入");
    println!("input: {} shape={:?}", inputs[0].name, inputs[0].output_shape);
}

#[test]
#[ignore = "需要本地拉取 vinxiangqi.onnx，CI 跳过"]
fn import_report_covers_four_rewrite_patterns() {
    if skip_if_no_model() {
        return;
    }
    let result = load_onnx(MODEL_PATH).expect("ONNX 导入失败");

    let report = &result.import_report;
    println!(
        "ImportReport: {} 条 rewrite, {} 条 warning",
        report.rewritten.len(),
        report.warnings.len()
    );

    // 按 pattern 分组统计
    use std::collections::BTreeMap;
    let mut counter: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &report.rewritten {
        *counter.entry(r.pattern).or_insert(0) += 1;
    }
    for (pat, cnt) in &counter {
        println!("  - {pat}: {cnt} 次");
    }

    // 4 种预期 rewrite 模式都应出现至少一次（plan §3.4-3.6 全覆盖）
    let expected_patterns = [
        "conv_with_bias_to_conv_plus_add",
        "constant_fold_into_reshape",
        "constant_fold_into_resize",
        "split_to_narrows",
    ];
    for &pat in &expected_patterns {
        assert!(
            counter.contains_key(pat),
            "ImportReport 缺少预期 rewrite 模式 \"{pat}\"（VinXiangQi 实测必含此模式）"
        );
    }

    // 数量下限校验（按 download_model.py 实测算子审计：60 Conv + 6 Reshape + 2 Resize + 3 Split）
    assert!(
        counter["conv_with_bias_to_conv_plus_add"] >= 50,
        "Conv+bias 拆分应 ≥50 次（实测 60），实际 {}",
        counter["conv_with_bias_to_conv_plus_add"]
    );
    assert!(
        counter["constant_fold_into_resize"] == 2,
        "Resize 折叠应正好 2 次（YOLOv5 PAN 上采样路径），实际 {}",
        counter["constant_fold_into_resize"]
    );
    assert!(
        counter["split_to_narrows"] == 3,
        "Split 重写应正好 3 次（YOLOv5 头部 anchor split），实际 {}",
        counter["split_to_narrows"]
    );
}

// TODO[forward-numeric-check]: 当 only_torch 修复 YOLOv5 PAN/FPN shape 传播 bug
// 后，在此处加 forward_numerical_match_with_onnxruntime 测试：
//   1. 加载 fixture_input.npy 作为输入
//   2. only_torch graph.from_descriptor + forward
//   3. 加载 fixture_output.npy 作为参考（onnxruntime 跑出的 ground truth）
//   4. element-wise 相对误差 < 1e-3
// 数据生成由 tests/onnx_models/yolov5_xiangqi/numeric_check.py 负责。
