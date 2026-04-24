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
//! 2. **rebuild 阶段**：`Graph::from_descriptor` 全过 spatial shape 传播,
//!    且通过 `explicit_output_ids` 精确还原 ONNX `graph.output` 声明的单个 `output` 节点
//!    (历史背景:修复前会被"无后继 = 输出"的拓扑推断误判为多个输出)
//!
//! 3. **forward shape 断言**:输入 `[1, 3, 640, 640]` 全零张量,output 应为
//!    `[1, 25200, 20]`(VinXiangQi YOLOv5 单一最终检测头)
//!
//! 4. **forward 数值对照**(backlog):numeric_check.py 用 onnxruntime 生成参考 `.npy`,
//!    断言 only_torch 输出与之 element-wise 相对误差 < 1e-3。当前 only_torch 推理路径
//!    NMS 后约 76-77 个检出 vs ORT 约 30 个,提示存在数值漂移(可能在 BatchNorm/Sigmoid),
//!    待诊断后启用此测试
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

/// rebuild 应成功完成 spatial shape 传播 + 通过 `explicit_output_ids` 精确选出单个输出
///
/// 历史背景:
/// - 早期 rebuild 卡在 PAN 处 `Concat: 父节点 1 在维度 2 大小不一致`(16×16 vs 期望 20×20),
///   根因 MaxPool2d 不读 ONNX `pads` / Constant 跳过 / placeholder 取第一个父——已修复
/// - 后续发现 `from_descriptor` 用"无后继 = 输出"拓扑推断,把常量折叠 + Split 重写后留下
///   的若干无后继中间节点都误当成输出节点,导致拿到 3 个输出而不是 ONNX 声明的 1 个
///   `output [1, 25200, 20]`——已通过 `explicit_output_ids` 修复
///
/// 修复后本测试应 PASS,作为 spatial 传播 + 输出节点精确还原的双重回归门。
#[test]
#[ignore = "需要本地拉取 vinxiangqi.onnx，CI 跳过"]
fn yolov5_xiangqi_rebuild_succeeds() {
    if skip_if_no_model() {
        return;
    }
    use only_torch::nn::Graph;
    let result = load_onnx(MODEL_PATH).expect("ONNX 导入失败");
    println!(
        "import OK，descriptor 节点数 = {}",
        result.descriptor.nodes.len()
    );
    let rebuilt = Graph::from_descriptor(&result.descriptor)
        .expect("rebuild 失败：spatial shape 传播 / Constant→Parameter / Concat placeholder 任一可能回退");
    println!(
        "rebuild OK，参数量 = {}, 输入 = {}, 输出 = {}",
        rebuilt.graph.parameter_count(),
        rebuilt.inputs.len(),
        rebuilt.outputs.len()
    );
    assert!(
        rebuilt.graph.parameter_count() > 0,
        "参数量应大于 0（含 Conv 权重 + BN 参数 + Constant 数值常量）"
    );

    // explicit_output_ids 修复回归:VinXiangQi 的 ONNX `graph.output` 只声明 1 个 `output`,
    // 拓扑推断会误判出 3 个(常量 Parameter + Split 拆出的中间 Narrow 等都是无后继的)。
    assert_eq!(
        rebuilt.inputs.len(),
        1,
        "VinXiangQi 模型 image 输入有且仅 1 个,实际 {}",
        rebuilt.inputs.len()
    );
    assert_eq!(
        rebuilt.outputs.len(),
        1,
        "VinXiangQi 模型 graph.output 显式声明 1 个 `output` 节点,\
         若拿到 >1 说明 explicit_output_ids 退化,实际 {}",
        rebuilt.outputs.len()
    );
}

// Note: 显式 forward + 输出 shape 断言由 example `chess_yolo_onnx_detect` 兼任
// (跑两个 sample 后会自动对比 FEN,FEN 不匹配立即报错,比 shape 断言更强)。
// 本文件保留 import + rebuild 的纯框架层断言,作 ImportReport / explicit_output_ids
// 等回归门;forward 数值对照仍是 backlog,见下方 TODO。
//
// TODO[forward-numeric-check]: 在 numeric_check.py 用 onnxruntime 生成
// fixture_input.npy + fixture_output.npy 后,加 forward_numerical_match_with_onnxruntime:
//   1. 加载 fixture_input.npy 作为输入
//   2. only_torch graph.from_descriptor + forward
//   3. 加载 fixture_output.npy 作为参考(onnxruntime 跑出的 ground truth)
//   4. element-wise 相对误差 < 1e-3
//
// 当前 only_torch 推理路径在 VinXiangQi 上 NMS 后约 76-77 个检出 vs ORT 约 30 个,
// 提示存在数值漂移(可能在 BatchNorm/Sigmoid),numeric_check 可帮助定位漂移源。
