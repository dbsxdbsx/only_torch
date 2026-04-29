//! yolov5_xiangqi fixture: ONNX 导入路径回归测试
//!
//! 对应 fixture: `tests/onnx_models/yolov5_xiangqi/`
//!
//! ## 测试覆盖
//!
//! 1. **import 阶段**（默认 `#[ignore]`，本地按需跑）：
//!    - descriptor 节点数 ≈ 423
//!    - ImportReport 包含 5 种预期 rewrite 模式
//!      （Conv+bias / Constant 折叠 ×2 / Split 重写 / Pow 常量指数折叠）
//!
//! 2. **rebuild 阶段**：`Graph::from_descriptor` 全过 spatial shape 传播,
//!    且通过 `explicit_output_ids` 精确还原 ONNX `graph.output` 声明的单个 `output` 节点
//!    (历史背景:修复前会被"无后继 = 输出"的拓扑推断误判为多个输出)
//!
//! 3. **forward 数值对照**:numeric_check.py 用 onnxruntime 生成参考 `.npy`,
//!    ignored 测试读取同一输入做 only_torch raw output 对照,输出 max abs / max rel /
//!    mean abs 统计。NMS 数量差异只作为现象记录,不预设具体算子根因。
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

use only_torch::nn::{Graph, load_onnx};
use only_torch::tensor::Tensor;
use std::path::Path;

const MODEL_PATH: &str = "models/vinxiangqi.onnx";
const FIXTURE_INPUT_PATH: &str = "tests/onnx_models/yolov5_xiangqi/fixture_input.npy";
const FIXTURE_OUTPUT_PATH: &str = "tests/onnx_models/yolov5_xiangqi/fixture_output.npy";
const ORT_INTERMEDIATE_DIR: &str = "target/yolov5_xiangqi_ort_intermediates";

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

/// 跳过条件：未生成 onnxruntime 数值参考 fixture 时跳过
fn skip_if_no_numeric_fixture() -> bool {
    let input_exists = Path::new(FIXTURE_INPUT_PATH).exists();
    let output_exists = Path::new(FIXTURE_OUTPUT_PATH).exists();
    if !input_exists || !output_exists {
        eprintln!(
            "[skip] 数值 fixture 不完整: input={input_exists}, output={output_exists}\n  \
             先运行: uv run --with onnxruntime --with numpy --with onnx python \
             tests/onnx_models/yolov5_xiangqi/numeric_check.py"
        );
        return true;
    }
    false
}

fn read_npy_tensor(path: &str) -> Tensor {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("读取 npy 失败 {path}: {e}"));
    assert!(
        bytes.starts_with(b"\x93NUMPY"),
        "不是有效的 npy 文件: {path}"
    );

    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10usize)
        }
        2 | 3 => {
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12usize)
        }
        _ => panic!("不支持的 npy 版本 {major}: {path}"),
    };
    let header_end = header_start + header_len;
    let header =
        std::str::from_utf8(&bytes[header_start..header_end]).expect("npy header 不是 UTF-8");

    assert!(
        header.contains("'descr': '<f4'")
            || header.contains("\"descr\": \"<f4\"")
            || header.contains("'descr': '|f4'")
            || header.contains("\"descr\": \"|f4\""),
        "仅支持 little-endian f32 npy fixture: {header}"
    );
    assert!(
        header.contains("'fortran_order': False") || header.contains("\"fortran_order\": false"),
        "仅支持 C-order npy fixture: {header}"
    );

    let shape_start = header
        .find('(')
        .unwrap_or_else(|| panic!("npy header 缺少 shape: {header}"));
    let shape_end = header[shape_start..]
        .find(')')
        .map(|idx| shape_start + idx)
        .unwrap_or_else(|| panic!("npy header shape 未闭合: {header}"));
    let shape: Vec<usize> = header[shape_start + 1..shape_end]
        .split(',')
        .filter_map(|part| {
            let part = part.trim();
            (!part.is_empty()).then(|| {
                part.parse::<usize>()
                    .unwrap_or_else(|e| panic!("解析 npy shape 失败 {part}: {e}"))
            })
        })
        .collect();
    assert!(!shape.is_empty(), "YOLO fixture 不应是标量 npy: {path}");

    let expected_values: usize = shape.iter().product();
    let data_bytes = &bytes[header_end..];
    assert_eq!(
        data_bytes.len(),
        expected_values * std::mem::size_of::<f32>(),
        "npy 数据长度与 shape 不匹配: {path}"
    );
    let data: Vec<f32> = data_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Tensor::new(&data, &shape)
}

fn sanitize_tensor_name(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn tensor_diff(got: &Tensor, want: &Tensor) -> Option<(f32, f32, f32, usize)> {
    if got.shape() != want.shape() {
        return None;
    }
    let got_data = got.to_vec();
    let want_data = want.to_vec();
    if got_data.len() != want_data.len() {
        return None;
    }

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mean_abs = 0.0f32;
    let mut max_abs_index = 0usize;
    for (idx, (&g, &w)) in got_data.iter().zip(want_data.iter()).enumerate() {
        let abs = (g - w).abs();
        let rel = abs / w.abs().max(1e-6);
        mean_abs += abs;
        if abs > max_abs {
            max_abs = abs;
            max_abs_index = idx;
        }
        max_rel = max_rel.max(rel);
    }
    mean_abs /= got_data.len() as f32;
    Some((max_abs, max_rel, mean_abs, max_abs_index))
}

#[test]
fn fixture_directory_exists() {
    // 元信息测试：fixture 目录骨架在 git 里（README + .gitignore + 脚本）
    let dir = Path::new("tests/onnx_models/yolov5_xiangqi");
    assert!(dir.exists(), "fixture 目录应存在");
    assert!(dir.join("README.md").exists(), "README.md 必填");
    assert!(dir.join("export.py").exists(), "export.py 必填");
    assert!(
        dir.join("numeric_check.py").exists(),
        "numeric_check.py 必填"
    );
    assert!(
        dir.join(".gitignore").exists(),
        ".gitignore 必填（防止 .onnx 入 git）"
    );
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
    println!(
        "input: {} shape={:?}",
        inputs[0].name, inputs[0].output_shape
    );

    let batch_norm_count = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::BatchNormOp { .. }))
        .count();
    assert_eq!(
        batch_norm_count, 0,
        "VinXiangQi v1.4.0 small ONNX 应为 fused 图，不应含独立 BatchNormalization"
    );
}

#[test]
#[ignore = "需要本地拉取 vinxiangqi.onnx，CI 跳过"]
fn import_report_covers_expected_rewrite_patterns() {
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

    // 5 种预期 rewrite 模式都应出现至少一次（YOLOv5 import 路径全覆盖）
    let expected_patterns = [
        "conv_with_bias_to_conv_plus_add",
        "constant_fold_into_reshape",
        "constant_fold_into_resize",
        "pow_const_exponent",
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
    assert!(
        counter["pow_const_exponent"] == 3,
        "Pow 常量指数折叠应正好 3 次（YOLOv5 三个检测头 width/height 解码），实际 {}",
        counter["pow_const_exponent"]
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
    let result = load_onnx(MODEL_PATH).expect("ONNX 导入失败");
    println!(
        "import OK，descriptor 节点数 = {}",
        result.descriptor.nodes.len()
    );
    let rebuilt = Graph::from_descriptor(&result.descriptor).expect(
        "rebuild 失败：spatial shape 传播 / Constant→Parameter / Concat placeholder 任一可能回退",
    );
    println!(
        "rebuild OK，参数量 = {}, 输入 = {}, 输出 = {}",
        rebuilt.graph.parameter_count(),
        rebuilt.inputs.len(),
        rebuilt.outputs.len()
    );
    assert!(
        rebuilt.graph.parameter_count() > 0,
        "参数量应大于 0（含 Conv 权重 + bias + Constant 数值常量）"
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

/// raw output 数值对照：同一随机输入下统计 only_torch 与 onnxruntime 的输出差异。
///
/// 该测试默认 ignored，因为需要本地模型和 `.npy` fixture。默认只断言 shape 与有限值,
/// 并打印误差统计；设置 `ONLY_TORCH_STRICT_YOLO_NUMERIC=1` 时才按阈值失败。
/// NMS 后检出数不是定位根因的可靠入口。
#[test]
#[ignore = "需要本地 vinxiangqi.onnx 与 numeric_check.py 生成的 .npy fixture"]
fn forward_numerical_match_with_onnxruntime() {
    if skip_if_no_model() || skip_if_no_numeric_fixture() {
        return;
    }

    let input = read_npy_tensor(FIXTURE_INPUT_PATH);
    let expected = read_npy_tensor(FIXTURE_OUTPUT_PATH);

    let result = Graph::from_onnx(MODEL_PATH).expect("ONNX 导入 + rebuild 失败");
    let input_var = &result.inputs[0].1;
    let output_var = &result.outputs[0];
    input_var.set_value(&input).expect("设置输入失败");
    result.graph.forward(output_var).expect("forward 失败");

    let actual = output_var
        .value()
        .expect("读取输出 value 失败")
        .expect("输出 value 为 None");

    assert_eq!(actual.shape(), expected.shape(), "raw output shape 应一致");

    let actual_data = actual.data_as_slice();
    let expected_data = expected.data_as_slice();
    assert_eq!(
        actual_data.len(),
        expected_data.len(),
        "raw output 元素数应一致"
    );

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mean_abs = 0.0f32;
    let mut max_abs_index = 0usize;

    for (idx, (&got, &want)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            got.is_finite(),
            "only_torch 输出含非有限值: idx={idx}, value={got}"
        );
        assert!(
            want.is_finite(),
            "ORT 输出含非有限值: idx={idx}, value={want}"
        );

        let abs = (got - want).abs();
        let rel = abs / want.abs().max(1e-6);
        mean_abs += abs;
        if abs > max_abs {
            max_abs = abs;
            max_abs_index = idx;
        }
        max_rel = max_rel.max(rel);
    }
    mean_abs /= actual_data.len() as f32;

    println!(
        "raw output diff: max_abs={max_abs:.6e} at idx={max_abs_index}, \
         max_rel={max_rel:.6e}, mean_abs={mean_abs:.6e}"
    );

    if std::env::var("ONLY_TORCH_STRICT_YOLO_NUMERIC").as_deref() == Ok("1") {
        assert!(
            max_abs <= 1e-3 || max_rel <= 1e-3,
            "raw output 与 ORT 差异过大: max_abs={max_abs:.6e}, \
             max_rel={max_rel:.6e}, mean_abs={mean_abs:.6e}, idx={max_abs_index}"
        );
    } else {
        println!(
            "strict numeric assertion disabled; set ONLY_TORCH_STRICT_YOLO_NUMERIC=1 to fail on drift"
        );
    }
}

/// 逐节点诊断：读取 Python/ORT 生成的中间张量 dump,定位 first drift。
///
/// 生成命令示例（只需本地调试时跑）：
/// `uv run --with onnxruntime --with onnx --with numpy python <dump script>`
#[test]
#[ignore = "诊断专用：需要 target/yolov5_xiangqi_ort_intermediates 下的 ORT 中间张量 dump"]
fn diagnose_first_intermediate_drift_against_onnxruntime() {
    if skip_if_no_model() || skip_if_no_numeric_fixture() {
        return;
    }
    if !Path::new(ORT_INTERMEDIATE_DIR).exists() {
        eprintln!("[skip] ORT 中间张量目录不存在: {ORT_INTERMEDIATE_DIR}");
        return;
    }

    let input = read_npy_tensor(FIXTURE_INPUT_PATH);
    let import = load_onnx(MODEL_PATH).expect("ONNX 导入失败");
    let result = Graph::from_onnx(MODEL_PATH).expect("ONNX 导入 + rebuild 失败");
    let input_var = &result.inputs[0].1;
    let output_var = &result.outputs[0];
    input_var.set_value(&input).expect("设置输入失败");
    result.graph.forward(output_var).expect("forward 失败");

    let mut compared = 0usize;
    let mut first_drift: Option<String> = None;
    let mut worst: Option<(String, f32, f32, f32, usize)> = None;

    for node_desc in &import.descriptor.nodes {
        let filename = format!("{}.npy", sanitize_tensor_name(&node_desc.name));
        let path = Path::new(ORT_INTERMEDIATE_DIR).join(filename);
        if !path.exists() {
            continue;
        }
        let Some(var) = result.node_map.get(&node_desc.id) else {
            continue;
        };
        let Some(got) = var.value().expect("读取 only_torch 中间节点值失败") else {
            continue;
        };
        let want = read_npy_tensor(path.to_str().expect("路径不是 UTF-8"));
        let Some((max_abs, max_rel, mean_abs, idx)) = tensor_diff(&got, &want) else {
            println!(
                "shape mismatch at {} {:?}: only_torch={:?}, ORT={:?}",
                node_desc.name,
                node_desc.node_type,
                got.shape(),
                want.shape()
            );
            continue;
        };
        compared += 1;

        if worst
            .as_ref()
            .is_none_or(|(_, prev_max_abs, _, _, _)| max_abs > *prev_max_abs)
        {
            worst = Some((node_desc.name.clone(), max_abs, max_rel, mean_abs, idx));
        }

        if first_drift.is_none() && (max_abs > 1e-3 && max_rel > 1e-3) {
            first_drift = Some(format!(
                "{} {:?}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, \
                 mean_abs={mean_abs:.6e}, idx={idx}",
                node_desc.name, node_desc.node_type
            ));
        }
    }

    println!("intermediate tensors compared: {compared}");
    if let Some(first) = first_drift {
        println!("first drift: {first}");
    } else {
        println!("first drift: none above threshold");
    }
    if let Some((name, max_abs, max_rel, mean_abs, idx)) = worst {
        println!(
            "worst drift: {name}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, \
             mean_abs={mean_abs:.6e}, idx={idx}"
        );
    }
}
