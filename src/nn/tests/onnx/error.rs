use crate::nn::graph::onnx_error::OnnxError;

#[test]
fn test_unsupported_operator_display() {
    let err = OnnxError::UnsupportedOperator {
        op_type: "CustomOp".to_string(),
        node_name: "layer1/custom".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("CustomOp"));
    assert!(msg.contains("layer1/custom"));
}

#[test]
fn test_unsupported_opset_display() {
    let err = OnnxError::UnsupportedOpsetVersion {
        version: 7,
        min_supported: 13,
        max_supported: 21,
    };
    let msg = format!("{err}");
    assert!(msg.contains("7"));
    assert!(msg.contains("13"));
    assert!(msg.contains("21"));
}

#[test]
fn test_unsupported_data_type_display() {
    let err = OnnxError::UnsupportedDataType {
        data_type: 10,
        context: "initializer weight_0".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("10"));
    assert!(msg.contains("float32"));
}

#[test]
fn test_unsupported_attribute_display() {
    let err = OnnxError::UnsupportedAttribute {
        op_type: "Gemm".to_string(),
        attribute: "alpha".to_string(),
        reason: "仅支持 alpha=1.0".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("Gemm"));
    assert!(msg.contains("alpha"));
}

#[test]
fn test_unsupported_conv_config_display() {
    let err = OnnxError::UnsupportedConvConfig {
        op_type: "Conv".to_string(),
        reason: "group > 1 的分组卷积暂不支持".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("Conv"));
    assert!(msg.contains("group"));
}

#[test]
fn test_weight_error_display() {
    let err = OnnxError::WeightError {
        tensor_name: "fc1.weight".to_string(),
        reason: "期望形状 [4, 2]，实际为空".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("fc1.weight"));
}

#[test]
fn test_training_node_in_export_path_display() {
    let err = OnnxError::TrainingNodeInExportPath {
        node_type: "SoftmaxCrossEntropy".to_string(),
        node_name: "loss".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("SoftmaxCrossEntropy"));
    assert!(msg.contains("推理子图"));
}

#[test]
fn test_io_error_conversion() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let onnx_err: OnnxError = io_err.into();
    assert!(matches!(onnx_err, OnnxError::Io(_)));
    let msg = format!("{onnx_err}");
    assert!(msg.contains("file not found"));
}

#[test]
fn test_parse_error_display() {
    let err = OnnxError::ParseError("unexpected EOF at byte 42".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("unexpected EOF"));
}

#[test]
fn test_invalid_graph_display() {
    let err = OnnxError::InvalidGraph("缺少图输入定义".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("缺少图输入定义"));
}

#[test]
fn test_descriptor_error_display() {
    let err = OnnxError::DescriptorError("节点 ID 冲突".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("节点 ID 冲突"));
}
