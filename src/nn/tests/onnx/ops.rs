use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::graph::onnx_error::OnnxError;
use crate::nn::graph::onnx_ops::{
    descriptor_to_export_category, onnx_op_to_descriptors, ExportCategory,
};
use onnx_rs::ast::{Attribute, OpType};

// ==================== 辅助构造函数 ====================

fn make_float_attr(name: &str, val: f32) -> Attribute<'static> {
    let name: &'static str = Box::leak(name.to_string().into_boxed_str());
    let mut attr = Attribute::default();
    attr.name = name;
    attr.f = val;
    attr
}

fn make_int_attr(name: &str, val: i64) -> Attribute<'static> {
    let name: &'static str = Box::leak(name.to_string().into_boxed_str());
    let mut attr = Attribute::default();
    attr.name = name;
    attr.i = val;
    attr
}

fn make_ints_attr(name: &str, vals: Vec<i64>) -> Attribute<'static> {
    let name: &'static str = Box::leak(name.to_string().into_boxed_str());
    let mut attr = Attribute::default();
    attr.name = name;
    attr.ints = vals;
    attr
}

// ==================== 导入方向测试 ====================

#[test]
fn test_import_relu() {
    let result = onnx_op_to_descriptors(&OpType::Relu, &[], "relu0").unwrap();
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], NodeTypeDescriptor::ReLU));
}

#[test]
fn test_import_sigmoid() {
    let result = onnx_op_to_descriptors(&OpType::Sigmoid, &[], "sig0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::Sigmoid));
}

#[test]
fn test_import_tanh() {
    let result = onnx_op_to_descriptors(&OpType::Tanh, &[], "tanh0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::Tanh));
}

#[test]
fn test_import_softmax() {
    let result = onnx_op_to_descriptors(&OpType::Softmax, &[], "sm0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::Softmax));
}

#[test]
fn test_import_log_softmax() {
    let result = onnx_op_to_descriptors(&OpType::LogSoftmax, &[], "lsm0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::LogSoftmax));
}

#[test]
fn test_import_gelu() {
    let result = onnx_op_to_descriptors(&OpType::Gelu, &[], "gelu0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::Gelu));
}

#[test]
fn test_import_selu() {
    let result = onnx_op_to_descriptors(&OpType::Selu, &[], "selu0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::Selu));
}

#[test]
fn test_import_mish() {
    let result = onnx_op_to_descriptors(&OpType::Mish, &[], "mish0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::Mish));
}

#[test]
fn test_import_hard_swish() {
    let result = onnx_op_to_descriptors(&OpType::HardSwish, &[], "hs0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::HardSwish));
}

#[test]
fn test_import_hard_sigmoid() {
    let result = onnx_op_to_descriptors(&OpType::HardSigmoid, &[], "hsg0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::HardSigmoid));
}

#[test]
fn test_import_softplus() {
    let result = onnx_op_to_descriptors(&OpType::Softplus, &[], "sp0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::SoftPlus));
}

#[test]
fn test_import_elu_default_alpha() {
    let result = onnx_op_to_descriptors(&OpType::Elu, &[], "elu0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::Elu { alpha } => assert!((alpha - 1.0).abs() < 1e-6),
        _ => panic!("expected Elu"),
    }
}

#[test]
fn test_import_elu_custom_alpha() {
    let attrs = vec![make_float_attr("alpha", 0.5)];
    let result = onnx_op_to_descriptors(&OpType::Elu, &attrs, "elu0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::Elu { alpha } => assert!((alpha - 0.5).abs() < 1e-6),
        _ => panic!("expected Elu"),
    }
}

#[test]
fn test_import_leaky_relu_default() {
    let result = onnx_op_to_descriptors(&OpType::LeakyRelu, &[], "lr0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::LeakyReLU { alpha } => assert!((alpha - 0.01).abs() < 1e-6),
        _ => panic!("expected LeakyReLU"),
    }
}

#[test]
fn test_import_leaky_relu_custom() {
    let attrs = vec![make_float_attr("alpha", 0.2)];
    let result = onnx_op_to_descriptors(&OpType::LeakyRelu, &attrs, "lr0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::LeakyReLU { alpha } => assert!((alpha - 0.2).abs() < 1e-6),
        _ => panic!("expected LeakyReLU"),
    }
}

#[test]
fn test_import_arithmetic_ops() {
    let cases = vec![
        (OpType::Add, NodeTypeDescriptor::Add),
        (OpType::Sub, NodeTypeDescriptor::Subtract),
        (OpType::Mul, NodeTypeDescriptor::Multiply),
        (OpType::Div, NodeTypeDescriptor::Divide),
        (OpType::Neg, NodeTypeDescriptor::Negate),
        (OpType::MatMul, NodeTypeDescriptor::MatMul),
    ];
    for (onnx_op, expected) in cases {
        let result = onnx_op_to_descriptors(&onnx_op, &[], "test").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            std::mem::discriminant(&result[0]),
            std::mem::discriminant(&expected),
            "failed for {:?}",
            onnx_op.as_str()
        );
    }
}

#[test]
fn test_import_math_ops() {
    let cases = vec![
        (OpType::Abs, NodeTypeDescriptor::Abs),
        (OpType::Exp, NodeTypeDescriptor::Exp),
        (OpType::Sqrt, NodeTypeDescriptor::Sqrt),
        (OpType::Log, NodeTypeDescriptor::Ln),
        (OpType::Sign, NodeTypeDescriptor::Sign),
        (OpType::Reciprocal, NodeTypeDescriptor::Reciprocal),
    ];
    for (onnx_op, expected) in cases {
        let result = onnx_op_to_descriptors(&onnx_op, &[], "test").unwrap();
        assert_eq!(
            std::mem::discriminant(&result[0]),
            std::mem::discriminant(&expected),
            "failed for {:?}",
            onnx_op.as_str()
        );
    }
}

#[test]
fn test_import_gemm_standard() {
    let attrs = vec![
        make_float_attr("alpha", 1.0),
        make_float_attr("beta", 1.0),
    ];
    let result = onnx_op_to_descriptors(&OpType::Gemm, &attrs, "gemm0").unwrap();
    assert_eq!(result.len(), 2);
    assert!(matches!(result[0], NodeTypeDescriptor::MatMul));
    assert!(matches!(result[1], NodeTypeDescriptor::Add));
}

#[test]
fn test_import_gemm_default_attrs() {
    let result = onnx_op_to_descriptors(&OpType::Gemm, &[], "gemm0").unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn test_import_gemm_non_standard_rejected() {
    let attrs = vec![make_float_attr("alpha", 2.0)];
    let result = onnx_op_to_descriptors(&OpType::Gemm, &attrs, "gemm0");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, OnnxError::UnsupportedAttribute { .. }));
}

#[test]
fn test_import_clip_relu6() {
    let attrs = vec![
        make_float_attr("min", 0.0),
        make_float_attr("max", 6.0),
    ];
    let result = onnx_op_to_descriptors(&OpType::Clip, &attrs, "clip0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::ReLU6));
}

#[test]
fn test_import_clip_generic() {
    let attrs = vec![
        make_float_attr("min", -1.0),
        make_float_attr("max", 1.0),
    ];
    let result = onnx_op_to_descriptors(&OpType::Clip, &attrs, "clip0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::Clip { min, max } => {
            assert!((*min - (-1.0)).abs() < 1e-6);
            assert!((*max - 1.0).abs() < 1e-6);
        }
        _ => panic!("expected Clip"),
    }
}

#[test]
fn test_import_flatten_axis1() {
    let attrs = vec![make_int_attr("axis", 1)];
    let result = onnx_op_to_descriptors(&OpType::Flatten, &attrs, "flat0").unwrap();
    assert!(matches!(
        result[0],
        NodeTypeDescriptor::Flatten { keep_first_dim: true }
    ));
}

#[test]
fn test_import_flatten_non_axis1_rejected() {
    let attrs = vec![make_int_attr("axis", 2)];
    let result = onnx_op_to_descriptors(&OpType::Flatten, &attrs, "flat0");
    assert!(result.is_err());
}

#[test]
fn test_import_concat() {
    let attrs = vec![make_int_attr("axis", 1)];
    let result = onnx_op_to_descriptors(&OpType::Concat, &attrs, "concat0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::Concat { axis } => assert_eq!(*axis, 1),
        _ => panic!("expected Concat"),
    }
}

#[test]
fn test_import_dropout() {
    let attrs = vec![make_float_attr("ratio", 0.3)];
    let result = onnx_op_to_descriptors(&OpType::Dropout, &attrs, "do0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::Dropout { p } => assert!((p - 0.3).abs() < 1e-6),
        _ => panic!("expected Dropout"),
    }
}

#[test]
fn test_import_conv2d() {
    let attrs = vec![
        make_ints_attr("kernel_shape", vec![3, 3]),
        make_ints_attr("strides", vec![1, 1]),
        make_ints_attr("pads", vec![1, 1, 1, 1]),
        make_int_attr("group", 1),
    ];
    let result = onnx_op_to_descriptors(&OpType::Conv, &attrs, "conv0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::Conv2d { stride, padding, dilation } => {
            assert_eq!(*stride, (1, 1));
            assert_eq!(*padding, (1, 1));
            assert_eq!(*dilation, (1, 1));
        }
        _ => panic!("expected Conv2d"),
    }
}

#[test]
fn test_import_conv_group_rejected() {
    let attrs = vec![
        make_ints_attr("kernel_shape", vec![3, 3]),
        make_int_attr("group", 4),
    ];
    let result = onnx_op_to_descriptors(&OpType::Conv, &attrs, "conv0");
    assert!(result.is_err());
}

#[test]
fn test_import_conv_dilation_accepted() {
    let attrs = vec![
        make_ints_attr("kernel_shape", vec![3, 3]),
        make_ints_attr("dilations", vec![2, 2]),
        make_ints_attr("strides", vec![1, 1]),
        make_ints_attr("pads", vec![2, 2, 2, 2]),
    ];
    let result = onnx_op_to_descriptors(&OpType::Conv, &attrs, "conv0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::Conv2d { stride, padding, dilation } => {
            assert_eq!(*stride, (1, 1));
            assert_eq!(*padding, (2, 2));
            assert_eq!(*dilation, (2, 2));
        }
        _ => panic!("expected Conv2d"),
    }
}

#[test]
fn test_import_maxpool() {
    let attrs = vec![
        make_ints_attr("kernel_shape", vec![2, 2]),
        make_ints_attr("strides", vec![2, 2]),
    ];
    let result = onnx_op_to_descriptors(&OpType::MaxPool, &attrs, "mp0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::MaxPool2d { kernel_size, stride } => {
            assert_eq!(*kernel_size, (2, 2));
            assert_eq!(*stride, (2, 2));
        }
        _ => panic!("expected MaxPool2d"),
    }
}

#[test]
fn test_import_avgpool() {
    let attrs = vec![
        make_ints_attr("kernel_shape", vec![3, 3]),
        make_ints_attr("strides", vec![1, 1]),
    ];
    let result = onnx_op_to_descriptors(&OpType::AveragePool, &attrs, "ap0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::AvgPool2d { kernel_size, stride } => {
            assert_eq!(*kernel_size, (3, 3));
            assert_eq!(*stride, (1, 1));
        }
        _ => panic!("expected AvgPool2d"),
    }
}

#[test]
fn test_import_pool_1d_rejected() {
    let attrs = vec![make_ints_attr("kernel_shape", vec![3])];
    let result = onnx_op_to_descriptors(&OpType::MaxPool, &attrs, "mp0");
    assert!(result.is_err());
}

#[test]
fn test_import_batchnorm() {
    let attrs = vec![
        make_float_attr("epsilon", 1e-5),
        make_float_attr("momentum", 0.1),
    ];
    let result = onnx_op_to_descriptors(&OpType::BatchNormalization, &attrs, "bn0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::BatchNormOp { eps, momentum, .. } => {
            assert!((*eps - 1e-5).abs() < 1e-8);
            assert!((*momentum - 0.1).abs() < 1e-6);
        }
        _ => panic!("expected BatchNormOp"),
    }
}

#[test]
fn test_import_identity() {
    let result = onnx_op_to_descriptors(&OpType::Identity, &[], "id0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::Identity));
}

#[test]
fn test_import_unsupported_operator() {
    let result = onnx_op_to_descriptors(&OpType::Custom("UnknownOp"), &[], "unk0");
    assert!(result.is_err());
    match result.unwrap_err() {
        OnnxError::UnsupportedOperator { op_type, .. } => {
            assert_eq!(op_type, "UnknownOp");
        }
        _ => panic!("expected UnsupportedOperator"),
    }
}

#[test]
fn test_import_reduce_sum_with_axis() {
    let attrs = vec![make_int_attr("axes", 1)];
    let result = onnx_op_to_descriptors(&OpType::ReduceSum, &attrs, "rs0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::Sum { axis } => assert_eq!(*axis, Some(1)),
        _ => panic!("expected Sum"),
    }
}

#[test]
fn test_import_reduce_mean_no_axis() {
    let result = onnx_op_to_descriptors(&OpType::ReduceMean, &[], "rm0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::Mean { axis } => assert_eq!(*axis, None),
        _ => panic!("expected Mean"),
    }
}

#[test]
fn test_import_element_wise_max_min() {
    let result = onnx_op_to_descriptors(&OpType::Max, &[], "max0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::Maximum));
    let result = onnx_op_to_descriptors(&OpType::Min, &[], "min0").unwrap();
    assert!(matches!(result[0], NodeTypeDescriptor::Minimum));
}

// ==================== 导出方向测试 ====================

#[test]
fn test_export_basic_input() {
    let cat = descriptor_to_export_category(&NodeTypeDescriptor::BasicInput);
    assert!(matches!(cat, ExportCategory::GraphInput));
}

#[test]
fn test_export_parameter() {
    let cat = descriptor_to_export_category(&NodeTypeDescriptor::Parameter);
    assert!(matches!(cat, ExportCategory::Initializer));
}

#[test]
fn test_export_state() {
    let cat = descriptor_to_export_category(&NodeTypeDescriptor::State);
    assert!(matches!(cat, ExportCategory::StateInitializer));
}

#[test]
fn test_export_training_only_nodes() {
    let training_nodes = vec![
        NodeTypeDescriptor::TargetInput,
        NodeTypeDescriptor::SoftmaxCrossEntropy,
        NodeTypeDescriptor::BCE {
            reduction: crate::nn::nodes::raw_node::Reduction::Mean,
        },
        NodeTypeDescriptor::MSE {
            reduction: crate::nn::nodes::raw_node::Reduction::Mean,
        },
        NodeTypeDescriptor::MAE {
            reduction: crate::nn::nodes::raw_node::Reduction::Mean,
        },
    ];
    for node in training_nodes {
        let cat = descriptor_to_export_category(&node);
        assert!(
            matches!(cat, ExportCategory::TrainingOnly),
            "expected TrainingOnly for {node:?}"
        );
    }
}

#[test]
fn test_export_activations() {
    let cases = vec![
        (NodeTypeDescriptor::ReLU, "Relu"),
        (NodeTypeDescriptor::Sigmoid, "Sigmoid"),
        (NodeTypeDescriptor::Tanh, "Tanh"),
        (NodeTypeDescriptor::Softmax, "Softmax"),
        (NodeTypeDescriptor::Gelu, "Gelu"),
        (NodeTypeDescriptor::Selu, "Selu"),
        (NodeTypeDescriptor::Mish, "Mish"),
        (NodeTypeDescriptor::HardSwish, "HardSwish"),
        (NodeTypeDescriptor::HardSigmoid, "HardSigmoid"),
        (NodeTypeDescriptor::SoftPlus, "Softplus"),
    ];
    for (desc, expected_op) in cases {
        match descriptor_to_export_category(&desc) {
            ExportCategory::Operator(op) => assert_eq!(op.op_type, expected_op, "for {desc:?}"),
            _ => panic!("expected Operator for {desc:?}"),
        }
    }
}

#[test]
fn test_export_leaky_relu_with_alpha() {
    let desc = NodeTypeDescriptor::LeakyReLU { alpha: 0.2 };
    match descriptor_to_export_category(&desc) {
        ExportCategory::Operator(op) => {
            assert_eq!(op.op_type, "LeakyRelu");
            assert_eq!(op.float_attrs.len(), 1);
            assert_eq!(op.float_attrs[0].0, "alpha");
            assert!((op.float_attrs[0].1 - 0.2).abs() < 1e-6);
        }
        _ => panic!("expected Operator"),
    }
}

#[test]
fn test_export_relu6_as_clip() {
    match descriptor_to_export_category(&NodeTypeDescriptor::ReLU6) {
        ExportCategory::Operator(op) => {
            assert_eq!(op.op_type, "Clip");
            assert_eq!(op.float_attrs.len(), 2);
        }
        _ => panic!("expected Operator Clip"),
    }
}

#[test]
fn test_export_conv2d() {
    let desc = NodeTypeDescriptor::Conv2d {
        stride: (2, 2),
        padding: (1, 1),
        dilation: (1, 1),
    };
    match descriptor_to_export_category(&desc) {
        ExportCategory::Operator(op) => {
            assert_eq!(op.op_type, "Conv");
            // dilation=(1,1) 不输出 → 仅 strides + pads
            assert_eq!(op.int_list_attrs.len(), 2);
        }
        _ => panic!("expected Operator Conv"),
    }

    // 非 (1,1) dilation 应导出 dilations 属性
    let desc_dil = NodeTypeDescriptor::Conv2d {
        stride: (1, 1),
        padding: (2, 2),
        dilation: (2, 2),
    };
    match descriptor_to_export_category(&desc_dil) {
        ExportCategory::Operator(op) => {
            assert_eq!(op.op_type, "Conv");
            assert_eq!(op.int_list_attrs.len(), 3, "应包含 strides + pads + dilations");
        }
        _ => panic!("expected Operator Conv"),
    }
}

#[test]
fn test_export_maxpool2d() {
    let desc = NodeTypeDescriptor::MaxPool2d {
        kernel_size: (2, 2),
        stride: (2, 2),
    };
    match descriptor_to_export_category(&desc) {
        ExportCategory::Operator(op) => {
            assert_eq!(op.op_type, "MaxPool");
        }
        _ => panic!("expected Operator MaxPool"),
    }
}

#[test]
fn test_export_batchnorm() {
    let desc = NodeTypeDescriptor::BatchNormOp {
        eps: 1e-5,
        momentum: 0.1,
        num_features: 64,
    };
    match descriptor_to_export_category(&desc) {
        ExportCategory::Operator(op) => {
            assert_eq!(op.op_type, "BatchNormalization");
            assert_eq!(op.float_attrs.len(), 2);
        }
        _ => panic!("expected Operator BatchNormalization"),
    }
}

#[test]
fn test_export_rnn_cells() {
    let rnn = NodeTypeDescriptor::CellRnn {
        input_size: 10,
        hidden_size: 20,
        return_sequences: false,
        seq_len: 5,
    };
    match descriptor_to_export_category(&rnn) {
        ExportCategory::Operator(op) => assert_eq!(op.op_type, "RNN"),
        _ => panic!("expected Operator RNN"),
    }

    let lstm = NodeTypeDescriptor::CellLstm {
        input_size: 10,
        hidden_size: 20,
        return_sequences: false,
        seq_len: 5,
    };
    match descriptor_to_export_category(&lstm) {
        ExportCategory::Operator(op) => assert_eq!(op.op_type, "LSTM"),
        _ => panic!("expected Operator LSTM"),
    }

    let gru = NodeTypeDescriptor::CellGru {
        input_size: 10,
        hidden_size: 20,
        return_sequences: false,
        seq_len: 5,
    };
    match descriptor_to_export_category(&gru) {
        ExportCategory::Operator(op) => assert_eq!(op.op_type, "GRU"),
        _ => panic!("expected Operator GRU"),
    }
}

#[test]
fn test_export_swish_unsupported() {
    match descriptor_to_export_category(&NodeTypeDescriptor::Swish) {
        ExportCategory::Unsupported(msg) => assert!(msg.contains("Swish")),
        _ => panic!("expected Unsupported for Swish"),
    }
}

#[test]
fn test_export_reduce_sum_with_axis() {
    let desc = NodeTypeDescriptor::Sum { axis: Some(1) };
    match descriptor_to_export_category(&desc) {
        ExportCategory::Operator(op) => {
            assert_eq!(op.op_type, "ReduceSum");
            assert_eq!(op.int_list_attrs.len(), 1);
            assert_eq!(op.int_list_attrs[0].1, vec![1]);
        }
        _ => panic!("expected Operator ReduceSum"),
    }
}

#[test]
fn test_export_reduce_sum_no_axis() {
    let desc = NodeTypeDescriptor::Sum { axis: None };
    match descriptor_to_export_category(&desc) {
        ExportCategory::Operator(op) => {
            assert_eq!(op.op_type, "ReduceSum");
            assert!(op.int_list_attrs.is_empty());
        }
        _ => panic!("expected Operator ReduceSum"),
    }
}

#[test]
fn test_export_flatten() {
    let desc = NodeTypeDescriptor::Flatten {
        keep_first_dim: true,
    };
    match descriptor_to_export_category(&desc) {
        ExportCategory::Operator(op) => {
            assert_eq!(op.op_type, "Flatten");
            assert_eq!(op.int_attrs, vec![("axis", 1)]);
        }
        _ => panic!("expected Operator Flatten"),
    }
}

#[test]
fn test_export_detach_as_identity() {
    match descriptor_to_export_category(&NodeTypeDescriptor::Detach) {
        ExportCategory::Operator(op) => assert_eq!(op.op_type, "Identity"),
        _ => panic!("expected Operator Identity"),
    }
}

// ==================== 往返一致性测试 ====================

#[test]
fn test_roundtrip_simple_activations() {
    let activations = vec![
        OpType::Relu,
        OpType::Sigmoid,
        OpType::Tanh,
        OpType::Softmax,
        OpType::Gelu,
        OpType::Selu,
        OpType::Mish,
    ];
    for onnx_op in activations {
        let imported = onnx_op_to_descriptors(&onnx_op, &[], "test").unwrap();
        assert_eq!(imported.len(), 1);
        let exported = descriptor_to_export_category(&imported[0]);
        match exported {
            ExportCategory::Operator(exp) => {
                assert_eq!(
                    exp.op_type,
                    onnx_op.as_str(),
                    "roundtrip mismatch for {:?}",
                    onnx_op.as_str()
                );
            }
            _ => panic!("roundtrip failed for {:?}", onnx_op.as_str()),
        }
    }
}

#[test]
fn test_roundtrip_arithmetic() {
    let ops = vec![
        (OpType::Add, "Add"),
        (OpType::Sub, "Sub"),
        (OpType::Mul, "Mul"),
        (OpType::Div, "Div"),
        (OpType::Neg, "Neg"),
        (OpType::MatMul, "MatMul"),
    ];
    for (onnx_op, expected_export_name) in ops {
        let imported = onnx_op_to_descriptors(&onnx_op, &[], "test").unwrap();
        let exported = descriptor_to_export_category(&imported[0]);
        match exported {
            ExportCategory::Operator(exp) => {
                assert_eq!(exp.op_type, expected_export_name);
            }
            _ => panic!("roundtrip failed for {expected_export_name}"),
        }
    }
}

#[test]
fn test_roundtrip_leaky_relu() {
    let attrs = vec![make_float_attr("alpha", 0.15)];
    let imported = onnx_op_to_descriptors(&OpType::LeakyRelu, &attrs, "test").unwrap();
    match descriptor_to_export_category(&imported[0]) {
        ExportCategory::Operator(exp) => {
            assert_eq!(exp.op_type, "LeakyRelu");
            assert!((exp.float_attrs[0].1 - 0.15).abs() < 1e-6);
        }
        _ => panic!("roundtrip failed for LeakyRelu"),
    }
}

#[test]
fn test_roundtrip_conv2d() {
    let attrs = vec![
        make_ints_attr("kernel_shape", vec![3, 3]),
        make_ints_attr("strides", vec![2, 2]),
        make_ints_attr("pads", vec![1, 1, 1, 1]),
        make_int_attr("group", 1),
    ];
    let imported = onnx_op_to_descriptors(&OpType::Conv, &attrs, "test").unwrap();
    match &imported[0] {
        NodeTypeDescriptor::Conv2d { stride, padding, dilation } => {
            assert_eq!(*stride, (2, 2));
            assert_eq!(*padding, (1, 1));
            assert_eq!(*dilation, (1, 1));
        }
        _ => panic!("expected Conv2d"),
    }
    match descriptor_to_export_category(&imported[0]) {
        ExportCategory::Operator(exp) => assert_eq!(exp.op_type, "Conv"),
        _ => panic!("roundtrip failed for Conv"),
    }
}

#[test]
fn test_roundtrip_batchnorm() {
    let attrs = vec![
        make_float_attr("epsilon", 1e-5),
        make_float_attr("momentum", 0.1),
    ];
    let imported =
        onnx_op_to_descriptors(&OpType::BatchNormalization, &attrs, "test").unwrap();
    match descriptor_to_export_category(&imported[0]) {
        ExportCategory::Operator(exp) => {
            assert_eq!(exp.op_type, "BatchNormalization");
        }
        _ => panic!("roundtrip failed for BatchNormalization"),
    }
}

// ==================== ConvTranspose2d 导入测试 ====================

#[test]
fn test_import_conv_transpose_default() {
    let attrs = vec![
        make_ints_attr("kernel_shape", vec![3, 3]),
    ];
    let result = onnx_op_to_descriptors(&OpType::ConvTranspose, &attrs, "deconv0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::ConvTranspose2d { stride, padding, output_padding } => {
            assert_eq!(*stride, (1, 1));
            assert_eq!(*padding, (0, 0));
            assert_eq!(*output_padding, (0, 0));
        }
        _ => panic!("expected ConvTranspose2d"),
    }
}

#[test]
fn test_import_conv_transpose_with_params() {
    let attrs = vec![
        make_ints_attr("kernel_shape", vec![3, 3]),
        make_ints_attr("strides", vec![2, 2]),
        make_ints_attr("pads", vec![1, 1, 1, 1]),
        make_ints_attr("output_padding", vec![1, 1]),
    ];
    let result = onnx_op_to_descriptors(&OpType::ConvTranspose, &attrs, "deconv0").unwrap();
    match &result[0] {
        NodeTypeDescriptor::ConvTranspose2d { stride, padding, output_padding } => {
            assert_eq!(*stride, (2, 2));
            assert_eq!(*padding, (1, 1));
            assert_eq!(*output_padding, (1, 1));
        }
        _ => panic!("expected ConvTranspose2d"),
    }
}

#[test]
fn test_import_conv_transpose_group_rejected() {
    let attrs = vec![
        make_ints_attr("kernel_shape", vec![3, 3]),
        make_int_attr("group", 4),
    ];
    let result = onnx_op_to_descriptors(&OpType::ConvTranspose, &attrs, "deconv0");
    assert!(result.is_err());
}

// ==================== ConvTranspose2d 导出测试 ====================

#[test]
fn test_export_conv_transpose2d() {
    let desc = NodeTypeDescriptor::ConvTranspose2d {
        stride: (2, 2),
        padding: (1, 1),
        output_padding: (0, 0),
    };
    match descriptor_to_export_category(&desc) {
        ExportCategory::Operator(op) => {
            assert_eq!(op.op_type, "ConvTranspose");
            // output_padding=(0,0) 不输出 → 仅 strides + pads
            assert_eq!(op.int_list_attrs.len(), 2);
        }
        _ => panic!("expected Operator ConvTranspose"),
    }

    // 非 (0,0) output_padding 应导出
    let desc_op = NodeTypeDescriptor::ConvTranspose2d {
        stride: (2, 2),
        padding: (1, 1),
        output_padding: (1, 1),
    };
    match descriptor_to_export_category(&desc_op) {
        ExportCategory::Operator(op) => {
            assert_eq!(op.op_type, "ConvTranspose");
            assert_eq!(op.int_list_attrs.len(), 3, "应包含 strides + pads + output_padding");
        }
        _ => panic!("expected Operator ConvTranspose"),
    }
}

// ==================== ConvTranspose2d 导入→导出 往返测试 ====================

#[test]
fn test_roundtrip_conv_transpose2d() {
    let attrs = vec![
        make_ints_attr("kernel_shape", vec![3, 3]),
        make_ints_attr("strides", vec![2, 2]),
        make_ints_attr("pads", vec![1, 1, 1, 1]),
        make_ints_attr("output_padding", vec![1, 1]),
    ];
    let imported =
        onnx_op_to_descriptors(&OpType::ConvTranspose, &attrs, "test").unwrap();
    match descriptor_to_export_category(&imported[0]) {
        ExportCategory::Operator(exp) => {
            assert_eq!(exp.op_type, "ConvTranspose");
        }
        _ => panic!("roundtrip failed for ConvTranspose"),
    }
}
