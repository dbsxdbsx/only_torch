use std::collections::HashSet;

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::graph::onnx_error::OnnxError;
use crate::nn::graph::onnx_import::load_onnx_from_bytes;
use onnx_rs::ast::*;

/// Input(2) → MatMul(W=[2,4]) → Add(b=[4]) → Relu → MatMul(W2=[4,1]) → Add(b2=[1])
fn build_minimal_mlp_bytes() -> Vec<u8> {
    let w1 = TensorProto::from_f32("W1", vec![2, 4], vec![0.1; 8]);
    let b1 = TensorProto::from_f32("b1", vec![4], vec![0.0; 4]);
    let w2 = TensorProto::from_f32("W2", vec![4, 1], vec![0.2; 4]);
    let b2 = TensorProto::from_f32("b2", vec![1], vec![0.0]);

    let input_vi = ValueInfo {
        name: "input",
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: Some(TensorShape {
                    dim: vec![
                        TensorShapeDimension {
                            value: Dimension::Param("batch"),
                            denotation: "",
                        },
                        TensorShapeDimension {
                            value: Dimension::Value(2),
                            denotation: "",
                        },
                    ],
                }),
            })),
            denotation: "",
        }),
        doc_string: "",
        metadata_props: vec![],
    };
    let output_vi = ValueInfo {
        name: "output",
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: Some(TensorShape {
                    dim: vec![
                        TensorShapeDimension {
                            value: Dimension::Param("batch"),
                            denotation: "",
                        },
                        TensorShapeDimension {
                            value: Dimension::Value(1),
                            denotation: "",
                        },
                    ],
                }),
            })),
            denotation: "",
        }),
        doc_string: "",
        metadata_props: vec![],
    };

    let matmul1 = Node {
        input: vec!["input", "W1"],
        output: vec!["matmul1_out"],
        name: "matmul1",
        op_type: OpType::MatMul,
        ..Default::default()
    };
    let add1 = Node {
        input: vec!["matmul1_out", "b1"],
        output: vec!["add1_out"],
        name: "add1",
        op_type: OpType::Add,
        ..Default::default()
    };
    let relu = Node {
        input: vec!["add1_out"],
        output: vec!["relu_out"],
        name: "relu1",
        op_type: OpType::Relu,
        ..Default::default()
    };
    let matmul2 = Node {
        input: vec!["relu_out", "W2"],
        output: vec!["matmul2_out"],
        name: "matmul2",
        op_type: OpType::MatMul,
        ..Default::default()
    };
    let add2 = Node {
        input: vec!["matmul2_out", "b2"],
        output: vec!["output"],
        name: "add2",
        op_type: OpType::Add,
        ..Default::default()
    };

    let graph = Graph {
        node: vec![matmul1, add1, relu, matmul2, add2],
        name: "mlp",
        initializer: vec![w1, b1, w2, b2],
        input: vec![input_vi],
        output: vec![output_vi],
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 17,
        }],
        graph: Some(graph),
        ..Default::default()
    };

    onnx_rs::encode(&model)
}

fn build_gemm_model_bytes() -> Vec<u8> {
    let w = TensorProto::from_f32("W", vec![4, 2], vec![0.1; 8]);
    let b = TensorProto::from_f32("b", vec![4], vec![0.0; 4]);

    let input_vi = ValueInfo {
        name: "X",
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: Some(TensorShape {
                    dim: vec![
                        TensorShapeDimension {
                            value: Dimension::Value(1),
                            denotation: "",
                        },
                        TensorShapeDimension {
                            value: Dimension::Value(2),
                            denotation: "",
                        },
                    ],
                }),
            })),
            denotation: "",
        }),
        doc_string: "",
        metadata_props: vec![],
    };
    let output_vi = ValueInfo {
        name: "Y",
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: None,
            })),
            denotation: "",
        }),
        doc_string: "",
        metadata_props: vec![],
    };

    let gemm = Node {
        input: vec!["X", "W", "b"],
        output: vec!["Y"],
        name: "gemm0",
        op_type: OpType::Gemm,
        attribute: vec![
            Attribute {
                name: "alpha",
                f: 1.0,
                ..Default::default()
            },
            Attribute {
                name: "beta",
                f: 1.0,
                ..Default::default()
            },
            Attribute {
                name: "transB",
                i: 1,
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    let graph = Graph {
        node: vec![gemm],
        name: "gemm_test",
        initializer: vec![w, b],
        input: vec![input_vi],
        output: vec![output_vi],
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 17,
        }],
        graph: Some(graph),
        ..Default::default()
    };

    onnx_rs::encode(&model)
}

#[test]
fn test_load_minimal_mlp() {
    let bytes = build_minimal_mlp_bytes();
    let result = load_onnx_from_bytes(&bytes).unwrap();

    assert_eq!(result.descriptor.nodes.len(), 10);
    assert_eq!(result.weights.len(), 4);

    let input_nodes: Vec<_> = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::BasicInput))
        .collect();
    assert_eq!(input_nodes.len(), 1);
    assert_eq!(input_nodes[0].name, "input");

    let param_nodes: Vec<_> = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::Parameter))
        .collect();
    assert_eq!(param_nodes.len(), 4);

    let relu_nodes: Vec<_> = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::ReLU))
        .collect();
    assert_eq!(relu_nodes.len(), 1);

    let matmul_nodes: Vec<_> = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::MatMul))
        .collect();
    assert_eq!(matmul_nodes.len(), 2);

    let add_nodes: Vec<_> = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::Add))
        .collect();
    assert_eq!(add_nodes.len(), 2);
}

#[test]
fn test_load_mlp_weights_correct() {
    let bytes = build_minimal_mlp_bytes();
    let result = load_onnx_from_bytes(&bytes).unwrap();

    let w1_node = result
        .descriptor
        .nodes
        .iter()
        .find(|n| n.name == "W1")
        .unwrap();
    assert_eq!(w1_node.output_shape, vec![2, 4]);

    let w1_tensor = result.weights.get(&w1_node.id).unwrap();
    assert_eq!(w1_tensor.shape(), &[2, 4]);
    let flat = w1_tensor.flatten_view();
    assert_eq!(flat.len(), 8);
    for &val in flat.iter() {
        assert!((val - 0.1).abs() < 1e-5);
    }
}

#[test]
fn test_load_mlp_topology() {
    let bytes = build_minimal_mlp_bytes();
    let result = load_onnx_from_bytes(&bytes).unwrap();

    let input_id = result
        .descriptor
        .nodes
        .iter()
        .find(|n| n.name == "input")
        .unwrap()
        .id;
    let w1_id = result
        .descriptor
        .nodes
        .iter()
        .find(|n| n.name == "W1")
        .unwrap()
        .id;
    let matmul1 = result
        .descriptor
        .nodes
        .iter()
        .find(|n| n.name == "matmul1_out")
        .unwrap();
    assert_eq!(matmul1.parents, vec![input_id, w1_id]);

    let b1_id = result
        .descriptor
        .nodes
        .iter()
        .find(|n| n.name == "b1")
        .unwrap()
        .id;
    let add1 = result
        .descriptor
        .nodes
        .iter()
        .find(|n| n.name == "add1_out")
        .unwrap();
    assert_eq!(add1.parents, vec![matmul1.id, b1_id]);

    let relu = result
        .descriptor
        .nodes
        .iter()
        .find(|n| matches!(n.node_type, NodeTypeDescriptor::ReLU))
        .unwrap();
    assert_eq!(relu.parents, vec![add1.id]);
}

#[test]
fn test_load_gemm_expansion() {
    let bytes = build_gemm_model_bytes();
    let result = load_onnx_from_bytes(&bytes).unwrap();

    assert_eq!(result.descriptor.nodes.len(), 5);

    let matmul_nodes: Vec<_> = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::MatMul))
        .collect();
    assert_eq!(matmul_nodes.len(), 1);

    let add_nodes: Vec<_> = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::Add))
        .collect();
    assert_eq!(add_nodes.len(), 1);

    let x_id = result
        .descriptor
        .nodes
        .iter()
        .find(|n| n.name == "X")
        .unwrap()
        .id;
    let w_id = result
        .descriptor
        .nodes
        .iter()
        .find(|n| n.name == "W")
        .unwrap()
        .id;
    assert_eq!(matmul_nodes[0].parents, vec![x_id, w_id]);

    let b_id = result
        .descriptor
        .nodes
        .iter()
        .find(|n| n.name == "b")
        .unwrap()
        .id;
    assert_eq!(add_nodes[0].parents, vec![matmul_nodes[0].id, b_id]);
}

#[test]
fn test_opset_version_too_low() {
    let model = Model {
        ir_version: 3,
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 7,
        }],
        graph: Some(Graph::default()),
        ..Default::default()
    };
    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes);
    assert!(result.is_err());
    match result.unwrap_err() {
        OnnxError::UnsupportedOpsetVersion { version, .. } => assert_eq!(version, 7),
        e => panic!("expected UnsupportedOpsetVersion, got: {e}"),
    }
}

#[test]
fn test_opset_version_too_high() {
    let model = Model {
        ir_version: 10,
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 25,
        }],
        graph: Some(Graph::default()),
        ..Default::default()
    };
    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_missing_graph() {
    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 17,
        }],
        graph: None,
        ..Default::default()
    };
    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), OnnxError::InvalidGraph(_)));
}

#[test]
fn test_unsupported_operator() {
    let input_vi = ValueInfo {
        name: "X",
        r#type: None,
        doc_string: "",
        metadata_props: vec![],
    };
    let unknown_node = Node {
        input: vec!["X"],
        output: vec!["Y"],
        name: "custom0",
        op_type: OpType::Custom("MyCustomOp"),
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 17,
        }],
        graph: Some(Graph {
            node: vec![unknown_node],
            input: vec![input_vi],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes);
    assert!(result.is_err());
    match result.unwrap_err() {
        OnnxError::UnsupportedOperator { op_type, .. } => {
            assert_eq!(op_type, "MyCustomOp");
        }
        e => panic!("expected UnsupportedOperator, got: {e}"),
    }
}

#[test]
fn test_input_shape_extraction() {
    let bytes = build_minimal_mlp_bytes();
    let result = load_onnx_from_bytes(&bytes).unwrap();
    let input_node = result
        .descriptor
        .nodes
        .iter()
        .find(|n| n.name == "input")
        .unwrap();
    assert_eq!(input_node.output_shape, vec![0, 2]);
}

#[test]
fn test_invalid_bytes() {
    let result = load_onnx_from_bytes(&[0xFF, 0xFE, 0xFD]);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), OnnxError::ParseError(_)));
}

#[test]
fn test_conv2d_import() {
    let w = TensorProto::from_f32("conv_w", vec![8, 1, 3, 3], vec![0.1; 72]);
    let b = TensorProto::from_f32("conv_b", vec![8], vec![0.0; 8]);
    let input_vi = ValueInfo {
        name: "img",
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: Some(TensorShape {
                    dim: vec![
                        TensorShapeDimension { value: Dimension::Value(1), denotation: "" },
                        TensorShapeDimension { value: Dimension::Value(1), denotation: "" },
                        TensorShapeDimension { value: Dimension::Value(28), denotation: "" },
                        TensorShapeDimension { value: Dimension::Value(28), denotation: "" },
                    ],
                }),
            })),
            denotation: "",
        }),
        doc_string: "",
        metadata_props: vec![],
    };

    let conv_node = Node {
        input: vec!["img", "conv_w", "conv_b"],
        output: vec!["conv_out"],
        name: "conv0",
        op_type: OpType::Conv,
        attribute: vec![
            Attribute { name: "kernel_shape", ints: vec![3, 3], ..Default::default() },
            Attribute { name: "strides", ints: vec![1, 1], ..Default::default() },
            Attribute { name: "pads", ints: vec![1, 1, 1, 1], ..Default::default() },
            Attribute { name: "group", i: 1, ..Default::default() },
        ],
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![conv_node],
            name: "conv_test",
            initializer: vec![w, b],
            input: vec![input_vi],
            output: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes).unwrap();

    let conv_nodes: Vec<_> = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::Conv2d { .. }))
        .collect();
    assert_eq!(conv_nodes.len(), 1);

    match &conv_nodes[0].node_type {
        NodeTypeDescriptor::Conv2d { stride, padding, dilation } => {
            assert_eq!(*stride, (1, 1));
            assert_eq!(*padding, (1, 1));
            assert_eq!(*dilation, (1, 1));
        }
        _ => panic!("expected Conv2d"),
    }

    assert_eq!(result.weights.len(), 2);
    let conv_w_node = result.descriptor.nodes.iter().find(|n| n.name == "conv_w").unwrap();
    let conv_w = result.weights.get(&conv_w_node.id).unwrap();
    assert_eq!(conv_w.shape(), &[8, 1, 3, 3]);
}

#[test]
fn test_empty_optional_input() {
    let input_vi = ValueInfo {
        name: "X",
        r#type: None,
        doc_string: "",
        metadata_props: vec![],
    };
    let dropout_node = Node {
        input: vec!["X", "", ""],
        output: vec!["Y"],
        name: "drop0",
        op_type: OpType::Dropout,
        attribute: vec![Attribute {
            name: "ratio",
            f: 0.5,
            ..Default::default()
        }],
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![dropout_node],
            input: vec![input_vi],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes).unwrap();

    let dropout: Vec<_> = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::Dropout { .. }))
        .collect();
    assert_eq!(dropout.len(), 1);
    assert_eq!(dropout[0].parents.len(), 1);
}

#[test]
fn test_initializer_in_input_list_not_duplicated() {
    let w = TensorProto::from_f32("W", vec![2, 3], vec![0.1; 6]);
    let input_vi = ValueInfo {
        name: "X",
        r#type: None,
        doc_string: "",
        metadata_props: vec![],
    };
    let w_vi = ValueInfo {
        name: "W",
        r#type: None,
        doc_string: "",
        metadata_props: vec![],
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![Node {
                input: vec!["X", "W"],
                output: vec!["Y"],
                name: "matmul0",
                op_type: OpType::MatMul,
                ..Default::default()
            }],
            name: "dedup_test",
            initializer: vec![w],
            input: vec![input_vi, w_vi],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes).unwrap();

    let w_nodes: Vec<_> = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| n.name == "W")
        .collect();
    assert_eq!(w_nodes.len(), 1);
    assert!(matches!(w_nodes[0].node_type, NodeTypeDescriptor::Parameter));

    let x_nodes: Vec<_> = result
        .descriptor
        .nodes
        .iter()
        .filter(|n| n.name == "X")
        .collect();
    assert_eq!(x_nodes.len(), 1);
    assert!(matches!(x_nodes[0].node_type, NodeTypeDescriptor::BasicInput));
}

#[test]
fn test_activation_chain() {
    let input_vi = ValueInfo {
        name: "X",
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: Some(TensorShape {
                    dim: vec![
                        TensorShapeDimension { value: Dimension::Value(1), denotation: "" },
                        TensorShapeDimension { value: Dimension::Value(10), denotation: "" },
                    ],
                }),
            })),
            denotation: "",
        }),
        doc_string: "",
        metadata_props: vec![],
    };

    let relu_node = Node {
        input: vec!["X"],
        output: vec!["r"],
        name: "relu0",
        op_type: OpType::Relu,
        ..Default::default()
    };
    let sigmoid_node = Node {
        input: vec!["r"],
        output: vec!["s"],
        name: "sig0",
        op_type: OpType::Sigmoid,
        ..Default::default()
    };
    let tanh_node = Node {
        input: vec!["s"],
        output: vec!["t"],
        name: "tanh0",
        op_type: OpType::Tanh,
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![relu_node, sigmoid_node, tanh_node],
            name: "activation_chain",
            input: vec![input_vi],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes).unwrap();

    assert_eq!(result.descriptor.nodes.len(), 4);

    let x_id = result.descriptor.nodes.iter().find(|n| n.name == "X").unwrap().id;
    let relu = result.descriptor.nodes.iter().find(|n| n.name == "r").unwrap();
    assert_eq!(relu.parents, vec![x_id]);
    assert!(matches!(relu.node_type, NodeTypeDescriptor::ReLU));

    let sig = result.descriptor.nodes.iter().find(|n| n.name == "s").unwrap();
    assert_eq!(sig.parents, vec![relu.id]);
    assert!(matches!(sig.node_type, NodeTypeDescriptor::Sigmoid));

    let tanh = result.descriptor.nodes.iter().find(|n| n.name == "t").unwrap();
    assert_eq!(tanh.parents, vec![sig.id]);
    assert!(matches!(tanh.node_type, NodeTypeDescriptor::Tanh));
}

#[test]
fn test_symbol_table_consistency() {
    let bytes = build_minimal_mlp_bytes();
    let result = load_onnx_from_bytes(&bytes).unwrap();

    let mut ids: Vec<u64> = result.descriptor.nodes.iter().map(|n| n.id).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), result.descriptor.nodes.len());

    let id_set: HashSet<u64> = result.descriptor.nodes.iter().map(|n| n.id).collect();
    for node in &result.descriptor.nodes {
        for &p in &node.parents {
            assert!(id_set.contains(&p), "节点 {} 引用了不存在的父节点 {}", node.name, p);
        }
    }
}

#[test]
fn test_graph_name_preserved() {
    let bytes = build_minimal_mlp_bytes();
    let result = load_onnx_from_bytes(&bytes).unwrap();
    assert_eq!(result.descriptor.name, "mlp");
}

#[test]
fn test_zero_weights() {
    let bytes = build_minimal_mlp_bytes();
    let result = load_onnx_from_bytes(&bytes).unwrap();

    let b1_node = result.descriptor.nodes.iter().find(|n| n.name == "b1").unwrap();
    let b1_tensor = result.weights.get(&b1_node.id).unwrap();
    assert_eq!(b1_tensor.shape(), &[1, 4]);
    for &val in b1_tensor.flatten_view().iter() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_dynamic_input_shape() {
    let bytes = build_minimal_mlp_bytes();
    let result = load_onnx_from_bytes(&bytes).unwrap();
    let input_node = result.descriptor.nodes.iter().find(|n| n.name == "input").unwrap();

    assert!(input_node.dynamic_shape.is_some());
    let dyn_shape = input_node.dynamic_shape.as_ref().unwrap();
    assert_eq!(dyn_shape[0], None);
    assert_eq!(dyn_shape[1], Some(2));
}

#[test]
fn test_graph_from_onnx_bytes_e2e() {
    use crate::nn::graph::Graph;

    let bytes = build_minimal_mlp_bytes();
    let result = Graph::from_onnx_bytes(&bytes).unwrap();

    assert_eq!(result.inputs.len(), 1);
    assert_eq!(result.inputs[0].0, "input");
    assert!(!result.outputs.is_empty());

    let w1_param = result
        .graph
        .inner()
        .get_parameter("W1");
    assert!(w1_param.is_some(), "W1 参数应在图中注册");
}

#[test]
fn test_genome_from_onnx_bytes() {
    use crate::nn::evolution::gene::NetworkGenome;

    let bytes = build_gemm_model_bytes();
    let genome = NetworkGenome::from_onnx_bytes(&bytes).unwrap();

    assert_eq!(genome.input_dim, 2);
    assert!(genome.seq_len.is_none());
    assert!(genome.input_spatial.is_none());
}

// ==================== ImportReport 骨架测试 ====================

#[test]
fn test_import_report_default_empty() {
    // 纯 MLP（无 Conv+bias、无 Gemm）：只有 MatMul + Add 直接映射，无 rewrite
    let bytes = build_minimal_mlp_bytes();
    let result = load_onnx_from_bytes(&bytes).unwrap();

    assert!(
        result.import_report.rewritten.is_empty(),
        "纯 MatMul+Add 图不应触发任何 rewrite"
    );
    assert!(result.import_report.warnings.is_empty());
}

#[test]
fn test_import_report_records_conv_bias_split() {
    // Conv with bias 模型（来自 test_conv2d_import 的同款图）
    let w = TensorProto::from_f32("conv_w", vec![8, 1, 3, 3], vec![0.1; 72]);
    let b = TensorProto::from_f32("conv_b", vec![8], vec![0.0; 8]);
    let input_vi = ValueInfo {
        name: "img",
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: Some(TensorShape {
                    dim: vec![
                        TensorShapeDimension { value: Dimension::Value(1), denotation: "" },
                        TensorShapeDimension { value: Dimension::Value(1), denotation: "" },
                        TensorShapeDimension { value: Dimension::Value(28), denotation: "" },
                        TensorShapeDimension { value: Dimension::Value(28), denotation: "" },
                    ],
                }),
            })),
            denotation: "",
        }),
        doc_string: "",
        metadata_props: vec![],
    };
    let conv_node = Node {
        input: vec!["img", "conv_w", "conv_b"],
        output: vec!["conv_out"],
        name: "conv0",
        op_type: OpType::Conv,
        attribute: vec![
            Attribute { name: "kernel_shape", ints: vec![3, 3], ..Default::default() },
            Attribute { name: "strides", ints: vec![1, 1], ..Default::default() },
            Attribute { name: "pads", ints: vec![1, 1, 1, 1], ..Default::default() },
            Attribute { name: "group", i: 1, ..Default::default() },
        ],
        ..Default::default()
    };
    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![conv_node],
            name: "conv_test",
            initializer: vec![w, b],
            input: vec![input_vi],
            output: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };
    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes).unwrap();

    // 必须含 1 条 conv_with_bias_to_conv_plus_add 记录
    assert_eq!(result.import_report.rewritten.len(), 1);
    let record = &result.import_report.rewritten[0];
    assert_eq!(record.pattern, "conv_with_bias_to_conv_plus_add");
    assert_eq!(record.consumed_onnx_nodes, vec!["conv0".to_string()]);
    assert_eq!(record.produced_descriptor_nodes.len(), 2, "应产出 Conv2d + Add 两个节点");

    // 反向验证：produced ID 真实存在于 descriptor 中
    let id_set: HashSet<u64> = result.descriptor.nodes.iter().map(|n| n.id).collect();
    for &id in &record.produced_descriptor_nodes {
        assert!(id_set.contains(&id), "produced_descriptor_nodes 中的 ID {id} 应存在于 descriptor");
    }
}

#[test]
fn test_import_report_records_gemm_split() {
    // Gemm 模型（已有 build_gemm_model_bytes 复用）
    let bytes = build_gemm_model_bytes();
    let result = load_onnx_from_bytes(&bytes).unwrap();

    assert_eq!(result.import_report.rewritten.len(), 1);
    let record = &result.import_report.rewritten[0];
    assert_eq!(record.pattern, "gemm_to_matmul_plus_add");
    assert_eq!(record.consumed_onnx_nodes, vec!["gemm0".to_string()]);
    assert_eq!(record.produced_descriptor_nodes.len(), 2);
}

// ==================== Constant 折叠 + Split 重写测试 ====================

/// 创建 i64 类型 TensorProto（用于 shape / split_sizes）
fn make_i64_tensor(name: &'static str, vals: Vec<i64>) -> TensorProto<'static> {
    let dims = vec![vals.len() as i64];
    TensorProto::from_i64(name, dims, vals)
}

/// 创建 f32 类型 TensorProto（用于 Resize scales）
fn make_f32_tensor(name: &'static str, vals: Vec<f32>) -> TensorProto<'static> {
    let dims = vec![vals.len() as i64];
    TensorProto::from_f32(name, dims, vals)
}

/// 通用 ValueInfo 构造
fn make_value_info<'a>(name: &'a str, dims: Vec<i64>) -> ValueInfo<'a> {
    ValueInfo {
        name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: Some(TensorShape {
                    dim: dims
                        .into_iter()
                        .map(|d| TensorShapeDimension {
                            value: Dimension::Value(d),
                            denotation: "",
                        })
                        .collect(),
                }),
            })),
            denotation: "",
        }),
        doc_string: "",
        metadata_props: vec![],
    }
}

#[test]
fn test_constant_fold_reshape_via_initializer() {
    // 构造 X(1×6) → Reshape(shape=initializer [2,3]) → Y
    let shape_init = make_i64_tensor("shape_init", vec![2, 3]);
    let input_vi = make_value_info("X", vec![1, 6]);
    let output_vi = ValueInfo {
        name: "Y",
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: None,
            })),
            denotation: "",
        }),
        doc_string: "",
        metadata_props: vec![],
    };
    let reshape_node = Node {
        input: vec!["X", "shape_init"],
        output: vec!["Y"],
        name: "reshape0",
        op_type: OpType::Reshape,
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![reshape_node],
            name: "reshape_const_fold",
            initializer: vec![shape_init],
            input: vec![input_vi],
            output: vec![output_vi],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes).unwrap();

    // Reshape descriptor 应已折叠 target_shape=[2,3]
    let reshape = result
        .descriptor
        .nodes
        .iter()
        .find(|n| n.name == "Y")
        .unwrap();
    match &reshape.node_type {
        NodeTypeDescriptor::Reshape { target_shape } => {
            assert_eq!(target_shape, &vec![2usize, 3]);
        }
        _ => panic!("expected Reshape"),
    }

    // shape_init 不应作为 Parameter 节点出现
    let shape_node = result.descriptor.nodes.iter().find(|n| n.name == "shape_init");
    assert!(shape_node.is_none(), "元信息 initializer 应被跳过");

    // ImportReport 记录
    assert!(result
        .import_report
        .rewritten
        .iter()
        .any(|r| r.pattern == "constant_fold_into_reshape"));
}

#[test]
fn test_constant_fold_reshape_via_constant_node() {
    // 与上一个相同，但 shape 来自 ONNX Constant 节点（不是 initializer）
    let const_value = TensorProto::from_i64("", vec![3], vec![1, 2, 3]);
    let shape_constant = Node {
        input: vec![],
        output: vec!["shape_out"],
        name: "shape_const",
        op_type: OpType::Constant,
        attribute: vec![Attribute {
            name: "value",
            t: Some(const_value),
            ..Default::default()
        }],
        ..Default::default()
    };
    let input_vi = make_value_info("X", vec![1, 6]);
    let output_vi = make_value_info("Y", vec![1, 2, 3]);
    let reshape_node = Node {
        input: vec!["X", "shape_out"],
        output: vec!["Y"],
        name: "reshape0",
        op_type: OpType::Reshape,
        ..Default::default()
    };
    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![shape_constant, reshape_node],
            name: "reshape_constant_node",
            input: vec![input_vi],
            output: vec![output_vi],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes).unwrap();

    // Constant 节点本身不应出现在 descriptor 中
    let const_node = result.descriptor.nodes.iter().find(|n| n.name == "shape_const");
    assert!(const_node.is_none(), "Constant 节点应被折叠消失");

    // Reshape 应已填好 target_shape
    let reshape = result.descriptor.nodes.iter().find(|n| n.name == "Y").unwrap();
    match &reshape.node_type {
        NodeTypeDescriptor::Reshape { target_shape } => {
            assert_eq!(target_shape, &vec![1usize, 2, 3]);
        }
        _ => panic!("expected Reshape"),
    }
}

#[test]
fn test_constant_fold_reshape_infers_negative_dim() {
    // shape 含一个 -1 应被静态推导：input(1×6) + shape=[-1, 3] → target_shape=[2, 3]
    let shape_init = make_i64_tensor("shape_init", vec![-1, 3]);
    let input_vi = make_value_info("X", vec![1, 6]);
    let reshape_node = Node {
        input: vec!["X", "shape_init"],
        output: vec!["Y"],
        name: "reshape_infer",
        op_type: OpType::Reshape,
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![reshape_node],
            name: "reshape_infer_neg_one",
            initializer: vec![shape_init],
            input: vec![input_vi],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes).unwrap();
    let reshape = result.descriptor.nodes.iter().find(|n| n.name == "Y").unwrap();
    match &reshape.node_type {
        NodeTypeDescriptor::Reshape { target_shape } => {
            assert_eq!(target_shape, &vec![2usize, 3], "应静态推导 -1 → 2");
        }
        _ => panic!("expected Reshape"),
    }
}

#[test]
fn test_constant_fold_reshape_rejects_multiple_neg_one() {
    // 多个 -1 违反 ONNX 规范，应明确报错
    let shape_init = make_i64_tensor("shape_init", vec![-1, -1, 3]);
    let input_vi = make_value_info("X", vec![1, 6]);
    let reshape_node = Node {
        input: vec!["X", "shape_init"],
        output: vec!["Y"],
        name: "reshape_double_neg",
        op_type: OpType::Reshape,
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![reshape_node],
            name: "reshape_multi_neg",
            initializer: vec![shape_init],
            input: vec![input_vi],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let err = load_onnx_from_bytes(&bytes).unwrap_err();
    assert!(matches!(err, OnnxError::UnsupportedAttribute { .. }));
}

#[test]
fn test_constant_fold_reshape_keeps_zero_dim() {
    // shape 含 0 应保留 parent 对应位置维度：input(2×3) + shape=[0, -1] → [2, 3]
    let shape_init = make_i64_tensor("shape_init", vec![0, -1]);
    let input_vi = make_value_info("X", vec![2, 3]);
    let reshape_node = Node {
        input: vec!["X", "shape_init"],
        output: vec!["Y"],
        name: "reshape_zero",
        op_type: OpType::Reshape,
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![reshape_node],
            name: "reshape_zero_dim",
            initializer: vec![shape_init],
            input: vec![input_vi],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes).unwrap();
    let reshape = result.descriptor.nodes.iter().find(|n| n.name == "Y").unwrap();
    match &reshape.node_type {
        NodeTypeDescriptor::Reshape { target_shape } => {
            assert_eq!(target_shape, &vec![2usize, 3], "shape[0]=0 应保留 parent[0]=2");
        }
        _ => panic!("expected Reshape"),
    }
}

#[test]
fn test_constant_fold_resize_scales() {
    // X(1×3×4×4) → Resize(scales=[1,1,2,2], nearest) → Y(1×3×8×8)
    // 用空字符串 "" 占位 roi（ONNX 标准做法：可选输入未提供）
    let scales_init = make_f32_tensor("scales", vec![1.0, 1.0, 2.0, 2.0]);
    let input_vi = make_value_info("X", vec![1, 3, 4, 4]);
    let resize_node = Node {
        input: vec!["X", "", "scales"],
        output: vec!["Y"],
        name: "resize0",
        op_type: OpType::Resize,
        attribute: vec![Attribute {
            name: "mode",
            s: b"nearest",
            ..Default::default()
        }],
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![resize_node],
            name: "resize_scales",
            initializer: vec![scales_init],
            input: vec![input_vi],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes).unwrap();

    let resize = result.descriptor.nodes.iter().find(|n| n.name == "Y").unwrap();
    match &resize.node_type {
        NodeTypeDescriptor::Upsample2d { scale_h, scale_w } => {
            assert_eq!(*scale_h, 2);
            assert_eq!(*scale_w, 2);
        }
        _ => panic!("expected Upsample2d"),
    }

    assert!(result
        .import_report
        .rewritten
        .iter()
        .any(|r| r.pattern == "constant_fold_into_resize"));
}

#[test]
fn test_constant_fold_resize_rejects_non_integer_scale() {
    // scale=1.5 应被拒绝（only_torch 仅支持整数倍 nearest）
    let scales_init = make_f32_tensor("scales", vec![1.0, 1.0, 1.5, 1.5]);
    let input_vi = make_value_info("X", vec![1, 3, 4, 4]);
    let resize_node = Node {
        input: vec!["X", "", "scales"],
        output: vec!["Y"],
        name: "resize_bad",
        op_type: OpType::Resize,
        attribute: vec![Attribute {
            name: "mode",
            s: b"nearest",
            ..Default::default()
        }],
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![resize_node],
            name: "resize_bad_scale",
            initializer: vec![scales_init],
            input: vec![input_vi],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let err = load_onnx_from_bytes(&bytes).unwrap_err();
    assert!(matches!(err, OnnxError::UnsupportedAttribute { .. }));
}

#[test]
fn test_split_to_narrows_via_constant_input() {
    // X(1×6) → Split(axis=1, split=Constant[2,4]) → Y1(1×2), Y2(1×4)
    let split_init = make_i64_tensor("split_sizes", vec![2, 4]);
    let input_vi = make_value_info("X", vec![1, 6]);
    let split_node = Node {
        input: vec!["X", "split_sizes"],
        output: vec!["Y1", "Y2"],
        name: "split0",
        op_type: OpType::Split,
        attribute: vec![Attribute {
            name: "axis",
            i: 1,
            ..Default::default()
        }],
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![split_node],
            name: "split_test",
            initializer: vec![split_init],
            input: vec![input_vi],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes).unwrap();

    let y1 = result.descriptor.nodes.iter().find(|n| n.name == "Y1").unwrap();
    match &y1.node_type {
        NodeTypeDescriptor::Narrow { axis, start, length } => {
            assert_eq!(*axis, 1);
            assert_eq!(*start, 0);
            assert_eq!(*length, 2);
        }
        _ => panic!("expected Narrow for Y1"),
    }

    let y2 = result.descriptor.nodes.iter().find(|n| n.name == "Y2").unwrap();
    match &y2.node_type {
        NodeTypeDescriptor::Narrow { axis, start, length } => {
            assert_eq!(*axis, 1);
            assert_eq!(*start, 2);
            assert_eq!(*length, 4);
        }
        _ => panic!("expected Narrow for Y2"),
    }

    let record = result
        .import_report
        .rewritten
        .iter()
        .find(|r| r.pattern == "split_to_narrows")
        .expect("应有 split_to_narrows 记录");
    assert_eq!(record.produced_descriptor_nodes.len(), 2);
}

#[test]
fn test_split_to_narrows_via_attribute() {
    // 测试 opset ≤12 风格：split 来自 attribute
    let input_vi = make_value_info("X", vec![1, 9]);
    let split_node = Node {
        input: vec!["X"],
        output: vec!["Y1", "Y2", "Y3"],
        name: "split0",
        op_type: OpType::Split,
        attribute: vec![
            Attribute { name: "axis", i: 1, ..Default::default() },
            Attribute {
                name: "split",
                ints: vec![3, 3, 3],
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId { domain: "", version: 17 }],
        graph: Some(Graph {
            node: vec![split_node],
            name: "split_attr_test",
            input: vec![input_vi],
            ..Default::default()
        }),
        ..Default::default()
    };

    let bytes = onnx_rs::encode(&model);
    let result = load_onnx_from_bytes(&bytes).unwrap();

    let outputs: Vec<_> = ["Y1", "Y2", "Y3"]
        .iter()
        .map(|n| result.descriptor.nodes.iter().find(|nd| nd.name == *n).unwrap())
        .collect();
    let expected_starts = [0usize, 3, 6];
    for (i, out) in outputs.iter().enumerate() {
        match &out.node_type {
            NodeTypeDescriptor::Narrow { axis, start, length } => {
                assert_eq!(*axis, 1);
                assert_eq!(*start, expected_starts[i]);
                assert_eq!(*length, 3);
            }
            _ => panic!("expected Narrow for Y{}", i + 1),
        }
    }
}
