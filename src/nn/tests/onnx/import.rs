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
