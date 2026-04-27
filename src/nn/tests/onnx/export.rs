use std::collections::HashMap;

use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
use crate::nn::graph::onnx_export::{EXPORT_OPSET_VERSION, export_to_bytes, save_onnx};
use crate::tensor::Tensor;

fn build_simple_mlp() -> (GraphDescriptor, HashMap<String, Tensor>) {
    let mut desc = GraphDescriptor::new("test_mlp");

    desc.add_node(NodeDescriptor::new(
        1,
        "input",
        NodeTypeDescriptor::BasicInput,
        vec![0, 2],
        Some(vec![None, Some(2)]),
        vec![],
    ));
    desc.add_node(NodeDescriptor::new(
        2,
        "W1",
        NodeTypeDescriptor::Parameter,
        vec![2, 4],
        None,
        vec![],
    ));
    desc.add_node(NodeDescriptor::new(
        3,
        "b1",
        NodeTypeDescriptor::Parameter,
        vec![1, 4],
        None,
        vec![],
    ));
    desc.add_node(NodeDescriptor::new(
        4,
        "matmul_out",
        NodeTypeDescriptor::MatMul,
        vec![0, 4],
        None,
        vec![1, 2],
    ));
    desc.add_node(NodeDescriptor::new(
        5,
        "add_out",
        NodeTypeDescriptor::Add,
        vec![0, 4],
        None,
        vec![4, 3],
    ));
    desc.add_node(NodeDescriptor::new(
        6,
        "relu_out",
        NodeTypeDescriptor::ReLU,
        vec![0, 4],
        None,
        vec![5],
    ));

    let mut weights = HashMap::new();
    weights.insert("W1".to_string(), Tensor::new(&vec![0.1; 8], &[2, 4]));
    weights.insert("b1".to_string(), Tensor::new(&vec![0.0; 4], &[1, 4]));

    (desc, weights)
}

#[test]
fn test_export_basic_roundtrip() {
    let (desc, weights) = build_simple_mlp();
    let bytes = export_to_bytes(&desc, &weights).unwrap();

    assert!(!bytes.is_empty());

    let model = onnx_rs::parse(&bytes).unwrap();
    assert_eq!(model.producer_name, "only_torch");
    assert!(model.graph.is_some());

    let graph = model.graph.as_ref().unwrap();
    assert_eq!(graph.name, "test_mlp");
}

#[test]
fn test_export_has_correct_initializers() {
    let (desc, weights) = build_simple_mlp();
    let bytes = export_to_bytes(&desc, &weights).unwrap();
    let model = onnx_rs::parse(&bytes).unwrap();
    let graph = model.graph.as_ref().unwrap();

    assert_eq!(graph.initializer.len(), 2);

    let w1 = graph.initializer.iter().find(|t| t.name() == "W1").unwrap();
    assert_eq!(w1.dims(), &[2, 4]);
    let w1_data = w1.as_f32().unwrap();
    assert_eq!(w1_data.len(), 8);
    for &v in w1_data.iter() {
        assert!((v - 0.1).abs() < 1e-5);
    }
}

#[test]
fn test_export_has_correct_inputs() {
    let (desc, weights) = build_simple_mlp();
    let bytes = export_to_bytes(&desc, &weights).unwrap();
    let model = onnx_rs::parse(&bytes).unwrap();
    let graph = model.graph.as_ref().unwrap();

    assert_eq!(graph.input.len(), 1);
    assert_eq!(graph.input[0].name, "input");
}

#[test]
fn test_export_has_correct_outputs() {
    let (desc, weights) = build_simple_mlp();
    let bytes = export_to_bytes(&desc, &weights).unwrap();
    let model = onnx_rs::parse(&bytes).unwrap();
    let graph = model.graph.as_ref().unwrap();

    assert_eq!(graph.output.len(), 1);
    assert_eq!(graph.output[0].name, "relu_out");
}

#[test]
fn test_export_has_correct_nodes() {
    let (desc, weights) = build_simple_mlp();
    let bytes = export_to_bytes(&desc, &weights).unwrap();
    let model = onnx_rs::parse(&bytes).unwrap();
    let graph = model.graph.as_ref().unwrap();

    assert_eq!(graph.node.len(), 3);

    let op_types: Vec<_> = graph
        .node
        .iter()
        .map(|n| format!("{:?}", n.op_type))
        .collect();
    assert!(op_types.iter().any(|o| o.contains("MatMul")));
    assert!(op_types.iter().any(|o| o.contains("Add")));
    assert!(op_types.iter().any(|o| o.contains("Relu")));
}

#[test]
fn test_export_node_connectivity() {
    let (desc, weights) = build_simple_mlp();
    let bytes = export_to_bytes(&desc, &weights).unwrap();
    let model = onnx_rs::parse(&bytes).unwrap();
    let graph = model.graph.as_ref().unwrap();

    let matmul = graph
        .node
        .iter()
        .find(|n| n.op_type == onnx_rs::ast::OpType::MatMul)
        .unwrap();
    assert!(matmul.input.contains(&"input"));
    assert!(matmul.input.contains(&"W1"));

    let add = graph
        .node
        .iter()
        .find(|n| n.op_type == onnx_rs::ast::OpType::Add)
        .unwrap();
    assert!(add.input.contains(&"matmul_out"));
    assert!(add.input.contains(&"b1"));

    let relu = graph
        .node
        .iter()
        .find(|n| n.op_type == onnx_rs::ast::OpType::Relu)
        .unwrap();
    assert!(relu.input.contains(&"add_out"));
}

#[test]
fn test_export_training_nodes_filtered() {
    let mut desc = GraphDescriptor::new("train_model");
    desc.add_node(NodeDescriptor::new(
        1,
        "input",
        NodeTypeDescriptor::BasicInput,
        vec![0, 2],
        None,
        vec![],
    ));
    desc.add_node(NodeDescriptor::new(
        2,
        "target",
        NodeTypeDescriptor::TargetInput,
        vec![0, 1],
        None,
        vec![],
    ));
    desc.add_node(NodeDescriptor::new(
        3,
        "relu",
        NodeTypeDescriptor::ReLU,
        vec![0, 2],
        None,
        vec![1],
    ));
    desc.add_node(NodeDescriptor::new(
        4,
        "loss",
        NodeTypeDescriptor::SoftmaxCrossEntropy,
        vec![1],
        None,
        vec![3, 2],
    ));

    let bytes = export_to_bytes(&desc, &HashMap::new()).unwrap();
    let model = onnx_rs::parse(&bytes).unwrap();
    let graph = model.graph.as_ref().unwrap();

    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.input.len(), 1);
}

#[test]
fn test_export_unsupported_node_error() {
    let mut desc = GraphDescriptor::new("bad_model");
    desc.add_node(NodeDescriptor::new(
        1,
        "input",
        NodeTypeDescriptor::BasicInput,
        vec![0, 2],
        None,
        vec![],
    ));
    desc.add_node(NodeDescriptor::new(
        2,
        "swish",
        NodeTypeDescriptor::Swish,
        vec![0, 2],
        None,
        vec![1],
    ));

    let result = export_to_bytes(&desc, &HashMap::new());
    assert!(result.is_err());
}

#[test]
fn test_export_conv2d_with_attributes() {
    let mut desc = GraphDescriptor::new("conv_model");
    desc.add_node(NodeDescriptor::new(
        1,
        "img",
        NodeTypeDescriptor::BasicInput,
        vec![1, 1, 28, 28],
        None,
        vec![],
    ));
    desc.add_node(NodeDescriptor::new(
        2,
        "conv_w",
        NodeTypeDescriptor::Parameter,
        vec![8, 1, 3, 3],
        None,
        vec![],
    ));
    desc.add_node(NodeDescriptor::new(
        3,
        "conv_out",
        NodeTypeDescriptor::Conv2d {
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
        },
        vec![1, 8, 28, 28],
        None,
        vec![1, 2],
    ));

    let mut weights = HashMap::new();
    weights.insert(
        "conv_w".to_string(),
        Tensor::new(&vec![0.1; 72], &[8, 1, 3, 3]),
    );

    let bytes = export_to_bytes(&desc, &weights).unwrap();
    let model = onnx_rs::parse(&bytes).unwrap();
    let graph = model.graph.as_ref().unwrap();

    let conv = &graph.node[0];
    assert_eq!(conv.op_type, onnx_rs::ast::OpType::Conv);

    let strides = conv.attribute.iter().find(|a| a.name == "strides").unwrap();
    assert_eq!(strides.ints, vec![1, 1]);

    let pads = conv.attribute.iter().find(|a| a.name == "pads").unwrap();
    assert_eq!(pads.ints, vec![1, 1, 1, 1]);
}

#[test]
fn test_export_leaky_relu_alpha() {
    let mut desc = GraphDescriptor::new("lrelu_model");
    desc.add_node(NodeDescriptor::new(
        1,
        "x",
        NodeTypeDescriptor::BasicInput,
        vec![0, 10],
        None,
        vec![],
    ));
    desc.add_node(NodeDescriptor::new(
        2,
        "lrelu",
        NodeTypeDescriptor::LeakyReLU { alpha: 0.2 },
        vec![0, 10],
        None,
        vec![1],
    ));

    let bytes = export_to_bytes(&desc, &HashMap::new()).unwrap();
    let model = onnx_rs::parse(&bytes).unwrap();
    let graph = model.graph.as_ref().unwrap();

    let lrelu = &graph.node[0];
    let alpha_attr = lrelu.attribute.iter().find(|a| a.name == "alpha").unwrap();
    assert!((alpha_attr.f - 0.2).abs() < 1e-5);
}

#[test]
fn test_export_opset_version() {
    let (desc, weights) = build_simple_mlp();
    let bytes = export_to_bytes(&desc, &weights).unwrap();
    let model = onnx_rs::parse(&bytes).unwrap();

    let opset = model
        .opset_import
        .iter()
        .find(|o| o.domain.is_empty())
        .unwrap();
    assert_eq!(opset.version, EXPORT_OPSET_VERSION);
}

#[test]
fn test_export_empty_graph() {
    let mut desc = GraphDescriptor::new("empty");
    desc.add_node(NodeDescriptor::new(
        1,
        "input",
        NodeTypeDescriptor::BasicInput,
        vec![0, 2],
        None,
        vec![],
    ));

    let bytes = export_to_bytes(&desc, &HashMap::new()).unwrap();
    let model = onnx_rs::parse(&bytes).unwrap();
    let graph = model.graph.as_ref().unwrap();

    assert_eq!(graph.node.len(), 0);
    assert_eq!(graph.input.len(), 1);
}

#[test]
fn test_export_import_roundtrip_structure() {
    let (desc, weights) = build_simple_mlp();
    let bytes = export_to_bytes(&desc, &weights).unwrap();

    let import_result = crate::nn::graph::onnx_import::load_onnx_from_bytes(&bytes).unwrap();
    let reimported = &import_result.descriptor;

    let orig_inputs = desc
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::BasicInput))
        .count();
    let re_inputs = reimported
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::BasicInput))
        .count();
    assert_eq!(orig_inputs, re_inputs);

    let orig_params = desc
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::Parameter))
        .count();
    let re_params = reimported
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::Parameter))
        .count();
    assert_eq!(orig_params, re_params);

    let orig_relu = desc
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::ReLU))
        .count();
    let re_relu = reimported
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::ReLU))
        .count();
    assert_eq!(orig_relu, re_relu);

    assert_eq!(import_result.weights.len(), 2);
}

#[test]
fn test_roundtrip_numerical_consistency() {
    use crate::nn::Init;
    use crate::nn::VarActivationOps;
    use crate::nn::VarMatrixOps;
    use crate::nn::graph::Graph;
    use crate::tensor::Tensor;

    let graph = Graph::new();
    let test_input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let input = graph.input(&test_input).unwrap();

    let w1 = graph.parameter(&[4, 8], Init::Constant(0.1), "w1").unwrap();
    let b1 = graph.parameter(&[1, 8], Init::Zeros, "b1").unwrap();
    let matmul1 = input.matmul(&w1).unwrap();
    let add1 = &matmul1 + &b1;
    let relu = add1.relu();

    let w2 = graph
        .parameter(&[8, 2], Init::Constant(0.05), "w2")
        .unwrap();
    let b2 = graph.parameter(&[1, 2], Init::Zeros, "b2").unwrap();
    let matmul2 = relu.matmul(&w2).unwrap();
    let output = &matmul2 + &b2;

    graph.forward(&output).unwrap();
    let original_output = output.value().unwrap().unwrap();

    let onnx_bytes = graph.export_onnx_bytes(&[&output]).unwrap();

    let reimported = Graph::from_onnx_bytes(&onnx_bytes).unwrap();
    assert_eq!(reimported.inputs.len(), 1);
    assert!(!reimported.outputs.is_empty());

    reimported.inputs[0].1.set_value(&test_input).unwrap();
    reimported.graph.forward(&reimported.outputs[0]).unwrap();
    let reimported_output = reimported.outputs[0].value().unwrap().unwrap();

    assert_eq!(original_output.shape(), reimported_output.shape());
    let orig_flat = original_output.flatten_view();
    let reimp_flat = reimported_output.flatten_view();
    assert_eq!(orig_flat.len(), reimp_flat.len());
    for (i, (&a, &b)) in orig_flat.iter().zip(reimp_flat.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "输出元素 [{}] 不一致: 原始={}, 重导入={}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_pytorch_cross_validation() {
    use crate::nn::graph::Graph;
    use crate::tensor::Tensor;

    let onnx_path = std::path::Path::new("tests/fixtures/pytorch_mlp.onnx");
    if !onnx_path.exists() {
        eprintln!("跳过 PyTorch 交叉验证（tests/fixtures/pytorch_mlp.onnx 不存在）");
        return;
    }

    let result = Graph::from_onnx(onnx_path).unwrap();

    assert_eq!(result.inputs.len(), 1);
    assert!(!result.outputs.is_empty());

    let test_input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    result.inputs[0].1.set_value(&test_input).unwrap();
    result.graph.forward(&result.outputs[0]).unwrap();

    let output = result.outputs[0].value().unwrap().unwrap();
    let flat = output.flatten_view();

    let expected = [0.4f32, 0.4];
    assert_eq!(flat.len(), expected.len(), "输出维度不一致");
    for (i, (&got, &want)) in flat.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "PyTorch 交叉验证失败：元素 [{}] got={}, want={}",
            i,
            got,
            want
        );
    }
}

#[test]
fn test_save_onnx_file() {
    let (desc, weights) = build_simple_mlp();
    let tmp_dir = std::env::temp_dir();
    let path = tmp_dir.join("test_export_only_torch.onnx");

    save_onnx(&path, &desc, &weights).unwrap();

    let metadata = std::fs::metadata(&path).unwrap();
    assert!(metadata.len() > 0);

    let bytes = std::fs::read(&path).unwrap();
    let model = onnx_rs::parse(&bytes).unwrap();
    assert!(model.graph.is_some());

    let _ = std::fs::remove_file(&path);
}
