use approx::assert_abs_diff_eq;

use crate::nn::optimizer::{Optimizer, SGD};
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

// ============================================================
// 基础创建测试
// ============================================================

#[test]
fn test_node_leaky_relu_creation() {
    let mut graph = Graph::new();

    // 1. 测试 LeakyReLU 节点创建（默认 slope）
    {
        let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
        let relu = graph
            .new_leaky_relu_node(input, 0.1, Some("leaky_relu"))
            .unwrap();
        assert_eq!(graph.get_node_name(relu).unwrap(), "leaky_relu");
        assert_eq!(graph.get_node_parents(relu).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(relu).unwrap().len(), 0);
    }

    // 2. 测试标准 ReLU 节点创建（slope=0）
    {
        let input = graph.new_input_node(&[2, 2], Some("input2")).unwrap();
        let relu = graph.new_relu_node(input, Some("standard_relu")).unwrap();
        assert_eq!(graph.get_node_name(relu).unwrap(), "standard_relu");
    }
}

#[test]
fn test_node_leaky_relu_name_generation() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();

    // 1. 测试节点显式命名
    let relu = graph
        .new_leaky_relu_node(input, 0.1, Some("explicit_relu"))
        .unwrap();
    assert_eq!(graph.get_node_name(relu).unwrap(), "explicit_relu");

    // 2. 测试节点自动命名
    let relu2 = graph.new_leaky_relu_node(input, 0.1, None).unwrap();
    assert_eq!(graph.get_node_name(relu2).unwrap(), "leaky_relu_1");

    // 3. 测试节点名称重复
    let result = graph.new_leaky_relu_node(input, 0.1, Some("explicit_relu"));
    assert_eq!(
        result,
        Err(GraphError::DuplicateNodeName(
            "节点explicit_relu在图default_graph中重复".to_string()
        ))
    );
}

#[test]
fn test_node_leaky_relu_invalid_slope() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();

    // 负的 negative_slope 应该失败
    let result = graph.new_leaky_relu_node(input, -0.1, Some("invalid_relu"));
    assert!(result.is_err());
}

#[test]
fn test_node_leaky_relu_manually_set_value() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let relu = graph.new_leaky_relu_node(input, 0.1, Some("relu")).unwrap();

    // 直接设置 LeakyReLU 节点的值应该失败
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(
        graph.set_node_value(relu, Some(&test_value)),
        Err(GraphError::InvalidOperation(
            "节点[id=2, name=relu, type=LeakyReLU]的值只能通过前向传播计算得到，不能直接设置"
                .into()
        ))
    );
}

// ============================================================
// 前向传播测试
// ============================================================

#[test]
fn test_node_standard_relu_forward() {
    // 标准 ReLU (negative_slope=0) 前向传播测试
    // 预期值来自 tests/calc_jacobi_by_pytorch/node_leaky_relu.py
    let mut graph = Graph::new();

    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();
    let relu = graph.new_relu_node(input, Some("relu")).unwrap();

    // 输入: [[0.5, -1.0], [0.0, 2.0]]
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&input_value)).unwrap();
    graph.forward_node(relu).unwrap();

    // 预期输出: [[0.5, 0.0], [0.0, 2.0]]
    let expected = Tensor::new(&[0.5, 0.0, 0.0, 2.0], &[2, 2]);
    let result = graph.get_node_value(relu).unwrap().unwrap();
    assert_eq!(result, &expected);
}

#[test]
fn test_node_leaky_relu_forward() {
    // Leaky ReLU (negative_slope=0.1) 前向传播测试
    // 预期值来自 tests/calc_jacobi_by_pytorch/node_leaky_relu.py
    let mut graph = Graph::new();

    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();
    let relu = graph
        .new_leaky_relu_node(input, 0.1, Some("leaky_relu"))
        .unwrap();

    // 输入: [[0.5, -1.0], [0.0, 2.0]]
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&input_value)).unwrap();
    graph.forward_node(relu).unwrap();

    // 预期输出: [[0.5, -0.1], [0.0, 2.0]]
    let expected = Tensor::new(&[0.5, -0.1, 0.0, 2.0], &[2, 2]);
    let result = graph.get_node_value(relu).unwrap().unwrap();
    assert_eq!(result, &expected);
}

#[test]
fn test_node_leaky_relu_forward_3x2() {
    // 3x2 矩阵的前向传播测试
    let mut graph = Graph::new();

    let input = graph.new_input_node(&[3, 2], Some("input")).unwrap();
    let relu = graph
        .new_leaky_relu_node(input, 0.1, Some("leaky_relu"))
        .unwrap();

    // 输入: [[1.0, -2.0], [-0.5, 0.5], [3.0, -1.5]]
    let input_value = Tensor::new(&[1.0, -2.0, -0.5, 0.5, 3.0, -1.5], &[3, 2]);
    graph.set_node_value(input, Some(&input_value)).unwrap();
    graph.forward_node(relu).unwrap();

    // 预期输出: [[1.0, -0.2], [-0.05, 0.5], [3.0, -0.15]]
    let expected = Tensor::new(&[1.0, -0.2, -0.05, 0.5, 3.0, -0.15], &[3, 2]);
    let result = graph.get_node_value(relu).unwrap().unwrap();
    assert_eq!(result, &expected);
}

// ============================================================
// 反向传播测试（Jacobi 模式）
// ============================================================

#[test]
fn test_node_standard_relu_backward() {
    // 标准 ReLU 反向传播测试
    // 预期值来自 tests/calc_jacobi_by_pytorch/node_leaky_relu.py
    let mut graph = Graph::new();

    let parent = graph.new_parameter_node(&[2, 2], Some("parent")).unwrap();
    let result = graph.new_relu_node(parent, Some("result")).unwrap();

    // 输入: [[0.5, -1.0], [0.0, 2.0]]
    let parent_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(parent, Some(&parent_value)).unwrap();
    graph.forward_node(result).unwrap();

    // 反向传播
    graph.backward_nodes(&[parent], result).unwrap();
    let parent_jacobi = graph.get_node_jacobi(parent).unwrap().unwrap();

    // 预期 Jacobi 对角矩阵: diag([1, 0, 0, 1])
    #[rustfmt::skip]
    let expected_jacobi = Tensor::new(
        &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
    );
    assert_eq!(parent_jacobi, &expected_jacobi);
}

#[test]
fn test_node_leaky_relu_backward() {
    // Leaky ReLU (slope=0.1) 反向传播测试
    // 预期值来自 tests/calc_jacobi_by_pytorch/node_leaky_relu.py
    let mut graph = Graph::new();

    let parent = graph.new_parameter_node(&[2, 2], Some("parent")).unwrap();
    let result = graph
        .new_leaky_relu_node(parent, 0.1, Some("result"))
        .unwrap();

    // 输入: [[0.5, -1.0], [0.0, 2.0]]
    let parent_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(parent, Some(&parent_value)).unwrap();
    graph.forward_node(result).unwrap();

    // 反向传播
    graph.backward_nodes(&[parent], result).unwrap();
    let parent_jacobi = graph.get_node_jacobi(parent).unwrap().unwrap();

    // 预期 Jacobi 对角矩阵: diag([1.0, 0.1, 0.1, 1.0])
    #[rustfmt::skip]
    let expected_jacobi = Tensor::new(
        &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.1, 0.0, 0.0,
            0.0, 0.0, 0.1, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
    );
    assert_eq!(parent_jacobi, &expected_jacobi);
}

#[test]
fn test_node_leaky_relu_backward_all_positive() {
    // 全正值输入的反向传播测试
    let mut graph = Graph::new();

    let parent = graph.new_parameter_node(&[2, 2], Some("parent")).unwrap();
    let result = graph
        .new_leaky_relu_node(parent, 0.1, Some("result"))
        .unwrap();

    // 全正值输入: [[1.0, 2.0], [3.0, 4.0]]
    let parent_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    graph.set_node_value(parent, Some(&parent_value)).unwrap();
    graph.forward_node(result).unwrap();
    graph.backward_nodes(&[parent], result).unwrap();

    let parent_jacobi = graph.get_node_jacobi(parent).unwrap().unwrap();

    // 全正值时 Jacobi 应该是单位矩阵
    let expected_jacobi = Tensor::eyes(4);
    assert_eq!(parent_jacobi, &expected_jacobi);
}

#[test]
fn test_node_leaky_relu_backward_all_negative() {
    // 全负值输入的反向传播测试
    let mut graph = Graph::new();

    let parent = graph.new_parameter_node(&[2, 2], Some("parent")).unwrap();
    let result = graph
        .new_leaky_relu_node(parent, 0.1, Some("result"))
        .unwrap();

    // 全负值输入: [[-1.0, -2.0], [-3.0, -4.0]]
    let parent_value = Tensor::new(&[-1.0, -2.0, -3.0, -4.0], &[2, 2]);
    graph.set_node_value(parent, Some(&parent_value)).unwrap();
    graph.forward_node(result).unwrap();
    graph.backward_nodes(&[parent], result).unwrap();

    let parent_jacobi = graph.get_node_jacobi(parent).unwrap().unwrap();

    // 全负值时 Jacobi 应该是 0.1 * 单位矩阵
    let expected_jacobi = &Tensor::eyes(4) * 0.1;
    assert_eq!(parent_jacobi, &expected_jacobi);
}

#[test]
fn test_node_leaky_relu_jacobi_accumulation() {
    // 测试 Jacobi 累积
    let mut graph = Graph::new();

    let parent = graph.new_parameter_node(&[2, 2], Some("parent")).unwrap();
    let result = graph
        .new_leaky_relu_node(parent, 0.1, Some("result"))
        .unwrap();

    let parent_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(parent, Some(&parent_value)).unwrap();
    graph.forward_node(result).unwrap();

    // 第一次反向传播
    graph.backward_nodes(&[parent], result).unwrap();
    #[rustfmt::skip]
    let expected_jacobi = Tensor::new(
        &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.1, 0.0, 0.0,
            0.0, 0.0, 0.1, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
    );

    // 第二次反向传播 - Jacobi 应该累积
    graph.backward_nodes(&[parent], result).unwrap();
    let parent_jacobi = graph.get_node_jacobi(parent).unwrap().unwrap();
    assert_eq!(parent_jacobi, &(&expected_jacobi * 2.0));

    // 清除后再次反向传播
    graph.clear_jacobi().unwrap();
    graph.backward_nodes(&[parent], result).unwrap();
    let parent_jacobi_after_clear = graph.get_node_jacobi(parent).unwrap().unwrap();
    assert_eq!(parent_jacobi_after_clear, &expected_jacobi);
}

// ============================================================
// 端到端网络测试
// ============================================================

#[test]
fn test_node_leaky_relu_simple_network() {
    // 简单网络: z -> leaky_relu -> output
    let mut graph = Graph::new_with_seed(42);

    let z = graph.new_parameter_node(&[1, 1], Some("z")).unwrap();
    let output = graph.new_leaky_relu_node(z, 0.1, Some("output")).unwrap();

    // 正值输入
    let z_value = Tensor::new(&[0.5], &[1, 1]);
    graph.set_node_value(z, Some(&z_value)).unwrap();
    graph.forward_node(output).unwrap();

    let output_val = graph.get_node_value(output).unwrap().unwrap()[[0, 0]];
    assert_abs_diff_eq!(output_val, 0.5, epsilon = 1e-10);

    // 负值输入
    let z_value_neg = Tensor::new(&[-2.0], &[1, 1]);
    graph.set_node_value(z, Some(&z_value_neg)).unwrap();
    graph.forward_node(output).unwrap();

    let output_val_neg = graph.get_node_value(output).unwrap().unwrap()[[0, 0]];
    assert_abs_diff_eq!(output_val_neg, -0.2, epsilon = 1e-10);

    // 反向传播
    graph.backward_nodes(&[z], output).unwrap();
    let z_jacobi = graph.get_node_jacobi(z).unwrap().unwrap();
    assert_eq!(z_jacobi.shape(), &[1, 1]);

    // 优化器测试
    let mut optimizer = SGD::new(&graph, 0.1).unwrap();
    optimizer.update(&mut graph).unwrap();
}

#[test]
fn test_node_leaky_relu_after_linear() {
    // 线性层后接 LeakyReLU: output = leaky_relu(w @ x + b)
    let mut graph = Graph::new_with_seed(42);

    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[1, 2], Some("w")).unwrap();
    let b = graph.new_parameter_node(&[1, 1], Some("b")).unwrap();

    let wx = graph.new_mat_mul_node(w, x, Some("wx")).unwrap();
    let z = graph.new_add_node(&[wx, b], Some("z")).unwrap();
    let output = graph.new_leaky_relu_node(z, 0.1, Some("output")).unwrap();

    // 设置输入
    let input_tensor = Tensor::new(&[1.0, -0.5], &[2, 1]);
    graph.set_node_value(x, Some(&input_tensor)).unwrap();
    graph.forward_node(output).unwrap();

    // 验证输出存在
    let output_value = graph.get_node_value(output).unwrap().unwrap();
    assert_eq!(output_value.shape(), &[1, 1]);

    // 反向传播
    graph.backward_nodes(&[w, b], output).unwrap();
    let w_jacobi = graph.get_node_jacobi(w).unwrap().unwrap();
    let b_jacobi = graph.get_node_jacobi(b).unwrap().unwrap();
    assert_eq!(w_jacobi.shape(), &[1, 2]);
    assert_eq!(b_jacobi.shape(), &[1, 1]);

    // 优化器测试
    let mut optimizer = SGD::new(&graph, 0.1).unwrap();
    optimizer.update(&mut graph).unwrap();
}

// ============================================================
// Batch 模式测试（2D 张量: batch_size x features）
// ============================================================

#[test]
fn test_node_leaky_relu_batch_forward_2d() {
    // Batch 前向传播测试（2D 张量）
    let mut graph = Graph::new();

    // 输入形状: batch_size=2, features=4
    let input = graph.new_input_node(&[2, 4], Some("input")).unwrap();
    let relu = graph
        .new_leaky_relu_node(input, 0.1, Some("leaky_relu"))
        .unwrap();

    // Batch 输入: 2 个样本，每个 4 个特征
    // 样本1: [0.5, -1.0, 0.0, 2.0]
    // 样本2: [-0.5, 1.0, -2.0, 0.5]
    let batch_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0, -0.5, 1.0, -2.0, 0.5], &[2, 4]);
    graph.set_node_value(input, Some(&batch_value)).unwrap();
    graph.forward_node(relu).unwrap();

    let result = graph.get_node_value(relu).unwrap().unwrap();
    // 样本1: [0.5, -0.1, 0.0, 2.0]
    // 样本2: [-0.05, 1.0, -0.2, 0.5]
    let expected = Tensor::new(&[0.5, -0.1, 0.0, 2.0, -0.05, 1.0, -0.2, 0.5], &[2, 4]);
    assert_eq!(result, &expected);
}

/// 测试 LeakyReLU 在 MLP 网络中的端到端训练
#[test]
fn test_node_leaky_relu_mlp_training() {
    let mut graph = Graph::new_with_seed(42);

    // 构建简单 MLP: x -> Linear -> LeakyReLU -> Linear -> output
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w1 = graph.new_parameter_node(&[3, 2], Some("w1")).unwrap();
    let b1 = graph.new_parameter_node(&[3, 1], Some("b1")).unwrap();

    let z1 = graph.new_mat_mul_node(w1, x, Some("z1")).unwrap(); // [3, 1]
    let h1 = graph.new_add_node(&[z1, b1], Some("h1")).unwrap(); // [3, 1]
    let a1 = graph.new_leaky_relu_node(h1, 0.1, Some("a1")).unwrap(); // LeakyReLU

    let w2 = graph.new_parameter_node(&[1, 3], Some("w2")).unwrap();
    let output = graph.new_mat_mul_node(w2, a1, Some("output")).unwrap(); // [1, 1]

    // 设置输入
    let input_value = Tensor::new(&[1.0, -0.5], &[2, 1]);
    graph.set_node_value(x, Some(&input_value)).unwrap();

    // 前向传播
    graph.forward_node(output).unwrap();
    let output_val = graph.get_node_value(output).unwrap().unwrap();
    assert_eq!(output_val.shape(), &[1, 1]);

    // 反向传播
    graph.backward_nodes(&[w1, b1, w2], output).unwrap();

    // 验证所有参数都有梯度
    assert!(graph.get_node_jacobi(w1).unwrap().is_some());
    assert!(graph.get_node_jacobi(b1).unwrap().is_some());
    assert!(graph.get_node_jacobi(w2).unwrap().is_some());

    // 优化器更新
    let mut optimizer = SGD::new(&graph, 0.01).unwrap();
    optimizer.update(&mut graph).unwrap();
}
