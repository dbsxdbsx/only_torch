use approx::assert_abs_diff_eq;

use crate::nn::optimizer::{Optimizer, SGD};
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

/// SoftPlus 节点测试
///
/// 预期值来自 tests/python/calc_jacobi_by_pytorch/node_softplus.py
///
/// 注意：本框架要求所有张量为 2-4 维，所以 "1D" 测试使用 [1, n] 形状

// ==================== 节点创建测试 ====================

#[test]
fn test_node_softplus_creation() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[1, 3], Some("input")).unwrap();

    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();

    assert_eq!(graph.get_node_name(softplus).unwrap(), "softplus");
    assert_eq!(graph.get_node_parents(softplus).unwrap().len(), 1);
    assert_eq!(graph.get_node_children(softplus).unwrap().len(), 0);
}

#[test]
fn test_node_softplus_name_generation() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[1, 3], Some("input")).unwrap();

    // 显式命名
    let sp1 = graph
        .new_softplus_node(input, Some("explicit_softplus"))
        .unwrap();
    assert_eq!(graph.get_node_name(sp1).unwrap(), "explicit_softplus");

    // 自动命名
    let sp2 = graph.new_softplus_node(input, None).unwrap();
    assert_eq!(graph.get_node_name(sp2).unwrap(), "softplus_1");

    // 重复命名应失败
    let result = graph.new_softplus_node(input, Some("explicit_softplus"));
    assert_eq!(
        result,
        Err(GraphError::DuplicateNodeName(
            "节点explicit_softplus在图default_graph中重复".to_string()
        ))
    );
}

#[test]
fn test_node_softplus_manually_set_value() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[1, 3], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();

    // 直接设置 SoftPlus 节点的值应该失败
    let test_value = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    assert_eq!(
        graph.set_node_value(softplus, Some(&test_value)),
        Err(GraphError::InvalidOperation(
            "节点[id=2, name=softplus, type=SoftPlus]的值只能通过前向传播计算得到，不能直接设置"
                .into()
        ))
    );
}

// ==================== 前向传播测试 ====================

#[test]
fn test_node_softplus_forward_1d() {
    // "1D" 向量前向传播测试 (实际使用 [1, 5] 形状)
    // 预期值来自 tests/python/calc_jacobi_by_pytorch/node_softplus.py
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[1, 5], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();

    // 设置输入: [-2.0, -1.0, 0.0, 1.0, 2.0]
    let input_data = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5]);
    graph.set_node_value(input, Some(&input_data)).unwrap();
    graph.forward_node(softplus).unwrap();

    let result = graph.get_node_value(softplus).unwrap().unwrap();

    // 预期输出 (来自 PyTorch)
    let expected = [0.12692800, 0.31326169, 0.69314718, 1.31326163, 2.12692809];
    for i in 0..5 {
        assert_abs_diff_eq!(result[[0, i]], expected[i], epsilon = 1e-5);
    }
}

#[test]
fn test_node_softplus_forward_2d() {
    // 2D 矩阵前向传播测试
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();

    // 设置输入: [[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]]
    let input_data = Tensor::new(&[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]);
    graph.set_node_value(input, Some(&input_data)).unwrap();
    graph.forward_node(softplus).unwrap();

    let result = graph.get_node_value(softplus).unwrap().unwrap();

    // 预期输出 (来自 PyTorch)
    // Row 0: softplus([-1.0, 0.0, 1.0]) = [0.31326169, 0.69314718, 1.31326163]
    // Row 1: softplus([2.0, -2.0, 0.5]) = [2.12692809, 0.12692800, 0.97407699]
    assert_abs_diff_eq!(result[[0, 0]], 0.31326169, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[0, 1]], 0.69314718, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[0, 2]], 1.31326163, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[1, 0]], 2.12692809, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[1, 1]], 0.12692800, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[1, 2]], 0.97407699, epsilon = 1e-5);
}

#[test]
fn test_node_softplus_numerical_stability() {
    // 数值稳定性测试：极端值
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[1, 5], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();

    // 设置极端输入: [-50.0, -20.0, 0.0, 20.0, 50.0]
    let input_data = Tensor::new(&[-50.0, -20.0, 0.0, 20.0, 50.0], &[1, 5]);
    graph.set_node_value(input, Some(&input_data)).unwrap();
    graph.forward_node(softplus).unwrap();

    let result = graph.get_node_value(softplus).unwrap().unwrap();

    // 大负数时 softplus(x) ≈ 0
    assert!(result[[0, 0]] < 1e-10, "softplus(-50) should be ≈ 0");
    assert!(result[[0, 1]] < 1e-5, "softplus(-20) should be ≈ 0");

    // softplus(0) = ln(2) ≈ 0.693
    assert_abs_diff_eq!(result[[0, 2]], 0.69314718, epsilon = 1e-5);

    // 大正数时 softplus(x) ≈ x
    assert_abs_diff_eq!(result[[0, 3]], 20.0, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[0, 4]], 50.0, epsilon = 1e-5);
}

// ==================== 反向传播测试（单样本 Jacobi 模式）====================

#[test]
fn test_node_softplus_backward_1d() {
    // "1D" 向量反向传播测试 (实际使用 [1, 5] 形状)
    // 预期值来自 tests/python/calc_jacobi_by_pytorch/node_softplus.py
    let mut graph = Graph::new();
    let parent = graph.new_parameter_node(&[1, 5], Some("parent")).unwrap();
    let result_node = graph.new_softplus_node(parent, Some("result")).unwrap();

    // 设置输入: [-2.0, -1.0, 0.0, 1.0, 2.0]
    let input_data = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5]);
    graph.set_node_value(parent, Some(&input_data)).unwrap();
    graph.forward_node(result_node).unwrap();

    // 反向传播
    graph.backward_nodes(&[parent], result_node).unwrap();
    let parent_jacobi = graph.get_node_jacobi(parent).unwrap().unwrap();

    // SoftPlus 的导数是 sigmoid(x)，Jacobian 是对角矩阵
    // 对角元素: [0.11920292, 0.26894143, 0.5, 0.73105860, 0.88079708]
    let expected_diag = [0.11920292, 0.26894143, 0.5, 0.73105860, 0.88079708];

    // 验证对角矩阵
    assert_eq!(parent_jacobi.shape(), &[5, 5]);
    for i in 0..5 {
        for j in 0..5 {
            if i == j {
                assert_abs_diff_eq!(parent_jacobi[[i, j]], expected_diag[i], epsilon = 1e-5);
            } else {
                assert_abs_diff_eq!(parent_jacobi[[i, j]], 0.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_node_softplus_backward_2d() {
    // 2D 矩阵反向传播测试
    let mut graph = Graph::new();
    let parent = graph.new_parameter_node(&[2, 3], Some("parent")).unwrap();
    let result_node = graph.new_softplus_node(parent, Some("result")).unwrap();

    // 设置输入: [[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]]
    let input_data = Tensor::new(&[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]);
    graph.set_node_value(parent, Some(&input_data)).unwrap();
    graph.forward_node(result_node).unwrap();

    // 反向传播
    graph.backward_nodes(&[parent], result_node).unwrap();
    let parent_jacobi = graph.get_node_jacobi(parent).unwrap().unwrap();

    // 预期对角元素 (来自 PyTorch): [0.26894143, 0.5, 0.73105860, 0.88079709, 0.11920292, 0.62245935]
    let expected_diag = [
        0.26894143, 0.5, 0.73105860, 0.88079709, 0.11920292, 0.62245935,
    ];

    // 验证对角矩阵 (6x6)
    assert_eq!(parent_jacobi.shape(), &[6, 6]);
    for i in 0..6 {
        assert_abs_diff_eq!(parent_jacobi[[i, i]], expected_diag[i], epsilon = 1e-5);
    }
}

// ==================== Batch 模式测试 ====================

#[test]
fn test_node_softplus_batch_forward() {
    // Batch 前向传播测试
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();

    // Batch 输入 (2 samples, 3 features)
    let input_data = Tensor::new(&[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]);
    graph.set_node_value(input, Some(&input_data)).unwrap();
    graph.forward_node(softplus).unwrap();

    let result = graph.get_node_value(softplus).unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 3]);

    // 预期输出
    assert_abs_diff_eq!(result[[0, 0]], 0.31326169, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[1, 0]], 2.12692809, epsilon = 1e-5);
}

#[test]
fn test_node_softplus_batch_backward() {
    // Batch 反向传播测试（Gradient-based）
    // 需要构建完整的网络到标量损失
    let mut graph = Graph::new();
    let input = graph.new_parameter_node(&[2, 3], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();
    // 添加一个简单的求和操作来得到标量损失
    let ones = graph.new_parameter_node(&[3, 1], Some("ones")).unwrap();
    let sum_rows = graph
        .new_mat_mul_node(softplus, ones, Some("sum_rows"))
        .unwrap(); // [2, 1]
    let ones2 = graph.new_parameter_node(&[1, 2], Some("ones2")).unwrap();
    let loss = graph
        .new_mat_mul_node(ones2, sum_rows, Some("loss"))
        .unwrap(); // [1, 1]

    // 设置输入
    let input_data = Tensor::new(&[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]);
    graph.set_node_value(input, Some(&input_data)).unwrap();
    graph
        .set_node_value(ones, Some(&Tensor::ones(&[3, 1])))
        .unwrap();
    graph
        .set_node_value(ones2, Some(&Tensor::ones(&[1, 2])))
        .unwrap();

    // 前向传播
    graph.forward_node(loss).unwrap();

    // Batch 反向传播
    graph.backward_batch(loss).unwrap();

    // 获取 input 的梯度
    let grad = graph.get_node_grad_batch(input).unwrap().unwrap();
    assert_eq!(grad.shape(), &[2, 3]);

    // 预期梯度 = sigmoid(x) (因为下游都是 ones)
    // Row 0: sigmoid([-1.0, 0.0, 1.0]) = [0.26894143, 0.5, 0.73105860]
    // Row 1: sigmoid([2.0, -2.0, 0.5]) = [0.88079709, 0.11920292, 0.62245935]
    assert_abs_diff_eq!(grad[[0, 0]], 0.26894143, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 0.5, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 2]], 0.73105860, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0]], 0.88079709, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1]], 0.11920292, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 2]], 0.62245935, epsilon = 1e-5);
}

// ==================== 链式传播测试 ====================

#[test]
fn test_node_softplus_simple_network() {
    // 简单网络: z -> softplus -> output
    let mut graph = Graph::new();
    let z = graph.new_parameter_node(&[1, 3], Some("z")).unwrap();
    let output = graph.new_softplus_node(z, Some("output")).unwrap();

    // 设置输入
    let z_value = Tensor::new(&[-1.0, 0.0, 1.0], &[1, 3]);
    graph.set_node_value(z, Some(&z_value)).unwrap();

    // 前向传播
    graph.forward_node(output).unwrap();

    // 验证输出
    let result = graph.get_node_value(output).unwrap().unwrap();
    assert_abs_diff_eq!(result[[0, 0]], 0.31326169, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[0, 1]], 0.69314718, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[0, 2]], 1.31326163, epsilon = 1e-5);

    // 反向传播
    graph.backward_nodes(&[z], output).unwrap();
    let jacobi = graph.get_node_jacobi(z).unwrap().unwrap();

    // 验证 Jacobian 对角元素 = sigmoid(z)
    let expected_diag = [0.26894143, 0.5, 0.73105860];
    for i in 0..3 {
        assert_abs_diff_eq!(jacobi[[i, i]], expected_diag[i], epsilon = 1e-5);
    }
}

#[test]
fn test_node_softplus_after_linear() {
    // 线性层后接 SoftPlus: output = softplus(x @ w)
    let mut graph = Graph::new();

    // 输入 x: [1, 2] (shape: [1, 2])
    let x = graph.new_input_node(&[1, 2], Some("x")).unwrap();
    // 权重 w: [[0.5, -0.5], [0.3, 0.7]] (shape: [2, 2])
    let w = graph.new_parameter_node(&[2, 2], Some("w")).unwrap();
    // z = x @ w
    let z = graph.new_mat_mul_node(x, w, Some("z")).unwrap();
    // output = softplus(z)
    let output = graph.new_softplus_node(z, Some("output")).unwrap();

    // 设置值
    let x_value = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let w_value = Tensor::new(&[0.5, -0.5, 0.3, 0.7], &[2, 2]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(w, Some(&w_value)).unwrap();

    // 前向传播
    graph.forward_node(output).unwrap();

    // 验证 z = x @ w = [1.0, 2.0] @ [[0.5, -0.5], [0.3, 0.7]] = [1.1, 0.9]
    let z_value = graph.get_node_value(z).unwrap().unwrap();
    assert_abs_diff_eq!(z_value[[0, 0]], 1.1, epsilon = 1e-5);
    assert_abs_diff_eq!(z_value[[0, 1]], 0.9, epsilon = 1e-5);

    // 验证 output = softplus([1.1, 0.9])
    let output_value = graph.get_node_value(output).unwrap().unwrap();
    // 预期值来自 PyTorch: [1.3873353, 1.2411538]
    assert_abs_diff_eq!(output_value[[0, 0]], 1.3873353, epsilon = 1e-5);
    assert_abs_diff_eq!(output_value[[0, 1]], 1.2411538, epsilon = 1e-5);
}

/// 测试 SoftPlus 在 MLP 网络中的端到端训练
#[test]
fn test_node_softplus_mlp_training() {
    let mut graph = Graph::new_with_seed(42);

    // 构建简单 MLP: x -> Linear -> SoftPlus -> Linear -> output
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w1 = graph.new_parameter_node(&[3, 2], Some("w1")).unwrap();
    let b1 = graph.new_parameter_node(&[3, 1], Some("b1")).unwrap();

    let z1 = graph.new_mat_mul_node(w1, x, Some("z1")).unwrap(); // [3, 1]
    let h1 = graph.new_add_node(&[z1, b1], Some("h1")).unwrap(); // [3, 1]
    let a1 = graph.new_softplus_node(h1, Some("a1")).unwrap(); // SoftPlus

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

// ==================== 与其他激活函数的对比测试 ====================

#[test]
fn test_softplus_vs_relu_smoothness() {
    // 验证 SoftPlus 在零点附近比 ReLU 更平滑
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[1, 5], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();
    let relu = graph.new_relu_node(input, Some("relu")).unwrap();

    // 零点附近的值
    let input_data = Tensor::new(&[-0.1, -0.01, 0.0, 0.01, 0.1], &[1, 5]);
    graph.set_node_value(input, Some(&input_data)).unwrap();

    graph.forward_node(softplus).unwrap();
    graph.forward_node(relu).unwrap();

    let softplus_output = graph.get_node_value(softplus).unwrap().unwrap();
    let relu_output = graph.get_node_value(relu).unwrap().unwrap();

    // ReLU 在负值处输出 0
    assert_eq!(relu_output[[0, 0]], 0.0);
    assert_eq!(relu_output[[0, 1]], 0.0);

    // SoftPlus 在负值处输出小正值（更平滑）
    assert!(softplus_output[[0, 0]] > 0.0);
    assert!(softplus_output[[0, 1]] > 0.0);

    // SoftPlus(0) = ln(2) ≈ 0.693, ReLU(0) = 0
    assert_abs_diff_eq!(softplus_output[[0, 2]], 0.69314718, epsilon = 1e-5);
    assert_eq!(relu_output[[0, 2]], 0.0);
}

#[test]
fn test_softplus_derivative_is_sigmoid() {
    // 验证 SoftPlus 的导数等于 Sigmoid
    // 使用两个独立的图来避免反向传播冲突
    let mut graph1 = Graph::new();
    let input1 = graph1.new_parameter_node(&[1, 5], Some("input")).unwrap();
    let softplus = graph1.new_softplus_node(input1, Some("softplus")).unwrap();

    let input_data = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5]);
    graph1.set_node_value(input1, Some(&input_data)).unwrap();

    // 前向传播
    graph1.forward_node(softplus).unwrap();

    // 获取 SoftPlus 的 Jacobian
    graph1.backward_nodes(&[input1], softplus).unwrap();
    let softplus_jacobi = graph1.get_node_jacobi(input1).unwrap().unwrap();

    // 独立计算 sigmoid 作为预期值
    let mut graph2 = Graph::new();
    let input2 = graph2.new_input_node(&[1, 5], Some("input")).unwrap();
    let sigmoid = graph2.new_sigmoid_node(input2, Some("sigmoid")).unwrap();
    graph2.set_node_value(input2, Some(&input_data)).unwrap();
    graph2.forward_node(sigmoid).unwrap();
    let sigmoid_output = graph2.get_node_value(sigmoid).unwrap().unwrap();

    // 验证: d(SoftPlus)/dx = sigmoid(x)
    // Jacobian 是对角矩阵，提取对角元素
    for i in 0..5 {
        let grad_diag = softplus_jacobi[[i, i]]; // 对角元素
        let sigmoid_val = sigmoid_output[[0, i]];
        assert_abs_diff_eq!(grad_diag, sigmoid_val, epsilon = 1e-5);
    }
}
