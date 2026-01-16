use approx::assert_abs_diff_eq;

use crate::assert_err;
use crate::nn::{GraphInner, GraphError};
use crate::tensor::Tensor;

/// SoftPlus 节点测试
///
/// 预期值来自 tests/python/calc_jacobi_by_pytorch/node_softplus.py
///
/// 注意：本框架要求所有张量为 2-4 维，所以 "1D" 测试使用 [1, n] 形状

// ==================== 节点创建测试 ====================

#[test]
fn test_node_softplus_creation() {
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[1, 3], Some("input")).unwrap();

    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();

    assert_eq!(graph.get_node_name(softplus).unwrap(), "softplus");
    assert_eq!(graph.get_node_parents(softplus).unwrap().len(), 1);
    assert_eq!(graph.get_node_children(softplus).unwrap().len(), 0);
}

#[test]
fn test_node_softplus_name_generation() {
    let mut graph = GraphInner::new();
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
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点explicit_softplus在图default_graph中重复")
    );
}

#[test]
fn test_node_softplus_manually_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[1, 3], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();

    // 直接设置 SoftPlus 节点的值应该失败
    let test_value = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    assert_err!(
        graph.set_node_value(softplus, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=softplus, type=SoftPlus]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

#[test]
fn test_node_softplus_forward_1d() {
    // "1D" 向量前向传播测试 (实际使用 [1, 5] 形状)
    // 预期值来自 tests/python/calc_jacobi_by_pytorch/node_softplus.py
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[1, 5], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();

    // 设置输入: [-2.0, -1.0, 0.0, 1.0, 2.0]
    let input_data = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5]);
    graph.set_node_value(input, Some(&input_data)).unwrap();
    graph.forward(softplus).unwrap();

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
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();

    // 设置输入: [[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]]
    let input_data = Tensor::new(&[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]);
    graph.set_node_value(input, Some(&input_data)).unwrap();
    graph.forward(softplus).unwrap();

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
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[1, 5], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();

    // 设置极端输入: [-50.0, -20.0, 0.0, 20.0, 50.0]
    let input_data = Tensor::new(&[-50.0, -20.0, 0.0, 20.0, 50.0], &[1, 5]);
    graph.set_node_value(input, Some(&input_data)).unwrap();
    graph.forward(softplus).unwrap();

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

// ==================== VJP单元测试（直接调用 calc_grad_to_parent）====================

/// 测试 SoftPlus VJP（单位上游梯度）
///
/// 对于 y = softplus(x) = ln(1 + e^x)，有 dy/dx = sigmoid(x) = 1/(1+e^(-x))
/// VJP: grad_to_input = upstream_grad ⊙ sigmoid(x)
#[test]
fn test_node_softplus_backward_vjp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    let input_id = graph.new_parameter_node(&[1, 5], Some("input"))?;
    let softplus_id = graph.new_softplus_node(input_id, Some("softplus"))?;

    // 设置输入: [-2.0, -1.0, 0.0, 1.0, 2.0]
    let input_data = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5]);
    graph.set_node_value(input_id, Some(&input_data))?;
    graph.forward(softplus_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[1, 5]);
    let softplus_node = graph.get_node(softplus_id)?;
    let input_node = graph.get_node(input_id)?;
    let grad = softplus_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // softplus 的导数是 sigmoid(x)
    // 预期: [0.11920292, 0.26894143, 0.5, 0.73105860, 0.88079708]
    let expected = [0.11920292, 0.26894143, 0.5, 0.73105860, 0.88079708];
    assert_eq!(grad.shape(), &[1, 5]);
    for i in 0..5 {
        assert_abs_diff_eq!(grad[[0, i]], expected[i], epsilon = 1e-5);
    }

    Ok(())
}

/// 测试 SoftPlus VJP（非单位上游梯度）
#[test]
fn test_node_softplus_backward_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    let input_id = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let softplus_id = graph.new_softplus_node(input_id, Some("softplus"))?;

    // 设置输入: [[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]]
    let input_data = Tensor::new(&[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]);
    graph.set_node_value(input_id, Some(&input_data))?;
    graph.forward(softplus_id)?;

    // 非单位上游梯度
    let upstream_grad = Tensor::new(&[2.0, 3.0, 1.0, 0.5, 4.0, 2.0], &[2, 3]);
    let softplus_node = graph.get_node(softplus_id)?;
    let input_node = graph.get_node(input_id)?;
    let grad = softplus_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // sigmoid 值: [0.26894143, 0.5, 0.73105860, 0.88079709, 0.11920292, 0.62245935]
    // grad = upstream ⊙ sigmoid
    let sigmoid_vals = [
        0.26894143, 0.5, 0.73105860, 0.88079709, 0.11920292, 0.62245935,
    ];
    let upstream_vals = [2.0, 3.0, 1.0, 0.5, 4.0, 2.0];
    assert_eq!(grad.shape(), &[2, 3]);
    for i in 0..6 {
        let expected = sigmoid_vals[i] * upstream_vals[i];
        assert_abs_diff_eq!(grad.data_as_slice()[i], expected, epsilon = 1e-5);
    }

    Ok(())
}

// ==================== 端到端反向传播测试（通过 graph.backward）====================

/// 测试 SoftPlus 通过 graph.backward() 的端到端反向传播
#[test]
fn test_node_softplus_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：output = softplus(input)
    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let softplus = graph.new_softplus_node(input, Some("softplus"))?;

    // loss = MSE(softplus, target)
    let target = graph.new_input_node(&[2, 3], Some("target"))?;
    let loss = graph.new_mse_loss_node(softplus, target, Some("loss"))?;

    // 设置值：input=[[-1,0,1],[2,-2,0.5]], target=zeros
    let input_data = Tensor::new(&[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]);
    graph.set_node_value(input, Some(&input_data))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 3])))?;

    // 前向传播
    graph.forward(loss)?;

    // softplus 输出: [0.31326169, 0.69314718, 1.31326163, 2.12692809, 0.12692800, 0.97407699]
    // loss = mean(softplus^2) 因为 target=0

    // 反向传播
    graph.zero_grad()?;
    let loss_returned = graph.backward(loss)?;
    assert!(loss_returned > 0.0); // loss 应为正值

    // 验证梯度存在且形状正确
    let input_grad = graph.get_node(input)?.grad().expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3]);

    // 梯度计算：∂loss/∂input = ∂loss/∂softplus * ∂softplus/∂input
    // ∂loss/∂softplus = 2 * softplus / n
    // ∂softplus/∂input = sigmoid(input)
    // 所以：grad = 2 * softplus * sigmoid / n
    let softplus_vals = [
        0.31326169, 0.69314718, 1.31326163, 2.12692809, 0.12692800, 0.97407699,
    ];
    let sigmoid_vals = [
        0.26894143, 0.5, 0.73105860, 0.88079709, 0.11920292, 0.62245935,
    ];
    let n = 6.0;
    for i in 0..6 {
        let expected = 2.0 * softplus_vals[i] * sigmoid_vals[i] / n;
        assert_abs_diff_eq!(input_grad.data_as_slice()[i], expected, epsilon = 1e-5);
    }

    Ok(())
}

/// 测试 SoftPlus 前向传播（batch 模式）
#[test]
fn test_node_softplus_batch_forward() {
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();

    // Batch 输入 (2 samples, 3 features)
    let input_data = Tensor::new(&[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]);
    graph.set_node_value(input, Some(&input_data)).unwrap();
    graph.forward(softplus).unwrap();

    let result = graph.get_node_value(softplus).unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 3]);

    // 预期输出
    assert_abs_diff_eq!(result[[0, 0]], 0.31326169, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[1, 0]], 2.12692809, epsilon = 1e-5);
}

/// 测试 SoftPlus 在线性层后的前向传播
#[test]
fn test_node_softplus_after_linear() {
    // 线性层后接 SoftPlus: output = softplus(x @ w)
    let mut graph = GraphInner::new();

    let x = graph.new_input_node(&[1, 2], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[2, 2], Some("w")).unwrap();
    let z = graph.new_mat_mul_node(x, w, Some("z")).unwrap();
    let output = graph.new_softplus_node(z, Some("output")).unwrap();

    // 设置值
    let x_value = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let w_value = Tensor::new(&[0.5, -0.5, 0.3, 0.7], &[2, 2]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(w, Some(&w_value)).unwrap();

    // 前向传播
    graph.forward(output).unwrap();

    // 验证 z = x @ w = [1.0, 2.0] @ [[0.5, -0.5], [0.3, 0.7]] = [1.1, 0.9]
    let z_value = graph.get_node_value(z).unwrap().unwrap();
    assert_abs_diff_eq!(z_value[[0, 0]], 1.1, epsilon = 1e-5);
    assert_abs_diff_eq!(z_value[[0, 1]], 0.9, epsilon = 1e-5);

    // 验证 output = softplus([1.1, 0.9])
    let output_value = graph.get_node_value(output).unwrap().unwrap();
    assert_abs_diff_eq!(output_value[[0, 0]], 1.3873353, epsilon = 1e-5);
    assert_abs_diff_eq!(output_value[[0, 1]], 1.2411538, epsilon = 1e-5);
}

// ==================== 梯度累积测试 ====================

/// 测试 SoftPlus 梯度累积
///
/// 验证语义：参数的 grad 在多次 backward 之间累积，直到调用 zero_grad()。
#[test]
fn test_node_softplus_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 3], Some("input"))?;
    let softplus = graph.new_softplus_node(input, Some("softplus"))?;
    let target = graph.new_input_node(&[2, 3], Some("target"))?;
    let loss = graph.new_mse_loss_node(softplus, target, Some("loss"))?;

    // 设置值
    let input_data = Tensor::new(&[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]);
    graph.set_node_value(input, Some(&input_data))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 3])))?;
    graph.forward(loss)?;

    // 第1次反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;
    let grad_first = graph.get_node(input)?.grad().unwrap().clone();

    // 第2次反向传播（梯度累积）- 需要重新 forward（PyTorch 语义）
    graph.forward(loss)?;
    graph.backward(loss)?;
    let grad_second = graph.get_node(input)?.grad().unwrap();
    assert_eq!(grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    graph.forward(loss)?;
    graph.backward(loss)?;
    let grad_after_clear = graph.get_node(input)?.grad().unwrap();
    assert_eq!(grad_after_clear, &grad_first);

    Ok(())
}

/// 测试 SoftPlus 在 MLP 网络中的端到端训练
#[test]
fn test_node_softplus_mlp_training() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 构建简单 MLP: x -> Linear -> SoftPlus -> Linear -> output -> MSE
    let x = graph.new_input_node(&[2, 1], Some("x"))?;
    let w1 = graph.new_parameter_node(&[3, 2], Some("w1"))?;
    let b1 = graph.new_parameter_node(&[3, 1], Some("b1"))?;

    let z1 = graph.new_mat_mul_node(w1, x, Some("z1"))?; // [3, 1]
    let h1 = graph.new_add_node(&[z1, b1], Some("h1"))?; // [3, 1]
    let a1 = graph.new_softplus_node(h1, Some("a1"))?; // SoftPlus

    let w2 = graph.new_parameter_node(&[1, 3], Some("w2"))?;
    let output = graph.new_mat_mul_node(w2, a1, Some("output"))?; // [1, 1]

    // 添加 MSE 损失
    let target = graph.new_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 设置输入
    let input_value = Tensor::new(&[1.0, -0.5], &[2, 1]);
    graph.set_node_value(x, Some(&input_value))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[1, 1])))?;

    // 前向传播
    graph.forward(loss)?;
    let output_val = graph.get_node_value(output)?.unwrap();
    assert_eq!(output_val.shape(), &[1, 1]);

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证所有参数都有梯度
    assert!(graph.get_node(w1)?.grad().is_some());
    assert!(graph.get_node(b1)?.grad().is_some());
    assert!(graph.get_node(w2)?.grad().is_some());

    // 手动 SGD 更新（验证梯度可用于参数更新）
    let lr = 0.01;
    for &param in &[w1, b1, w2] {
        let val = graph.get_node_value(param)?.unwrap();
        let grad = graph.get_node_grad(param)?.unwrap();
        let new_val = val - lr * &grad;
        graph.set_node_value(param, Some(&new_val))?;
    }

    Ok(())
}

// ==================== 与其他激活函数的对比测试 ====================

#[test]
fn test_softplus_vs_relu_smoothness() {
    // 验证 SoftPlus 在零点附近比 ReLU 更平滑
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[1, 5], Some("input")).unwrap();
    let softplus = graph.new_softplus_node(input, Some("softplus")).unwrap();
    let relu = graph.new_relu_node(input, Some("relu")).unwrap();

    // 零点附近的值
    let input_data = Tensor::new(&[-0.1, -0.01, 0.0, 0.01, 0.1], &[1, 5]);
    graph.set_node_value(input, Some(&input_data)).unwrap();

    graph.forward(softplus).unwrap();
    graph.forward(relu).unwrap();

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

/// 验证 SoftPlus 的导数等于 Sigmoid（使用 VJP 验证）
#[test]
fn test_softplus_derivative_is_sigmoid() -> Result<(), GraphError> {
    // 通过 VJP 验证 SoftPlus 的导数等于 Sigmoid
    let mut graph = GraphInner::new();
    let input_id = graph.new_parameter_node(&[1, 5], Some("input"))?;
    let softplus_id = graph.new_softplus_node(input_id, Some("softplus"))?;

    let input_data = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5]);
    graph.set_node_value(input_id, Some(&input_data))?;
    graph.forward(softplus_id)?;

    // 使用单位上游梯度计算 VJP
    let upstream_grad = Tensor::ones(&[1, 5]);
    let softplus_node = graph.get_node(softplus_id)?;
    let input_node = graph.get_node(input_id)?;
    let grad = softplus_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 独立计算 sigmoid 作为预期值
    let mut graph2 = GraphInner::new();
    let input2 = graph2.new_input_node(&[1, 5], Some("input"))?;
    let sigmoid = graph2.new_sigmoid_node(input2, Some("sigmoid"))?;
    graph2.set_node_value(input2, Some(&input_data))?;
    graph2.forward(sigmoid)?;
    let sigmoid_output = graph2.get_node_value(sigmoid)?.unwrap();

    // 验证: d(SoftPlus)/dx = sigmoid(x)
    for i in 0..5 {
        assert_abs_diff_eq!(grad[[0, i]], sigmoid_output[[0, i]], epsilon = 1e-5);
    }

    Ok(())
}
