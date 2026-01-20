/*
 * @Author       : 老董
 * @Description  : LeakyReLU 节点单元测试
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）
 * 2. 前向传播测试
 * 3. VJP 单元测试（直接调用 calc_grad_to_parent）
 * 4. 端到端反向传播测试（通过 graph.backward）
 * 5. 梯度累积测试
 */

use crate::assert_err;
use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 LeakyReLU 节点创建
#[test]
fn test_leaky_relu_creation() {
    let mut graph = GraphInner::new();

    // 1. Input 节点作为父节点（默认 slope=0.1）
    {
        let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
        let relu = graph
            .new_leaky_relu_node(input, 0.1, Some("leaky_relu_input"))
            .unwrap();

        assert_eq!(graph.get_node_name(relu).unwrap(), "leaky_relu_input");
        assert_eq!(graph.get_node_parents(relu).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(relu).unwrap().len(), 0);
        assert_eq!(graph.get_node_value_expected_shape(relu).unwrap(), &[2, 2]);
    }

    // 2. Parameter 节点作为父节点
    {
        let param = graph.new_parameter_node(&[2, 3], Some("param1")).unwrap();
        let relu = graph
            .new_leaky_relu_node(param, 0.2, Some("leaky_relu_param"))
            .unwrap();

        assert_eq!(graph.get_node_name(relu).unwrap(), "leaky_relu_param");
        assert_eq!(graph.get_node_parents(relu).unwrap().len(), 1);
        assert_eq!(graph.get_node_value_expected_shape(relu).unwrap(), &[2, 3]);
    }

    // 3. 标准 ReLU (slope=0)
    {
        let input = graph.new_input_node(&[2, 2], Some("input2")).unwrap();
        let relu = graph.new_relu_node(input, Some("standard_relu")).unwrap();

        assert_eq!(graph.get_node_name(relu).unwrap(), "standard_relu");
    }
}

/// 测试 LeakyReLU 节点命名
#[test]
fn test_leaky_relu_name_generation() {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();

    // 1. 显式命名
    let relu1 = graph
        .new_leaky_relu_node(input, 0.1, Some("my_relu"))
        .unwrap();
    assert_eq!(graph.get_node_name(relu1).unwrap(), "my_relu");

    // 2. 自动命名
    let relu2 = graph.new_leaky_relu_node(input, 0.1, None).unwrap();
    assert_eq!(graph.get_node_name(relu2).unwrap(), "leaky_relu_1");

    // 3. 名称重复
    let result = graph.new_leaky_relu_node(input, 0.1, Some("my_relu"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_relu在图default_graph中重复")
    );
}

/// 测试 LeakyReLU 无效 slope
#[test]
fn test_leaky_relu_invalid_slope() {
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();

    // 负的 negative_slope 应该失败
    let result = graph.new_leaky_relu_node(input, -0.1, Some("invalid_relu"));
    assert!(result.is_err());
}

/// 测试 LeakyReLU 节点不能直接设置值
#[test]
fn test_leaky_relu_cannot_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();
    let relu = graph.new_leaky_relu_node(input, 0.1, Some("relu")).unwrap();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_err!(
        graph.set_node_value(relu, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=relu, type=LeakyReLU]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

/// 测试标准 ReLU (slope=0) 前向传播
#[test]
fn test_relu_forward() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 2], Some("input")).unwrap();
    let relu = graph.new_relu_node(input, Some("relu")).unwrap();

    // 输入: [[0.5, -1.0], [0.0, 2.0]]
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(relu).unwrap();

    // 预期: [[0.5, 0.0], [0.0, 2.0]]
    let output = graph.get_node_value(relu).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 2.0, epsilon = 1e-6);
}

/// 测试 LeakyReLU (slope=0.1) 前向传播
#[test]
fn test_leaky_relu_forward() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 2], Some("input")).unwrap();
    let relu = graph
        .new_leaky_relu_node(input, 0.1, Some("leaky_relu"))
        .unwrap();

    // 输入: [[0.5, -1.0], [0.0, 2.0]]
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(relu).unwrap();

    // 预期: [[0.5, -0.1], [0.0, 2.0]]
    let output = graph.get_node_value(relu).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], -0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 2.0, epsilon = 1e-6);
}

/// 测试 LeakyReLU 前向传播（3x2 矩阵）
#[test]
fn test_leaky_relu_forward_3x2() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[3, 2], Some("input")).unwrap();
    let relu = graph
        .new_leaky_relu_node(input, 0.1, Some("leaky_relu"))
        .unwrap();

    // 输入: [[1.0, -2.0], [-0.5, 0.5], [3.0, -1.5]]
    let input_value = Tensor::new(&[1.0, -2.0, -0.5, 0.5, 3.0, -1.5], &[3, 2]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(relu).unwrap();

    // 预期: [[1.0, -0.2], [-0.05, 0.5], [3.0, -0.15]]
    let output = graph.get_node_value(relu).unwrap().unwrap();
    assert_eq!(output.shape(), &[3, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], -0.2, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], -0.05, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2, 1]], -0.15, epsilon = 1e-6);
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试标准 ReLU 对父节点的梯度计算
///
/// 对于 y = ReLU(x)，有：
/// - dy/dx = 1 if x > 0 else 0
/// - VJP: grad_to_parent = upstream_grad * (1 if x > 0 else 0)
#[test]
fn test_relu_backward_vjp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let relu_id = graph.new_relu_node(input_id, Some("relu"))?;

    // 设置值：[[0.5, -1.0], [0.0, 2.0]]
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(relu_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 2]);
    let relu_node = graph.get_node(relu_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = relu_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // grad = upstream_grad * (1 if x > 0 else 0)
    // 对于 [0.5, -1.0, 0.0, 2.0]：梯度为 [1, 0, 0, 1]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6); // x=0.5 > 0
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-6); // x=-1.0 < 0
    assert_abs_diff_eq!(grad[[1, 0]], 0.0, epsilon = 1e-6); // x=0.0 <= 0
    assert_abs_diff_eq!(grad[[1, 1]], 1.0, epsilon = 1e-6); // x=2.0 > 0

    Ok(())
}

/// 测试 LeakyReLU 对父节点的梯度计算
///
/// 对于 y = LeakyReLU(x)，有：
/// - dy/dx = 1 if x > 0 else negative_slope
/// - VJP: grad_to_parent = upstream_grad * (1 if x > 0 else negative_slope)
#[test]
fn test_leaky_relu_backward_vjp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let relu_id = graph.new_leaky_relu_node(input_id, 0.1, Some("leaky_relu"))?;

    // 设置值：[[0.5, -1.0], [0.0, 2.0]]
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(relu_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 2]);
    let relu_node = graph.get_node(relu_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = relu_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // grad = upstream_grad * (1 if x > 0 else 0.1)
    // 对于 [0.5, -1.0, 0.0, 2.0]：梯度为 [1, 0.1, 0.1, 1]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6); // x=0.5 > 0
    assert_abs_diff_eq!(grad[[0, 1]], 0.1, epsilon = 1e-6); // x=-1.0 < 0
    assert_abs_diff_eq!(grad[[1, 0]], 0.1, epsilon = 1e-6); // x=0.0 <= 0
    assert_abs_diff_eq!(grad[[1, 1]], 1.0, epsilon = 1e-6); // x=2.0 > 0

    Ok(())
}

/// 测试 LeakyReLU 梯度计算（非单位 upstream_grad）
#[test]
fn test_leaky_relu_backward_with_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let relu_id = graph.new_leaky_relu_node(input_id, 0.1, Some("leaky_relu"))?;

    // 设置值
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(relu_id)?;

    // upstream_grad = [[2,3],[4,5]]（非全1）
    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let relu_node = graph.get_node(relu_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = relu_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // grad = upstream_grad * (1 if x > 0 else 0.1)
    // = [2*1, 3*0.1, 4*0.1, 5*1] = [2, 0.3, 0.4, 5]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0 * 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 3.0 * 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 4.0 * 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 5.0 * 1.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 LeakyReLU 梯度计算（全正值）
#[test]
fn test_leaky_relu_backward_all_positive() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let relu_id = graph.new_leaky_relu_node(input_id, 0.1, Some("leaky_relu"))?;

    // 全正值输入
    let input_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(relu_id)?;

    let upstream_grad = Tensor::ones(&[2, 2]);
    let relu_node = graph.get_node(relu_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = relu_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 全正值时梯度全为 1
    assert_eq!(&grad, &Tensor::ones(&[2, 2]));

    Ok(())
}

/// 测试 LeakyReLU 梯度计算（全负值）
#[test]
fn test_leaky_relu_backward_all_negative() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let relu_id = graph.new_leaky_relu_node(input_id, 0.1, Some("leaky_relu"))?;

    // 全负值输入
    let input_value = Tensor::new(&[-1.0, -2.0, -3.0, -4.0], &[2, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(relu_id)?;

    let upstream_grad = Tensor::ones(&[2, 2]);
    let relu_node = graph.get_node(relu_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = relu_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 全负值时梯度全为 0.1
    let expected = Tensor::new(&[0.1, 0.1, 0.1, 0.1], &[2, 2]);
    assert_eq!(&grad, &expected);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 LeakyReLU 通过 graph.backward() 的端到端反向传播
///
/// 构建简单图：result = leaky_relu(input) → loss = MSE(result, target)
#[test]
fn test_leaky_relu_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = leaky_relu(input)
    let input = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let result = graph.new_leaky_relu_node(input, 0.1, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：input = [[0.5, -1.0], [0.0, 2.0]], target = [[0, 0], [0, 0]]
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&input_value))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;

    // 前向传播
    graph.forward(loss)?;

    // result = leaky_relu(input) = [[0.5, -0.1], [0.0, 2.0]]
    // loss = mean(result²) = mean([0.25, 0.01, 0.0, 4.0]) = 1.065
    let loss_value = graph.get_node_value(loss)?.unwrap();
    let expected_loss =
        (0.5_f32.powi(2) + 0.1_f32.powi(2) + 0.0_f32.powi(2) + 2.0_f32.powi(2)) / 4.0;
    assert_abs_diff_eq!(
        loss_value.get_data_number().unwrap(),
        expected_loss,
        epsilon = 1e-5
    );

    // 反向传播
    graph.zero_grad()?;
    let loss_returned = graph.backward(loss)?;
    assert_abs_diff_eq!(loss_returned, expected_loss, epsilon = 1e-5);

    // 验证梯度存在且形状正确
    let input_grad = graph.get_node(input)?.grad().expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = result/2
    // ∂loss/∂input = ∂loss/∂result * leaky_relu'(x)
    // = [0.5/2 * 1, -0.1/2 * 0.1, 0.0/2 * 0.1, 2.0/2 * 1]
    // = [0.25, -0.005, 0.0, 1.0]
    assert_abs_diff_eq!(input_grad[[0, 0]], 0.25, epsilon = 1e-5);
    assert_abs_diff_eq!(input_grad[[0, 1]], -0.005, epsilon = 1e-5);
    assert_abs_diff_eq!(input_grad[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(input_grad[[1, 1]], 1.0, epsilon = 1e-5);

    Ok(())
}

/// 测试 LeakyReLU 在链式网络中的端到端反向传播
///
/// 网络结构: x -> MatMul(w) -> Add(b) -> LeakyReLU -> output
#[test]
fn test_leaky_relu_backward_e2e_chain() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 构建网络: output = leaky_relu(w @ x + b)
    let x = graph.new_input_node(&[2, 1], Some("x"))?;
    let w = graph.new_parameter_node(&[2, 2], Some("w"))?;
    let b = graph.new_parameter_node(&[2, 1], Some("b"))?;
    let wx = graph.new_mat_mul_node(w, x, Some("wx"))?;
    let z = graph.new_add_node(&[wx, b], Some("z"))?;
    let output = graph.new_leaky_relu_node(z, 0.1, Some("output"))?;

    // loss = MSE(output, target)
    let target = graph.new_input_node(&[2, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 设置输入
    graph.set_node_value(x, Some(&Tensor::new(&[1.0, 0.5], &[2, 1])))?;
    graph.set_node_value(w, Some(&Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[2, 2])))?;
    graph.set_node_value(b, Some(&Tensor::new(&[0.0, 0.0], &[2, 1])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[0.5, 0.5], &[2, 1])))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证 w 和 b 的梯度存在且形状正确
    let w_grad = graph.get_node(w)?.grad().expect("w 应有 grad");
    let b_grad = graph.get_node(b)?.grad().expect("b 应有 grad");
    assert_eq!(w_grad.shape(), &[2, 2]);
    assert_eq!(b_grad.shape(), &[2, 1]);

    Ok(())
}

/// 测试 LeakyReLU 在 MLP 网络中的端到端训练
#[test]
fn test_leaky_relu_backward_e2e_mlp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 构建简单 MLP: x -> Linear -> LeakyReLU -> Linear -> output
    let x = graph.new_input_node(&[2, 1], Some("x"))?;
    let w1 = graph.new_parameter_node(&[3, 2], Some("w1"))?;
    let b1 = graph.new_parameter_node(&[3, 1], Some("b1"))?;

    let z1 = graph.new_mat_mul_node(w1, x, Some("z1"))?; // [3, 1]
    let h1 = graph.new_add_node(&[z1, b1], Some("h1"))?; // [3, 1]
    let a1 = graph.new_leaky_relu_node(h1, 0.1, Some("a1"))?; // LeakyReLU

    let w2 = graph.new_parameter_node(&[1, 3], Some("w2"))?;
    let output = graph.new_mat_mul_node(w2, a1, Some("output"))?; // [1, 1]

    // loss = MSE(output, target)
    let target = graph.new_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 设置输入
    graph.set_node_value(x, Some(&Tensor::new(&[1.0, -0.5], &[2, 1])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证所有参数都有梯度
    assert!(graph.get_node(w1)?.grad().is_some());
    assert!(graph.get_node(b1)?.grad().is_some());
    assert!(graph.get_node(w2)?.grad().is_some());

    Ok(())
}

// ==================== 梯度累积测试 ====================

/// 测试 LeakyReLU 梯度累积
///
/// 验证语义：参数的 grad 在多次 backward 之间累积，直到调用 zero_grad()。
#[test]
fn test_leaky_relu_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let result = graph.new_leaky_relu_node(input, 0.1, Some("result"))?;
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(input, Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;
    graph.forward(loss)?;

    // 第 1 次反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;
    let grad_first = graph.get_node(input)?.grad().unwrap().clone();

    // 第 2 次反向传播（梯度累积）- 需要重新 forward（PyTorch 语义）
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

// ==================== Batch 模式测试 ====================

/// 测试 LeakyReLU Batch 前向传播
#[test]
fn test_leaky_relu_batch_forward() {
    let mut graph = GraphInner::new();

    // 输入形状: batch_size=2, features=4
    let input = graph.new_parameter_node(&[2, 4], Some("input")).unwrap();
    let relu = graph
        .new_leaky_relu_node(input, 0.1, Some("leaky_relu"))
        .unwrap();

    // Batch 输入: 2 个样本，每个 4 个特征
    // 样本1: [0.5, -1.0, 0.0, 2.0]
    // 样本2: [-0.5, 1.0, -2.0, 0.5]
    let batch_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0, -0.5, 1.0, -2.0, 0.5], &[2, 4]);
    graph.set_node_value(input, Some(&batch_value)).unwrap();
    graph.forward(relu).unwrap();

    let result = graph.get_node_value(relu).unwrap().unwrap();
    // 样本1: [0.5, -0.1, 0.0, 2.0]
    // 样本2: [-0.05, 1.0, -0.2, 0.5]
    let expected = Tensor::new(&[0.5, -0.1, 0.0, 2.0, -0.05, 1.0, -0.2, 0.5], &[2, 4]);
    assert_eq!(result, &expected);
}
