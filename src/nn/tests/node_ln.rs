/*
 * @Author       : 老董
 * @Description  : Ln（自然对数）节点单元测试
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

/// 测试 Ln 节点创建
#[test]
fn test_ln_creation() {
    let mut graph = GraphInner::new();

    // 1. Input 节点作为父节点
    {
        let input = graph.new_basic_input_node(&[2, 2], Some("input1")).unwrap();
        let ln = graph.new_ln_node(input, Some("ln_with_input")).unwrap();

        assert_eq!(graph.get_node_name(ln).unwrap(), "ln_with_input");
        assert_eq!(graph.get_node_parents(ln).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(ln).unwrap().len(), 0);
        assert_eq!(graph.get_node_value_expected_shape(ln).unwrap(), &[2, 2]);
    }

    // 2. Parameter 节点作为父节点
    {
        let param = graph.new_parameter_node(&[2, 3], Some("param1")).unwrap();
        let ln = graph.new_ln_node(param, Some("ln_with_param")).unwrap();

        assert_eq!(graph.get_node_name(ln).unwrap(), "ln_with_param");
        assert_eq!(graph.get_node_parents(ln).unwrap().len(), 1);
        assert_eq!(graph.get_node_value_expected_shape(ln).unwrap(), &[2, 3]);
    }
}

/// 测试 Ln 节点命名
#[test]
fn test_ln_name_generation() {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 2], Some("input")).unwrap();

    // 1. 显式命名
    let ln1 = graph.new_ln_node(input, Some("my_ln")).unwrap();
    assert_eq!(graph.get_node_name(ln1).unwrap(), "my_ln");

    // 2. 自动命名
    let ln2 = graph.new_ln_node(input, None).unwrap();
    assert_eq!(graph.get_node_name(ln2).unwrap(), "ln_1");

    // 3. 名称重复
    let result = graph.new_ln_node(input, Some("my_ln"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_ln在图default_graph中重复")
    );
}

/// 测试 Ln 节点不能直接设置值
#[test]
fn test_ln_cannot_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_basic_input_node(&[2, 2], Some("input")).unwrap();
    let ln = graph.new_ln_node(input, Some("ln")).unwrap();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_err!(
        graph.set_node_value(ln, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=ln, type=Ln]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

/// 测试 Ln 前向传播
#[test]
fn test_ln_forward() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 2], Some("input")).unwrap();
    let ln = graph.new_ln_node(input, Some("ln")).unwrap();

    // 测试数据：ln(1) = 0, ln(e) = 1, ln(e^2) ≈ 2, ln(0.5) ≈ -0.693
    let e = std::f32::consts::E;
    let input_value = Tensor::new(&[1.0, e, e * e, 0.5], &[2, 2]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(ln).unwrap();

    let output = graph.get_node_value(ln).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 1e-6); // ln(1) = 0
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-6); // ln(e) = 1
    assert_abs_diff_eq!(output[[1, 0]], 2.0, epsilon = 1e-5); // ln(e^2) = 2
    assert_abs_diff_eq!(output[[1, 1]], -0.6931472, epsilon = 1e-6); // ln(0.5) ≈ -0.693
}

/// 测试 Ln 前向传播（边界值）
#[test]
fn test_ln_forward_edge_cases() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[1, 4], Some("input")).unwrap();
    let ln = graph.new_ln_node(input, Some("ln")).unwrap();

    // 边界值：接近 0 → 大负数，大正数 → 大正数
    let input_value = Tensor::new(&[0.001, 1.0, 10.0, 100.0], &[1, 4]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(ln).unwrap();

    let output = graph.get_node_value(ln).unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], -6.907755, epsilon = 1e-5); // ln(0.001)
    assert_abs_diff_eq!(output[[0, 1]], 0.0, epsilon = 1e-6); // ln(1) = 0
    assert_abs_diff_eq!(output[[0, 2]], 2.302585, epsilon = 1e-5); // ln(10)
    assert_abs_diff_eq!(output[[0, 3]], 4.60517, epsilon = 1e-4); // ln(100)
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试 Ln 对父节点的梯度计算
///
/// 对于 y = ln(x)，有：
/// - dy/dx = 1/x
/// - VJP: grad_to_parent = upstream_grad / x
#[test]
fn test_ln_backward_vjp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let ln_id = graph.new_ln_node(input_id, Some("ln"))?;

    // 设置值（必须为正数）
    let input_value = Tensor::new(&[1.0, 2.0, 4.0, 0.5], &[2, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(ln_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 2]);
    let ln_node = graph.get_node(ln_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = ln_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // grad = upstream_grad / x = 1 / x
    // 1/[1, 2, 4, 0.5] = [1, 0.5, 0.25, 2]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 2.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Ln 梯度计算（非单位 upstream_grad）
#[test]
fn test_ln_backward_with_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let ln_id = graph.new_ln_node(input_id, Some("ln"))?;

    // 设置值
    let input_value = Tensor::new(&[1.0, 2.0, 4.0, 0.5], &[2, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(ln_id)?;

    // upstream_grad = [[2,3],[4,5]]（非全1）
    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let ln_node = graph.get_node(ln_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = ln_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // grad = upstream_grad / x
    // [2, 3, 4, 5] / [1, 2, 4, 0.5] = [2, 1.5, 1, 10]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 1.5, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 10.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Ln 梯度计算（接近 0 时梯度较大）
#[test]
fn test_ln_backward_small_input() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 2], Some("input"))?;
    let ln_id = graph.new_ln_node(input_id, Some("ln"))?;

    // 小值输入：梯度会很大
    let input_value = Tensor::new(&[0.1, 0.01], &[1, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(ln_id)?;

    let upstream_grad = Tensor::ones(&[1, 2]);
    let ln_node = graph.get_node(ln_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = ln_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // grad = 1/x
    // 1/[0.1, 0.01] = [10, 100]
    assert_abs_diff_eq!(grad[[0, 0]], 10.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 100.0, epsilon = 1e-4);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 Ln 通过 graph.backward() 的端到端反向传播
///
/// 构建简单图：result = ln(input) → loss = MSE(result, target)
#[test]
fn test_ln_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = ln(input)
    let input = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let result = graph.new_ln_node(input, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_basic_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：input = [[1, 2], [4, 0.5]], target = [[0, 0.5], [1.5, -0.5]]
    let input_value = Tensor::new(&[1.0, 2.0, 4.0, 0.5], &[2, 2]);
    let target_value = Tensor::new(&[0.0, 0.5, 1.5, -0.5], &[2, 2]);
    graph.set_node_value(input, Some(&input_value))?;
    graph.set_node_value(target, Some(&target_value))?;

    // 前向传播
    graph.forward(loss)?;

    // result = ln(input) = [[0, 0.693], [1.386, -0.693]]
    // diff = result - target = [[0, 0.193], [-0.114, -0.193]]
    // loss = mean(diff²)

    // 反向传播
    graph.zero_grad()?;
    let loss_returned = graph.backward(loss)?;
    assert!(loss_returned >= 0.0);

    // 验证梯度存在且形状正确
    let input_grad = graph.get_node(input)?.grad().expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = (result - target)/2
    // ∂loss/∂input = ∂loss/∂result * 1/x
    // 梯度应该非零
    assert!(input_grad[[0, 0]].abs() > 1e-6 || input_grad[[0, 0]] == 0.0); // ln(1)=0, target=0, diff=0
    assert!(input_grad[[0, 1]].abs() > 1e-6);
    assert!(input_grad[[1, 0]].abs() > 1e-6);
    assert!(input_grad[[1, 1]].abs() > 1e-6);

    Ok(())
}

/// 测试 Ln 在链式网络中的端到端反向传播
///
/// 网络结构: x -> MatMul(w) -> Add(b) -> Sigmoid -> Ln -> output
/// 注意：Sigmoid 输出在 (0,1) 区间，适合作为 Ln 的输入
#[test]
fn test_ln_backward_e2e_chain() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 构建网络: output = ln(sigmoid(w @ x + b))
    let x = graph.new_basic_input_node(&[2, 1], Some("x"))?;
    let w = graph.new_parameter_node(&[2, 2], Some("w"))?;
    let b = graph.new_parameter_node(&[2, 1], Some("b"))?;
    let wx = graph.new_mat_mul_node(w, x, Some("wx"))?;
    let z = graph.new_add_node(&[wx, b], Some("z"))?;
    let sig = graph.new_sigmoid_node(z, Some("sig"))?;
    let output = graph.new_ln_node(sig, Some("output"))?;

    // loss = MSE(output, target)
    let target = graph.new_basic_input_node(&[2, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 设置输入
    graph.set_node_value(x, Some(&Tensor::new(&[1.0, 0.5], &[2, 1])))?;
    graph.set_node_value(w, Some(&Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[2, 2])))?;
    graph.set_node_value(b, Some(&Tensor::new(&[0.0, 0.0], &[2, 1])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[-0.5, -0.5], &[2, 1])))?;

    // 前向传播
    graph.forward(loss)?;

    // 验证 sigmoid 输出在 (0, 1) 范围内
    let sig_val = graph.get_node_value(sig)?.unwrap();
    assert!(sig_val[[0, 0]] > 0.0 && sig_val[[0, 0]] < 1.0);
    assert!(sig_val[[1, 0]] > 0.0 && sig_val[[1, 0]] < 1.0);

    // 验证 ln 输出为负数（因为 sigmoid 输出 < 1）
    let output_val = graph.get_node_value(output)?.unwrap();
    assert!(output_val[[0, 0]] < 0.0);
    assert!(output_val[[1, 0]] < 0.0);

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

// ==================== 梯度累积测试 ====================

/// 测试 Ln 梯度累积
///
/// 验证语义：参数的 grad 在多次 backward 之间累积，直到调用 zero_grad()。
#[test]
fn test_ln_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let result = graph.new_ln_node(input, Some("result"))?;
    let target = graph.new_basic_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值（正数）
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 2.0, 4.0, 0.5], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[0.0, 0.5, 1.5, -0.5], &[2, 2])))?;
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

// ==================== 动态形状测试 ====================

/// 测试 Ln 节点的动态形状传播
#[test]
fn test_ln_dynamic_shape_propagation() {
    use crate::nn::Graph;
    use crate::nn::var_ops::VarActivationOps;

    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用正数）
    let x = graph.input(&Tensor::ones(&[4, 8])).unwrap();

    // 直接用 x 来测试，因为 x 是正数
    let result = x.ln();

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(8), "特征维度应该是 8");
}

/// 测试 Ln 节点在不同 batch_size 下的前向计算
#[test]
fn test_ln_dynamic_batch_forward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::VarActivationOps;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点（正数输入）
    let x = graph.input(&Tensor::ones(&[2, 8])).unwrap();

    // Ln: x -> ln(x)
    let result = x.ln();

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 8], "第一次 forward: batch=2");
    // ln(1) = 0
    assert_abs_diff_eq!(value1[[0, 0]], 0.0, epsilon = 1e-6);

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::ones(&[8, 8])).unwrap();

    // 第二次 forward：batch=8
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 8], "第二次 forward: batch=8");
}

/// 测试 Ln 节点在不同 batch_size 下的反向传播
#[test]
fn test_ln_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::{VarActivationOps, VarLossOps};

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    // 使用 e 的幂作为输入，确保输出在合理范围
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], &[2, 4]))
        .unwrap();

    // Ln: x -> ln(x) -> [?, 4]
    let result = x.ln();

    // 创建目标和损失
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::new(
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        &[6, 4],
    ))
    .unwrap();
    target.set_value(&Tensor::zeros(&[6, 4])).unwrap();

    // 第二次 forward + backward：batch=6
    loss.forward().unwrap();
    assert_eq!(
        result.value().unwrap().unwrap().shape(),
        &[6, 4],
        "第二次 forward: batch=6"
    );
    loss.backward().unwrap();
}
