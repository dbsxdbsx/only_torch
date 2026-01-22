/*
 * @Author       : 老董
 * @Description  : Tanh 节点单元测试
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

/// 测试 Tanh 节点创建
#[test]
fn test_tanh_creation() {
    let mut graph = GraphInner::new();

    // 1. Input 节点作为父节点
    {
        let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
        let tanh = graph.new_tanh_node(input, Some("tanh_with_input")).unwrap();

        assert_eq!(graph.get_node_name(tanh).unwrap(), "tanh_with_input");
        assert_eq!(graph.get_node_parents(tanh).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(tanh).unwrap().len(), 0);
        assert_eq!(graph.get_node_value_expected_shape(tanh).unwrap(), &[2, 2]);
    }

    // 2. Parameter 节点作为父节点
    {
        let param = graph.new_parameter_node(&[2, 3], Some("param1")).unwrap();
        let tanh = graph.new_tanh_node(param, Some("tanh_with_param")).unwrap();

        assert_eq!(graph.get_node_name(tanh).unwrap(), "tanh_with_param");
        assert_eq!(graph.get_node_parents(tanh).unwrap().len(), 1);
        assert_eq!(graph.get_node_value_expected_shape(tanh).unwrap(), &[2, 3]);
    }
}

/// 测试 Tanh 节点命名
#[test]
fn test_tanh_name_generation() {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();

    // 1. 显式命名
    let tanh1 = graph.new_tanh_node(input, Some("my_tanh")).unwrap();
    assert_eq!(graph.get_node_name(tanh1).unwrap(), "my_tanh");

    // 2. 自动命名
    let tanh2 = graph.new_tanh_node(input, None).unwrap();
    assert_eq!(graph.get_node_name(tanh2).unwrap(), "tanh_1");

    // 3. 名称重复
    let result = graph.new_tanh_node(input, Some("my_tanh"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_tanh在图default_graph中重复")
    );
}

/// 测试 Tanh 节点不能直接设置值
#[test]
fn test_tanh_cannot_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();
    let tanh = graph.new_tanh_node(input, Some("tanh")).unwrap();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_err!(
        graph.set_node_value(tanh, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=tanh, type=Tanh]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

/// 测试 Tanh 前向传播
#[test]
fn test_tanh_forward() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 2], Some("input")).unwrap();
    let tanh = graph.new_tanh_node(input, Some("tanh")).unwrap();

    // 测试数据（与 Python 参考一致）
    // tanh(0.5) ≈ 0.4621, tanh(-1.0) ≈ -0.7616, tanh(0.0) = 0.0, tanh(2.0) ≈ 0.9640
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(tanh).unwrap();

    let output = graph.get_node_value(tanh).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.46211716, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], -0.76159418, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 0.96402758, epsilon = 1e-6);
}

/// 测试 Tanh 前向传播（边界值）
#[test]
fn test_tanh_forward_edge_cases() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[1, 4], Some("input")).unwrap();
    let tanh = graph.new_tanh_node(input, Some("tanh")).unwrap();

    // 边界值：大正数 → 接近 1，大负数 → 接近 -1
    let input_value = Tensor::new(&[0.0, 10.0, -10.0, 0.001], &[1, 4]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(tanh).unwrap();

    let output = graph.get_node_value(tanh).unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 1e-6); // tanh(0) = 0
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-6); // tanh(10) ≈ 1
    assert_abs_diff_eq!(output[[0, 2]], -1.0, epsilon = 1e-6); // tanh(-10) ≈ -1
    assert_abs_diff_eq!(output[[0, 3]], 0.001, epsilon = 1e-5); // tanh(x) ≈ x for small x
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试 Tanh 对父节点的梯度计算
///
/// 对于 y = tanh(x)，有：
/// - dy/dx = 1 - tanh²(x) = 1 - y²
/// - VJP: grad_to_parent = upstream_grad * (1 - y²)
#[test]
fn test_tanh_backward_vjp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let tanh_id = graph.new_tanh_node(input_id, Some("tanh"))?;

    // 设置值
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(tanh_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 2]);
    let tanh_node = graph.get_node(tanh_id)?;
    let input_node = graph.get_node(input_id)?;

    // Tanh 不需要 assistant_parent
    let grad = tanh_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // grad = upstream_grad * (1 - tanh(x)²)
    // tanh([0.5, -1.0, 0.0, 2.0]) = [0.4621, -0.7616, 0.0, 0.9640]
    // 1 - tanh² = [0.7864, 0.4200, 1.0, 0.0707]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 0.78644770, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.41997433, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 0.07065082, epsilon = 1e-6);

    Ok(())
}

/// 测试 Tanh 梯度计算（非单位 upstream_grad）
#[test]
fn test_tanh_backward_with_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let tanh_id = graph.new_tanh_node(input_id, Some("tanh"))?;

    // 设置值
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(tanh_id)?;

    // upstream_grad = [[2,3],[4,5]]（非全1）
    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let tanh_node = graph.get_node(tanh_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = tanh_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // grad = upstream_grad * (1 - tanh²)
    // (1 - tanh²) = [0.7864, 0.4200, 1.0, 0.0707]
    // grad = [2*0.7864, 3*0.4200, 4*1.0, 5*0.0707] = [1.5729, 1.2599, 4.0, 0.3533]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0 * 0.78644770, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 3.0 * 0.41997433, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0]], 4.0 * 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1]], 5.0 * 0.07065082, epsilon = 1e-5);

    Ok(())
}

/// 测试 Tanh 梯度计算（接近饱和区）
///
/// 当输入绝对值很大时，tanh 接近 ±1，梯度接近 0（梯度消失）
#[test]
fn test_tanh_backward_saturation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 2], Some("input"))?;
    let tanh_id = graph.new_tanh_node(input_id, Some("tanh"))?;

    // 饱和区输入：大正数和大负数
    let input_value = Tensor::new(&[5.0, -5.0], &[1, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(tanh_id)?;

    let upstream_grad = Tensor::ones(&[1, 2]);
    let tanh_node = graph.get_node(tanh_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = tanh_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // tanh(±5) ≈ ±0.9999，1 - tanh² ≈ 0.0002（梯度接近 0）
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-3);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 Tanh 通过 graph.backward() 的端到端反向传播
///
/// 构建简单图：result = tanh(input) → loss = MSE(result, target)
#[test]
fn test_tanh_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = tanh(input)
    let input = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let result = graph.new_tanh_node(input, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：input = [[0.5, -1.0], [0.0, 2.0]], target = [[0, 0], [0, 0]]
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&input_value))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;

    // 前向传播
    graph.forward(loss)?;

    // result = tanh(input) = [[0.4621, -0.7616], [0.0, 0.9640]]
    // loss = mean(result²) = mean([0.2135, 0.5800, 0.0, 0.9293]) = 0.4307
    let loss_value = graph.get_node_value(loss)?.unwrap();
    let expected_loss = (0.46211716_f32.powi(2)
        + 0.76159418_f32.powi(2)
        + 0.0_f32.powi(2)
        + 0.96402758_f32.powi(2))
        / 4.0;
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
    // ∂loss/∂input = ∂loss/∂result * (1 - tanh²)
    // = [0.4621/2 * 0.7864, -0.7616/2 * 0.4200, 0.0/2 * 1.0, 0.9640/2 * 0.0707]
    // = [0.1817, -0.1599, 0.0, 0.0341]
    assert_abs_diff_eq!(input_grad[[0, 0]], 0.18165, epsilon = 1e-4);
    assert_abs_diff_eq!(input_grad[[0, 1]], -0.15993, epsilon = 1e-4);
    assert_abs_diff_eq!(input_grad[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(input_grad[[1, 1]], 0.03409, epsilon = 1e-4);

    Ok(())
}

/// 测试 Tanh 在链式网络中的端到端反向传播
///
/// 网络结构: x -> MatMul(w) -> Tanh -> output
#[test]
fn test_tanh_backward_e2e_chain() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 构建网络: output = tanh(w @ x)
    let x = graph.new_input_node(&[2, 1], Some("x"))?;
    let w = graph.new_parameter_node(&[2, 2], Some("w"))?;
    let wx = graph.new_mat_mul_node(w, x, Some("wx"))?;
    let output = graph.new_tanh_node(wx, Some("output"))?;

    // loss = MSE(output, target)
    let target = graph.new_input_node(&[2, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 设置输入
    graph.set_node_value(x, Some(&Tensor::new(&[1.0, 0.5], &[2, 1])))?;
    graph.set_node_value(w, Some(&Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 1])))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证 w 的梯度存在且形状正确
    let w_grad = graph.get_node(w)?.grad().expect("w 应有 grad");
    assert_eq!(w_grad.shape(), &[2, 2]);

    Ok(())
}

// ==================== 梯度累积测试 ====================

/// 测试 Tanh 梯度累积
///
/// 验证语义：参数的 grad 在多次 backward 之间累积，直到调用 zero_grad()。
#[test]
fn test_tanh_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let result = graph.new_tanh_node(input, Some("result"))?;
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

// ==================== 动态形状测试 ====================

/// 测试 Tanh 节点的动态形状传播
#[test]
fn test_tanh_dynamic_shape_propagation() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建 Tanh: h0 -> tanh(h0) -> [?, 16]
    use crate::nn::var_ops::VarActivationOps;
    let result = h0.tanh();

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试 Tanh 节点在不同 batch_size 下的前向计算
#[test]
fn test_tanh_dynamic_batch_forward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::VarActivationOps;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // Tanh: h0 -> tanh(h0)
    let result = h0.tanh();

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 16], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();

    // 第二次 forward：batch=8
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 16], "第二次 forward: batch=8");
}

/// 测试 Tanh 节点在不同 batch_size 下的反向传播
#[test]
fn test_tanh_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::{VarActivationOps, VarLossOps};

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]

    // Tanh: h0 -> tanh(h0) -> [?, 4]
    let result = h0.tanh();

    // 创建目标和损失
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[6, 8])).unwrap();
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
