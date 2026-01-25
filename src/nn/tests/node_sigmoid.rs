/*
 * @Author       : 老董
 * @Description  : Sigmoid 节点单元测试
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

/// 测试 Sigmoid 节点创建
#[test]
fn test_sigmoid_creation() {
    let mut graph = GraphInner::new();

    // 1. Input 节点作为父节点
    {
        let input = graph.new_basic_input_node(&[2, 2], Some("input1")).unwrap();
        let sigmoid = graph
            .new_sigmoid_node(input, Some("sigmoid_with_input"))
            .unwrap();

        assert_eq!(graph.get_node_name(sigmoid).unwrap(), "sigmoid_with_input");
        assert_eq!(graph.get_node_parents(sigmoid).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(sigmoid).unwrap().len(), 0);
        assert_eq!(
            graph.get_node_value_expected_shape(sigmoid).unwrap(),
            &[2, 2]
        );
    }

    // 2. Parameter 节点作为父节点
    {
        let param = graph.new_parameter_node(&[2, 3], Some("param1")).unwrap();
        let sigmoid = graph
            .new_sigmoid_node(param, Some("sigmoid_with_param"))
            .unwrap();

        assert_eq!(graph.get_node_name(sigmoid).unwrap(), "sigmoid_with_param");
        assert_eq!(graph.get_node_parents(sigmoid).unwrap().len(), 1);
        assert_eq!(
            graph.get_node_value_expected_shape(sigmoid).unwrap(),
            &[2, 3]
        );
    }
}

/// 测试 Sigmoid 节点命名
#[test]
fn test_sigmoid_name_generation() {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 2], Some("input")).unwrap();

    // 1. 显式命名
    let sigmoid1 = graph.new_sigmoid_node(input, Some("my_sigmoid")).unwrap();
    assert_eq!(graph.get_node_name(sigmoid1).unwrap(), "my_sigmoid");

    // 2. 自动命名
    let sigmoid2 = graph.new_sigmoid_node(input, None).unwrap();
    assert_eq!(graph.get_node_name(sigmoid2).unwrap(), "sigmoid_1");

    // 3. 名称重复
    let result = graph.new_sigmoid_node(input, Some("my_sigmoid"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_sigmoid在图default_graph中重复")
    );
}

/// 测试 Sigmoid 节点不能直接设置值
#[test]
fn test_sigmoid_cannot_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_basic_input_node(&[2, 2], Some("input")).unwrap();
    let sigmoid = graph.new_sigmoid_node(input, Some("sigmoid")).unwrap();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_err!(
        graph.set_node_value(sigmoid, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=sigmoid, type=Sigmoid]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

/// 测试 Sigmoid 前向传播
#[test]
fn test_sigmoid_forward() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 2], Some("input")).unwrap();
    let sigmoid = graph.new_sigmoid_node(input, Some("sigmoid")).unwrap();

    // 测试数据（与 Python 参考一致）
    // sigmoid(0.5) ≈ 0.6225, sigmoid(-1.0) ≈ 0.2689, sigmoid(0.0) = 0.5, sigmoid(2.0) ≈ 0.8808
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(sigmoid).unwrap();

    let output = graph.get_node_value(sigmoid).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.62245935, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 0.26894143, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 0.88079703, epsilon = 1e-6);
}

/// 测试 Sigmoid 前向传播（边界值）
#[test]
fn test_sigmoid_forward_edge_cases() {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[1, 4], Some("input")).unwrap();
    let sigmoid = graph.new_sigmoid_node(input, Some("sigmoid")).unwrap();

    // 边界值：大正数 → 接近 1，大负数 → 接近 0
    let input_value = Tensor::new(&[0.0, 10.0, -10.0, 0.001], &[1, 4]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    graph.forward(sigmoid).unwrap();

    let output = graph.get_node_value(sigmoid).unwrap().unwrap();
    assert_abs_diff_eq!(output[[0, 0]], 0.5, epsilon = 1e-6); // sigmoid(0) = 0.5
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-4); // sigmoid(10) ≈ 1
    assert_abs_diff_eq!(output[[0, 2]], 0.0, epsilon = 1e-4); // sigmoid(-10) ≈ 0
    assert_abs_diff_eq!(output[[0, 3]], 0.50025, epsilon = 1e-4); // sigmoid(0.001) ≈ 0.50025
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试 Sigmoid 对父节点的梯度计算
///
/// 对于 y = sigmoid(x)，有：
/// - dy/dx = sigmoid(x) * (1 - sigmoid(x)) = y * (1 - y)
/// - VJP: grad_to_parent = upstream_grad * y * (1 - y)
#[test]
fn test_sigmoid_backward_vjp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let sigmoid_id = graph.new_sigmoid_node(input_id, Some("sigmoid"))?;

    // 设置值
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(sigmoid_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 2]);
    let sigmoid_node = graph.get_node(sigmoid_id)?;
    let input_node = graph.get_node(input_id)?;

    // Sigmoid 不需要 assistant_parent
    let grad = sigmoid_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // grad = upstream_grad * sigmoid(x) * (1 - sigmoid(x))
    // sigmoid([0.5, -1.0, 0.0, 2.0]) = [0.6225, 0.2689, 0.5, 0.8808]
    // y * (1-y) = [0.2350, 0.1966, 0.25, 0.1050]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 0.23500371, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.19661194, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 0.10499363, epsilon = 1e-6);

    Ok(())
}

/// 测试 Sigmoid 梯度计算（非单位 upstream_grad）
#[test]
fn test_sigmoid_backward_with_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let sigmoid_id = graph.new_sigmoid_node(input_id, Some("sigmoid"))?;

    // 设置值
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(sigmoid_id)?;

    // upstream_grad = [[2,3],[4,5]]（非全1）
    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let sigmoid_node = graph.get_node(sigmoid_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = sigmoid_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // grad = upstream_grad * y * (1 - y)
    // y * (1-y) = [0.2350, 0.1966, 0.25, 0.1050]
    // grad = [2*0.2350, 3*0.1966, 4*0.25, 5*0.1050]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 2.0 * 0.23500371, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 3.0 * 0.19661194, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 0]], 4.0 * 0.25, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[1, 1]], 5.0 * 0.10499363, epsilon = 1e-5);

    Ok(())
}

/// 测试 Sigmoid 梯度计算（接近饱和区）
///
/// 当输入绝对值很大时，sigmoid 接近 0 或 1，梯度接近 0（梯度消失）
#[test]
fn test_sigmoid_backward_saturation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 2], Some("input"))?;
    let sigmoid_id = graph.new_sigmoid_node(input_id, Some("sigmoid"))?;

    // 饱和区输入：大正数和大负数
    let input_value = Tensor::new(&[5.0, -5.0], &[1, 2]);
    graph.set_node_value(input_id, Some(&input_value))?;
    graph.forward(sigmoid_id)?;

    let upstream_grad = Tensor::ones(&[1, 2]);
    let sigmoid_node = graph.get_node(sigmoid_id)?;
    let input_node = graph.get_node(input_id)?;

    let grad = sigmoid_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // sigmoid(5) ≈ 0.9933，sigmoid(-5) ≈ 0.0067
    // y*(1-y) ≈ 0.0066（两者都接近 0）
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-2);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-2);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 Sigmoid 通过 graph.backward() 的端到端反向传播
///
/// 构建简单图：result = sigmoid(input) → loss = MSE(result, target)
#[test]
fn test_sigmoid_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = sigmoid(input)
    let input = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let result = graph.new_sigmoid_node(input, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_basic_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：input = [[0.5, -1.0], [0.0, 2.0]], target = [[0.5, 0.5], [0.5, 0.5]]
    let input_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    let target_value = Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2]);
    graph.set_node_value(input, Some(&input_value))?;
    graph.set_node_value(target, Some(&target_value))?;

    // 前向传播
    graph.forward(loss)?;

    // result = sigmoid(input) = [[0.6225, 0.2689], [0.5, 0.8808]]
    // diff = result - target = [[0.1225, -0.2311], [0.0, 0.3808]]
    // loss = mean(diff²)
    let loss_value = graph.get_node_value(loss)?.unwrap();
    let diff: [f32; 4] = [
        0.62245935 - 0.5,
        0.26894143 - 0.5,
        0.5 - 0.5,
        0.88079703 - 0.5,
    ];
    let expected_loss =
        (diff[0].powi(2) + diff[1].powi(2) + diff[2].powi(2) + diff[3].powi(2)) / 4.0;
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

    // ∂loss/∂result = 2*(result - target)/n = (result - target)/2
    // ∂loss/∂input = ∂loss/∂result * sigmoid'(x) = ∂loss/∂result * y * (1-y)
    // 验证梯度非零（除了 sigmoid(0) 点，梯度应该非零）
    assert!(input_grad[[0, 0]].abs() > 1e-6);
    assert!(input_grad[[0, 1]].abs() > 1e-6);
    // input[1,0] = 0，sigmoid(0) = 0.5，target = 0.5，所以 diff = 0，grad 应为 0
    assert_abs_diff_eq!(input_grad[[1, 0]], 0.0, epsilon = 1e-6);
    assert!(input_grad[[1, 1]].abs() > 1e-6);

    Ok(())
}

/// 测试 Sigmoid 在链式网络中的端到端反向传播
///
/// 网络结构: x -> MatMul(w) -> Add(b) -> Sigmoid -> output
#[test]
fn test_sigmoid_backward_e2e_chain() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 构建网络: output = sigmoid(w @ x + b)
    let x = graph.new_basic_input_node(&[2, 1], Some("x"))?;
    let w = graph.new_parameter_node(&[2, 2], Some("w"))?;
    let b = graph.new_parameter_node(&[2, 1], Some("b"))?;
    let wx = graph.new_mat_mul_node(w, x, Some("wx"))?;
    let z = graph.new_add_node(&[wx, b], Some("z"))?;
    let output = graph.new_sigmoid_node(z, Some("output"))?;

    // loss = MSE(output, target)
    let target = graph.new_basic_input_node(&[2, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 设置输入
    graph.set_node_value(x, Some(&Tensor::new(&[1.0, 0.5], &[2, 1])))?;
    graph.set_node_value(w, Some(&Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[2, 2])))?;
    graph.set_node_value(b, Some(&Tensor::new(&[0.0, 0.0], &[2, 1])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[0.5, 0.5], &[2, 1])))?;

    // 前向传播
    graph.forward(loss)?;

    // 验证输出在 (0, 1) 范围内
    let output_val = graph.get_node_value(output)?.unwrap();
    assert!(output_val[[0, 0]] > 0.0 && output_val[[0, 0]] < 1.0);
    assert!(output_val[[1, 0]] > 0.0 && output_val[[1, 0]] < 1.0);

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

/// 测试 Sigmoid 梯度累积
///
/// 验证语义：参数的 grad 在多次 backward 之间累积，直到调用 zero_grad()。
#[test]
fn test_sigmoid_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_parameter_node(&[2, 2], Some("input"))?;
    let result = graph.new_sigmoid_node(input, Some("result"))?;
    let target = graph.new_basic_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(input, Some(&Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2])))?;
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

/// 测试 Sigmoid 节点的动态形状传播
#[test]
fn test_sigmoid_dynamic_shape_propagation() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建 Sigmoid: h0 -> sigmoid(h0) -> [?, 16]
    use crate::nn::var_ops::VarActivationOps;
    let result = h0.sigmoid();

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试 Sigmoid 节点在不同 batch_size 下的前向计算
#[test]
fn test_sigmoid_dynamic_batch_forward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::VarActivationOps;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // Sigmoid: h0 -> sigmoid(h0)
    let result = h0.sigmoid();

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

/// 测试 Sigmoid 节点在不同 batch_size 下的反向传播
#[test]
fn test_sigmoid_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::{VarActivationOps, VarLossOps};

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]

    // Sigmoid: h0 -> sigmoid(h0) -> [?, 4]
    let result = h0.sigmoid();

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
