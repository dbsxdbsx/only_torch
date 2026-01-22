/*
 * @Author       : 老董
 * @Description  : Multiply 节点单元测试（逐元素乘法 Hadamard 积）
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

/// 测试 Multiply 节点创建
#[test]
fn test_multiply_creation() {
    let mut graph = GraphInner::new();

    // 1. 矩阵(2x3) ⊙ 矩阵(2x3)
    {
        let left = graph.new_parameter_node(&[2, 3], Some("left")).unwrap();
        let right = graph.new_input_node(&[2, 3], Some("right")).unwrap();
        let result = graph.new_multiply_node(left, right, Some("mul")).unwrap();

        assert_eq!(graph.get_node_name(result).unwrap(), "mul");
        assert_eq!(graph.get_node_parents(result).unwrap().len(), 2);
        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[2, 3]
        );
    }

    // 2. 向量(1xN) ⊙ 向量(1xN)
    {
        let v1 = graph.new_parameter_node(&[1, 5], Some("v1")).unwrap();
        let v2 = graph.new_parameter_node(&[1, 5], Some("v2")).unwrap();
        let result = graph.new_multiply_node(v1, v2, Some("mul_vec")).unwrap();

        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[1, 5]
        );
    }

    // 3. 标量(1x1) ⊙ 标量(1x1)
    {
        let s1 = graph.new_parameter_node(&[1, 1], Some("s1")).unwrap();
        let s2 = graph.new_parameter_node(&[1, 1], Some("s2")).unwrap();
        let result = graph.new_multiply_node(s1, s2, Some("mul_scalar")).unwrap();

        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[1, 1]
        );
    }
}

/// 测试 Multiply 创建时的形状校验（广播不兼容的情况）
#[test]
fn test_multiply_creation_invalid_shape() {
    let mut graph = GraphInner::new();

    // 1. 无法广播的形状：[2, 3] + [3, 4]（两个维度都不兼容）
    let left = graph.new_parameter_node(&[2, 3], Some("left")).unwrap();
    let right = graph.new_input_node(&[3, 4], Some("right")).unwrap();

    let result = graph.new_multiply_node(left, right, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 3], [3, 4], "Multiply节点的父节点形状无法广播")
    );

    // 2. 无法广播的形状：[2, 3] + [2, 4]（第二维 3 != 4 且都不是 1）
    let right2 = graph.new_input_node(&[2, 4], Some("right2")).unwrap();
    let result = graph.new_multiply_node(left, right2, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 3], [2, 4], "Multiply节点的父节点形状无法广播")
    );
}

/// 测试 Multiply 节点命名
#[test]
fn test_multiply_name_generation() {
    let mut graph = GraphInner::new();

    let left = graph.new_parameter_node(&[2, 3], Some("l")).unwrap();
    let right = graph.new_parameter_node(&[2, 3], Some("r")).unwrap();

    // 1. 显式命名
    let result1 = graph
        .new_multiply_node(left, right, Some("my_mul"))
        .unwrap();
    assert_eq!(graph.get_node_name(result1).unwrap(), "my_mul");

    // 2. 自动命名
    let result2 = graph.new_multiply_node(left, right, None).unwrap();
    assert_eq!(graph.get_node_name(result2).unwrap(), "multiply_1");

    // 3. 名称重复
    let result = graph.new_multiply_node(left, right, Some("my_mul"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_mul在图default_graph中重复")
    );
}

/// 测试 Multiply 节点不能直接设置值
#[test]
fn test_multiply_cannot_set_value() {
    let mut graph = GraphInner::new();
    let left = graph.new_parameter_node(&[2, 3], Some("l")).unwrap();
    let right = graph.new_parameter_node(&[2, 3], Some("r")).unwrap();
    let result = graph.new_multiply_node(left, right, Some("mul")).unwrap();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_err!(
        graph.set_node_value(result, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=3, name=mul, type=Multiply]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

/// 测试 Multiply 前向传播
#[test]
fn test_multiply_forward() {
    let mut graph = GraphInner::new();

    let left = graph.new_parameter_node(&[2, 3], Some("left")).unwrap();
    let right = graph.new_parameter_node(&[2, 3], Some("right")).unwrap();
    let result = graph
        .new_multiply_node(left, right, Some("result"))
        .unwrap();

    // left=[1,2,3,4,5,6], right=[2,3,4,5,6,7]
    // result = [1*2, 2*3, 3*4, 4*5, 5*6, 6*7] = [2,6,12,20,30,42]
    graph
        .set_node_value(
            left,
            Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])),
        )
        .unwrap();
    graph
        .set_node_value(
            right,
            Some(&Tensor::new(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 3])),
        )
        .unwrap();

    graph.forward(result).unwrap();

    let output = graph.get_node_value(result).unwrap().unwrap();
    let expected = Tensor::new(&[2.0, 6.0, 12.0, 20.0, 30.0, 42.0], &[2, 3]);
    assert_eq!(output, &expected);
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试 Multiply 对 left 父节点的梯度计算
///
/// 对于 result = A ⊙ B，有 ∂result/∂A = diag(B)
/// VJP: grad_to_left = upstream_grad ⊙ B
#[test]
fn test_multiply_backward_to_left() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let left_id = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right_id = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result_id = graph.new_multiply_node(left_id, right_id, Some("result"))?;

    // 设置值
    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let right_value = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    graph.set_node_value(left_id, Some(&left_value))?;
    graph.set_node_value(right_id, Some(&right_value))?;
    graph.forward(result_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;
    let right_node = graph.get_node(right_id)?;

    // Multiply 需要 assistant_parent（另一个操作数）
    let grad = result_node.calc_grad_to_parent(left_node, &upstream_grad, Some(right_node))?;

    // grad_to_left = upstream ⊙ right = ones ⊙ [5,6,7,8] = [5,6,7,8]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_eq!(&grad, &right_value);

    Ok(())
}

/// 测试 Multiply 对 right 父节点的梯度计算
///
/// 对于 result = A ⊙ B，有 ∂result/∂B = diag(A)
/// VJP: grad_to_right = upstream_grad ⊙ A
#[test]
fn test_multiply_backward_to_right() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let left_id = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right_id = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result_id = graph.new_multiply_node(left_id, right_id, Some("result"))?;

    // 设置值
    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let right_value = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    graph.set_node_value(left_id, Some(&left_value))?;
    graph.set_node_value(right_id, Some(&right_value))?;
    graph.forward(result_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;
    let right_node = graph.get_node(right_id)?;

    let grad = result_node.calc_grad_to_parent(right_node, &upstream_grad, Some(left_node))?;

    // grad_to_right = upstream ⊙ left = ones ⊙ [1,2,3,4] = [1,2,3,4]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_eq!(&grad, &left_value);

    Ok(())
}

/// 测试 Multiply 梯度计算（非单位 upstream_grad）
#[test]
fn test_multiply_backward_with_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let left_id = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right_id = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result_id = graph.new_multiply_node(left_id, right_id, Some("result"))?;

    // left=[[1,2],[3,4]], right=[[5,6],[7,8]]
    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let right_value = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    graph.set_node_value(left_id, Some(&left_value))?;
    graph.set_node_value(right_id, Some(&right_value))?;
    graph.forward(result_id)?;

    // upstream_grad = [[2,2],[2,2]]
    let upstream_grad = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;
    let right_node = graph.get_node(right_id)?;

    // 对 left 的梯度：upstream ⊙ right = [2,2,2,2] ⊙ [5,6,7,8] = [10,12,14,16]
    let grad_to_left =
        result_node.calc_grad_to_parent(left_node, &upstream_grad, Some(right_node))?;
    let expected_left = Tensor::new(&[10.0, 12.0, 14.0, 16.0], &[2, 2]);
    assert_eq!(&grad_to_left, &expected_left);

    // 对 right 的梯度：upstream ⊙ left = [2,2,2,2] ⊙ [1,2,3,4] = [2,4,6,8]
    let grad_to_right =
        result_node.calc_grad_to_parent(right_node, &upstream_grad, Some(left_node))?;
    let expected_right = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    assert_eq!(&grad_to_right, &expected_right);

    Ok(())
}

/// 测试 Multiply 梯度计算（负数值）
///
/// 验证 VJP 在负数值场景下的正确性
#[test]
fn test_multiply_backward_with_negative_values() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let left_id = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right_id = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result_id = graph.new_multiply_node(left_id, right_id, Some("result"))?;

    // left=[[-1,2],[-3,4]], right=[[5,-6],[7,-8]]
    let left_value = Tensor::new(&[-1.0, 2.0, -3.0, 4.0], &[2, 2]);
    let right_value = Tensor::new(&[5.0, -6.0, 7.0, -8.0], &[2, 2]);
    graph.set_node_value(left_id, Some(&left_value))?;
    graph.set_node_value(right_id, Some(&right_value))?;
    graph.forward(result_id)?;

    // 验证前向传播：[[-1*5, 2*-6], [-3*7, 4*-8]] = [[-5,-12],[-21,-32]]
    let output = graph.get_node_value(result_id)?.unwrap();
    let expected_output = Tensor::new(&[-5.0, -12.0, -21.0, -32.0], &[2, 2]);
    assert_eq!(output, &expected_output);

    // upstream_grad = [[1,1],[1,1]]
    let upstream_grad = Tensor::ones(&[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;
    let right_node = graph.get_node(right_id)?;

    // grad_to_left = upstream ⊙ right = [[5,-6],[7,-8]]
    let grad_to_left =
        result_node.calc_grad_to_parent(left_node, &upstream_grad, Some(right_node))?;
    assert_eq!(&grad_to_left, &right_value);

    // grad_to_right = upstream ⊙ left = [[-1,2],[-3,4]]
    let grad_to_right =
        result_node.calc_grad_to_parent(right_node, &upstream_grad, Some(left_node))?;
    assert_eq!(&grad_to_right, &left_value);

    Ok(())
}

/// 测试 Multiply 梯度计算（含零值）
///
/// 零值是重要边界情况：0*x=0，但梯度仍应正确传播
#[test]
fn test_multiply_backward_with_zero_value() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let left_id = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right_id = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result_id = graph.new_multiply_node(left_id, right_id, Some("result"))?;

    // left=[[0,2],[3,0]], right=[[5,0],[0,8]]
    let left_value = Tensor::new(&[0.0, 2.0, 3.0, 0.0], &[2, 2]);
    let right_value = Tensor::new(&[5.0, 0.0, 0.0, 8.0], &[2, 2]);
    graph.set_node_value(left_id, Some(&left_value))?;
    graph.set_node_value(right_id, Some(&right_value))?;
    graph.forward(result_id)?;

    // 验证前向传播：[[0*5, 2*0], [3*0, 0*8]] = [[0,0],[0,0]]
    let output = graph.get_node_value(result_id)?.unwrap();
    assert_eq!(output, &Tensor::zeros(&[2, 2]));

    // upstream_grad = [[1,1],[1,1]]
    let upstream_grad = Tensor::ones(&[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;
    let right_node = graph.get_node(right_id)?;

    // 即使输出全为 0，梯度仍应正确计算
    // grad_to_left = upstream ⊙ right = [[5,0],[0,8]]
    let grad_to_left =
        result_node.calc_grad_to_parent(left_node, &upstream_grad, Some(right_node))?;
    assert_eq!(&grad_to_left, &right_value);

    // grad_to_right = upstream ⊙ left = [[0,2],[3,0]]
    let grad_to_right =
        result_node.calc_grad_to_parent(right_node, &upstream_grad, Some(left_node))?;
    assert_eq!(&grad_to_right, &left_value);

    Ok(())
}

/// 测试 Multiply 梯度计算缺少 assistant_parent 时报错
#[test]
fn test_multiply_backward_missing_assistant_parent() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let left_id = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right_id = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result_id = graph.new_multiply_node(left_id, right_id, Some("result"))?;

    // 设置值并前向传播
    graph.set_node_value(left_id, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(right_id, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))?;
    graph.forward(result_id)?;

    // 直接测试 VJP，不传 assistant_parent（应该报错）
    let upstream_grad = Tensor::ones(&[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;

    let result = result_node.calc_grad_to_parent(left_node, &upstream_grad, None);
    assert_err!(
        result,
        GraphError::ComputationError("Multiply 节点计算梯度需要辅助父节点")
    );

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 Multiply 通过 graph.backward() 的端到端反向传播
///
/// 构建简单图：result = left ⊙ right → loss = MSE(result, target)
#[test]
fn test_multiply_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = left ⊙ right
    let left = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result = graph.new_multiply_node(left, right, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：left=[[1,2],[3,4]], right=[[2,2],[2,2]], target=[[0,0],[0,0]]
    graph.set_node_value(left, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(right, Some(&Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;

    // 前向传播
    graph.forward(loss)?;

    // result = [[2,4],[6,8]]
    // loss = mean((result - 0)^2) = mean([4,16,36,64]) = 120/4 = 30
    let loss_value = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(loss_value.get_data_number().unwrap(), 30.0, epsilon = 1e-6);

    // 反向传播（验证 backward 返回 loss 标量值 —— Phase 2 API 契约）
    graph.zero_grad()?;
    let loss_returned = graph.backward(loss)?;
    assert_abs_diff_eq!(loss_returned, 30.0, epsilon = 1e-6);

    // 验证梯度存在且形状正确
    let left_grad = graph.get_node(left)?.grad().expect("left 应有 grad");
    let right_grad = graph.get_node(right)?.grad().expect("right 应有 grad");
    assert_eq!(left_grad.shape(), &[2, 2]);
    assert_eq!(right_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = 2*result/4 = result/2 = [[1,2],[3,4]]
    // ∂loss/∂left = ∂loss/∂result ⊙ right = [[1,2],[3,4]] ⊙ [[2,2],[2,2]] = [[2,4],[6,8]]
    let expected_left_grad = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    assert_eq!(left_grad, &expected_left_grad);

    // ∂loss/∂right = ∂loss/∂result ⊙ left = [[1,2],[3,4]] ⊙ [[1,2],[3,4]] = [[1,4],[9,16]]
    let expected_right_grad = Tensor::new(&[1.0, 4.0, 9.0, 16.0], &[2, 2]);
    assert_eq!(right_grad, &expected_right_grad);

    Ok(())
}

// ==================== 梯度累积测试 ====================

/// 测试 Multiply 梯度累积
///
/// 验证语义：参数的 grad 在多次 backward 之间累积，直到调用 zero_grad()。
/// 这是 PyTorch 兼容的行为，支持"micro-batch 梯度累积"场景。
#[test]
fn test_multiply_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let left = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result = graph.new_multiply_node(left, right, Some("result"))?;
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(left, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(right, Some(&Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;
    graph.forward(loss)?;

    // 第1次反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;
    let grad_first = graph.get_node(left)?.grad().unwrap().clone();

    // 第2次反向传播（梯度累积）- 需要重新 forward（PyTorch 语义）
    graph.forward(loss)?;
    graph.backward(loss)?;
    let grad_second = graph.get_node(left)?.grad().unwrap();
    assert_eq!(grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    graph.forward(loss)?;
    graph.backward(loss)?;
    let grad_after_clear = graph.get_node(left)?.grad().unwrap();
    assert_eq!(grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 广播测试 ====================

/// 测试 Multiply 节点支持广播的创建
#[test]
fn test_multiply_broadcast_creation() {
    let mut graph = GraphInner::new();

    // 1. [3, 4] ⊙ [1, 4] -> [3, 4]（行广播）
    {
        let left = graph.new_input_node(&[3, 4], Some("left1")).unwrap();
        let right = graph.new_input_node(&[1, 4], Some("right1")).unwrap();
        let result = graph.new_multiply_node(left, right, Some("mul1")).unwrap();
        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[3, 4]
        );
    }

    // 2. [3, 1] ⊙ [1, 4] -> [3, 4]（双向广播）
    {
        let left = graph.new_input_node(&[3, 1], Some("left2")).unwrap();
        let right = graph.new_input_node(&[1, 4], Some("right2")).unwrap();
        let result = graph.new_multiply_node(left, right, Some("mul2")).unwrap();
        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[3, 4]
        );
    }

    // 3. [2, 3, 4] ⊙ [1, 1, 4] -> [2, 3, 4]（高维广播）
    {
        let left = graph.new_input_node(&[2, 3, 4], Some("left3")).unwrap();
        let right = graph.new_input_node(&[1, 1, 4], Some("right3")).unwrap();
        let result = graph.new_multiply_node(left, right, Some("mul3")).unwrap();
        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[2, 3, 4]
        );
    }
}

/// 测试 Multiply 广播前向传播
///
/// [3, 4] ⊙ [1, 4] -> [3, 4]
#[test]
fn test_multiply_broadcast_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建节点：[2, 3] ⊙ [1, 3] -> [2, 3]
    let matrix = graph.new_parameter_node(&[2, 3], Some("matrix"))?;
    let scale = graph.new_parameter_node(&[1, 3], Some("scale"))?;
    let result = graph.new_multiply_node(matrix, scale, Some("result"))?;

    // matrix = [[1,2,3], [4,5,6]]
    // scale = [[2, 3, 4]]
    // result = [[1*2,2*3,3*4], [4*2,5*3,6*4]] = [[2,6,12], [8,15,24]]
    graph.set_node_value(
        matrix,
        Some(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])),
    )?;
    graph.set_node_value(scale, Some(&Tensor::new(&[2., 3., 4.], &[1, 3])))?;

    graph.forward(result)?;

    let output = graph.get_node_value(result)?.unwrap();
    let expected = Tensor::new(&[2., 6., 12., 8., 15., 24.], &[2, 3]);
    assert_eq!(output, &expected);

    Ok(())
}

/// 测试 Multiply 广播反向传播（关键测试）
///
/// [2, 3] ⊙ [1, 3] -> [2, 3]
/// 反向传播时，对 [1, 3] 的梯度需要沿 axis=0 求和
#[test]
fn test_multiply_broadcast_backward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建节点：result = matrix ⊙ scale
    let matrix = graph.new_parameter_node(&[2, 3], Some("matrix"))?;
    let scale = graph.new_parameter_node(&[1, 3], Some("scale"))?;
    let result = graph.new_multiply_node(matrix, scale, Some("result"))?;

    // matrix = [[1,2,3], [4,5,6]]
    // scale = [[2, 3, 4]]
    graph.set_node_value(
        matrix,
        Some(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])),
    )?;
    graph.set_node_value(scale, Some(&Tensor::new(&[2., 3., 4.], &[1, 3])))?;
    graph.forward(result)?;

    // 直接测试 VJP
    // upstream_grad = [[1,1,1], [1,1,1]] (全1)
    let upstream_grad = Tensor::ones(&[2, 3]);
    let result_node = graph.get_node(result)?;
    let matrix_node = graph.get_node(matrix)?;
    let scale_node = graph.get_node(scale)?;

    // 对 matrix [2,3] 的梯度：upstream ⊙ scale（广播后）
    // upstream ⊙ [[2,3,4],[2,3,4]] = [[2,3,4],[2,3,4]]
    let grad_to_matrix =
        result_node.calc_grad_to_parent(matrix_node, &upstream_grad, Some(scale_node))?;
    assert_eq!(grad_to_matrix.shape(), &[2, 3]);
    let expected_matrix_grad = Tensor::new(&[2., 3., 4., 2., 3., 4.], &[2, 3]);
    assert_eq!(&grad_to_matrix, &expected_matrix_grad);

    // 对 scale [1,3] 的梯度：upstream ⊙ matrix，然后沿 axis=0 求和
    // upstream ⊙ [[1,2,3],[4,5,6]] = [[1,2,3],[4,5,6]]
    // sum([[1,2,3],[4,5,6]], axis=0) = [[5,7,9]]
    let grad_to_scale =
        result_node.calc_grad_to_parent(scale_node, &upstream_grad, Some(matrix_node))?;
    assert_eq!(grad_to_scale.shape(), &[1, 3]);
    let expected_scale_grad = Tensor::new(&[5., 7., 9.], &[1, 3]);
    assert_eq!(&grad_to_scale, &expected_scale_grad);

    Ok(())
}

/// 测试 Multiply 广播反向传播（非全 1 上游梯度）
///
/// 实际训练中，upstream_grad 几乎不会是全 1，而是由链式法则层层计算得到的各种值。
/// 此测试验证 sum_to_shape 在这种真实场景下的正确性。
#[test]
fn test_multiply_broadcast_backward_non_unit() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建节点：result = matrix ⊙ scale
    let matrix = graph.new_parameter_node(&[2, 3], Some("matrix"))?;
    let scale = graph.new_parameter_node(&[1, 3], Some("scale"))?;
    let result = graph.new_multiply_node(matrix, scale, Some("result"))?;

    // matrix = [[1,2,3], [4,5,6]]
    // scale = [[2, 3, 4]]
    graph.set_node_value(
        matrix,
        Some(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])),
    )?;
    graph.set_node_value(scale, Some(&Tensor::new(&[2., 3., 4.], &[1, 3])))?;
    graph.forward(result)?;

    // upstream_grad = [[1,2,3], [4,5,6]]（非全 1）
    let upstream_grad = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let result_node = graph.get_node(result)?;
    let matrix_node = graph.get_node(matrix)?;
    let scale_node = graph.get_node(scale)?;

    // 对 scale [1,3] 的梯度：upstream ⊙ matrix，然后沿 axis=0 求和
    // [[1,2,3],[4,5,6]] ⊙ [[1,2,3],[4,5,6]] = [[1,4,9],[16,25,36]]
    // sum([[1,4,9],[16,25,36]], axis=0) = [[17,29,45]]
    let grad_to_scale =
        result_node.calc_grad_to_parent(scale_node, &upstream_grad, Some(matrix_node))?;
    assert_eq!(grad_to_scale.shape(), &[1, 3]);
    let expected = Tensor::new(&[17., 29., 45.], &[1, 3]);
    assert_eq!(&grad_to_scale, &expected);

    Ok(())
}

/// 测试 Multiply 广播端到端反向传播
///
/// 验证广播在完整训练场景中的正确性
#[test]
fn test_multiply_broadcast_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 模拟特征缩放：result = features ⊙ scale
    // features [2,3], scale [1,3]
    let features = graph.new_parameter_node(&[2, 3], Some("features"))?;
    let scale = graph.new_parameter_node(&[1, 3], Some("scale"))?;
    let result = graph.new_multiply_node(features, scale, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_input_node(&[2, 3], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    // features = [[1,2,3], [4,5,6]]
    // scale = [[1,1,1]]（初始为1）
    // result = features（因为 scale 全为 1）
    // target = [[0,0,0], [0,0,0]]
    graph.set_node_value(
        features,
        Some(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])),
    )?;
    graph.set_node_value(scale, Some(&Tensor::ones(&[1, 3])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 3])))?;

    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证梯度形状
    let features_grad = graph
        .get_node(features)?
        .grad()
        .expect("features 应有 grad");
    let scale_grad = graph.get_node(scale)?.grad().expect("scale 应有 grad");

    assert_eq!(
        features_grad.shape(),
        &[2, 3],
        "features 梯度形状应为 [2,3]"
    );
    assert_eq!(scale_grad.shape(), &[1, 3], "scale 梯度形状应为 [1,3]");

    // result = [[1,2,3], [4,5,6]]
    // ∂loss/∂result = 2*(result - target)/n = result/3
    //               = [[1/3, 2/3, 1], [4/3, 5/3, 2]]
    //
    // ∂loss/∂scale = sum(∂loss/∂result ⊙ features, axis=0)
    //              = sum([[1*1/3, 2*2/3, 3*1], [4*4/3, 5*5/3, 6*2]], axis=0)
    //              = sum([[1/3, 4/3, 3], [16/3, 25/3, 12]], axis=0)
    //              = [[(1+16)/3, (4+25)/3, (3+12)]]
    //              = [[17/3, 29/3, 15]]
    let expected_scale_grad = Tensor::new(&[17. / 3., 29. / 3., 15.], &[1, 3]);
    assert_abs_diff_eq!(scale_grad, &expected_scale_grad, epsilon = 1e-4);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 Multiply 节点的动态形状传播
#[test]
fn test_multiply_dynamic_shape_propagation() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建一个固定形状的参数
    let scale = graph
        .parameter(&[1, 16], crate::nn::Init::Ones, "scale")
        .unwrap();

    // Multiply: h0 * scale
    let result = &h0 * &scale;

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "feature 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "feature 维度应该是 16");
}

/// 测试 Multiply 节点在不同 batch_size 下的前向计算
#[test]
fn test_multiply_dynamic_batch_forward() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::ones(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]
    let scale = graph
        .parameter(&[1, 16], crate::nn::Init::Ones, "scale")
        .unwrap();

    // Multiply: h0 * scale（结果全零，因为 h0 是零）
    let result = &h0 * &scale;

    // 第一次 forward：batch=2
    result.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 16], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::ones(&[8, 8])).unwrap();

    // 第二次 forward：batch=8
    result.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 16], "第二次 forward: batch=8");
}

/// 测试 Multiply 节点在不同 batch_size 下的反向传播
#[test]
fn test_multiply_dynamic_batch_backward() {
    use crate::nn::var_ops::VarLossOps;
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::ones(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]
    let scale = graph
        .parameter(&[1, 4], crate::nn::Init::Ones, "scale")
        .unwrap();

    // Multiply: h0 * scale
    let result = &h0 * &scale;

    // 创建目标和损失
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    // 验证 scale 有梯度
    let scale_grad1 = scale.grad().unwrap().unwrap();
    assert_eq!(scale_grad1.shape(), &[1, 4], "scale 梯度形状应为 [1, 4]");

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::ones(&[6, 8])).unwrap();
    target.set_value(&Tensor::zeros(&[6, 4])).unwrap();

    // 第二次 forward + backward：batch=6
    loss.forward().unwrap();
    assert_eq!(
        result.value().unwrap().unwrap().shape(),
        &[6, 4],
        "第二次 forward: batch=6"
    );
    loss.backward().unwrap();

    // 验证 scale 仍有正确形状的梯度
    let scale_grad2 = scale.grad().unwrap().unwrap();
    assert_eq!(scale_grad2.shape(), &[1, 4], "scale 梯度形状仍应为 [1, 4]");
}
