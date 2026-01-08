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
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 Multiply 节点创建
#[test]
fn test_multiply_creation() {
    let mut graph = Graph::new();

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

/// 测试 Multiply 创建时的形状校验
#[test]
fn test_multiply_creation_invalid_shape() {
    let mut graph = Graph::new();

    // 1. 形状完全不同
    let left = graph.new_parameter_node(&[2, 3], Some("left")).unwrap();
    let right = graph.new_input_node(&[3, 4], Some("right")).unwrap();

    let result = graph.new_multiply_node(left, right, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 3], [3, 4], "Multiply节点的两个父节点形状必须相同")
    );

    // 2. 行数相同但列数不同
    let right2 = graph.new_input_node(&[2, 4], Some("right2")).unwrap();
    let result = graph.new_multiply_node(left, right2, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 3], [2, 4], "Multiply节点的两个父节点形状必须相同")
    );
}

/// 测试 Multiply 节点命名
#[test]
fn test_multiply_name_generation() {
    let mut graph = Graph::new();

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
    let mut graph = Graph::new();
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
    let mut graph = Graph::new();

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
    let mut graph = Graph::new();

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
    let mut graph = Graph::new();

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
    let mut graph = Graph::new();

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
    let mut graph = Graph::new();

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
    let mut graph = Graph::new();

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
    let mut graph = Graph::new();

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
    let mut graph = Graph::new();

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
    let mut graph = Graph::new();

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
