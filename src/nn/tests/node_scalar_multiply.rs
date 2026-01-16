/*
 * @Author       : 老董
 * @Description  : ScalarMultiply 节点单元测试
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）
 * 2. 前向传播测试
 * 3. VJP 单元测试（直接调用 calc_grad_to_parent）
 * 4. 端到端反向传播测试（通过 graph.backward）
 * 5. 梯度累积测试
 */

use crate::assert_err;
use crate::nn::{GraphInner, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 ScalarMultiply 节点创建
#[test]
fn test_scalar_multiply_creation() {
    let mut graph = GraphInner::new();

    // 1. 标量(1x1) * 矩阵(2x3)
    {
        let scalar = graph.new_parameter_node(&[1, 1], Some("scalar")).unwrap();
        let matrix = graph.new_input_node(&[2, 3], Some("matrix")).unwrap();
        let result = graph
            .new_scalar_multiply_node(scalar, matrix, Some("scalar_mul"))
            .unwrap();

        assert_eq!(graph.get_node_name(result).unwrap(), "scalar_mul");
        assert_eq!(graph.get_node_parents(result).unwrap().len(), 2);
        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[2, 3]
        );
    }

    // 2. 标量 * 向量(1xN)
    {
        let scalar = graph.new_parameter_node(&[1, 1], Some("scalar2")).unwrap();
        let vector = graph.new_input_node(&[1, 5], Some("vector")).unwrap();
        let result = graph
            .new_scalar_multiply_node(scalar, vector, Some("scalar_mul_vec"))
            .unwrap();

        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[1, 5]
        );
    }

    // 3. 标量 * 标量(1x1)
    {
        let scalar1 = graph.new_parameter_node(&[1, 1], Some("scalar_a")).unwrap();
        let scalar2 = graph.new_parameter_node(&[1, 1], Some("scalar_b")).unwrap();
        let result = graph
            .new_scalar_multiply_node(scalar1, scalar2, Some("scalar_mul_scalar"))
            .unwrap();

        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[1, 1]
        );
    }
}

/// 测试 ScalarMultiply 创建时的形状校验
#[test]
fn test_scalar_multiply_creation_invalid_shape() {
    let mut graph = GraphInner::new();

    // 1. 第1个参数不是标量（应该失败）
    let non_scalar = graph
        .new_parameter_node(&[2, 3], Some("non_scalar"))
        .unwrap();
    let matrix = graph.new_input_node(&[3, 4], Some("matrix")).unwrap();

    let result = graph.new_scalar_multiply_node(non_scalar, matrix, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch(
            [1, 1],
            [2, 3],
            "ScalarMultiply的第1个父节点必须是标量(形状为[1,1])"
        )
    );

    // 2. 第1个参数是向量(1xN)而非标量（应该失败）
    let vector = graph.new_parameter_node(&[1, 3], Some("vector")).unwrap();
    let result = graph.new_scalar_multiply_node(vector, matrix, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch(
            [1, 1],
            [1, 3],
            "ScalarMultiply的第1个父节点必须是标量(形状为[1,1])"
        )
    );
}

/// 测试 ScalarMultiply 节点命名
#[test]
fn test_scalar_multiply_name_generation() {
    let mut graph = GraphInner::new();

    let scalar = graph.new_parameter_node(&[1, 1], Some("s")).unwrap();
    let matrix = graph.new_input_node(&[2, 3], Some("m")).unwrap();

    // 1. 显式命名
    let result1 = graph
        .new_scalar_multiply_node(scalar, matrix, Some("my_scalar_mul"))
        .unwrap();
    assert_eq!(graph.get_node_name(result1).unwrap(), "my_scalar_mul");

    // 2. 自动命名
    let result2 = graph
        .new_scalar_multiply_node(scalar, matrix, None)
        .unwrap();
    assert_eq!(graph.get_node_name(result2).unwrap(), "scalar_multiply_1");

    // 3. 名称重复
    let result = graph.new_scalar_multiply_node(scalar, matrix, Some("my_scalar_mul"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_scalar_mul在图default_graph中重复")
    );
}

/// 测试 ScalarMultiply 节点不能直接设置值
#[test]
fn test_scalar_multiply_cannot_set_value() {
    let mut graph = GraphInner::new();
    let scalar = graph.new_parameter_node(&[1, 1], Some("s")).unwrap();
    let matrix = graph.new_input_node(&[2, 3], Some("m")).unwrap();
    let result = graph
        .new_scalar_multiply_node(scalar, matrix, Some("sm"))
        .unwrap();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_err!(
        graph.set_node_value(result, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=3, name=sm, type=ScalarMultiply]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

/// 测试 ScalarMultiply 前向传播
#[test]
fn test_scalar_multiply_forward() {
    let mut graph = GraphInner::new();

    let scalar = graph.new_parameter_node(&[1, 1], Some("scalar")).unwrap();
    let matrix = graph.new_input_node(&[2, 3], Some("matrix")).unwrap();
    let result = graph
        .new_scalar_multiply_node(scalar, matrix, Some("result"))
        .unwrap();

    // s=2, M=[1,2,3,4,5,6] → result=[2,4,6,8,10,12]
    graph
        .set_node_value(scalar, Some(&Tensor::new(&[2.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(
            matrix,
            Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])),
        )
        .unwrap();

    graph.forward(result).unwrap();

    let output = graph.get_node_value(result).unwrap().unwrap();
    let expected = Tensor::new(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0], &[2, 3]);
    assert_eq!(output, &expected);
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试 ScalarMultiply 对 scalar 的梯度计算
///
/// 对于 result = s * M，有 ∂result/∂s = M
/// VJP: grad_to_scalar = sum(upstream_grad * M)
#[test]
fn test_scalar_multiply_backward_to_scalar() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let scalar_id = graph.new_parameter_node(&[1, 1], Some("scalar"))?;
    let matrix_id = graph.new_parameter_node(&[2, 3], Some("matrix"))?;
    let result_id = graph.new_scalar_multiply_node(scalar_id, matrix_id, Some("result"))?;

    // 设置值
    let scalar_value = Tensor::new(&[2.0], &[1, 1]);
    let matrix_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(scalar_id, Some(&scalar_value))?;
    graph.set_node_value(matrix_id, Some(&matrix_value))?;
    graph.forward(result_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 3]);
    let result_node = graph.get_node(result_id)?;
    let scalar_node = graph.get_node(scalar_id)?;
    let matrix_node = graph.get_node(matrix_id)?;

    let grad = result_node.calc_grad_to_parent(scalar_node, &upstream_grad, Some(matrix_node))?;

    // grad_to_scalar = sum(upstream * M) = sum([1,2,3,4,5,6]) = 21
    assert_eq!(grad.shape(), &[1, 1]);
    assert_abs_diff_eq!(grad.get_data_number().unwrap(), 21.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 ScalarMultiply 对 matrix 的梯度计算
///
/// 对于 result = s * M，有 ∂result/∂M = s * I（逐元素）
/// VJP: grad_to_matrix = s * upstream_grad
#[test]
fn test_scalar_multiply_backward_to_matrix() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let scalar_id = graph.new_parameter_node(&[1, 1], Some("scalar"))?;
    let matrix_id = graph.new_parameter_node(&[2, 3], Some("matrix"))?;
    let result_id = graph.new_scalar_multiply_node(scalar_id, matrix_id, Some("result"))?;

    // 设置值
    let scalar_value = Tensor::new(&[2.0], &[1, 1]);
    let matrix_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    graph.set_node_value(scalar_id, Some(&scalar_value))?;
    graph.set_node_value(matrix_id, Some(&matrix_value))?;
    graph.forward(result_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 3]);
    let result_node = graph.get_node(result_id)?;
    let scalar_node = graph.get_node(scalar_id)?;
    let matrix_node = graph.get_node(matrix_id)?;

    let grad = result_node.calc_grad_to_parent(matrix_node, &upstream_grad, Some(scalar_node))?;

    // grad_to_matrix = s * upstream = 2 * ones = [2,2,2,2,2,2]
    assert_eq!(grad.shape(), &[2, 3]);
    let expected = Tensor::new(&[2.0; 6], &[2, 3]);
    assert_eq!(&grad, &expected);

    Ok(())
}

/// 测试 ScalarMultiply 梯度计算（非单位 upstream_grad）
#[test]
fn test_scalar_multiply_backward_with_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let scalar_id = graph.new_parameter_node(&[1, 1], Some("scalar"))?;
    let matrix_id = graph.new_parameter_node(&[2, 2], Some("matrix"))?;
    let result_id = graph.new_scalar_multiply_node(scalar_id, matrix_id, Some("result"))?;

    // s=3, M=[[1,2],[3,4]]
    graph.set_node_value(scalar_id, Some(&Tensor::new(&[3.0], &[1, 1])))?;
    graph.set_node_value(
        matrix_id,
        Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])),
    )?;
    graph.forward(result_id)?;

    // upstream_grad = [[1,2],[3,4]]（非全1）
    let upstream_grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let scalar_node = graph.get_node(scalar_id)?;
    let matrix_node = graph.get_node(matrix_id)?;

    // 对 scalar 的梯度：sum(upstream * M) = 1*1 + 2*2 + 3*3 + 4*4 = 1+4+9+16 = 30
    let grad_to_scalar =
        result_node.calc_grad_to_parent(scalar_node, &upstream_grad, Some(matrix_node))?;
    assert_abs_diff_eq!(
        grad_to_scalar.get_data_number().unwrap(),
        30.0,
        epsilon = 1e-6
    );

    // 对 matrix 的梯度：s * upstream = 3 * [[1,2],[3,4]] = [[3,6],[9,12]]
    let grad_to_matrix =
        result_node.calc_grad_to_parent(matrix_node, &upstream_grad, Some(scalar_node))?;
    let expected = Tensor::new(&[3.0, 6.0, 9.0, 12.0], &[2, 2]);
    assert_eq!(&grad_to_matrix, &expected);

    Ok(())
}

/// 测试 ScalarMultiply 梯度计算（负数值）
///
/// 验证 VJP 在负数值场景下的正确性
#[test]
fn test_scalar_multiply_backward_with_negative_values() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let scalar_id = graph.new_parameter_node(&[1, 1], Some("scalar"))?;
    let matrix_id = graph.new_parameter_node(&[2, 2], Some("matrix"))?;
    let result_id = graph.new_scalar_multiply_node(scalar_id, matrix_id, Some("result"))?;

    // s=-2, M=[[-1,2],[-3,4]]
    let scalar_value = Tensor::new(&[-2.0], &[1, 1]);
    let matrix_value = Tensor::new(&[-1.0, 2.0, -3.0, 4.0], &[2, 2]);
    graph.set_node_value(scalar_id, Some(&scalar_value))?;
    graph.set_node_value(matrix_id, Some(&matrix_value))?;
    graph.forward(result_id)?;

    // 验证前向传播：-2 * [[-1,2],[-3,4]] = [[2,-4],[6,-8]]
    let output = graph.get_node_value(result_id)?.unwrap();
    let expected_output = Tensor::new(&[2.0, -4.0, 6.0, -8.0], &[2, 2]);
    assert_eq!(output, &expected_output);

    // upstream_grad = [[1,1],[1,1]]
    let upstream_grad = Tensor::ones(&[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let scalar_node = graph.get_node(scalar_id)?;
    let matrix_node = graph.get_node(matrix_id)?;

    // grad_to_scalar = sum(upstream * M) = sum([[-1,2],[-3,4]]) = -1+2-3+4 = 2
    let grad_to_scalar =
        result_node.calc_grad_to_parent(scalar_node, &upstream_grad, Some(matrix_node))?;
    assert_abs_diff_eq!(
        grad_to_scalar.get_data_number().unwrap(),
        2.0,
        epsilon = 1e-6
    );

    // grad_to_matrix = s * upstream = -2 * ones = [[-2,-2],[-2,-2]]
    let grad_to_matrix =
        result_node.calc_grad_to_parent(matrix_node, &upstream_grad, Some(scalar_node))?;
    let expected_matrix_grad = Tensor::new(&[-2.0; 4], &[2, 2]);
    assert_eq!(&grad_to_matrix, &expected_matrix_grad);

    Ok(())
}

/// 测试 ScalarMultiply 梯度计算（零标量）
///
/// 零标量是重要边界情况：0*M=0，但梯度仍应正确传播
#[test]
fn test_scalar_multiply_backward_with_zero_scalar() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let scalar_id = graph.new_parameter_node(&[1, 1], Some("scalar"))?;
    let matrix_id = graph.new_parameter_node(&[2, 2], Some("matrix"))?;
    let result_id = graph.new_scalar_multiply_node(scalar_id, matrix_id, Some("result"))?;

    // s=0, M=[[1,2],[3,4]]
    let scalar_value = Tensor::new(&[0.0], &[1, 1]);
    let matrix_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    graph.set_node_value(scalar_id, Some(&scalar_value))?;
    graph.set_node_value(matrix_id, Some(&matrix_value))?;
    graph.forward(result_id)?;

    // 验证前向传播：0 * [[1,2],[3,4]] = [[0,0],[0,0]]
    let output = graph.get_node_value(result_id)?.unwrap();
    assert_eq!(output, &Tensor::zeros(&[2, 2]));

    // upstream_grad = [[1,1],[1,1]]
    let upstream_grad = Tensor::ones(&[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let scalar_node = graph.get_node(scalar_id)?;
    let matrix_node = graph.get_node(matrix_id)?;

    // 即使输出全为 0，梯度仍应正确计算
    // grad_to_scalar = sum(upstream * M) = sum([[1,2],[3,4]]) = 10
    let grad_to_scalar =
        result_node.calc_grad_to_parent(scalar_node, &upstream_grad, Some(matrix_node))?;
    assert_abs_diff_eq!(
        grad_to_scalar.get_data_number().unwrap(),
        10.0,
        epsilon = 1e-6
    );

    // grad_to_matrix = s * upstream = 0 * ones = [[0,0],[0,0]]
    let grad_to_matrix =
        result_node.calc_grad_to_parent(matrix_node, &upstream_grad, Some(scalar_node))?;
    assert_eq!(&grad_to_matrix, &Tensor::zeros(&[2, 2]));

    Ok(())
}

/// 测试 ScalarMultiply 梯度计算缺少 assistant_parent 时报错
#[test]
fn test_scalar_multiply_backward_missing_assistant_parent() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let scalar_id = graph.new_parameter_node(&[1, 1], Some("scalar"))?;
    let matrix_id = graph.new_parameter_node(&[2, 3], Some("matrix"))?;
    let result_id = graph.new_scalar_multiply_node(scalar_id, matrix_id, Some("result"))?;

    // 设置值并前向传播
    graph.set_node_value(scalar_id, Some(&Tensor::new(&[2.0], &[1, 1])))?;
    graph.set_node_value(
        matrix_id,
        Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])),
    )?;
    graph.forward(result_id)?;

    // 直接测试 VJP，不传 assistant_parent（应该报错）
    let upstream_grad = Tensor::ones(&[2, 3]);
    let result_node = graph.get_node(result_id)?;
    let scalar_node = graph.get_node(scalar_id)?;

    let result = result_node.calc_grad_to_parent(scalar_node, &upstream_grad, None);
    assert_err!(
        result,
        GraphError::ComputationError("ScalarMultiply 节点计算梯度需要辅助父节点")
    );

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 ScalarMultiply 通过 graph.backward() 的端到端反向传播
///
/// 构建简单图：result = s * M → loss = MSE(result, target)
#[test]
fn test_scalar_multiply_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建简单的计算图：result = s * M
    let scalar = graph.new_parameter_node(&[1, 1], Some("scalar"))?;
    let matrix = graph.new_parameter_node(&[2, 2], Some("matrix"))?;
    let result = graph.new_scalar_multiply_node(scalar, matrix, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：s=2, M=[[1,2],[3,4]], target=[[0,0],[0,0]]
    graph.set_node_value(scalar, Some(&Tensor::new(&[2.0], &[1, 1])))?;
    graph.set_node_value(matrix, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
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
    let scalar_grad = graph.get_node(scalar)?.grad().expect("scalar 应有 grad");
    let matrix_grad = graph.get_node(matrix)?.grad().expect("matrix 应有 grad");
    assert_eq!(scalar_grad.shape(), &[1, 1]);
    assert_eq!(matrix_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = 2*result/4 = result/2 = [[1,2],[3,4]]
    // ∂loss/∂scalar = sum(∂loss/∂result * M) = sum([[1,2],[3,4]] * [[1,2],[3,4]]) = 1+4+9+16 = 30
    assert_abs_diff_eq!(scalar_grad.get_data_number().unwrap(), 30.0, epsilon = 1e-6);

    // ∂loss/∂matrix = s * ∂loss/∂result = 2 * [[1,2],[3,4]] = [[2,4],[6,8]]
    let expected_matrix_grad = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    assert_eq!(matrix_grad, &expected_matrix_grad);

    Ok(())
}

// ==================== 梯度累积测试 ====================

/// 测试 ScalarMultiply 梯度累积
///
/// 验证语义：参数的 grad 在多次 backward 之间累积，直到调用 zero_grad()。
/// 这是 PyTorch 兼容的行为，支持"micro-batch 梯度累积"场景。
#[test]
fn test_scalar_multiply_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let scalar = graph.new_parameter_node(&[1, 1], Some("scalar"))?;
    let matrix = graph.new_parameter_node(&[2, 2], Some("matrix"))?;
    let result = graph.new_scalar_multiply_node(scalar, matrix, Some("result"))?;
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(scalar, Some(&Tensor::new(&[2.0], &[1, 1])))?;
    graph.set_node_value(matrix, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;
    graph.forward(loss)?;

    // 第1次反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;
    let grad_first = graph.get_node(scalar)?.grad().unwrap().clone();

    // 第2次反向传播（梯度累积）- 需要重新 forward（PyTorch 语义）
    graph.forward(loss)?;
    graph.backward(loss)?;
    let grad_second = graph.get_node(scalar)?.grad().unwrap();
    assert_eq!(grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    graph.forward(loss)?;
    graph.backward(loss)?;
    let grad_after_clear = graph.get_node(scalar)?.grad().unwrap();
    assert_eq!(grad_after_clear, &grad_first);

    Ok(())
}
