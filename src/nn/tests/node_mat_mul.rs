/*
 * @Author       : 老董
 * @Description  : MatMul 节点单元测试（矩阵乘法）
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

/// 测试 MatMul 节点创建
#[test]
fn test_mat_mul_creation() {
    let mut graph = Graph::new();

    // 1. 两个 Input 节点相乘
    {
        let input1 = graph.new_input_node(&[2, 3], Some("input1")).unwrap();
        let input2 = graph.new_input_node(&[3, 4], Some("input2")).unwrap();
        let mat_mul = graph
            .new_mat_mul_node(input1, input2, Some("mat_mul_inputs"))
            .unwrap();

        assert_eq!(graph.get_node_name(mat_mul).unwrap(), "mat_mul_inputs");
        assert_eq!(graph.get_node_parents(mat_mul).unwrap().len(), 2);
        assert_eq!(
            graph.get_node_value_expected_shape(mat_mul).unwrap(),
            &[2, 4]
        );
    }

    // 2. 两个 Parameter 节点相乘
    {
        let param1 = graph.new_parameter_node(&[2, 3], Some("param1")).unwrap();
        let param2 = graph.new_parameter_node(&[3, 4], Some("param2")).unwrap();
        let mat_mul = graph
            .new_mat_mul_node(param1, param2, Some("mat_mul_params"))
            .unwrap();

        assert_eq!(graph.get_node_name(mat_mul).unwrap(), "mat_mul_params");
        assert_eq!(graph.get_node_parents(mat_mul).unwrap().len(), 2);
        assert_eq!(
            graph.get_node_value_expected_shape(mat_mul).unwrap(),
            &[2, 4]
        );
    }

    // 3. 混合 Input 和 Parameter 节点相乘
    {
        let input = graph.new_input_node(&[2, 3], Some("input3")).unwrap();
        let param = graph.new_parameter_node(&[3, 4], Some("param3")).unwrap();
        let mat_mul = graph
            .new_mat_mul_node(input, param, Some("mat_mul_mixed"))
            .unwrap();

        assert_eq!(graph.get_node_name(mat_mul).unwrap(), "mat_mul_mixed");
        assert_eq!(graph.get_node_parents(mat_mul).unwrap().len(), 2);
        assert_eq!(
            graph.get_node_value_expected_shape(mat_mul).unwrap(),
            &[2, 4]
        );
    }

    // 4. 向量乘矩阵 (1xN) @ (NxM) = (1xM)
    {
        let vec = graph.new_parameter_node(&[1, 3], Some("vec")).unwrap();
        let mat = graph.new_parameter_node(&[3, 5], Some("mat")).unwrap();
        let result = graph
            .new_mat_mul_node(vec, mat, Some("vec_mat_mul"))
            .unwrap();

        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[1, 5]
        );
    }
}

/// 测试 MatMul 创建时的形状校验
#[test]
fn test_mat_mul_creation_invalid_shape() {
    let mut graph = Graph::new();

    // 1. 列数与行数不匹配：[2,3] @ [2,4]（3 ≠ 2）
    let left = graph.new_input_node(&[2, 3], Some("left")).unwrap();
    let right = graph.new_input_node(&[2, 4], Some("right")).unwrap();

    let result = graph.new_mat_mul_node(left, right, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch(
            [2, 4],
            [3, 2],
            "MatMul节点的2个父节点形状不兼容：父节点1的列数(3)与父节点2的行数(2)不相等。"
        )
    );

    // 2. 另一种不匹配：[2,3] @ [4,3]（3 ≠ 4）
    let right2 = graph.new_input_node(&[4, 3], Some("right2")).unwrap();
    let result = graph.new_mat_mul_node(left, right2, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch(
            [2, 3],
            [3, 4],
            "MatMul节点的2个父节点形状不兼容：父节点1的列数(3)与父节点2的行数(4)不相等。"
        )
    );
}

/// 测试 MatMul 节点命名
#[test]
fn test_mat_mul_name_generation() {
    let mut graph = Graph::new();

    let left = graph.new_input_node(&[2, 3], Some("l")).unwrap();
    let right = graph.new_input_node(&[3, 4], Some("r")).unwrap();

    // 1. 显式命名
    let result1 = graph
        .new_mat_mul_node(left, right, Some("my_matmul"))
        .unwrap();
    assert_eq!(graph.get_node_name(result1).unwrap(), "my_matmul");

    // 2. 自动命名
    let result2 = graph.new_mat_mul_node(left, right, None).unwrap();
    assert_eq!(graph.get_node_name(result2).unwrap(), "mat_mul_1");

    // 3. 名称重复
    let result = graph.new_mat_mul_node(left, right, Some("my_matmul"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_matmul在图default_graph中重复")
    );
}

/// 测试 MatMul 节点不能直接设置值
#[test]
fn test_mat_mul_cannot_set_value() {
    let mut graph = Graph::new();
    let left = graph.new_input_node(&[2, 3], Some("l")).unwrap();
    let right = graph.new_input_node(&[3, 4], Some("r")).unwrap();
    let result = graph.new_mat_mul_node(left, right, Some("matmul")).unwrap();

    let test_value = Tensor::new(&[1.0; 8], &[2, 4]);
    assert_err!(
        graph.set_node_value(result, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=3, name=matmul, type=MatMul]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

/// 测试 MatMul 前向传播
#[test]
fn test_mat_mul_forward() {
    let mut graph = Graph::new();

    let left = graph.new_parameter_node(&[2, 3], Some("left")).unwrap();
    let right = graph.new_parameter_node(&[3, 4], Some("right")).unwrap();
    let result = graph.new_mat_mul_node(left, right, Some("result")).unwrap();

    // left=[[1,2,3],[4,5,6]], right=[[7,8,9,10],[11,12,13,14],[15,16,17,18]]
    // result = [[1*7+2*11+3*15, ...], [...]] = [[74,80,86,92],[173,188,203,218]]
    graph
        .set_node_value(
            left,
            Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])),
        )
        .unwrap();
    graph
        .set_node_value(
            right,
            Some(&Tensor::new(
                &[
                    7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                ],
                &[3, 4],
            )),
        )
        .unwrap();

    graph.forward(result).unwrap();

    let output = graph.get_node_value(result).unwrap().unwrap();
    let expected = Tensor::new(
        &[74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0],
        &[2, 4],
    );
    assert_eq!(output, &expected);
}

/// 测试 MatMul 前向传播（向量乘矩阵）
#[test]
fn test_mat_mul_forward_vector() {
    let mut graph = Graph::new();

    let vec = graph.new_parameter_node(&[1, 3], Some("vec")).unwrap();
    let mat = graph.new_parameter_node(&[3, 2], Some("mat")).unwrap();
    let result = graph.new_mat_mul_node(vec, mat, Some("result")).unwrap();

    // vec=[1,2,3], mat=[[1,2],[3,4],[5,6]]
    // result = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    graph
        .set_node_value(vec, Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(
            mat,
            Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])),
        )
        .unwrap();

    graph.forward(result).unwrap();

    let output = graph.get_node_value(result).unwrap().unwrap();
    let expected = Tensor::new(&[22.0, 28.0], &[1, 2]);
    assert_eq!(output, &expected);
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试 MatMul 对 left 父节点的梯度计算
///
/// 对于 C = A @ B，有：
/// - ∂C/∂A 的 VJP: grad_to_A = upstream_grad @ B^T
#[test]
fn test_mat_mul_backward_to_left() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let left_id = graph.new_parameter_node(&[2, 3], Some("left"))?;
    let right_id = graph.new_parameter_node(&[3, 4], Some("right"))?;
    let result_id = graph.new_mat_mul_node(left_id, right_id, Some("result"))?;

    // 设置值
    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let right_value = Tensor::new(
        &[
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[3, 4],
    );
    graph.set_node_value(left_id, Some(&left_value))?;
    graph.set_node_value(right_id, Some(&right_value))?;
    graph.forward(result_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 4]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;
    let right_node = graph.get_node(right_id)?;

    // MatMul 需要 assistant_parent（另一个操作数）
    let grad = result_node.calc_grad_to_parent(left_node, &upstream_grad, Some(right_node))?;

    // grad_to_left = upstream @ right^T
    // upstream=[2,4], right^T=[4,3] → grad=[2,3]
    let expected = upstream_grad.mat_mul(&right_value.transpose());
    assert_eq!(grad.shape(), &[2, 3]);
    assert_eq!(&grad, &expected);

    Ok(())
}

/// 测试 MatMul 对 right 父节点的梯度计算
///
/// 对于 C = A @ B，有：
/// - ∂C/∂B 的 VJP: grad_to_B = A^T @ upstream_grad
#[test]
fn test_mat_mul_backward_to_right() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let left_id = graph.new_parameter_node(&[2, 3], Some("left"))?;
    let right_id = graph.new_parameter_node(&[3, 4], Some("right"))?;
    let result_id = graph.new_mat_mul_node(left_id, right_id, Some("result"))?;

    // 设置值
    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let right_value = Tensor::new(
        &[
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[3, 4],
    );
    graph.set_node_value(left_id, Some(&left_value))?;
    graph.set_node_value(right_id, Some(&right_value))?;
    graph.forward(result_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 4]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;
    let right_node = graph.get_node(right_id)?;

    let grad = result_node.calc_grad_to_parent(right_node, &upstream_grad, Some(left_node))?;

    // grad_to_right = left^T @ upstream
    // left^T=[3,2], upstream=[2,4] → grad=[3,4]
    let expected = left_value.transpose().mat_mul(&upstream_grad);
    assert_eq!(grad.shape(), &[3, 4]);
    assert_eq!(&grad, &expected);

    Ok(())
}

/// 测试 MatMul 梯度计算（非单位 upstream_grad）
#[test]
fn test_mat_mul_backward_with_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let left_id = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right_id = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result_id = graph.new_mat_mul_node(left_id, right_id, Some("result"))?;

    // left=[[1,2],[3,4]], right=[[5,6],[7,8]]
    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let right_value = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    graph.set_node_value(left_id, Some(&left_value))?;
    graph.set_node_value(right_id, Some(&right_value))?;
    graph.forward(result_id)?;

    // upstream_grad = [[1,2],[3,4]]（非全1）
    let upstream_grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;
    let right_node = graph.get_node(right_id)?;

    // 对 left 的梯度：upstream @ right^T
    let grad_to_left =
        result_node.calc_grad_to_parent(left_node, &upstream_grad, Some(right_node))?;
    let expected_left = upstream_grad.mat_mul(&right_value.transpose());
    assert_eq!(&grad_to_left, &expected_left);

    // 对 right 的梯度：left^T @ upstream
    let grad_to_right =
        result_node.calc_grad_to_parent(right_node, &upstream_grad, Some(left_node))?;
    let expected_right = left_value.transpose().mat_mul(&upstream_grad);
    assert_eq!(&grad_to_right, &expected_right);

    Ok(())
}

/// 测试 MatMul 梯度计算（负数值）
///
/// 验证 VJP 在负数值场景下的正确性
#[test]
fn test_mat_mul_backward_with_negative_values() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let left_id = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right_id = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result_id = graph.new_mat_mul_node(left_id, right_id, Some("result"))?;

    // left=[[-1,2],[-3,4]], right=[[5,-6],[7,-8]]
    let left_value = Tensor::new(&[-1.0, 2.0, -3.0, 4.0], &[2, 2]);
    let right_value = Tensor::new(&[5.0, -6.0, 7.0, -8.0], &[2, 2]);
    graph.set_node_value(left_id, Some(&left_value))?;
    graph.set_node_value(right_id, Some(&right_value))?;
    graph.forward(result_id)?;

    // 验证前向传播
    // [[-1,2],[-3,4]] @ [[5,-6],[7,-8]]
    // = [[-1*5+2*7, -1*-6+2*-8], [-3*5+4*7, -3*-6+4*-8]]
    // = [[9, -10], [13, -14]]
    let output = graph.get_node_value(result_id)?.unwrap();
    let expected_output = Tensor::new(&[9.0, -10.0, 13.0, -14.0], &[2, 2]);
    assert_eq!(output, &expected_output);

    // upstream_grad = [[1,1],[1,1]]
    let upstream_grad = Tensor::ones(&[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;
    let right_node = graph.get_node(right_id)?;

    // grad_to_left = upstream @ right^T
    let grad_to_left =
        result_node.calc_grad_to_parent(left_node, &upstream_grad, Some(right_node))?;
    let expected_left = upstream_grad.mat_mul(&right_value.transpose());
    assert_eq!(&grad_to_left, &expected_left);

    // grad_to_right = left^T @ upstream
    let grad_to_right =
        result_node.calc_grad_to_parent(right_node, &upstream_grad, Some(left_node))?;
    let expected_right = left_value.transpose().mat_mul(&upstream_grad);
    assert_eq!(&grad_to_right, &expected_right);

    Ok(())
}

/// 测试 MatMul 梯度计算（含零值）
///
/// 零值是重要边界情况：A @ B 中的零元素，梯度仍应正确传播
#[test]
fn test_mat_mul_backward_with_zero_value() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let left_id = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right_id = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result_id = graph.new_mat_mul_node(left_id, right_id, Some("result"))?;

    // left=[[0,1],[2,0]], right=[[1,0],[0,1]]（单位矩阵变体）
    let left_value = Tensor::new(&[0.0, 1.0, 2.0, 0.0], &[2, 2]);
    let right_value = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    graph.set_node_value(left_id, Some(&left_value))?;
    graph.set_node_value(right_id, Some(&right_value))?;
    graph.forward(result_id)?;

    // 验证前向传播：left @ I' = [[0,1],[2,0]]
    let output = graph.get_node_value(result_id)?.unwrap();
    let expected_output = Tensor::new(&[0.0, 1.0, 2.0, 0.0], &[2, 2]);
    assert_eq!(output, &expected_output);

    // upstream_grad = [[1,1],[1,1]]
    let upstream_grad = Tensor::ones(&[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;
    let right_node = graph.get_node(right_id)?;

    // 即使有零值，梯度仍应正确计算
    let grad_to_left =
        result_node.calc_grad_to_parent(left_node, &upstream_grad, Some(right_node))?;
    let expected_left = upstream_grad.mat_mul(&right_value.transpose());
    assert_eq!(&grad_to_left, &expected_left);

    let grad_to_right =
        result_node.calc_grad_to_parent(right_node, &upstream_grad, Some(left_node))?;
    let expected_right = left_value.transpose().mat_mul(&upstream_grad);
    assert_eq!(&grad_to_right, &expected_right);

    Ok(())
}

/// 测试 MatMul 梯度计算缺少 assistant_parent 时报错
#[test]
fn test_mat_mul_backward_missing_assistant_parent() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let left_id = graph.new_parameter_node(&[2, 3], Some("left"))?;
    let right_id = graph.new_parameter_node(&[3, 4], Some("right"))?;
    let result_id = graph.new_mat_mul_node(left_id, right_id, Some("result"))?;

    // 设置值并前向传播
    graph.set_node_value(
        left_id,
        Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])),
    )?;
    graph.set_node_value(right_id, Some(&Tensor::new(&[1.0; 12], &[3, 4])))?;
    graph.forward(result_id)?;

    // 直接测试 VJP，不传 assistant_parent（应该报错）
    let upstream_grad = Tensor::ones(&[2, 4]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;

    let result = result_node.calc_grad_to_parent(left_node, &upstream_grad, None);
    assert_err!(
        result,
        GraphError::ComputationError("MatMul 需要辅助父节点来计算梯度")
    );

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 MatMul 通过 graph.backward() 的端到端反向传播
///
/// 构建简单图：result = left @ right → loss = MSE(result, target)
#[test]
fn test_mat_mul_backward_e2e() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    // 创建计算图：result = left @ right
    let left = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result = graph.new_mat_mul_node(left, right, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：left=[[1,2],[3,4]], right=[[1,0],[0,1]]（单位矩阵）, target=[[0,0],[0,0]]
    let left_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let right_value = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    graph.set_node_value(left, Some(&left_value))?;
    graph.set_node_value(right, Some(&right_value))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;

    // 前向传播
    graph.forward(loss)?;

    // result = left @ I = left = [[1,2],[3,4]]
    // loss = mean((result - 0)^2) = mean([1,4,9,16]) = 30/4 = 7.5
    let loss_value = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(loss_value.get_data_number().unwrap(), 7.5, epsilon = 1e-6);

    // 反向传播（验证 backward 返回 loss 标量值 —— Phase 2 API 契约）
    graph.zero_grad()?;
    let loss_returned = graph.backward(loss)?;
    assert_abs_diff_eq!(loss_returned, 7.5, epsilon = 1e-6);

    // 验证梯度存在且形状正确
    let left_grad = graph.get_node(left)?.grad().expect("left 应有 grad");
    let right_grad = graph.get_node(right)?.grad().expect("right 应有 grad");
    assert_eq!(left_grad.shape(), &[2, 2]);
    assert_eq!(right_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = 2*result/4 = result/2 = [[0.5,1],[1.5,2]]
    // ∂loss/∂left = ∂loss/∂result @ right^T = [[0.5,1],[1.5,2]] @ I = [[0.5,1],[1.5,2]]
    let expected_left_grad = Tensor::new(&[0.5, 1.0, 1.5, 2.0], &[2, 2]);
    assert_eq!(left_grad, &expected_left_grad);

    // ∂loss/∂right = left^T @ ∂loss/∂result = [[1,3],[2,4]] @ [[0.5,1],[1.5,2]] = [[5,7],[7,10]]
    let expected_right_grad = Tensor::new(&[5.0, 7.0, 7.0, 10.0], &[2, 2]);
    assert_eq!(right_grad, &expected_right_grad);

    Ok(())
}

// ==================== 梯度累积测试 ====================

/// 测试 MatMul 梯度累积
///
/// 验证语义：参数的 grad 在多次 backward 之间累积，直到调用 zero_grad()。
/// 这是 PyTorch 兼容的行为，支持"micro-batch 梯度累积"场景。
#[test]
fn test_mat_mul_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let left = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result = graph.new_mat_mul_node(left, right, Some("result"))?;
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(left, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(right, Some(&Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2])))?;
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
