/*
 * @Author       : 老董
 * @Description  : Subtract 节点单元测试（逐元素减法）
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）
 * 2. 前向传播测试
 * 3. VJP 单元测试（直接调用 calc_grad_to_parent）
 * 4. 端到端反向传播测试（通过 graph.backward）
 * 5. 广播测试
 */

use crate::assert_err;
use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 Subtract 节点创建
#[test]
fn test_subtract_creation() {
    let mut graph = GraphInner::new();

    // 1. 矩阵(2x3) - 矩阵(2x3)
    {
        let left = graph.new_parameter_node(&[2, 3], Some("left")).unwrap();
        let right = graph.new_basic_input_node(&[2, 3], Some("right")).unwrap();
        let result = graph.new_subtract_node(left, right, Some("sub")).unwrap();

        assert_eq!(graph.get_node_name(result).unwrap(), "sub");
        assert_eq!(graph.get_node_parents(result).unwrap().len(), 2);
        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[2, 3]
        );
    }

    // 2. 标量(1x1) - 标量(1x1)
    {
        let s1 = graph.new_parameter_node(&[1, 1], Some("s1")).unwrap();
        let s2 = graph.new_parameter_node(&[1, 1], Some("s2")).unwrap();
        let result = graph.new_subtract_node(s1, s2, Some("sub_scalar")).unwrap();

        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[1, 1]
        );
    }
}

/// 测试 Subtract 创建时的形状校验（广播不兼容的情况）
#[test]
fn test_subtract_creation_invalid_shape() {
    let mut graph = GraphInner::new();

    // 1. 无法广播的形状：[2, 3] - [3, 4]
    let left = graph.new_parameter_node(&[2, 3], Some("left")).unwrap();
    let right = graph.new_basic_input_node(&[3, 4], Some("right")).unwrap();

    let result = graph.new_subtract_node(left, right, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 3], [3, 4], "Subtract 节点的父节点形状无法广播")
    );

    // 2. 无法广播的形状：[2, 3] - [2, 4]
    let right2 = graph.new_basic_input_node(&[2, 4], Some("right2")).unwrap();
    let result = graph.new_subtract_node(left, right2, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 3], [2, 4], "Subtract 节点的父节点形状无法广播")
    );
}

/// 测试 Subtract 节点命名
#[test]
fn test_subtract_name_generation() {
    let mut graph = GraphInner::new();

    let left = graph.new_parameter_node(&[2, 3], Some("l")).unwrap();
    let right = graph.new_parameter_node(&[2, 3], Some("r")).unwrap();

    // 1. 显式命名
    let result1 = graph
        .new_subtract_node(left, right, Some("my_sub"))
        .unwrap();
    assert_eq!(graph.get_node_name(result1).unwrap(), "my_sub");

    // 2. 自动命名
    let result2 = graph.new_subtract_node(left, right, None).unwrap();
    assert_eq!(graph.get_node_name(result2).unwrap(), "subtract_1");

    // 3. 名称重复
    let result = graph.new_subtract_node(left, right, Some("my_sub"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_sub在图default_graph中重复")
    );
}

// ==================== 前向传播测试 ====================

/// 测试 Subtract 前向传播
#[test]
fn test_subtract_forward() {
    let mut graph = GraphInner::new();

    let left = graph.new_parameter_node(&[2, 3], Some("left")).unwrap();
    let right = graph.new_parameter_node(&[2, 3], Some("right")).unwrap();
    let result = graph
        .new_subtract_node(left, right, Some("result"))
        .unwrap();

    // left=[6,8,12,20,30,42], right=[2,4,3,5,6,7]
    // result = [6-2, 8-4, 12-3, 20-5, 30-6, 42-7] = [4,4,9,15,24,35]
    graph
        .set_node_value(
            left,
            Some(&Tensor::new(&[6.0, 8.0, 12.0, 20.0, 30.0, 42.0], &[2, 3])),
        )
        .unwrap();
    graph
        .set_node_value(
            right,
            Some(&Tensor::new(&[2.0, 4.0, 3.0, 5.0, 6.0, 7.0], &[2, 3])),
        )
        .unwrap();

    graph.forward(result).unwrap();

    let output = graph.get_node_value(result).unwrap().unwrap();
    let expected = Tensor::new(&[4.0, 4.0, 9.0, 15.0, 24.0, 35.0], &[2, 3]);
    assert_eq!(output, &expected);
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试 Subtract 对 left（被减数）的梯度计算
///
/// 对于 result = A - B：
/// ∂L/∂A = upstream_grad
#[test]
fn test_subtract_backward_to_left() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let left_id = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right_id = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result_id = graph.new_subtract_node(left_id, right_id, Some("result"))?;

    // 设置值
    graph.set_node_value(left_id, Some(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2])))?;
    graph.set_node_value(right_id, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.forward(result_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;

    // Subtract 不需要 assistant_parent
    let grad = result_node.calc_grad_to_parent(left_node, &upstream_grad, None)?;

    // grad_to_left = upstream = ones
    assert_eq!(grad.shape(), &[2, 2]);
    assert_eq!(&grad, &upstream_grad);

    Ok(())
}

/// 测试 Subtract 对 right（减数）的梯度计算
///
/// 对于 result = A - B：
/// ∂L/∂B = -upstream_grad
#[test]
fn test_subtract_backward_to_right() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let left_id = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right_id = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result_id = graph.new_subtract_node(left_id, right_id, Some("result"))?;

    // 设置值
    graph.set_node_value(left_id, Some(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2])))?;
    graph.set_node_value(right_id, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.forward(result_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let right_node = graph.get_node(right_id)?;

    let grad = result_node.calc_grad_to_parent(right_node, &upstream_grad, None)?;

    // grad_to_right = -upstream = -ones = [[-1,-1],[-1,-1]]
    assert_eq!(grad.shape(), &[2, 2]);
    let expected = Tensor::new(&[-1.0, -1.0, -1.0, -1.0], &[2, 2]);
    assert_eq!(&grad, &expected);

    Ok(())
}

/// 测试 Subtract 梯度计算（非全 1 上游梯度）
#[test]
fn test_subtract_backward_with_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let left_id = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right_id = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result_id = graph.new_subtract_node(left_id, right_id, Some("result"))?;

    // 设置值
    graph.set_node_value(left_id, Some(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2])))?;
    graph.set_node_value(right_id, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.forward(result_id)?;

    // upstream_grad = [[2,3],[4,5]]
    let upstream_grad = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let result_node = graph.get_node(result_id)?;
    let left_node = graph.get_node(left_id)?;
    let right_node = graph.get_node(right_id)?;

    // grad_to_left = upstream = [[2,3],[4,5]]
    let grad_to_left = result_node.calc_grad_to_parent(left_node, &upstream_grad, None)?;
    assert_eq!(&grad_to_left, &upstream_grad);

    // grad_to_right = -upstream = [[-2,-3],[-4,-5]]
    let grad_to_right = result_node.calc_grad_to_parent(right_node, &upstream_grad, None)?;
    let expected_right = Tensor::new(&[-2.0, -3.0, -4.0, -5.0], &[2, 2]);
    assert_eq!(&grad_to_right, &expected_right);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 Subtract 通过 graph.backward() 的端到端反向传播
#[test]
fn test_subtract_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = left - right
    let left = graph.new_parameter_node(&[2, 2], Some("left"))?;
    let right = graph.new_parameter_node(&[2, 2], Some("right"))?;
    let result = graph.new_subtract_node(left, right, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_basic_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：left=[[4,6],[8,10]], right=[[1,2],[3,4]], target=[[0,0],[0,0]]
    // result = [[3,4],[5,6]]
    graph.set_node_value(left, Some(&Tensor::new(&[4.0, 6.0, 8.0, 10.0], &[2, 2])))?;
    graph.set_node_value(right, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;

    // 前向传播
    graph.forward(loss)?;

    // result = [[3,4],[5,6]]
    // loss = mean((result - 0)^2) = mean([9,16,25,36]) = 86/4 = 21.5
    let loss_value = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(loss_value.get_data_number().unwrap(), 21.5, epsilon = 1e-6);

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证梯度存在且形状正确
    let left_grad = graph.get_node(left)?.grad().expect("left 应有 grad");
    let right_grad = graph.get_node(right)?.grad().expect("right 应有 grad");
    assert_eq!(left_grad.shape(), &[2, 2]);
    assert_eq!(right_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = result/2 = [[1.5,2],[2.5,3]]
    // ∂loss/∂left = ∂loss/∂result = [[1.5,2],[2.5,3]]
    // ∂loss/∂right = -∂loss/∂result = [[-1.5,-2],[-2.5,-3]]
    let expected_left_grad = Tensor::new(&[1.5, 2.0, 2.5, 3.0], &[2, 2]);
    let expected_right_grad = Tensor::new(&[-1.5, -2.0, -2.5, -3.0], &[2, 2]);
    assert_abs_diff_eq!(left_grad, &expected_left_grad, epsilon = 1e-6);
    assert_abs_diff_eq!(right_grad, &expected_right_grad, epsilon = 1e-6);

    Ok(())
}

// ==================== 广播测试 ====================

/// 测试 Subtract 节点支持广播的创建
#[test]
fn test_subtract_broadcast_creation() {
    let mut graph = GraphInner::new();

    // 1. [3, 4] - [1, 4] -> [3, 4]（行广播）
    {
        let left = graph.new_basic_input_node(&[3, 4], Some("left1")).unwrap();
        let right = graph.new_basic_input_node(&[1, 4], Some("right1")).unwrap();
        let result = graph.new_subtract_node(left, right, Some("sub1")).unwrap();
        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[3, 4]
        );
    }

    // 2. [3, 1] - [1, 4] -> [3, 4]（双向广播）
    {
        let left = graph.new_basic_input_node(&[3, 1], Some("left2")).unwrap();
        let right = graph.new_basic_input_node(&[1, 4], Some("right2")).unwrap();
        let result = graph.new_subtract_node(left, right, Some("sub2")).unwrap();
        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[3, 4]
        );
    }

    // 3. [2, 3, 4] - [1, 1, 4] -> [2, 3, 4]（高维广播）
    {
        let left = graph
            .new_basic_input_node(&[2, 3, 4], Some("left3"))
            .unwrap();
        let right = graph
            .new_basic_input_node(&[1, 1, 4], Some("right3"))
            .unwrap();
        let result = graph.new_subtract_node(left, right, Some("sub3")).unwrap();
        assert_eq!(
            graph.get_node_value_expected_shape(result).unwrap(),
            &[2, 3, 4]
        );
    }
}

/// 测试 Subtract 广播前向传播
///
/// [2, 3] - [1, 3] -> [2, 3]
#[test]
fn test_subtract_broadcast_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建节点：[2, 3] - [1, 3] -> [2, 3]
    let matrix = graph.new_parameter_node(&[2, 3], Some("matrix"))?;
    let bias = graph.new_parameter_node(&[1, 3], Some("bias"))?;
    let result = graph.new_subtract_node(matrix, bias, Some("result"))?;

    // matrix = [[10,20,30], [40,50,60]]
    // bias = [[1, 2, 3]]
    // result = [[10-1,20-2,30-3], [40-1,50-2,60-3]] = [[9,18,27], [39,48,57]]
    graph.set_node_value(
        matrix,
        Some(&Tensor::new(&[10., 20., 30., 40., 50., 60.], &[2, 3])),
    )?;
    graph.set_node_value(bias, Some(&Tensor::new(&[1., 2., 3.], &[1, 3])))?;

    graph.forward(result)?;

    let output = graph.get_node_value(result)?.unwrap();
    let expected = Tensor::new(&[9., 18., 27., 39., 48., 57.], &[2, 3]);
    assert_eq!(output, &expected);

    Ok(())
}

/// 测试 Subtract 广播反向传播（关键测试）
///
/// [2, 3] - [1, 3] -> [2, 3]
/// 反向传播时，对 [1, 3] 的梯度需要沿 axis=0 求和
#[test]
fn test_subtract_broadcast_backward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建节点：result = matrix - bias
    let matrix = graph.new_parameter_node(&[2, 3], Some("matrix"))?;
    let bias = graph.new_parameter_node(&[1, 3], Some("bias"))?;
    let result = graph.new_subtract_node(matrix, bias, Some("result"))?;

    // 设置值（具体值不重要，Subtract 的梯度与值无关）
    graph.set_node_value(
        matrix,
        Some(&Tensor::new(&[10., 20., 30., 40., 50., 60.], &[2, 3])),
    )?;
    graph.set_node_value(bias, Some(&Tensor::new(&[1., 2., 3.], &[1, 3])))?;
    graph.forward(result)?;

    // 直接测试 VJP
    // upstream_grad = [[1,1,1], [1,1,1]] (全1)
    let upstream_grad = Tensor::ones(&[2, 3]);
    let result_node = graph.get_node(result)?;
    let matrix_node = graph.get_node(matrix)?;
    let bias_node = graph.get_node(bias)?;

    // 对 matrix [2,3] 的梯度：直接传递 upstream_grad
    let grad_to_matrix = result_node.calc_grad_to_parent(matrix_node, &upstream_grad, None)?;
    assert_eq!(grad_to_matrix.shape(), &[2, 3]);
    assert_eq!(&grad_to_matrix, &upstream_grad);

    // 对 bias [1,3] 的梯度：-upstream_grad，然后沿 axis=0 求和
    // -[[1,1,1],[1,1,1]] = [[-1,-1,-1],[-1,-1,-1]]
    // sum([[-1,-1,-1],[-1,-1,-1]], axis=0) = [[-2,-2,-2]]
    let grad_to_bias = result_node.calc_grad_to_parent(bias_node, &upstream_grad, None)?;
    assert_eq!(grad_to_bias.shape(), &[1, 3]);
    let expected_bias_grad = Tensor::new(&[-2., -2., -2.], &[1, 3]);
    assert_eq!(&grad_to_bias, &expected_bias_grad);

    Ok(())
}

/// 测试 Subtract 广播反向传播（非全 1 上游梯度）
///
/// 实际训练中，upstream_grad 几乎不会是全 1，而是由链式法则层层计算得到的各种值。
/// 此测试验证 sum_to_shape 在这种真实场景下的正确性。
#[test]
fn test_subtract_broadcast_backward_non_unit() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建节点：result = matrix - bias
    let matrix = graph.new_parameter_node(&[2, 3], Some("matrix"))?;
    let bias = graph.new_parameter_node(&[1, 3], Some("bias"))?;
    let result = graph.new_subtract_node(matrix, bias, Some("result"))?;

    // 设置值
    graph.set_node_value(
        matrix,
        Some(&Tensor::new(&[10., 20., 30., 40., 50., 60.], &[2, 3])),
    )?;
    graph.set_node_value(bias, Some(&Tensor::new(&[1., 2., 3.], &[1, 3])))?;
    graph.forward(result)?;

    // upstream_grad = [[1,2,3], [4,5,6]]（非全 1）
    let upstream_grad = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let result_node = graph.get_node(result)?;
    let bias_node = graph.get_node(bias)?;

    // 对 bias [1,3] 的梯度：-upstream_grad，然后沿 axis=0 求和
    // -[[1,2,3],[4,5,6]] = [[-1,-2,-3],[-4,-5,-6]]
    // sum([[-1,-2,-3],[-4,-5,-6]], axis=0) = [[-5,-7,-9]]
    let grad_to_bias = result_node.calc_grad_to_parent(bias_node, &upstream_grad, None)?;
    assert_eq!(grad_to_bias.shape(), &[1, 3]);
    let expected = Tensor::new(&[-5., -7., -9.], &[1, 3]);
    assert_eq!(&grad_to_bias, &expected);

    Ok(())
}

/// 测试 Subtract 广播端到端反向传播
///
/// 验证广播在完整训练场景中的正确性
#[test]
fn test_subtract_broadcast_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 模拟偏置减法：result = features - bias
    // features [2,3], bias [1,3]
    let features = graph.new_parameter_node(&[2, 3], Some("features"))?;
    let bias = graph.new_parameter_node(&[1, 3], Some("bias"))?;
    let result = graph.new_subtract_node(features, bias, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_basic_input_node(&[2, 3], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    // features = [[1,2,3], [4,5,6]]
    // bias = [[1,1,1]]
    // result = [[0,1,2], [3,4,5]]
    // target = [[0,0,0], [0,0,0]]
    graph.set_node_value(
        features,
        Some(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])),
    )?;
    graph.set_node_value(bias, Some(&Tensor::ones(&[1, 3])))?;
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
    let bias_grad = graph.get_node(bias)?.grad().expect("bias 应有 grad");

    assert_eq!(
        features_grad.shape(),
        &[2, 3],
        "features 梯度形状应为 [2,3]"
    );
    assert_eq!(bias_grad.shape(), &[1, 3], "bias 梯度形状应为 [1,3]");

    // result = [[0,1,2], [3,4,5]]
    // ∂loss/∂result = 2*(result - target)/n = result/3
    //               = [[0, 1/3, 2/3], [1, 4/3, 5/3]]
    //
    // ∂loss/∂features = ∂loss/∂result = [[0, 1/3, 2/3], [1, 4/3, 5/3]]
    // ∂loss/∂bias = -sum(∂loss/∂result, axis=0)
    //             = -[[(0+1), (1/3+4/3), (2/3+5/3)]]
    //             = -[[1, 5/3, 7/3]]
    //             = [[-1, -5/3, -7/3]]
    let expected_bias_grad = Tensor::new(&[-1., -5. / 3., -7. / 3.], &[1, 3]);
    assert_abs_diff_eq!(bias_grad, &expected_bias_grad, epsilon = 1e-4);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 Subtract 节点的动态形状传播
#[test]
fn test_subtract_dynamic_shape_propagation() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建一个固定形状的参数
    let bias = graph
        .parameter(&[1, 16], crate::nn::Init::Zeros, "bias")
        .unwrap();

    // Subtract: h0 - bias
    let result = &h0 - &bias;

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "feature 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "feature 维度应该是 16");
}

/// 测试 Subtract 节点在不同 batch_size 下的前向计算
#[test]
fn test_subtract_dynamic_batch_forward() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]
    let bias = graph
        .parameter(&[1, 16], crate::nn::Init::Ones, "bias")
        .unwrap();

    // Subtract: h0 - bias
    let result = &h0 - &bias;

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

/// 测试 Subtract 节点在不同 batch_size 下的反向传播
#[test]
fn test_subtract_dynamic_batch_backward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::VarLossOps;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]
    let bias = graph
        .parameter(&[1, 4], crate::nn::Init::Ones, "bias")
        .unwrap();

    // Subtract: h0 - bias
    let result = &h0 - &bias;

    // 创建目标和损失
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward + backward：batch=2
    loss.forward().unwrap();
    assert_eq!(result.value().unwrap().unwrap().shape(), &[2, 4]);
    loss.backward().unwrap();

    // 验证 bias 有梯度
    let bias_grad1 = bias.grad().unwrap().unwrap();
    assert_eq!(bias_grad1.shape(), &[1, 4], "bias 梯度形状应为 [1, 4]");

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

    // 验证 bias 仍有正确形状的梯度
    let bias_grad2 = bias.grad().unwrap().unwrap();
    assert_eq!(bias_grad2.shape(), &[1, 4], "bias 梯度形状仍应为 [1, 4]");
}
