/*
 * @Author       : 老董
 * @Description  : Add 节点单元测试
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

/// 测试 Add 节点创建
#[test]
fn test_add_creation() {
    let mut graph = GraphInner::new();

    // 1. 两个 Input 节点相加
    {
        let input1 = graph.new_input_node(&[2, 3], Some("input1")).unwrap();
        let input2 = graph.new_input_node(&[2, 3], Some("input2")).unwrap();
        let add = graph
            .new_add_node(&[input1, input2], Some("add_inputs"))
            .unwrap();

        assert_eq!(graph.get_node_name(add).unwrap(), "add_inputs");
        assert_eq!(graph.get_node_parents(add).unwrap().len(), 2);
        assert_eq!(graph.get_node_value_expected_shape(add).unwrap(), &[2, 3]);
    }

    // 2. 两个 Parameter 节点相加
    {
        let param1 = graph.new_parameter_node(&[2, 3], Some("param1")).unwrap();
        let param2 = graph.new_parameter_node(&[2, 3], Some("param2")).unwrap();
        let add = graph
            .new_add_node(&[param1, param2], Some("add_params"))
            .unwrap();

        assert_eq!(graph.get_node_name(add).unwrap(), "add_params");
        assert_eq!(graph.get_node_parents(add).unwrap().len(), 2);
        assert_eq!(graph.get_node_value_expected_shape(add).unwrap(), &[2, 3]);
    }

    // 3. 混合 Input 和 Parameter 节点相加
    {
        let input = graph.new_input_node(&[2, 3], Some("input3")).unwrap();
        let param = graph.new_parameter_node(&[2, 3], Some("param3")).unwrap();
        let add = graph
            .new_add_node(&[input, param], Some("add_mixed"))
            .unwrap();

        assert_eq!(graph.get_node_name(add).unwrap(), "add_mixed");
        assert_eq!(graph.get_node_parents(add).unwrap().len(), 2);
        assert_eq!(graph.get_node_value_expected_shape(add).unwrap(), &[2, 3]);
    }

    // 4. 三个父节点相加（Add 支持多父节点）
    {
        let p1 = graph.new_parameter_node(&[2, 2], Some("p1")).unwrap();
        let p2 = graph.new_parameter_node(&[2, 2], Some("p2")).unwrap();
        let p3 = graph.new_parameter_node(&[2, 2], Some("p3")).unwrap();
        let add = graph
            .new_add_node(&[p1, p2, p3], Some("add_three"))
            .unwrap();

        assert_eq!(graph.get_node_parents(add).unwrap().len(), 3);
        assert_eq!(graph.get_node_value_expected_shape(add).unwrap(), &[2, 2]);
    }
}

/// 测试 Add 创建时的形状校验（广播不兼容的情况）
#[test]
fn test_add_creation_invalid_shape() {
    let mut graph = GraphInner::new();

    // 1. 无法广播的形状：[2, 2] + [3, 2]（第一维 2 != 3 且都不是 1）
    let input1 = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let input2 = graph.new_input_node(&[3, 2], Some("input2")).unwrap();

    let result = graph.new_add_node(&[input1, input2], None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 2], [3, 2], "Add节点的父节点形状无法广播")
    );

    // 2. 无法广播的形状：[2, 2] + [2, 3]（第二维 2 != 3 且都不是 1）
    let input3 = graph.new_input_node(&[2, 3], Some("input3")).unwrap();
    let result = graph.new_add_node(&[input1, input3], None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 2], [2, 3], "Add节点的父节点形状无法广播")
    );

    // 3. 三个父节点中有无法广播的
    let input4 = graph.new_input_node(&[2, 2], Some("input4")).unwrap();
    let input5 = graph.new_input_node(&[3, 2], Some("input5")).unwrap();
    let result = graph.new_add_node(&[input1, input4, input5], None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 2], [3, 2], "Add节点的父节点形状无法广播")
    );
}

/// 测试 Add 创建时父节点数量不足
#[test]
fn test_add_creation_insufficient_parents() {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();
    let result = graph.new_add_node(&[input], None);
    assert_err!(
        result,
        GraphError::InvalidOperation("Add节点至少需要2个父节点")
    );
}

/// 测试 Add 节点命名
#[test]
fn test_add_name_generation() {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 3], Some("p1")).unwrap();
    let p2 = graph.new_parameter_node(&[2, 3], Some("p2")).unwrap();

    // 1. 显式命名
    let add1 = graph.new_add_node(&[p1, p2], Some("my_add")).unwrap();
    assert_eq!(graph.get_node_name(add1).unwrap(), "my_add");

    // 2. 自动命名
    let add2 = graph.new_add_node(&[p1, p2], None).unwrap();
    assert_eq!(graph.get_node_name(add2).unwrap(), "add_1");

    // 3. 名称重复
    let result = graph.new_add_node(&[p1, p2], Some("my_add"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_add在图default_graph中重复")
    );
}

/// 测试 Add 节点不能直接设置值
#[test]
fn test_add_cannot_set_value() {
    let mut graph = GraphInner::new();
    let p1 = graph.new_parameter_node(&[2, 2], Some("p1")).unwrap();
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2")).unwrap();
    let add = graph.new_add_node(&[p1, p2], Some("add")).unwrap();

    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_err!(
        graph.set_node_value(add, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=3, name=add, type=Add]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

// ==================== 前向传播测试 ====================

/// 测试 Add 前向传播（两个父节点）
#[test]
fn test_add_forward() {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1")).unwrap();
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2")).unwrap();
    let add = graph.new_add_node(&[p1, p2], Some("add")).unwrap();

    // p1=[[1,2],[3,4]], p2=[[5,6],[7,8]] → add=[[6,8],[10,12]]
    graph
        .set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    graph
        .set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))
        .unwrap();

    graph.forward(add).unwrap();

    let output = graph.get_node_value(add).unwrap().unwrap();
    let expected = Tensor::new(&[6.0, 8.0, 10.0, 12.0], &[2, 2]);
    assert_eq!(output, &expected);
}

/// 测试 Add 前向传播（三个父节点）
#[test]
fn test_add_forward_three_parents() {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1")).unwrap();
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2")).unwrap();
    let p3 = graph.new_parameter_node(&[2, 2], Some("p3")).unwrap();
    let add = graph.new_add_node(&[p1, p2, p3], Some("add")).unwrap();

    // p1=[[1,2],[3,4]], p2=[[5,6],[7,8]], p3=[[10,10],[10,10]]
    // → add=[[16,18],[20,22]]
    graph
        .set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    graph
        .set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))
        .unwrap();
    graph
        .set_node_value(p3, Some(&Tensor::new(&[10.0, 10.0, 10.0, 10.0], &[2, 2])))
        .unwrap();

    graph.forward(add).unwrap();

    let output = graph.get_node_value(add).unwrap().unwrap();
    let expected = Tensor::new(&[16.0, 18.0, 20.0, 22.0], &[2, 2]);
    assert_eq!(output, &expected);
}

// ==================== 节点级反向传播测试（直接调用 calc_grad_to_parent）====================

/// 测试 Add 对第一个父节点的梯度计算
///
/// 对于 result = p1 + p2，有 ∂result/∂p1 = I（恒等映射）
/// VJP: grad_to_p1 = upstream_grad
#[test]
fn test_add_backward_to_first_parent() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2"))?;
    let add = graph.new_add_node(&[p1, p2], Some("add"))?;

    // 设置值并前向传播
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))?;
    graph.forward(add)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 2]);
    let add_node = graph.get_node(add)?;
    let p1_node = graph.get_node(p1)?;

    // Add 不需要 assistant_parent，梯度直接传递
    let grad = add_node.calc_grad_to_parent(p1_node, &upstream_grad, None)?;

    // grad_to_p1 = upstream_grad（恒等传递）
    assert_eq!(grad.shape(), &[2, 2]);
    assert_eq!(&grad, &upstream_grad);

    Ok(())
}

/// 测试 Add 对第二个父节点的梯度计算
///
/// 对于 result = p1 + p2，有 ∂result/∂p2 = I（恒等映射）
/// VJP: grad_to_p2 = upstream_grad
#[test]
fn test_add_backward_to_second_parent() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2"))?;
    let add = graph.new_add_node(&[p1, p2], Some("add"))?;

    // 设置值并前向传播
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))?;
    graph.forward(add)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 2]);
    let add_node = graph.get_node(add)?;
    let p2_node = graph.get_node(p2)?;

    let grad = add_node.calc_grad_to_parent(p2_node, &upstream_grad, None)?;

    // grad_to_p2 = upstream_grad（恒等传递）
    assert_eq!(grad.shape(), &[2, 2]);
    assert_eq!(&grad, &upstream_grad);

    Ok(())
}

/// 测试 Add 梯度计算（非单位 upstream_grad）
#[test]
fn test_add_backward_with_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2"))?;
    let add = graph.new_add_node(&[p1, p2], Some("add"))?;

    // 设置值并前向传播
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))?;
    graph.forward(add)?;

    // upstream_grad = [[1,2],[3,4]]（非全1）
    let upstream_grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let add_node = graph.get_node(add)?;
    let p1_node = graph.get_node(p1)?;
    let p2_node = graph.get_node(p2)?;

    // 对 p1 的梯度：直接传递 upstream_grad
    let grad_to_p1 = add_node.calc_grad_to_parent(p1_node, &upstream_grad, None)?;
    assert_eq!(&grad_to_p1, &upstream_grad);

    // 对 p2 的梯度：直接传递 upstream_grad
    let grad_to_p2 = add_node.calc_grad_to_parent(p2_node, &upstream_grad, None)?;
    assert_eq!(&grad_to_p2, &upstream_grad);

    Ok(())
}

/// 测试 Add 梯度计算（负数值）
///
/// 验证 VJP 在负数值场景下的正确性
#[test]
fn test_add_backward_with_negative_values() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2"))?;
    let add = graph.new_add_node(&[p1, p2], Some("add"))?;

    // p1=[[-1,-2],[-3,-4]], p2=[[5,-6],[7,-8]]
    graph.set_node_value(p1, Some(&Tensor::new(&[-1.0, -2.0, -3.0, -4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, -6.0, 7.0, -8.0], &[2, 2])))?;
    graph.forward(add)?;

    // 验证前向传播：[[-1+5, -2-6], [-3+7, -4-8]] = [[4,-8],[4,-12]]
    let output = graph.get_node_value(add)?.unwrap();
    let expected_output = Tensor::new(&[4.0, -8.0, 4.0, -12.0], &[2, 2]);
    assert_eq!(output, &expected_output);

    // upstream_grad = [[-1,2],[-3,4]]（含负数）
    let upstream_grad = Tensor::new(&[-1.0, 2.0, -3.0, 4.0], &[2, 2]);
    let add_node = graph.get_node(add)?;
    let p1_node = graph.get_node(p1)?;
    let p2_node = graph.get_node(p2)?;

    // Add 的梯度恒等传递，与值无关
    let grad_to_p1 = add_node.calc_grad_to_parent(p1_node, &upstream_grad, None)?;
    let grad_to_p2 = add_node.calc_grad_to_parent(p2_node, &upstream_grad, None)?;
    assert_eq!(&grad_to_p1, &upstream_grad);
    assert_eq!(&grad_to_p2, &upstream_grad);

    Ok(())
}

/// 测试 Add 对三个父节点的梯度计算
#[test]
fn test_add_backward_three_parents() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2"))?;
    let p3 = graph.new_parameter_node(&[2, 2], Some("p3"))?;
    let add = graph.new_add_node(&[p1, p2, p3], Some("add"))?;

    // 设置值并前向传播
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))?;
    graph.set_node_value(p3, Some(&Tensor::new(&[10.0, 10.0, 10.0, 10.0], &[2, 2])))?;
    graph.forward(add)?;

    // upstream_grad = [[2,4],[6,8]]
    let upstream_grad = Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    let add_node = graph.get_node(add)?;
    let p1_node = graph.get_node(p1)?;
    let p2_node = graph.get_node(p2)?;
    let p3_node = graph.get_node(p3)?;

    // 所有父节点的梯度都相同，等于 upstream_grad
    let grad_to_p1 = add_node.calc_grad_to_parent(p1_node, &upstream_grad, None)?;
    let grad_to_p2 = add_node.calc_grad_to_parent(p2_node, &upstream_grad, None)?;
    let grad_to_p3 = add_node.calc_grad_to_parent(p3_node, &upstream_grad, None)?;

    assert_eq!(&grad_to_p1, &upstream_grad);
    assert_eq!(&grad_to_p2, &upstream_grad);
    assert_eq!(&grad_to_p3, &upstream_grad);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 Add 通过 graph.backward() 的端到端反向传播
///
/// 构建简单图：result = p1 + p2 → loss = MSE(result, target)
#[test]
fn test_add_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = p1 + p2
    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2"))?;
    let result = graph.new_add_node(&[p1, p2], Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：p1=[[1,2],[3,4]], p2=[[5,6],[7,8]], target=[[0,0],[0,0]]
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;

    // 前向传播
    graph.forward(loss)?;

    // result = [[6,8],[10,12]]
    // loss = mean((result - 0)^2) = mean([36,64,100,144]) = 344/4 = 86
    let loss_value = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(loss_value.get_data_number().unwrap(), 86.0, epsilon = 1e-6);

    // 反向传播（验证 backward 返回 loss 标量值 —— Phase 2 API 契约）
    graph.zero_grad()?;
    let loss_returned = graph.backward(loss)?;
    assert_abs_diff_eq!(loss_returned, 86.0, epsilon = 1e-6);

    // 验证梯度存在且形状正确
    let p1_grad = graph.get_node(p1)?.grad().expect("p1 应有 grad");
    let p2_grad = graph.get_node(p2)?.grad().expect("p2 应有 grad");
    assert_eq!(p1_grad.shape(), &[2, 2]);
    assert_eq!(p2_grad.shape(), &[2, 2]);

    // ∂loss/∂result = 2*(result - target)/n = 2*result/4 = result/2 = [[3,4],[5,6]]
    // ∂loss/∂p1 = ∂loss/∂result (因为 ∂result/∂p1 = I)
    // ∂loss/∂p2 = ∂loss/∂result (因为 ∂result/∂p2 = I)
    let expected_grad = Tensor::new(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);
    assert_eq!(p1_grad, &expected_grad);
    assert_eq!(p2_grad, &expected_grad);

    // Add 节点对两个输入的梯度应该相同
    assert_eq!(p1_grad, p2_grad);

    Ok(())
}

/// 测试 Add 端到端反向传播（三个父节点）
#[test]
fn test_add_backward_e2e_three_parents() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = p1 + p2 + p3
    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2"))?;
    let p3 = graph.new_parameter_node(&[2, 2], Some("p3"))?;
    let result = graph.new_add_node(&[p1, p2, p3], Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：p1=[[1,1],[1,1]], p2=[[2,2],[2,2]], p3=[[3,3],[3,3]], target=[[0,0],[0,0]]
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0; 4], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[2.0; 4], &[2, 2])))?;
    graph.set_node_value(p3, Some(&Tensor::new(&[3.0; 4], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;

    // 前向传播
    graph.forward(loss)?;

    // result = [[6,6],[6,6]]
    // loss = mean((6-0)^2) = 36
    let loss_value = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(loss_value.get_data_number().unwrap(), 36.0, epsilon = 1e-6);

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // ∂loss/∂result = 2*(result - target)/n = 2*6/4 = 3
    // 所有父节点的梯度相同
    let p1_grad = graph.get_node(p1)?.grad().expect("p1 应有 grad");
    let p2_grad = graph.get_node(p2)?.grad().expect("p2 应有 grad");
    let p3_grad = graph.get_node(p3)?.grad().expect("p3 应有 grad");

    let expected_grad = Tensor::new(&[3.0; 4], &[2, 2]);
    assert_eq!(p1_grad, &expected_grad);
    assert_eq!(p2_grad, &expected_grad);
    assert_eq!(p3_grad, &expected_grad);

    Ok(())
}

// ==================== 梯度累积测试 ====================

/// 测试 Add 梯度累积
///
/// 验证语义：参数的 grad 在多次 backward 之间累积，直到调用 zero_grad()。
/// 这是 PyTorch 兼容的行为，支持"micro-batch 梯度累积"场景。
#[test]
fn test_add_gradient_accumulation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2"))?;
    let result = graph.new_add_node(&[p1, p2], Some("result"))?;
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;
    graph.forward(loss)?;

    // 第1次反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;
    let grad_first = graph.get_node(p1)?.grad().unwrap().clone();

    // 第2次反向传播（梯度累积）- 需要重新 forward（PyTorch 语义）
    graph.forward(loss)?;
    graph.backward(loss)?;
    let grad_second = graph.get_node(p1)?.grad().unwrap();
    assert_eq!(grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    graph.forward(loss)?;
    graph.backward(loss)?;
    let grad_after_clear = graph.get_node(p1)?.grad().unwrap();
    assert_eq!(grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 广播测试 ====================

/// 测试 Add 节点支持广播的创建
#[test]
fn test_add_broadcast_creation() {
    let mut graph = GraphInner::new();

    // 1. [3, 4] + [1, 4] -> [3, 4]（行广播）
    {
        let input1 = graph.new_input_node(&[3, 4], Some("input1")).unwrap();
        let input2 = graph.new_input_node(&[1, 4], Some("input2")).unwrap();
        let add = graph.new_add_node(&[input1, input2], Some("add1")).unwrap();
        assert_eq!(graph.get_node_value_expected_shape(add).unwrap(), &[3, 4]);
    }

    // 2. [3, 4] + [3, 1] -> [3, 4]（列广播）
    {
        let input1 = graph.new_input_node(&[3, 4], Some("input3")).unwrap();
        let input2 = graph.new_input_node(&[3, 1], Some("input4")).unwrap();
        let add = graph.new_add_node(&[input1, input2], Some("add2")).unwrap();
        assert_eq!(graph.get_node_value_expected_shape(add).unwrap(), &[3, 4]);
    }

    // 3. [3, 1] + [1, 4] -> [3, 4]（双向广播）
    {
        let input1 = graph.new_input_node(&[3, 1], Some("input5")).unwrap();
        let input2 = graph.new_input_node(&[1, 4], Some("input6")).unwrap();
        let add = graph.new_add_node(&[input1, input2], Some("add3")).unwrap();
        assert_eq!(graph.get_node_value_expected_shape(add).unwrap(), &[3, 4]);
    }

    // 4. [2, 3, 4] + [1, 1, 4] -> [2, 3, 4]（高维广播）
    {
        let input1 = graph.new_input_node(&[2, 3, 4], Some("input7")).unwrap();
        let input2 = graph.new_input_node(&[1, 1, 4], Some("input8")).unwrap();
        let add = graph.new_add_node(&[input1, input2], Some("add4")).unwrap();
        assert_eq!(
            graph.get_node_value_expected_shape(add).unwrap(),
            &[2, 3, 4]
        );
    }
}

/// 测试 Add 广播前向传播
///
/// [3, 4] + [1, 4] -> [3, 4]
#[test]
fn test_add_broadcast_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建节点：[3, 4] + [1, 4] -> [3, 4]
    let p1 = graph.new_parameter_node(&[3, 4], Some("matrix"))?;
    let p2 = graph.new_parameter_node(&[1, 4], Some("bias"))?;
    let add = graph.new_add_node(&[p1, p2], Some("add"))?;

    // p1 = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    // p2 = [[10, 20, 30, 40]]
    // add = [[11,22,33,44], [15,26,37,48], [19,30,41,52]]
    graph.set_node_value(
        p1,
        Some(&Tensor::new(
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            &[3, 4],
        )),
    )?;
    graph.set_node_value(p2, Some(&Tensor::new(&[10., 20., 30., 40.], &[1, 4])))?;

    graph.forward(add)?;

    let output = graph.get_node_value(add)?.unwrap();
    let expected = Tensor::new(
        &[11., 22., 33., 44., 15., 26., 37., 48., 19., 30., 41., 52.],
        &[3, 4],
    );
    assert_eq!(output, &expected);

    Ok(())
}

/// 测试 Add 广播反向传播（关键测试）
///
/// [3, 4] + [1, 4] -> [3, 4]
/// 反向传播时，对 [1, 4] 的梯度需要沿 axis=0 求和
#[test]
fn test_add_broadcast_backward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建节点：result = matrix + bias
    let matrix = graph.new_parameter_node(&[3, 4], Some("matrix"))?;
    let bias = graph.new_parameter_node(&[1, 4], Some("bias"))?;
    let result = graph.new_add_node(&[matrix, bias], Some("result"))?;

    // 设置值
    graph.set_node_value(
        matrix,
        Some(&Tensor::new(
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            &[3, 4],
        )),
    )?;
    graph.set_node_value(bias, Some(&Tensor::new(&[10., 20., 30., 40.], &[1, 4])))?;
    graph.forward(result)?;

    // 直接测试 VJP
    // upstream_grad = [[1,1,1,1], [1,1,1,1], [1,1,1,1]] (全1)
    let upstream_grad = Tensor::ones(&[3, 4]);
    let result_node = graph.get_node(result)?;
    let matrix_node = graph.get_node(matrix)?;
    let bias_node = graph.get_node(bias)?;

    // 对 matrix [3,4] 的梯度：直接传递 upstream_grad
    let grad_to_matrix = result_node.calc_grad_to_parent(matrix_node, &upstream_grad, None)?;
    assert_eq!(grad_to_matrix.shape(), &[3, 4]);
    assert_eq!(&grad_to_matrix, &upstream_grad);

    // 对 bias [1,4] 的梯度：沿 axis=0 求和
    // sum([[1,1,1,1], [1,1,1,1], [1,1,1,1]], axis=0) = [[3,3,3,3]]
    let grad_to_bias = result_node.calc_grad_to_parent(bias_node, &upstream_grad, None)?;
    assert_eq!(grad_to_bias.shape(), &[1, 4]);
    let expected_bias_grad = Tensor::new(&[3., 3., 3., 3.], &[1, 4]);
    assert_eq!(&grad_to_bias, &expected_bias_grad);

    Ok(())
}

/// 测试 Add 广播反向传播（非全 1 上游梯度）
///
/// 实际训练中，upstream_grad 几乎不会是全 1，而是由链式法则层层计算得到的各种值。
/// 此测试验证 sum_to_shape 在这种真实场景下的正确性。
#[test]
fn test_add_broadcast_backward_non_unit() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建节点：result = matrix + bias
    let matrix = graph.new_parameter_node(&[2, 3], Some("matrix"))?;
    let bias = graph.new_parameter_node(&[1, 3], Some("bias"))?;
    let result = graph.new_add_node(&[matrix, bias], Some("result"))?;

    // 设置值（具体值不重要，Add 的梯度与值无关）
    graph.set_node_value(
        matrix,
        Some(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])),
    )?;
    graph.set_node_value(bias, Some(&Tensor::new(&[10., 20., 30.], &[1, 3])))?;
    graph.forward(result)?;

    // upstream_grad = [[1,2,3], [4,5,6]]
    let upstream_grad = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let result_node = graph.get_node(result)?;
    let bias_node = graph.get_node(bias)?;

    // 对 bias [1,3] 的梯度：沿 axis=0 求和
    // sum([[1,2,3], [4,5,6]], axis=0) = [[5,7,9]]
    let grad_to_bias = result_node.calc_grad_to_parent(bias_node, &upstream_grad, None)?;
    assert_eq!(grad_to_bias.shape(), &[1, 3]);
    let expected = Tensor::new(&[5., 7., 9.], &[1, 3]);
    assert_eq!(&grad_to_bias, &expected);

    Ok(())
}

/// 测试 Add 广播端到端反向传播
///
/// 这是最重要的测试：验证广播在完整训练场景中的正确性
#[test]
fn test_add_broadcast_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 模拟 Linear 层：output = X @ W + bias
    // 简化为：result = matrix + bias，其中 matrix [2,3], bias [1,3]
    let matrix = graph.new_parameter_node(&[2, 3], Some("matrix"))?;
    let bias = graph.new_parameter_node(&[1, 3], Some("bias"))?;
    let result = graph.new_add_node(&[matrix, bias], Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_input_node(&[2, 3], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值
    // matrix = [[1,2,3], [4,5,6]]
    // bias = [[10,20,30]]
    // result = [[11,22,33], [14,25,36]]
    // target = [[0,0,0], [0,0,0]]
    graph.set_node_value(
        matrix,
        Some(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])),
    )?;
    graph.set_node_value(bias, Some(&Tensor::new(&[10., 20., 30.], &[1, 3])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 3])))?;

    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证梯度形状
    let matrix_grad = graph.get_node(matrix)?.grad().expect("matrix 应有 grad");
    let bias_grad = graph.get_node(bias)?.grad().expect("bias 应有 grad");

    assert_eq!(matrix_grad.shape(), &[2, 3], "matrix 梯度形状应为 [2,3]");
    assert_eq!(bias_grad.shape(), &[1, 3], "bias 梯度形状应为 [1,3]");

    // ∂loss/∂result = 2*(result - target)/n = result/3
    // result = [[11,22,33], [14,25,36]]
    // ∂loss/∂result = [[11/3, 22/3, 33/3], [14/3, 25/3, 36/3]]
    //               ≈ [[3.667, 7.333, 11], [4.667, 8.333, 12]]
    //
    // ∂loss/∂matrix = ∂loss/∂result（形状相同，直接传递）
    // ∂loss/∂bias = sum(∂loss/∂result, axis=0)
    //             = [[(11+14)/3, (22+25)/3, (33+36)/3]]
    //             = [[25/3, 47/3, 69/3]]
    //             ≈ [[8.333, 15.667, 23]]

    let expected_bias_grad = Tensor::new(&[25. / 3., 47. / 3., 69. / 3.], &[1, 3]);
    assert_abs_diff_eq!(bias_grad, &expected_bias_grad, epsilon = 1e-4);

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 Add 节点的动态形状传播
#[test]
fn test_add_dynamic_shape_propagation() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建一个固定形状的参数
    let bias = graph.parameter(&[1, 16], crate::nn::Init::Zeros, "bias").unwrap();

    // Add: h0 + bias
    let result = &h0 + &bias;

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "feature 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "feature 维度应该是 16");
}

/// 测试 Add 节点在不同 batch_size 下的前向计算
#[test]
fn test_add_dynamic_batch_forward() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]
    let bias = graph.parameter(&[1, 16], crate::nn::Init::Ones, "bias").unwrap();

    // Add: h0 + bias
    let result = &h0 + &bias;

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

/// 测试 Add 节点在不同 batch_size 下的反向传播
#[test]
fn test_add_dynamic_batch_backward() {
    use crate::nn::var_ops::VarLossOps;
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]
    let bias = graph.parameter(&[1, 4], crate::nn::Init::Ones, "bias").unwrap();

    // Add: h0 + bias
    let result = &h0 + &bias;

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
