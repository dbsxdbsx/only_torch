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
use crate::nn::{GraphInner, GraphError};
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

/// 测试 Add 创建时的形状校验
#[test]
fn test_add_creation_invalid_shape() {
    let mut graph = GraphInner::new();

    // 1. 第一个和第二个父节点形状不同（行数不同）
    let input1 = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let input2 = graph.new_input_node(&[3, 2], Some("input2")).unwrap();

    let result = graph.new_add_node(&[input1, input2], None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 2], [3, 2], "Add节点的所有父节点形状必须相同")
    );

    // 2. 列数不同
    let input3 = graph.new_input_node(&[2, 3], Some("input3")).unwrap();
    let result = graph.new_add_node(&[input1, input3], None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 2], [2, 3], "Add节点的所有父节点形状必须相同")
    );

    // 3. 三个父节点中有形状不同的
    let input4 = graph.new_input_node(&[2, 2], Some("input4")).unwrap();
    let input5 = graph.new_input_node(&[3, 2], Some("input5")).unwrap();
    let result = graph.new_add_node(&[input1, input4, input5], None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 2], [3, 2], "Add节点的所有父节点形状必须相同")
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
