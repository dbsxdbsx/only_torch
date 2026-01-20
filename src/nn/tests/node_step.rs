use crate::assert_err;
use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;

#[test]
fn test_node_step_creation() {
    let mut graph = GraphInner::new();

    // 1. 测试Input节点作为父节点
    {
        let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
        let step = graph.new_step_node(input, Some("step_with_input")).unwrap();
        // 1.1 验证基本属性
        assert_eq!(graph.get_node_name(step).unwrap(), "step_with_input");
        assert_eq!(graph.get_node_parents(step).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(step).unwrap().len(), 0);
    }

    // 2. 测试Parameter节点作为父节点
    {
        let param = graph.new_parameter_node(&[2, 2], Some("param1")).unwrap();
        let step = graph.new_step_node(param, Some("step_with_param")).unwrap();
        assert_eq!(graph.get_node_name(step).unwrap(), "step_with_param");
        assert_eq!(graph.get_node_parents(step).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(step).unwrap().len(), 0);
    }
}

#[test]
fn test_node_step_name_generation() {
    let mut graph = GraphInner::new();

    // 1. 测试节点显式命名
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let step = graph.new_step_node(input, Some("explicit_step")).unwrap();
    assert_eq!(graph.get_node_name(step).unwrap(), "explicit_step");

    // 2. 测试节点自动命名
    let step2 = graph.new_step_node(input, None).unwrap();
    assert_eq!(graph.get_node_name(step2).unwrap(), "step_1");

    // 3. 测试节点名称重复
    let result = graph.new_step_node(input, Some("explicit_step"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点explicit_step在图default_graph中重复")
    );
}

#[test]
fn test_node_step_manually_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let step = graph.new_step_node(input, Some("step")).unwrap();

    // 1. 测试直接设置Step节点的值（应该失败）
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_err!(
        graph.set_node_value(step, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=step, type=Step]的值只能通过前向传播计算得到，不能直接设置"
        )
    );

    // 2. 测试清除Step节点的值（也应该失败）
    assert_err!(
        graph.set_node_value(step, None),
        GraphError::InvalidOperation(
            "节点[id=2, name=step, type=Step]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

#[test]
fn test_node_step_expected_shape() {
    let mut graph = GraphInner::new();

    // 1. 测试基本的Step节点预期形状
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let step = graph.new_step_node(input, Some("step")).unwrap();
    assert_eq!(graph.get_node_value_expected_shape(step).unwrap(), &[2, 2]);
    assert_eq!(graph.get_node_value_shape(step).unwrap(), None); // 实际值形状为None（未计算）

    // 2. 测试前向传播后的形状
    let value = Tensor::zeros(&[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();
    graph.forward(step).unwrap();

    // 2.1 验证前向传播后的形状
    assert_eq!(graph.get_node_value_shape(step).unwrap().unwrap(), &[2, 2]); // 实际值形状
    assert_eq!(graph.get_node_value_expected_shape(step).unwrap(), &[2, 2]); // 预期形状保持不变

    // 2.2 测试父节点值在首次前向传播后，再次设置新值后的形状检查
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();

    // 2.2.1 验证预期形状和实际形状
    assert_eq!(graph.get_node_value_expected_shape(step).unwrap(), &[2, 2]);
    assert_eq!(graph.get_node_value_shape(step).unwrap().unwrap(), &[2, 2]); // 虽然值已过期，但由于值仍然存在，所以形状不变
}

#[test]
fn test_node_step_forward_propagation() {
    // 1. 准备测试数据 (与Python测试tests\calc_jacobi_by_pytorch\node_step.py保持一致)
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    let expected = Tensor::new(&[1.0, 0.0, 1.0, 1.0], &[2, 2]);

    // 2. 测试不同节点类型组合的前向传播
    let node_types = ["input", "parameter"];
    for parent_type in node_types {
        let mut graph = GraphInner::new();

        // 创建parent节点
        let parent = match parent_type {
            "input" => graph.new_input_node(&[2, 2], Some("input_1")).unwrap(),
            "parameter" => graph
                .new_parameter_node(&[2, 2], Some("parameter_1"))
                .unwrap(),
            _ => unreachable!(),
        };

        // Step节点总是可训练的
        let step = graph.new_step_node(parent, Some("step")).unwrap();

        // 如果节点是parameter，因创建时其值已隐式初始化过了，所以前向传播应成功
        if parent_type == "parameter" {
            graph.forward(step).unwrap();
        } else {
            // 如果是input节点，因创建时其值未初始化，所以前向传播应失败
            assert_err!(
                graph.forward(step),
                GraphError::InvalidOperation(msg) if msg.contains("不能直接前向传播")
            );

            // 设置input节点的值
            graph.set_node_value(parent, Some(&value)).unwrap();

            // 设置值后前向传播应成功
            graph.forward(step).unwrap();
            let result = graph.get_node_value(step).unwrap().unwrap();

            // 只有当节点是input时才检查输出值
            if parent_type == "input" {
                assert_eq!(result, &expected);
            }
        }
    }
}

#[test]
fn test_node_step_forward_values() {
    // 测试 Step 节点的具体输出值
    let mut graph = GraphInner::new();

    // 1. 测试包含正数、负数、零的情况（使用 2D 张量，框架要求 2-4 维）
    // Step: x >= 0 → 1, x < 0 → 0
    let input = graph.new_input_node(&[5, 1], Some("input")).unwrap();
    let step = graph.new_step_node(input, Some("step")).unwrap();

    let value = Tensor::new(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[5, 1]);
    graph.set_node_value(input, Some(&value)).unwrap();
    graph.forward(step).unwrap();

    let result = graph.get_node_value(step).unwrap().unwrap();
    // 注意：0.0 >= 0 为 true，所以输出 1.0
    let expected = Tensor::new(&[0.0, 0.0, 1.0, 1.0, 1.0], &[5, 1]);
    assert_eq!(result, &expected);

    // 2. 测试极端值
    let extreme_value = Tensor::new(
        &[f32::INFINITY, f32::NEG_INFINITY, f32::MIN, f32::MAX],
        &[2, 2],
    );
    let input2 = graph.new_input_node(&[2, 2], Some("input2")).unwrap();
    let step2 = graph.new_step_node(input2, Some("step2")).unwrap();

    graph.set_node_value(input2, Some(&extreme_value)).unwrap();
    graph.forward(step2).unwrap();

    let result2 = graph.get_node_value(step2).unwrap().unwrap();
    // INFINITY >= 0 → 1, NEG_INFINITY < 0 → 0, MIN < 0 → 0, MAX >= 0 → 1
    let expected2 = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assert_eq!(result2, &expected2);
}

/// 测试 Step 节点的反向传播（VJP 模式）
///
/// Step 是不可微节点，VJP 返回 0（梯度不流经此节点）。
/// Step 函数：x >= 0 → 1, x < 0 → 0（Heaviside 阶跃函数）
#[test]
fn test_node_step_backward_propagation() {
    use approx::assert_abs_diff_eq;

    let mut graph = GraphInner::new();

    // 1. 构建计算图: parent -> step -> mse_loss
    let parent = graph.new_parameter_node(&[2, 2], Some("parent")).unwrap();
    let step = graph.new_step_node(parent, Some("step")).unwrap();
    let target = graph.new_input_node(&[2, 2], Some("target")).unwrap();
    let loss = graph.new_mse_loss_node(step, target, None).unwrap();

    // 2. 设置输入值
    // parent = [0.5, -1.0, 0.0, 2.0]
    // step(parent) = [1, 0, 1, 1]
    let parent_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    let target_value = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
    graph.set_node_value(parent, Some(&parent_value)).unwrap();
    graph.set_node_value(target, Some(&target_value)).unwrap();

    // 3. 前向传播
    graph.forward(loss).unwrap();

    // step(parent) = [1, 0, 1, 1]
    // loss = mean((step - target)^2) = mean([1, 0, 1, 1]) = 0.75
    let loss_val = graph
        .get_node_value(loss)
        .unwrap()
        .unwrap()
        .get_data_number()
        .unwrap();
    assert_abs_diff_eq!(loss_val, 0.75, epsilon = 1e-6);

    // 4. 初始时梯度应为空
    assert!(graph.get_node_grad(parent).unwrap().is_none());

    // 5. 反向传播
    graph.zero_grad().unwrap();
    graph.backward(loss).unwrap();

    // 6. 验证 parent 的梯度应该全为 0（Step 不可微）
    let grad = graph.get_node_grad(parent).unwrap().unwrap();
    assert_eq!(grad.shape(), &[2, 2]);

    // Step 的 VJP 返回 0，所以梯度不会传播到 parent
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 0.0, epsilon = 1e-6);

    // 7. 测试梯度累积（0 + 0 = 0）- 需要重新 forward（PyTorch 语义）
    graph.forward(loss).unwrap();
    graph.backward(loss).unwrap();
    let grad_accumulated = graph.get_node_grad(parent).unwrap().unwrap();
    assert_abs_diff_eq!(grad_accumulated[[0, 0]], 0.0, epsilon = 1e-6);

    // 8. zero_grad 后梯度应清零
    graph.zero_grad().unwrap();
    assert!(graph.get_node_grad(parent).unwrap().is_none());
}
