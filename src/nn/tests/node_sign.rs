use crate::assert_err;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_node_sign_creation() {
    let mut graph = Graph::new();

    // 1. 测试Input节点作为父节点
    {
        let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
        let sign = graph.new_sign_node(input, Some("sign_with_input")).unwrap();
        // 1.1 验证基本属性
        assert_eq!(graph.get_node_name(sign).unwrap(), "sign_with_input");
        assert_eq!(graph.get_node_parents(sign).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(sign).unwrap().len(), 0);
    }

    // 2. 测试Parameter节点作为父节点
    {
        let param = graph.new_parameter_node(&[2, 2], Some("param1")).unwrap();
        let sign = graph.new_sign_node(param, Some("sign_with_param")).unwrap();
        assert_eq!(graph.get_node_name(sign).unwrap(), "sign_with_param");
        assert_eq!(graph.get_node_parents(sign).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(sign).unwrap().len(), 0);
    }
}

#[test]
fn test_node_sign_name_generation() {
    let mut graph = Graph::new();

    // 1. 测试节点显式命名
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let sign = graph.new_sign_node(input, Some("explicit_sign")).unwrap();
    assert_eq!(graph.get_node_name(sign).unwrap(), "explicit_sign");

    // 2. 测试节点自动命名
    let sign2 = graph.new_sign_node(input, None).unwrap();
    assert_eq!(graph.get_node_name(sign2).unwrap(), "sign_1");

    // 3. 测试节点名称重复
    let result = graph.new_sign_node(input, Some("explicit_sign"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点explicit_sign在图default_graph中重复")
    );
}

#[test]
fn test_node_sign_manually_set_value() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let sign = graph.new_sign_node(input, Some("sign")).unwrap();

    // 1. 测试直接设置Sign节点的值（应该失败）
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_err!(
        graph.set_node_value(sign, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=sign, type=Sign]的值只能通过前向传播计算得到，不能直接设置"
        )
    );

    // 2. 测试清除Sign节点的值（也应该失败）
    assert_err!(
        graph.set_node_value(sign, None),
        GraphError::InvalidOperation(
            "节点[id=2, name=sign, type=Sign]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

#[test]
fn test_node_sign_expected_shape() {
    let mut graph = Graph::new();

    // 1. 测试基本的Sign节点预期形状
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let sign = graph.new_sign_node(input, Some("sign")).unwrap();
    assert_eq!(graph.get_node_value_expected_shape(sign).unwrap(), &[2, 2]);
    assert_eq!(graph.get_node_value_shape(sign).unwrap(), None); // 实际值形状为None（未计算）

    // 2. 测试前向传播后的形状
    let value = Tensor::zeros(&[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();
    graph.forward(sign).unwrap();

    // 2.1 验证前向传播后的形状
    assert_eq!(graph.get_node_value_shape(sign).unwrap().unwrap(), &[2, 2]); // 实际值形状
    assert_eq!(graph.get_node_value_expected_shape(sign).unwrap(), &[2, 2]); // 预期形状保持不变

    // 2.2 测试父节点值在首次前向传播后，再次设置新值后的形状检查
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();

    // 2.2.1 验证预期形状和实际形状
    assert_eq!(graph.get_node_value_expected_shape(sign).unwrap(), &[2, 2]);
    assert_eq!(graph.get_node_value_shape(sign).unwrap().unwrap(), &[2, 2]); // 虽然值已过期，但由于值仍然存在，所以形状不变
}

#[test]
fn test_node_sign_forward_propagation() {
    // 1. 准备测试数据
    // Sign: 正数→1, 负数→-1, 零→0
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    let expected = Tensor::new(&[1.0, -1.0, 0.0, 1.0], &[2, 2]);

    // 2. 测试不同节点类型组合的前向传播
    let node_types = ["input", "parameter"];
    for parent_type in node_types {
        let mut graph = Graph::new();

        // 创建parent节点
        let parent = match parent_type {
            "input" => graph.new_input_node(&[2, 2], Some("input_1")).unwrap(),
            "parameter" => graph
                .new_parameter_node(&[2, 2], Some("parameter_1"))
                .unwrap(),
            _ => unreachable!(),
        };

        // Sign节点
        let sign = graph.new_sign_node(parent, Some("sign")).unwrap();

        // 如果节点是parameter，因创建时其值已隐式初始化过了，所以前向传播应成功
        if parent_type == "parameter" {
            graph.forward(sign).unwrap();
        } else {
            // 如果是input节点，因创建时其值未初始化，所以前向传播应失败
            assert_err!(
                graph.forward(sign),
                GraphError::InvalidOperation(msg) if msg.contains("不能直接前向传播")
            );

            // 设置input节点的值
            graph.set_node_value(parent, Some(&value)).unwrap();

            // 设置值后前向传播应成功
            graph.forward(sign).unwrap();
            let result = graph.get_node_value(sign).unwrap().unwrap();

            // 只有当节点是input时才检查输出值
            if parent_type == "input" {
                assert_eq!(result, &expected);
            }
        }
    }
}

#[test]
fn test_node_sign_forward_values() {
    // 测试 Sign 节点的具体输出值
    let mut graph = Graph::new();

    // 1. 测试包含正数、负数、零的情况（使用 2D 张量，框架要求 2-4 维）
    let input = graph.new_input_node(&[5, 1], Some("input")).unwrap();
    let sign = graph.new_sign_node(input, Some("sign")).unwrap();

    let value = Tensor::new(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[5, 1]);
    graph.set_node_value(input, Some(&value)).unwrap();
    graph.forward(sign).unwrap();

    let result = graph.get_node_value(sign).unwrap().unwrap();
    let expected = Tensor::new(&[-1.0, -1.0, 0.0, 1.0, 1.0], &[5, 1]);
    assert_eq!(result, &expected);

    // 2. 测试极端值
    let extreme_value = Tensor::new(
        &[f32::INFINITY, f32::NEG_INFINITY, f32::MIN, f32::MAX],
        &[2, 2],
    );
    let input2 = graph.new_input_node(&[2, 2], Some("input2")).unwrap();
    let sign2 = graph.new_sign_node(input2, Some("sign2")).unwrap();

    graph.set_node_value(input2, Some(&extreme_value)).unwrap();
    graph.forward(sign2).unwrap();

    let result2 = graph.get_node_value(sign2).unwrap().unwrap();
    // INFINITY → 1, NEG_INFINITY → -1, MIN → -1, MAX → 1
    let expected2 = Tensor::new(&[1.0, -1.0, -1.0, 1.0], &[2, 2]);
    assert_eq!(result2, &expected2);
}

/// 测试 Sign 节点的反向传播（VJP 模式）
///
/// Sign 是不可微节点，VJP 返回 0（梯度不流经此节点）。
/// 这与 PyTorch 行为一致：`torch.sign(x).backward()` 时 x.grad 为 0。
#[test]
fn test_node_sign_backward_propagation() {
    use approx::assert_abs_diff_eq;

    let mut graph = Graph::new();

    // 1. 构建计算图: parent -> sign -> mse_loss
    let parent = graph.new_parameter_node(&[2, 2], Some("parent")).unwrap();
    let sign = graph.new_sign_node(parent, Some("sign")).unwrap();
    let target = graph.new_input_node(&[2, 2], Some("target")).unwrap();
    let loss = graph.new_mse_loss_node(sign, target, None).unwrap();

    // 2. 设置输入值
    let parent_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    let target_value = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
    graph.set_node_value(parent, Some(&parent_value)).unwrap();
    graph.set_node_value(target, Some(&target_value)).unwrap();

    // 3. 前向传播
    graph.forward(loss).unwrap();

    // sign(parent) = [1, -1, 0, 1]
    // loss = mean((sign - target)^2) = mean([1, 1, 0, 1]) = 0.75
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

    // 6. 验证 parent 的梯度应该全为 0（Sign 不可微）
    let grad = graph.get_node_grad(parent).unwrap().unwrap();
    assert_eq!(grad.shape(), &[2, 2]);

    // Sign 的 VJP 返回 0，所以梯度不会传播到 parent
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
