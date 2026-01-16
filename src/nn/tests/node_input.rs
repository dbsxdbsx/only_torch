use crate::assert_err;
use crate::nn::{GraphInner, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_node_input_creation() {
    let mut graph = GraphInner::new();

    // 1. 测试基本创建
    let input = graph.new_input_node(&[2, 3], Some("input1")).unwrap();

    // 1.1 验证基本属性
    assert_eq!(graph.get_node_name(input).unwrap(), "input1");
    assert_eq!(graph.get_node_parents(input).unwrap().len(), 0);
    assert_eq!(graph.get_node_children(input).unwrap().len(), 0);
    assert!(!graph.is_node_inited(input).unwrap()); // Input节点创建时未初始化
}

#[test]
fn test_node_input_creation_with_invalid_shape() {
    let mut graph = GraphInner::new();

    // 测试不同维度的形状（支持 2-4 维，0/1/5 维应该失败）
    for dims in [0, 1, 5] {
        let shape = match dims {
            0 => vec![],
            1 => vec![2],
            5 => vec![2, 2, 2, 2, 2],
            _ => unreachable!(),
        };

        let result = graph.new_input_node(&shape, None);
        assert_err!(
            result,
            GraphError::DimensionMismatch { expected, got, message }
                if *expected == 2 && *got == dims && message == &format!(
                    "节点张量必须是 2-4 维（支持 FC 和 CNN），但收到的维度是 {} 维。",
                    dims
                )
        );
    }

    // 3D 和 4D 现在应该成功（CNN 支持）
    assert!(graph.new_input_node(&[3, 28, 28], Some("input_3d")).is_ok());
    assert!(
        graph
            .new_input_node(&[4, 3, 28, 28], Some("input_4d"))
            .is_ok()
    );
}

#[test]
fn test_node_input_name_generation() {
    let mut graph = GraphInner::new();

    // 1. 测试节点显式命名
    let input1 = graph
        .new_input_node(&[2, 2], Some("explicit_input"))
        .unwrap();
    assert_eq!(graph.get_node_name(input1).unwrap(), "explicit_input");

    // 2. 测试节点自动命名
    let input2 = graph.new_input_node(&[2, 2], None).unwrap();
    assert_eq!(graph.get_node_name(input2).unwrap(), "input_1");

    // 3. 测试节点名称重复
    let result = graph.new_input_node(&[2, 2], Some("explicit_input"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点explicit_input在图default_graph中重复")
    );
}

#[test]
fn test_node_input_manually_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[2, 2], Some("test_input")).unwrap();

    // 1. 测试有效赋值
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    {
        let cloned_tensor = test_value.clone();
        graph.set_node_value(input, Some(&cloned_tensor)).unwrap();
    } // cloned_tensor在这里被释放

    // 1.1 验证节点状态
    assert!(graph.is_node_inited(input).unwrap());
    assert_eq!(graph.get_node_value(input).unwrap().unwrap(), &test_value);

    // 2. 测试错误形状的赋值
    let invalid_cases = [
        Tensor::new(&[1.0], &[1, 1]),
        Tensor::new(&[1.0, 2.0], &[2, 1]),
        Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]),
    ];
    for value in invalid_cases {
        let shape = value.shape().to_vec();
        assert_err!(
            graph.set_node_value(input, Some(&value)),
            GraphError::ShapeMismatch { expected, got, message }
                if expected == &[2, 2] && got == &shape
                    && message == &format!(
                        "新张量的形状 {:?} 与节点 '{}' 现有张量的形状 {:?} 不匹配。",
                        shape,
                        "test_input",
                        &[2, 2]
                    )
        );
    }

    // 3. 测试设置空值（清除值）
    graph.set_node_value(input, None).unwrap();
    assert!(!graph.is_node_inited(input).unwrap());
    assert!(graph.get_node_value(input).unwrap().is_none());
}

#[test]
fn test_node_input_expected_shape() {
    let mut graph = GraphInner::new();

    // 1. 测试基本的Input节点预期形状
    let input = graph.new_input_node(&[2, 3], Some("input")).unwrap();
    assert_eq!(graph.get_node_value_shape(input).unwrap(), None); // 实际值形状为None
    assert_eq!(graph.get_node_value_expected_shape(input).unwrap(), &[2, 3]); // 预期形状已确定

    // 2. 设置值后检查
    let value = Tensor::zeros(&[2, 3]);
    graph.set_node_value(input, Some(&value)).unwrap();
    assert_eq!(graph.get_node_value_shape(input).unwrap().unwrap(), &[2, 3]); // 设置值后实际形状
    assert_eq!(graph.get_node_value_expected_shape(input).unwrap(), &[2, 3]); // 预期形状保持不变

    // 3. 清除值后检查
    graph.set_node_value(input, None).unwrap();
    assert_eq!(graph.get_node_value_shape(input).unwrap(), None); // 清除后实际值形状为None
    assert_eq!(graph.get_node_value_expected_shape(input).unwrap(), &[2, 3]); // 预期形状仍然保持
}

#[test]
fn test_node_input_forward_propagation() {
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();

    // 1. 测试前向传播（应该失败，因为Input节点不支持前向传播）
    assert_err!(
        graph.forward(input),
        GraphError::InvalidOperation(
            "节点[id=1, name=input, type=Input]是输入/参数/状态节点，其值应通过set_value设置，而不是通过父节点前向传播计算"
        )
    );

    // 2. 设置值后仍然不能前向传播
    let value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();
    assert_err!(
        graph.forward(input),
        GraphError::InvalidOperation(
            "节点[id=1, name=input, type=Input]是输入/参数/状态节点，其值应通过set_value设置，而不是通过父节点前向传播计算"
        )
    );
}

/// 测试 Input 节点不应该有梯度（VJP 模式）
///
/// Input 节点是输入数据，不是可学习参数，因此不应该有梯度。
#[test]
fn test_node_input_no_grad() {
    let mut graph = GraphInner::new();

    // 1. 创建输入节点
    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();
    let value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // 2. 对 Input 节点调用 get_node_grad 应该返回错误
    assert_err!(
        graph.get_node_grad(input),
        GraphError::InvalidOperation(msg) if msg == "输入节点[id=1, name=input, type=Input]不应该有梯度"
    );

    // 3. 设置值后，get_node_grad 仍应返回错误
    graph.set_node_value(input, Some(&value)).unwrap();
    assert_err!(
        graph.get_node_grad(input),
        GraphError::InvalidOperation(msg) if msg == "输入节点[id=1, name=input, type=Input]不应该有梯度"
    );

    // 4. zero_grad 后，get_node_grad 仍应返回错误
    graph.zero_grad().unwrap();
    assert_err!(
        graph.get_node_grad(input),
        GraphError::InvalidOperation(msg) if msg == "输入节点[id=1, name=input, type=Input]不应该有梯度"
    );
}

/// 测试 Input 节点在正常计算图反向传播后的行为
///
/// 在完整的计算图中，反向传播到 Input 节点时会无害跳过（Input 是"梯度汇点"），
/// 调用 get_node_grad(input) 仍然返回错误。
#[test]
fn test_node_input_in_computation_graph() {
    let mut graph = GraphInner::new();

    // 1. 构建简单计算图: input -> param -> mse_loss
    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();
    let param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();
    let target = graph.new_input_node(&[2, 2], Some("target")).unwrap();

    // input * param
    let mul = graph.new_multiply_node(input, param, None).unwrap();
    // MSE Loss
    let loss = graph.new_mse_loss_node(mul, target, None).unwrap();

    // 2. 设置输入值
    let input_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let target_val = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    graph.set_node_value(input, Some(&input_val)).unwrap();
    graph.set_node_value(target, Some(&target_val)).unwrap();

    // 3. 前向传播
    graph.forward(loss).unwrap();

    // 4. 反向传播
    graph.zero_grad().unwrap();
    let _loss_val = graph.backward(loss).unwrap();

    // 5. 验证：Parameter 节点应该有梯度
    assert!(graph.get_node_grad(param).unwrap().is_some());

    // 6. 验证：Input 节点仍然不应该有梯度（调用 get_node_grad 返回错误）
    //    反向传播到 Input 节点时是无害跳过的，不会报错，但 get_node_grad 仍返回错误
    assert_err!(
        graph.get_node_grad(input),
        GraphError::InvalidOperation(msg) if msg == "输入节点[id=1, name=input, type=Input]不应该有梯度"
    );
    assert_err!(
        graph.get_node_grad(target),
        GraphError::InvalidOperation(msg) if msg == "输入节点[id=3, name=target, type=Input]不应该有梯度"
    );
}
