use crate::assert_err;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_node_perception_loss_creation() {
    let mut graph = Graph::new();

    // 1. 测试Input节点作为父节点
    {
        let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
        let loss = graph
            .new_perception_loss_node(input, Some("perception_loss_with_input"))
            .unwrap();
        // 1.1 验证基本属性
        assert_eq!(
            graph.get_node_name(loss).unwrap(),
            "perception_loss_with_input"
        );
        assert_eq!(graph.get_node_parents(loss).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(loss).unwrap().len(), 0);
    }

    // 2. 测试Parameter节点作为父节点
    {
        let param = graph.new_parameter_node(&[2, 2], Some("param1")).unwrap();
        let loss = graph
            .new_perception_loss_node(param, Some("perception_loss_with_param"))
            .unwrap();
        assert_eq!(
            graph.get_node_name(loss).unwrap(),
            "perception_loss_with_param"
        );
        assert_eq!(graph.get_node_parents(loss).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(loss).unwrap().len(), 0);
    }
}

#[test]
fn test_node_perception_loss_name_generation() {
    let mut graph = Graph::new();

    // 1. 测试节点显式命名
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let loss = graph
        .new_perception_loss_node(input, Some("explicit_perception_loss"))
        .unwrap();
    assert_eq!(
        graph.get_node_name(loss).unwrap(),
        "explicit_perception_loss"
    );

    // 2. 测试节点自动命名
    let loss2 = graph.new_perception_loss_node(input, None).unwrap();
    assert_eq!(graph.get_node_name(loss2).unwrap(), "perception_loss_1");

    // 3. 测试节点名称重复
    let result = graph.new_perception_loss_node(input, Some("explicit_perception_loss"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点explicit_perception_loss在图default_graph中重复")
    );
}

#[test]
fn test_node_perception_loss_manually_set_value() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let loss = graph.new_perception_loss_node(input, Some("loss")).unwrap();

    // 1. 测试直接设置Loss节点的值（应该失败）
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_err!(
        graph.set_node_value(loss, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=loss, type=PerceptionLoss]的值只能通过前向传播计算得到，不能直接设置"
        )
    );

    // 2. 测试清除Loss节点的值（也应该失败）
    assert_err!(
        graph.set_node_value(loss, None),
        GraphError::InvalidOperation(
            "节点[id=2, name=loss, type=PerceptionLoss]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

#[test]
fn test_node_perception_loss_expected_shape() {
    let mut graph = Graph::new();

    // 1. 测试基本的 Loss 节点预期形状（标量 [1, 1]）
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let loss = graph.new_perception_loss_node(input, Some("loss")).unwrap();
    assert_eq!(graph.get_node_value_expected_shape(loss).unwrap(), &[1, 1]); // 标量输出
    assert_eq!(graph.get_node_value_shape(loss).unwrap(), None); // 实际值形状为 None（未计算）

    // 2. 测试前向传播后的形状
    let value = Tensor::zeros(&[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();
    graph.forward(loss).unwrap();

    // 2.1 验证前向传播后的形状（标量 [1, 1]）
    assert_eq!(graph.get_node_value_shape(loss).unwrap().unwrap(), &[1, 1]);
    assert_eq!(graph.get_node_value_expected_shape(loss).unwrap(), &[1, 1]);

    // 2.2 测试父节点值在首次前向传播后，再次设置新值后的形状检查
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();

    // 2.2.1 验证预期形状和实际形状
    assert_eq!(graph.get_node_value_expected_shape(loss).unwrap(), &[1, 1]);
    assert_eq!(graph.get_node_value_shape(loss).unwrap().unwrap(), &[1, 1]); // 虽然值已过期，但由于值仍然存在，所以形状不变
}

/// 测试 PerceptionLoss 的前向传播
///
/// PerceptionLoss 公式: loss = mean(max(0, -x))
/// - x >= 0 时元素损失为 0
/// - x < 0 时元素损失为 -x
/// - 输出为标量 [1, 1]
#[test]
fn test_node_perception_loss_forward_propagation() {
    // 1. 准备测试数据
    // 输入: [0.5, -1.0, 0.0, 2.0]
    // 元素损失: [0.0, 1.0, 0.0, 0.0]
    // 输出: mean([0.0, 1.0, 0.0, 0.0]) = 0.25
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    let expected_loss = 0.25_f32; // mean(max(0, -x))

    // 2. 测试不同节点类型组合的前向传播
    let node_types = ["input", "parameter"];
    for parent_type in node_types {
        let mut graph = Graph::new();

        // 创建 parent 节点
        let parent = match parent_type {
            "input" => graph.new_input_node(&[2, 2], Some("input_1")).unwrap(),
            "parameter" => graph
                .new_parameter_node(&[2, 2], Some("parameter_1"))
                .unwrap(),
            _ => unreachable!(),
        };

        let loss = graph
            .new_perception_loss_node(parent, Some("loss"))
            .unwrap();

        // 如果节点是 parameter，因创建时其值已隐式初始化过了，所以前向传播应成功
        if parent_type == "parameter" {
            graph.forward(loss).unwrap();
            // Parameter 节点初始化值接近 0，所以 loss 应该接近 0
            let result = graph.get_node_value(loss).unwrap().unwrap();
            assert_eq!(result.shape(), &[1, 1]); // 标量输出
        } else {
            // 如果是 input 节点，因创建时其值未初始化，所以前向传播应失败
            assert_err!(
                graph.forward(loss),
                GraphError::InvalidOperation(
                    "节点[id=1, name=input_1, type=Input]不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为0，而图的前向传播次数为1"
                )
            );

            // 设置 input 节点的值
            graph.set_node_value(parent, Some(&value)).unwrap();

            // 设置值后前向传播应成功
            graph.forward(loss).unwrap();
            let result = graph.get_node_value(loss).unwrap().unwrap();

            // 验证输出是标量 [1, 1] 且值正确
            assert_eq!(result.shape(), &[1, 1]);
            let loss_val = result.get_data_number().unwrap();
            assert!(
                (loss_val - expected_loss).abs() < 1e-6,
                "期望 loss = {}, 实际 loss = {}",
                expected_loss,
                loss_val
            );
        }
    }
}

/// 测试 PerceptionLoss 的反向传播（VJP 模式）
///
/// PerceptionLoss 梯度（mean reduction）:
/// - x >= 0 时梯度为 0
/// - x < 0 时梯度为 -1/n（n 为元素数）
#[test]
fn test_node_perception_loss_backward_propagation() {
    use approx::assert_abs_diff_eq;

    let mut graph = Graph::new();

    // 1. 创建计算图: parent -> perception_loss
    let parent = graph.new_parameter_node(&[2, 3], Some("parent")).unwrap();
    let loss = graph
        .new_perception_loss_node(parent, Some("loss"))
        .unwrap();

    // 2. 测试在前向传播之前进行反向传播（应该失败）
    assert_err!(
        graph.backward(loss),
        GraphError::ComputationError(
            "损失节点 节点[id=2, name=loss, type=PerceptionLoss] 没有值，请先执行 forward"
        )
    );

    // 3. 设置输入值
    // parent = [0.5, -1.0, 1.5, 0.0, -3.0, -2.0]
    // 元素损失 = [0, 1, 0, 0, 3, 2]
    // loss = mean([0, 1, 0, 0, 3, 2]) = 1.0
    let parent_value = Tensor::new(&[0.5, -1.0, 1.5, 0.0, -3.0, -2.0], &[2, 3]);
    graph.set_node_value(parent, Some(&parent_value)).unwrap();

    // 4. 前向传播
    graph.forward(loss).unwrap();

    // 验证 loss 值
    let loss_val = graph
        .get_node_value(loss)
        .unwrap()
        .unwrap()
        .get_data_number()
        .unwrap();
    assert_abs_diff_eq!(loss_val, 1.0, epsilon = 1e-6);

    // 5. 初始时梯度应为空
    assert!(graph.get_node_grad(parent).unwrap().is_none());

    // 6. 反向传播
    graph.zero_grad().unwrap();
    graph.backward(loss).unwrap();

    // 7. 验证 parent 的梯度
    // 梯度 = (x >= 0 ? 0 : -1/n)，n = 6
    // parent = [0.5, -1.0, 1.5, 0.0, -3.0, -2.0]
    // 梯度 = [0, -1/6, 0, 0, -1/6, -1/6]
    let grad = graph.get_node_grad(parent).unwrap().unwrap();
    assert_eq!(grad.shape(), &[2, 3]);

    let n = 6.0_f32;
    assert_abs_diff_eq!(grad[[0, 0]], 0.0, epsilon = 1e-6); // 0.5 >= 0
    assert_abs_diff_eq!(grad[[0, 1]], -1.0 / n, epsilon = 1e-6); // -1.0 < 0
    assert_abs_diff_eq!(grad[[0, 2]], 0.0, epsilon = 1e-6); // 1.5 >= 0
    assert_abs_diff_eq!(grad[[1, 0]], 0.0, epsilon = 1e-6); // 0.0 >= 0
    assert_abs_diff_eq!(grad[[1, 1]], -1.0 / n, epsilon = 1e-6); // -3.0 < 0
    assert_abs_diff_eq!(grad[[1, 2]], -1.0 / n, epsilon = 1e-6); // -2.0 < 0

    // 8. 测试梯度累积 - 需要重新 forward（PyTorch 语义）
    graph.forward(loss).unwrap();
    graph.backward(loss).unwrap();
    let grad_accumulated = graph.get_node_grad(parent).unwrap().unwrap();
    assert_abs_diff_eq!(grad_accumulated[[0, 1]], -2.0 / n, epsilon = 1e-6);

    // 9. zero_grad 后梯度应清零
    graph.zero_grad().unwrap();
    assert!(graph.get_node_grad(parent).unwrap().is_none());

    // 10. 再次反向传播 - 仍应正常工作（需要重新 forward）
    graph.forward(loss).unwrap();
    graph.backward(loss).unwrap();
    let grad_after_clear = graph.get_node_grad(parent).unwrap().unwrap();
    assert_abs_diff_eq!(grad_after_clear[[0, 1]], -1.0 / n, epsilon = 1e-6);
}
