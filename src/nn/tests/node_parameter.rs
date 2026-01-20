use approx::assert_abs_diff_eq;

use crate::assert_err;
use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;

#[test]
fn test_node_parameter_creation() {
    let mut graph = GraphInner::new();

    // 1. 测试基本创建
    let param = graph.new_parameter_node(&[2, 3], Some("param1")).unwrap();

    // 1.1 验证基本属性
    assert_eq!(graph.get_node_name(param).unwrap(), "param1");
    assert_eq!(graph.get_node_parents(param).unwrap().len(), 0);
    assert_eq!(graph.get_node_children(param).unwrap().len(), 0);
    assert!(graph.is_node_inited(param).unwrap()); // Parameter节点创建时已初始化

    // 1.2 验证初始化值
    let value = graph.get_node_value(param).unwrap().unwrap();
    let mean = value.mean();
    let std_dev = value.std_dev();
    assert_abs_diff_eq!(mean, 0.0, epsilon = 0.1); // 均值应接近0
    assert_abs_diff_eq!(std_dev, 0.001, epsilon = 0.001); // 标准差应接近0.001
}

#[test]
fn test_node_parameter_creation_with_invalid_shape() {
    let mut graph = GraphInner::new();

    // 测试不同维度的形状（支持 2-4 维，0/1/5 维应该失败）
    for dims in [0, 1, 5] {
        let shape = match dims {
            0 => vec![],
            1 => vec![2],
            5 => vec![2, 2, 2, 2, 2],
            _ => unreachable!(),
        };

        let result = graph.new_parameter_node(&shape, None);
        assert_err!(
            result,
            GraphError::DimensionMismatch { expected, got, message }
                if *expected == 2 && *got == dims && message == &format!(
                    "参数张量必须是 2-4 维（支持 FC 权重和 CNN 卷积核），但收到的维度是 {} 维。",
                    dims
                )
        );
    }

    // 3D 和 4D 现在应该成功（CNN 卷积核支持）
    assert!(
        graph
            .new_parameter_node(&[16, 3, 3], Some("param_3d"))
            .is_ok()
    );
    assert!(
        graph
            .new_parameter_node(&[32, 16, 3, 3], Some("conv_kernel"))
            .is_ok()
    );
}

#[test]
fn test_node_parameter_name_generation() {
    let mut graph = GraphInner::new();

    // 1. 测试节点显式命名
    let param1 = graph
        .new_parameter_node(&[2, 2], Some("explicit_param"))
        .unwrap();
    assert_eq!(graph.get_node_name(param1).unwrap(), "explicit_param");

    // 2. 测试节点自动命名
    let param2 = graph.new_parameter_node(&[2, 2], None).unwrap();
    assert_eq!(graph.get_node_name(param2).unwrap(), "parameter_1");

    // 3. 测试节点名称重复
    let result = graph.new_parameter_node(&[2, 2], Some("explicit_param"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点explicit_param在图default_graph中重复")
    );
}

#[test]
fn test_node_parameter_manually_set_value() {
    let mut graph = GraphInner::new();
    let param = graph
        .new_parameter_node(&[2, 2], Some("test_param"))
        .unwrap();

    // 1. 测试有效赋值
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    {
        let cloned_tensor = test_value.clone();
        graph.set_node_value(param, Some(&cloned_tensor)).unwrap();
    } // cloned_tensor在这里被释放

    // 1.1 验证节点状态
    assert!(graph.is_node_inited(param).unwrap());
    assert_eq!(graph.get_node_value(param).unwrap().unwrap(), &test_value);

    // 2. 测试错误形状的赋值
    let invalid_cases = [
        Tensor::new(&[1.0], &[1, 1]),
        Tensor::new(&[1.0, 2.0], &[2, 1]),
        Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]),
    ];
    for value in invalid_cases {
        let shape = value.shape().to_vec();
        assert_err!(
            graph.set_node_value(param, Some(&value)),
            GraphError::ShapeMismatch { expected, got, message }
                if expected == &[2, 2] && got == &shape
                    && message == &format!(
                        "新张量的形状 {:?} 与节点 '{}' 现有张量的形状 {:?} 不匹配。",
                        shape,
                        "test_param",
                        &[2, 2]
                    )
        );
    }

    // 3. 测试设置空值（清除值）
    graph.set_node_value(param, None).unwrap();
    assert!(!graph.is_node_inited(param).unwrap());
    assert!(graph.get_node_value(param).unwrap().is_none());
}

#[test]
fn test_node_parameter_expected_shape() {
    let mut graph = GraphInner::new();

    // 1. 测试基本的Parameter节点预期形状
    let param = graph.new_parameter_node(&[2, 3], Some("param")).unwrap();
    assert_eq!(graph.get_node_value_shape(param).unwrap().unwrap(), &[2, 3]); // 实际值形状（已初始化）
    assert_eq!(graph.get_node_value_expected_shape(param).unwrap(), &[2, 3]); // 预期形状已确定

    // 2. 设置新值后检查
    let value = Tensor::zeros(&[2, 3]);
    graph.set_node_value(param, Some(&value)).unwrap();
    assert_eq!(graph.get_node_value_shape(param).unwrap().unwrap(), &[2, 3]); // 设置值后实际形状
    assert_eq!(graph.get_node_value_expected_shape(param).unwrap(), &[2, 3]); // 预期形状保持不变

    // 3. 清除值后检查
    graph.set_node_value(param, None).unwrap();
    assert_eq!(graph.get_node_value_shape(param).unwrap(), None); // 清除后实际值形状为None
    assert_eq!(graph.get_node_value_expected_shape(param).unwrap(), &[2, 3]); // 预期形状仍然保持
}

#[test]
fn test_node_parameter_forward_propagation() {
    let mut graph = GraphInner::new();
    let param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();

    // 1. 测试前向传播（应该失败，因为Parameter节点不支持前向传播）
    assert_err!(
        graph.forward(param),
        GraphError::InvalidOperation(
            "节点[id=1, name=param, type=Parameter]是输入/参数/状态节点，其值应通过set_value设置，而不是通过父节点前向传播计算"
        )
    );

    // 2. 设置新值后仍然不能前向传播
    let value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    graph.set_node_value(param, Some(&value)).unwrap();
    assert_err!(
        graph.forward(param),
        GraphError::InvalidOperation(
            "节点[id=1, name=param, type=Parameter]是输入/参数/状态节点，其值应通过set_value设置，而不是通过父节点前向传播计算"
        )
    );
}

/// 测试 Parameter 节点在完整计算图中的反向传播行为
///
/// Parameter 节点是可学习参数，在反向传播后应该有梯度。
#[test]
fn test_node_parameter_backward_propagation() {
    let mut graph = GraphInner::new();

    // 1. 构建计算图: input * param -> mse_loss
    let input = graph.new_input_node(&[2, 2], Some("input")).unwrap();
    let param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();
    let target = graph.new_input_node(&[2, 2], Some("target")).unwrap();

    let mul = graph.new_multiply_node(input, param, None).unwrap();
    let loss = graph.new_mse_loss_node(mul, target, None).unwrap();

    // 2. 设置输入值
    let input_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let param_val = Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2]);
    let target_val = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    graph.set_node_value(input, Some(&input_val)).unwrap();
    graph.set_node_value(param, Some(&param_val)).unwrap();
    graph.set_node_value(target, Some(&target_val)).unwrap();

    // 3. 初始时梯度应为空
    assert!(graph.get_node_grad(param).unwrap().is_none());

    // 4. 前向传播 + 反向传播
    graph.forward(loss).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss).unwrap();

    // 5. Parameter 节点应该有梯度
    let grad = graph.get_node_grad(param).unwrap();
    assert!(grad.is_some(), "Parameter 节点应该有梯度");

    // 6. 清除梯度并验证
    graph.zero_grad().unwrap();
    assert!(graph.get_node_grad(param).unwrap().is_none());
}

/// 测试 Parameter 节点的梯度值正确性
#[test]
fn test_node_parameter_gradient_correctness() {
    let mut graph = GraphInner::new();

    // 简单计算图: param -> mse_loss(param, target)
    // loss = mean((param - target)^2)
    // d_loss/d_param = 2 * (param - target) / n
    let param = graph.new_parameter_node(&[1, 2], Some("param")).unwrap();
    let target = graph.new_input_node(&[1, 2], Some("target")).unwrap();
    let loss = graph.new_mse_loss_node(param, target, None).unwrap();

    // 设置值: param = [1.0, 2.0], target = [0.0, 0.0]
    let param_val = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let target_val = Tensor::new(&[0.0, 0.0], &[1, 2]);
    graph.set_node_value(param, Some(&param_val)).unwrap();
    graph.set_node_value(target, Some(&target_val)).unwrap();

    // 前向 + 反向
    graph.forward(loss).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss).unwrap();

    // 验证梯度
    // loss = mean((1-0)^2 + (2-0)^2) = mean(1 + 4) = 2.5
    // d_loss/d_param = 2 * (param - target) / 2 = (param - target)
    // d_loss/d_param = [1.0, 2.0]
    let grad = graph.get_node_grad(param).unwrap().unwrap();
    assert_eq!(grad.shape(), &[1, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 2.0, epsilon = 1e-5);
}
