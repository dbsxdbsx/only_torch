use crate::assert_err;
use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;

#[test]
fn test_node_abs_creation() {
    let mut graph = GraphInner::new();

    // 1. 测试 Input 节点作为父节点
    {
        let input = graph.new_basic_input_node(&[2, 2], Some("input1")).unwrap();
        let abs = graph.new_abs_node(input, Some("abs_with_input")).unwrap();
        // 1.1 验证基本属性
        assert_eq!(graph.get_node_name(abs).unwrap(), "abs_with_input");
        assert_eq!(graph.get_node_parents(abs).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(abs).unwrap().len(), 0);
    }

    // 2. 测试 Parameter 节点作为父节点
    {
        let param = graph.new_parameter_node(&[2, 2], Some("param1")).unwrap();
        let abs = graph.new_abs_node(param, Some("abs_with_param")).unwrap();
        assert_eq!(graph.get_node_name(abs).unwrap(), "abs_with_param");
        assert_eq!(graph.get_node_parents(abs).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(abs).unwrap().len(), 0);
    }
}

#[test]
fn test_node_abs_name_generation() {
    let mut graph = GraphInner::new();

    // 1. 测试节点显式命名
    let input = graph.new_basic_input_node(&[2, 2], Some("input1")).unwrap();
    let abs = graph.new_abs_node(input, Some("explicit_abs")).unwrap();
    assert_eq!(graph.get_node_name(abs).unwrap(), "explicit_abs");

    // 2. 测试节点自动命名
    let abs2 = graph.new_abs_node(input, None).unwrap();
    assert_eq!(graph.get_node_name(abs2).unwrap(), "abs_1");

    // 3. 测试节点名称重复
    let result = graph.new_abs_node(input, Some("explicit_abs"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点explicit_abs在图default_graph中重复")
    );
}

#[test]
fn test_node_abs_manually_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_basic_input_node(&[2, 2], Some("input1")).unwrap();
    let abs = graph.new_abs_node(input, Some("abs")).unwrap();

    // 1. 测试直接设置 Abs 节点的值（应该失败）
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_err!(
        graph.set_node_value(abs, Some(&test_value)),
        GraphError::InvalidOperation(
            "节点[id=2, name=abs, type=Abs]的值只能通过前向传播计算得到，不能直接设置"
        )
    );

    // 2. 测试清除 Abs 节点的值（也应该失败）
    assert_err!(
        graph.set_node_value(abs, None),
        GraphError::InvalidOperation(
            "节点[id=2, name=abs, type=Abs]的值只能通过前向传播计算得到，不能直接设置"
        )
    );
}

#[test]
fn test_node_abs_expected_shape() {
    let mut graph = GraphInner::new();

    // 1. 测试基本的 Abs 节点预期形状
    let input = graph.new_basic_input_node(&[2, 2], Some("input1")).unwrap();
    let abs = graph.new_abs_node(input, Some("abs")).unwrap();
    assert_eq!(graph.get_node_value_expected_shape(abs).unwrap(), &[2, 2]);
    assert_eq!(graph.get_node_value_shape(abs).unwrap(), None); // 实际值形状为 None（未计算）

    // 2. 测试前向传播后的形状
    let value = Tensor::zeros(&[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();
    graph.forward(abs).unwrap();

    // 2.1 验证前向传播后的形状
    assert_eq!(graph.get_node_value_shape(abs).unwrap().unwrap(), &[2, 2]); // 实际值形状
    assert_eq!(graph.get_node_value_expected_shape(abs).unwrap(), &[2, 2]); // 预期形状保持不变

    // 2.2 测试父节点值在首次前向传播后，再次设置新值后的形状检查
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();

    // 2.2.1 验证预期形状和实际形状
    assert_eq!(graph.get_node_value_expected_shape(abs).unwrap(), &[2, 2]);
    assert_eq!(graph.get_node_value_shape(abs).unwrap().unwrap(), &[2, 2]);
}

#[test]
fn test_node_abs_forward_propagation() {
    // 1. 准备测试数据
    // Abs: |x|
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    let expected = Tensor::new(&[0.5, 1.0, 0.0, 2.0], &[2, 2]);

    // 2. 测试不同节点类型组合的前向传播
    let node_types = ["input", "parameter"];
    for parent_type in node_types {
        let mut graph = GraphInner::new();

        // 创建 parent 节点
        let parent = match parent_type {
            "input" => graph
                .new_basic_input_node(&[2, 2], Some("input_1"))
                .unwrap(),
            "parameter" => graph
                .new_parameter_node(&[2, 2], Some("parameter_1"))
                .unwrap(),
            _ => unreachable!(),
        };

        // Abs 节点
        let abs = graph.new_abs_node(parent, Some("abs")).unwrap();

        // 如果节点是 parameter，因创建时其值已隐式初始化过了，所以前向传播应成功
        if parent_type == "parameter" {
            graph.forward(abs).unwrap();
        } else {
            // 如果是 input 节点，因创建时其值未初始化，所以前向传播应失败
            assert_err!(
                graph.forward(abs),
                GraphError::InvalidOperation(msg) if msg.contains("不能直接前向传播")
            );

            // 设置 input 节点的值
            graph.set_node_value(parent, Some(&value)).unwrap();

            // 设置值后前向传播应成功
            graph.forward(abs).unwrap();
            let result = graph.get_node_value(abs).unwrap().unwrap();

            // 只有当节点是 input 时才检查输出值
            if parent_type == "input" {
                assert_eq!(result, &expected);
            }
        }
    }
}

#[test]
fn test_node_abs_forward_values() {
    // 测试 Abs 节点的具体输出值
    let mut graph = GraphInner::new();

    // 1. 测试包含正数、负数、零的情况（使用 2D 张量，框架要求 2-4 维）
    let input = graph.new_basic_input_node(&[5, 1], Some("input")).unwrap();
    let abs = graph.new_abs_node(input, Some("abs")).unwrap();

    let value = Tensor::new(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[5, 1]);
    graph.set_node_value(input, Some(&value)).unwrap();
    graph.forward(abs).unwrap();

    let result = graph.get_node_value(abs).unwrap().unwrap();
    let expected = Tensor::new(&[2.0, 0.5, 0.0, 0.5, 2.0], &[5, 1]);
    assert_eq!(result, &expected);

    // 2. 测试极端值
    let extreme_value = Tensor::new(
        &[f32::INFINITY, f32::NEG_INFINITY, f32::MIN, f32::MAX],
        &[2, 2],
    );
    let input2 = graph.new_basic_input_node(&[2, 2], Some("input2")).unwrap();
    let abs2 = graph.new_abs_node(input2, Some("abs2")).unwrap();

    graph.set_node_value(input2, Some(&extreme_value)).unwrap();
    graph.forward(abs2).unwrap();

    let result2 = graph.get_node_value(abs2).unwrap().unwrap();
    // INFINITY → INFINITY, NEG_INFINITY → INFINITY, MIN → abs(MIN), MAX → MAX
    // 注意：f32::MIN 是最小的负数，其绝对值约等于 MAX
    let expected2 = Tensor::new(
        &[f32::INFINITY, f32::INFINITY, f32::MIN.abs(), f32::MAX.abs()],
        &[2, 2],
    );
    assert_eq!(result2, &expected2);
}

/// 测试 Abs 节点的反向传播（VJP 模式）
///
/// Abs 的梯度是 sign(x)：
/// - x > 0 时，梯度 = 1
/// - x < 0 时，梯度 = -1
/// - x = 0 时，梯度 = 0（与 PyTorch 行为一致）
#[test]
fn test_node_abs_backward_propagation() {
    use approx::assert_abs_diff_eq;

    let mut graph = GraphInner::new();

    // 1. 构建计算图: parent -> abs -> mse_loss
    let parent = graph.new_parameter_node(&[2, 2], Some("parent")).unwrap();
    let abs = graph.new_abs_node(parent, Some("abs")).unwrap();
    let target = graph.new_basic_input_node(&[2, 2], Some("target")).unwrap();
    let loss = graph.new_mse_loss_node(abs, target, None).unwrap();

    // 2. 设置输入值
    // parent = [0.5, -1.0, 0.0, 2.0]
    // abs(parent) = [0.5, 1.0, 0.0, 2.0]
    // target = [0.0, 0.0, 0.0, 0.0]
    let parent_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    let target_value = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
    graph.set_node_value(parent, Some(&parent_value)).unwrap();
    graph.set_node_value(target, Some(&target_value)).unwrap();

    // 3. 前向传播
    graph.forward(loss).unwrap();

    // abs(parent) = [0.5, 1.0, 0.0, 2.0]
    // loss = mean((abs - target)^2) = mean([0.25, 1.0, 0.0, 4.0]) = 1.3125
    let loss_val = graph
        .get_node_value(loss)
        .unwrap()
        .unwrap()
        .get_data_number()
        .unwrap();
    assert_abs_diff_eq!(loss_val, 1.3125, epsilon = 1e-6);

    // 4. 初始时梯度应为空
    assert!(graph.get_node_grad(parent).unwrap().is_none());

    // 5. 反向传播
    graph.zero_grad().unwrap();
    graph.backward(loss).unwrap();

    // 6. 验证 parent 的梯度
    // dL/d_abs = 2 * (abs - target) / N = [0.25, 0.5, 0.0, 1.0]
    // d_abs/d_parent = sign(parent) = [1, -1, 0, 1]
    // dL/d_parent = dL/d_abs * d_abs/d_parent = [0.25, -0.5, 0.0, 1.0]
    let grad = graph.get_node_grad(parent).unwrap().unwrap();
    assert_eq!(grad.shape(), &[2, 2]);

    assert_abs_diff_eq!(grad[[0, 0]], 0.25, epsilon = 1e-6); // 0.5 > 0, sign = 1
    assert_abs_diff_eq!(grad[[0, 1]], -0.5, epsilon = 1e-6); // -1.0 < 0, sign = -1
    assert_abs_diff_eq!(grad[[1, 0]], 0.0, epsilon = 1e-6); // 0.0 = 0, sign = 0
    assert_abs_diff_eq!(grad[[1, 1]], 1.0, epsilon = 1e-6); // 2.0 > 0, sign = 1

    // 7. zero_grad 后梯度应清零
    graph.zero_grad().unwrap();
    assert!(graph.get_node_grad(parent).unwrap().is_none());
}

/// 测试 Abs 作为 L1 损失的核心组件
/// L1 Loss = mean(|pred - target|)
#[test]
fn test_node_abs_as_l1_loss_component() {
    use approx::assert_abs_diff_eq;

    let mut graph = GraphInner::new();

    // 构建计算图: pred - target -> abs -> mean (通过 mse_loss 近似)
    let pred = graph.new_parameter_node(&[2, 2], Some("pred")).unwrap();
    let target = graph.new_basic_input_node(&[2, 2], Some("target")).unwrap();
    let diff = graph.new_subtract_node(pred, target, Some("diff")).unwrap();
    let abs_diff = graph.new_abs_node(diff, Some("abs_diff")).unwrap();

    // 设置值
    // pred = [1.0, 2.0, 3.0, 4.0], target = [0.5, 2.5, 2.0, 5.0]
    // diff = [0.5, -0.5, 1.0, -1.0]
    // abs_diff = [0.5, 0.5, 1.0, 1.0]
    let pred_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let target_value = Tensor::new(&[0.5, 2.5, 2.0, 5.0], &[2, 2]);
    graph.set_node_value(pred, Some(&pred_value)).unwrap();
    graph.set_node_value(target, Some(&target_value)).unwrap();

    // 前向传播
    graph.forward(abs_diff).unwrap();

    let result = graph.get_node_value(abs_diff).unwrap().unwrap();
    let expected = Tensor::new(&[0.5, 0.5, 1.0, 1.0], &[2, 2]);
    assert_abs_diff_eq!(result, &expected, epsilon = 1e-6);
}

// ==================== 动态形状测试 ====================

/// 测试 Abs 节点的动态形状传播
#[test]
fn test_abs_dynamic_shape_propagation() {
    use crate::nn::Graph;

    let graph = Graph::new();

    // 创建一个支持动态 batch 的输入（使用 ZerosLike）
    let x = graph.input(&Tensor::zeros(&[4, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[16], None).unwrap(); // [?, 16]

    // 创建 Abs
    use crate::nn::var_ops::VarActivationOps;
    let result = h0.abs();

    // 验证动态形状传播
    let dyn_shape = result.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试 Abs 节点在不同 batch_size 下的前向和反向计算
#[test]
fn test_abs_dynamic_batch_forward_backward() {
    use crate::nn::Graph;
    use crate::nn::var_ops::{VarActivationOps, VarLossOps};

    let graph = Graph::new();

    // 创建支持动态 batch 的节点
    let x = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let h0 = graph.zeros_like(&x, &[4], None).unwrap(); // [?, 4]

    // Abs
    let result = h0.abs();

    // 创建目标和损失
    let target = graph.input(&Tensor::zeros(&[2, 4])).unwrap();
    let loss = result.mse_loss(&target).unwrap();

    // 第一次 forward：batch=2
    loss.forward().unwrap();
    let value1 = result.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 4], "第一次 forward: batch=2");

    // 第一次 backward
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::zeros(&[8, 8])).unwrap();
    target.set_value(&Tensor::zeros(&[8, 4])).unwrap();

    // 第二次 forward：batch=8
    loss.forward().unwrap();
    let value2 = result.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[8, 4], "第二次 forward: batch=8");

    // 第二次 backward
    loss.backward().unwrap();
}

/// 测试 Abs 的幂等性：abs(abs(x)) == abs(x)
#[test]
fn test_node_abs_idempotent() {
    use approx::assert_abs_diff_eq;

    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 2], Some("input")).unwrap();
    let abs1 = graph.new_abs_node(input, Some("abs1")).unwrap();
    let abs2 = graph.new_abs_node(abs1, Some("abs2")).unwrap();

    let value = Tensor::new(&[-2.0, -1.0, 1.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();

    graph.forward(abs2).unwrap();

    let result1 = graph.get_node_value(abs1).unwrap().unwrap();
    let result2 = graph.get_node_value(abs2).unwrap().unwrap();

    // abs(abs(x)) == abs(x)
    assert_abs_diff_eq!(result1, result2, epsilon = 1e-6);
}
