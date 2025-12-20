/*
 * @Author       : 老董
 * @Date         : 2025-12-20
 * @Description  : 优化器模块单元测试
 */

use crate::nn::optimizer::{Adam, GradientAccumulator, Optimizer, OptimizerState, SGD};
use crate::nn::{Graph, NodeId};
use crate::tensor::Tensor;

// ============================================================================
// GradientAccumulator 测试
// ============================================================================

#[test]
fn test_gradient_accumulator_creation() {
    let accumulator = GradientAccumulator::new();
    assert_eq!(accumulator.sample_count(), 0);
}

#[test]
fn test_gradient_accumulator_single_accumulate() {
    let mut accumulator = GradientAccumulator::new();
    let node_id = NodeId(1);
    let gradient = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // 累积一个梯度
    accumulator.accumulate(node_id, &gradient).unwrap();
    accumulator.increment_sample_count();

    // 验证样本数
    assert_eq!(accumulator.sample_count(), 1);

    // 验证平均梯度（只有一个样本，平均值等于原值）
    let avg_gradient = accumulator.get_average_gradient(node_id).unwrap();
    assert_eq!(avg_gradient, gradient);
}

#[test]
fn test_gradient_accumulator_multiple_accumulate() {
    let mut accumulator = GradientAccumulator::new();
    let node_id = NodeId(1);

    // 累积多个梯度
    let gradient1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let gradient2 = Tensor::new(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);
    let gradient3 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);

    accumulator.accumulate(node_id, &gradient1).unwrap();
    accumulator.increment_sample_count();
    accumulator.accumulate(node_id, &gradient2).unwrap();
    accumulator.increment_sample_count();
    accumulator.accumulate(node_id, &gradient3).unwrap();
    accumulator.increment_sample_count();

    // 验证样本数
    assert_eq!(accumulator.sample_count(), 3);

    // 验证平均梯度：(1+3+2)/3=2, (2+4+3)/3=3, (3+5+4)/3=4, (4+6+5)/3=5
    let avg_gradient = accumulator.get_average_gradient(node_id).unwrap();
    let expected = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    assert_eq!(avg_gradient, expected);
}

#[test]
fn test_gradient_accumulator_multiple_nodes() {
    let mut accumulator = GradientAccumulator::new();
    let node_id_1 = NodeId(1);
    let node_id_2 = NodeId(2);

    let gradient1 = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let gradient2 = Tensor::new(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);

    accumulator.accumulate(node_id_1, &gradient1).unwrap();
    accumulator.accumulate(node_id_2, &gradient2).unwrap();
    accumulator.increment_sample_count();

    // 每个节点都有自己的累积梯度
    let avg1 = accumulator.get_average_gradient(node_id_1).unwrap();
    let avg2 = accumulator.get_average_gradient(node_id_2).unwrap();

    assert_eq!(avg1, gradient1);
    assert_eq!(avg2, gradient2);
}

#[test]
fn test_gradient_accumulator_clear() {
    let mut accumulator = GradientAccumulator::new();
    let node_id = NodeId(1);
    let gradient = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    accumulator.accumulate(node_id, &gradient).unwrap();
    accumulator.increment_sample_count();

    // 验证累积状态
    assert_eq!(accumulator.sample_count(), 1);
    assert!(accumulator.get_average_gradient(node_id).is_some());

    // 清除
    accumulator.clear();

    // 验证清除后的状态
    assert_eq!(accumulator.sample_count(), 0);
    assert!(accumulator.get_average_gradient(node_id).is_none());
}

#[test]
fn test_gradient_accumulator_get_nonexistent_node() {
    let accumulator = GradientAccumulator::new();
    let node_id = NodeId(999);

    // 不存在的节点应返回None
    assert!(accumulator.get_average_gradient(node_id).is_none());
}

#[test]
fn test_gradient_accumulator_zero_sample_count() {
    let mut accumulator = GradientAccumulator::new();
    let node_id = NodeId(1);
    let gradient = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // 只累积梯度，不增加样本数
    accumulator.accumulate(node_id, &gradient).unwrap();

    // 样本数为0时，应返回None（避免除以0）
    assert_eq!(accumulator.sample_count(), 0);
    assert!(accumulator.get_average_gradient(node_id).is_none());
}

// ============================================================================
// OptimizerState 测试
// ============================================================================

#[test]
fn test_optimizer_state_creation() {
    let mut graph = Graph::new();
    let _input = graph.new_input_node(&[2, 2], Some("input")).unwrap();
    let param1 = graph.new_parameter_node(&[2, 2], Some("param1")).unwrap();
    let param2 = graph.new_parameter_node(&[1, 1], Some("param2")).unwrap();

    let state = OptimizerState::new(&graph, 0.01).unwrap();

    // 验证可训练节点（只有Parameter节点）
    let trainable = state.trainable_nodes();
    assert_eq!(trainable.len(), 2);
    assert!(trainable.contains(&param1));
    assert!(trainable.contains(&param2));

    // 验证学习率
    assert_eq!(state.learning_rate(), 0.01);
}

#[test]
fn test_optimizer_state_learning_rate() {
    let graph = Graph::new();
    let mut state = OptimizerState::new(&graph, 0.01).unwrap();

    // 验证初始学习率
    assert_eq!(state.learning_rate(), 0.01);

    // 修改学习率
    state.set_learning_rate(0.001);
    assert_eq!(state.learning_rate(), 0.001);

    // 设置为0
    state.set_learning_rate(0.0);
    assert_eq!(state.learning_rate(), 0.0);
}

#[test]
fn test_optimizer_state_reset() {
    let mut graph = Graph::new();
    let _param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();
    let mut state = OptimizerState::new(&graph, 0.01).unwrap();

    // 手动向累积器添加一些数据
    let node_id = NodeId(1);
    let gradient = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    state
        .gradient_accumulator_mut()
        .accumulate(node_id, &gradient)
        .unwrap();
    state.gradient_accumulator_mut().increment_sample_count();

    // 验证累积状态
    assert_eq!(state.gradient_accumulator().sample_count(), 1);

    // 重置
    state.reset();

    // 验证重置后状态
    assert_eq!(state.gradient_accumulator().sample_count(), 0);
}

// ============================================================================
// SGD 优化器测试
// ============================================================================

#[test]
fn test_sgd_creation() {
    let mut graph = Graph::new();
    let _param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();

    let sgd = SGD::new(&graph, 0.01).unwrap();
    assert_eq!(sgd.learning_rate(), 0.01);
}

#[test]
fn test_sgd_learning_rate_modification() {
    let mut graph = Graph::new();
    let _param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();

    let mut sgd = SGD::new(&graph, 0.01).unwrap();
    assert_eq!(sgd.learning_rate(), 0.01);

    sgd.set_learning_rate(0.001);
    assert_eq!(sgd.learning_rate(), 0.001);
}

#[test]
fn test_sgd_update_formula() {
    // 测试SGD更新公式：θ_new = θ_old - α * ∇θ
    // 使用ADALINE结构：loss_input = label * output，当预测错误时梯度不为0
    let mut graph = Graph::new();

    // 创建计算图：output = w * x, loss_input = label * output
    let x = graph.new_input_node(&[1, 1], Some("x")).unwrap();
    let label = graph.new_input_node(&[1, 1], Some("label")).unwrap();
    let w = graph.new_parameter_node(&[1, 1], Some("w")).unwrap();
    let output = graph.new_mat_mul_node(w, x, Some("output")).unwrap();
    let loss_input = graph
        .new_mat_mul_node(label, output, Some("loss_input"))
        .unwrap();
    let loss = graph
        .new_perception_loss_node(loss_input, Some("loss"))
        .unwrap();

    // 设置初始值：w=1, x=1, label=-1 => output=1, loss_input=-1 (负数，会产生梯度)
    let initial_w = Tensor::new(&[1.0], &[1, 1]);
    let x_value = Tensor::new(&[1.0], &[1, 1]);
    let label_value = Tensor::new(&[-1.0], &[1, 1]); // 负标签使loss_input为负
    graph.set_node_value(w, Some(&initial_w)).unwrap();
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(label, Some(&label_value)).unwrap();

    // 创建SGD优化器
    let learning_rate = 0.1;
    let mut sgd = SGD::new(&graph, learning_rate).unwrap();

    // 执行一步训练
    sgd.one_step(&mut graph, loss).unwrap();
    sgd.update(&mut graph).unwrap();

    // 验证参数已更新
    let new_w = graph.get_node_value(w).unwrap().unwrap();
    let new_w_value = new_w.get(&[0, 0]).get_data_number().unwrap();

    // 参数应该变化（因为loss_input=-1 < 0，梯度不为0）
    assert_ne!(new_w_value, 1.0, "参数w应该在优化后改变");
}

#[test]
fn test_sgd_reset() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[1, 1], Some("input")).unwrap();
    let param = graph.new_parameter_node(&[1, 1], Some("param")).unwrap();
    let output = graph
        .new_mat_mul_node(param, input, Some("output"))
        .unwrap();
    let loss = graph
        .new_perception_loss_node(output, Some("loss"))
        .unwrap();

    let input_value = Tensor::new(&[1.0], &[1, 1]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    let mut sgd = SGD::new(&graph, 0.01).unwrap();

    // 执行一步训练（会累积梯度）
    sgd.one_step(&mut graph, loss).unwrap();

    // 重置
    sgd.reset();

    // 重置后，update不应该改变参数（因为没有累积的梯度）
    let param_before = graph.get_node_value(param).unwrap().unwrap().clone();
    sgd.update(&mut graph).unwrap();
    let param_after = graph.get_node_value(param).unwrap().unwrap();

    assert_eq!(&param_before, param_after);
}

#[test]
fn test_sgd_zero_learning_rate() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[1, 1], Some("input")).unwrap();
    let param = graph.new_parameter_node(&[1, 1], Some("param")).unwrap();
    let output = graph
        .new_mat_mul_node(param, input, Some("output"))
        .unwrap();
    let loss = graph
        .new_perception_loss_node(output, Some("loss"))
        .unwrap();

    let input_value = Tensor::new(&[1.0], &[1, 1]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    // 学习率为0
    let mut sgd = SGD::new(&graph, 0.0).unwrap();

    // 记录原始参数值
    let param_before = graph.get_node_value(param).unwrap().unwrap().clone();

    // 执行训练
    sgd.one_step(&mut graph, loss).unwrap();
    sgd.update(&mut graph).unwrap();

    // 参数应该不变（因为学习率为0）
    let param_after = graph.get_node_value(param).unwrap().unwrap();
    assert_eq!(&param_before, param_after);
}

#[test]
fn test_sgd_gradient_accumulation() {
    // 测试多次one_step后的梯度累积
    // 使用ADALINE结构确保梯度不为0
    let mut graph = Graph::new();
    let x = graph.new_input_node(&[1, 1], Some("x")).unwrap();
    let label = graph.new_input_node(&[1, 1], Some("label")).unwrap();
    let w = graph.new_parameter_node(&[1, 1], Some("w")).unwrap();
    let output = graph.new_mat_mul_node(w, x, Some("output")).unwrap();
    let loss_input = graph
        .new_mat_mul_node(label, output, Some("loss_input"))
        .unwrap();
    let loss = graph
        .new_perception_loss_node(loss_input, Some("loss"))
        .unwrap();

    let initial_w = Tensor::new(&[1.0], &[1, 1]);
    graph.set_node_value(w, Some(&initial_w)).unwrap();

    let x_value = Tensor::new(&[2.0], &[1, 1]);
    let label_value = Tensor::new(&[-1.0], &[1, 1]); // 负标签确保梯度不为0

    let mut sgd = SGD::new(&graph, 0.1).unwrap();

    // 执行3次one_step（梯度累积）
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(label, Some(&label_value)).unwrap();
    sgd.one_step(&mut graph, loss).unwrap();

    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(label, Some(&label_value)).unwrap();
    sgd.one_step(&mut graph, loss).unwrap();

    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(label, Some(&label_value)).unwrap();
    sgd.one_step(&mut graph, loss).unwrap();

    // 更新参数
    sgd.update(&mut graph).unwrap();

    // 参数应该改变
    let new_w = graph.get_node_value(w).unwrap().unwrap();
    let new_w_value = new_w.get(&[0, 0]).get_data_number().unwrap();
    assert_ne!(new_w_value, 1.0, "参数w应该在梯度累积后改变");
}

// ============================================================================
// Adam 优化器测试
// ============================================================================

#[test]
fn test_adam_creation() {
    let mut graph = Graph::new();
    let _param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();

    // 测试自定义参数
    let adam = Adam::new(&graph, 0.001, 0.9, 0.999, 1e-8).unwrap();
    assert_eq!(adam.learning_rate(), 0.001);

    // 测试默认参数
    let adam_default = Adam::new_default(&graph, 0.001).unwrap();
    assert_eq!(adam_default.learning_rate(), 0.001);
}

#[test]
fn test_adam_learning_rate_modification() {
    let mut graph = Graph::new();
    let _param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();

    let mut adam = Adam::new_default(&graph, 0.001).unwrap();
    assert_eq!(adam.learning_rate(), 0.001);

    adam.set_learning_rate(0.0001);
    assert_eq!(adam.learning_rate(), 0.0001);
}

#[test]
fn test_adam_update() {
    // 使用ADALINE结构确保梯度不为0
    let mut graph = Graph::new();
    let x = graph.new_input_node(&[1, 1], Some("x")).unwrap();
    let label = graph.new_input_node(&[1, 1], Some("label")).unwrap();
    let w = graph.new_parameter_node(&[1, 1], Some("w")).unwrap();
    let output = graph.new_mat_mul_node(w, x, Some("output")).unwrap();
    let loss_input = graph
        .new_mat_mul_node(label, output, Some("loss_input"))
        .unwrap();
    let loss = graph
        .new_perception_loss_node(loss_input, Some("loss"))
        .unwrap();

    // w=2, x=3, label=-1 => output=6, loss_input=-6 (负数，梯度不为0)
    let initial_w = Tensor::new(&[2.0], &[1, 1]);
    let x_value = Tensor::new(&[3.0], &[1, 1]);
    let label_value = Tensor::new(&[-1.0], &[1, 1]);
    graph.set_node_value(w, Some(&initial_w)).unwrap();
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(label, Some(&label_value)).unwrap();

    let mut adam = Adam::new_default(&graph, 0.1).unwrap();

    // 执行一步训练
    adam.one_step(&mut graph, loss).unwrap();
    adam.update(&mut graph).unwrap();

    // 验证参数已更新
    let new_w = graph.get_node_value(w).unwrap().unwrap();
    let new_w_value = new_w.get(&[0, 0]).get_data_number().unwrap();
    assert_ne!(new_w_value, 2.0, "参数w应该在Adam优化后改变");
}

#[test]
fn test_adam_reset() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[1, 1], Some("input")).unwrap();
    let param = graph.new_parameter_node(&[1, 1], Some("param")).unwrap();
    let output = graph
        .new_mat_mul_node(param, input, Some("output"))
        .unwrap();
    let loss = graph
        .new_perception_loss_node(output, Some("loss"))
        .unwrap();

    let input_value = Tensor::new(&[1.0], &[1, 1]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    let mut adam = Adam::new_default(&graph, 0.01).unwrap();

    // 执行几步训练
    adam.one_step(&mut graph, loss).unwrap();
    adam.update(&mut graph).unwrap();

    // 重置（清除矩估计和时间步）
    adam.reset();

    // 重置后update不应该改变参数
    let param_before = graph.get_node_value(param).unwrap().unwrap().clone();
    adam.update(&mut graph).unwrap();
    let param_after = graph.get_node_value(param).unwrap().unwrap();

    assert_eq!(&param_before, param_after);
}

#[test]
fn test_adam_momentum_accumulation() {
    // 测试Adam的动量累积（多次更新后，一阶矩和二阶矩应该有累积效果）
    // 使用ADALINE结构确保梯度不为0
    let mut graph = Graph::new();
    let x = graph.new_input_node(&[1, 1], Some("x")).unwrap();
    let label = graph.new_input_node(&[1, 1], Some("label")).unwrap();
    let w = graph.new_parameter_node(&[1, 1], Some("w")).unwrap();
    let output = graph.new_mat_mul_node(w, x, Some("output")).unwrap();
    let loss_input = graph
        .new_mat_mul_node(label, output, Some("loss_input"))
        .unwrap();
    let loss = graph
        .new_perception_loss_node(loss_input, Some("loss"))
        .unwrap();

    let initial_w = Tensor::new(&[1.0], &[1, 1]);
    graph.set_node_value(w, Some(&initial_w)).unwrap();

    let x_value = Tensor::new(&[2.0], &[1, 1]);
    let label_value = Tensor::new(&[-1.0], &[1, 1]);

    let mut adam = Adam::new_default(&graph, 0.01).unwrap();

    // 记录每次更新后的参数值
    let mut param_history = Vec::new();
    param_history.push(
        graph
            .get_node_value(w)
            .unwrap()
            .unwrap()
            .get(&[0, 0])
            .get_data_number()
            .unwrap(),
    );

    // 执行多次更新
    for _ in 0..5 {
        graph.set_node_value(x, Some(&x_value)).unwrap();
        graph.set_node_value(label, Some(&label_value)).unwrap();
        adam.one_step(&mut graph, loss).unwrap();
        adam.update(&mut graph).unwrap();

        param_history.push(
            graph
                .get_node_value(w)
                .unwrap()
                .unwrap()
                .get(&[0, 0])
                .get_data_number()
                .unwrap(),
        );
    }

    // 验证参数在持续变化
    for i in 1..param_history.len() {
        assert_ne!(
            param_history[i],
            param_history[i - 1],
            "参数在第{}次更新后应该变化",
            i
        );
    }
}

#[test]
fn test_adam_zero_learning_rate() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[1, 1], Some("input")).unwrap();
    let param = graph.new_parameter_node(&[1, 1], Some("param")).unwrap();
    let output = graph
        .new_mat_mul_node(param, input, Some("output"))
        .unwrap();
    let loss = graph
        .new_perception_loss_node(output, Some("loss"))
        .unwrap();

    let input_value = Tensor::new(&[1.0], &[1, 1]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    // 学习率为0
    let mut adam = Adam::new(&graph, 0.0, 0.9, 0.999, 1e-8).unwrap();

    let param_before = graph.get_node_value(param).unwrap().unwrap().clone();

    adam.one_step(&mut graph, loss).unwrap();
    adam.update(&mut graph).unwrap();

    // 参数应该不变
    let param_after = graph.get_node_value(param).unwrap().unwrap();
    assert_eq!(&param_before, param_after);
}

// ============================================================================
// Optimizer trait 通用行为测试
// ============================================================================

#[test]
fn test_optimizer_trait_implementations() {
    // 验证SGD和Adam都正确实现了Optimizer trait
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[1, 1], Some("input")).unwrap();
    let param = graph.new_parameter_node(&[1, 1], Some("param")).unwrap();
    let output = graph
        .new_mat_mul_node(param, input, Some("output"))
        .unwrap();
    let _loss = graph
        .new_perception_loss_node(output, Some("loss"))
        .unwrap();

    let input_value = Tensor::new(&[1.0], &[1, 1]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    // 测试SGD作为trait object
    let mut sgd: Box<dyn Optimizer> = Box::new(SGD::new(&graph, 0.01).unwrap());
    assert_eq!(sgd.learning_rate(), 0.01);
    sgd.set_learning_rate(0.02);
    assert_eq!(sgd.learning_rate(), 0.02);

    // 测试Adam作为trait object
    let mut adam: Box<dyn Optimizer> = Box::new(Adam::new_default(&graph, 0.001).unwrap());
    assert_eq!(adam.learning_rate(), 0.001);
    adam.set_learning_rate(0.002);
    assert_eq!(adam.learning_rate(), 0.002);
}

#[test]
fn test_optimizer_with_multiple_parameters() {
    // 测试优化器能正确处理多个参数
    // 使用ADALINE结构确保梯度不为0
    let mut graph = Graph::new();
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let label = graph.new_input_node(&[1, 1], Some("label")).unwrap();
    let w = graph.new_parameter_node(&[1, 2], Some("w")).unwrap();
    let b = graph.new_parameter_node(&[1, 1], Some("b")).unwrap();

    let wx = graph.new_mat_mul_node(w, x, Some("wx")).unwrap();
    let output = graph.new_add_node(&[wx, b], Some("output")).unwrap();
    let loss_input = graph
        .new_mat_mul_node(label, output, Some("loss_input"))
        .unwrap();
    let loss = graph
        .new_perception_loss_node(loss_input, Some("loss"))
        .unwrap();

    // 显式设置参数值，确保output > 0，使得loss_input = label * output < 0
    // 这样PerceptionLoss的梯度才不为0
    let w_init = Tensor::new(&[1.0, 1.0], &[1, 2]); // w = [1, 1]
    let b_init = Tensor::new(&[0.5], &[1, 1]); // b = 0.5
    graph.set_node_value(w, Some(&w_init)).unwrap();
    graph.set_node_value(b, Some(&b_init)).unwrap();

    // 设置输入值：x=[1,2]^T, label=-1
    // output = w @ x + b = [1,1] @ [1,2]^T + 0.5 = 3 + 0.5 = 3.5
    // loss_input = label * output = -1 * 3.5 = -3.5 < 0，梯度不为0
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let label_value = Tensor::new(&[-1.0], &[1, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(label, Some(&label_value)).unwrap();

    let initial_w = graph.get_node_value(w).unwrap().unwrap().clone();
    let initial_b = graph.get_node_value(b).unwrap().unwrap().clone();

    // 使用SGD优化
    let mut sgd = SGD::new(&graph, 0.01).unwrap();
    sgd.one_step(&mut graph, loss).unwrap();
    sgd.update(&mut graph).unwrap();

    // 两个参数都应该更新
    let new_w = graph.get_node_value(w).unwrap().unwrap();
    let new_b = graph.get_node_value(b).unwrap().unwrap();

    // 两个参数都应该变化
    let w_changed = new_w != &initial_w;
    let b_changed = new_b != &initial_b;
    assert!(w_changed, "参数w应该在优化后改变");
    assert!(b_changed, "参数b应该在优化后改变");
}

#[test]
fn test_optimizer_update_without_one_step() {
    // 测试在没有调用one_step的情况下调用update
    let mut graph = Graph::new();
    let _param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();

    let mut sgd = SGD::new(&graph, 0.01).unwrap();

    // 直接调用update（没有累积任何梯度）
    // 应该不报错，参数保持不变
    let result = sgd.update(&mut graph);
    assert!(result.is_ok());
}
