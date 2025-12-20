/*
 * @Author       : 老董
 * @Date         : 2025-12-20
 * @Description  : Adam 优化器测试
 */

use crate::nn::optimizer::{Adam, Optimizer};
use crate::nn::Graph;
use crate::tensor::Tensor;

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
    // 测试Adam更新公式
    // 预期值通过 PyTorch 验证，见 tests/calc_jacobi_by_pytorch/optimizer_test_values.py
    //
    // 计算图: output = w @ x, loss_input = label @ output, loss = perception_loss(loss_input)
    // 初始值: w=2, x=3, label=-1 => output=6, loss_input=-6, loss=6
    // 梯度: d(loss)/d(w) = (-1) * (-1) * 3 = 3
    // Adam更新 (beta1=0.9, beta2=0.999, eps=1e-8, lr=0.1):
    //   m_1 = 0.1 * 3 = 0.3
    //   v_1 = 0.001 * 9 = 0.009
    //   m_hat = 0.3 / (1-0.9) = 3.0
    //   v_hat = 0.009 / (1-0.999) = 9.0
    //   update = 0.1 * 3.0 / (sqrt(9.0) + 1e-8) ≈ 0.1
    //   w_new = 2.0 - 0.1 ≈ 1.9
    // PyTorch验证: w_new = 1.899999976158142
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

    // 设置初始值（PyTorch验证用的相同值）
    graph
        .set_node_value(w, Some(&Tensor::new(&[2.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(x, Some(&Tensor::new(&[3.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(label, Some(&Tensor::new(&[-1.0], &[1, 1])))
        .unwrap();

    let mut adam = Adam::new_default(&graph, 0.1).unwrap();
    adam.one_step(&mut graph, loss).unwrap();
    adam.update(&mut graph).unwrap();

    // 验证：PyTorch 计算结果 w_new ≈ 1.9
    let new_w = graph.get_node_value(w).unwrap().unwrap();
    let new_w_value = new_w.get(&[0, 0]).get_data_number().unwrap();
    let expected = 1.9;
    let tolerance = 1e-5;
    assert!(
        (new_w_value - expected).abs() < tolerance,
        "Adam更新后w应该约等于1.9（PyTorch验证），实际值: {}",
        new_w_value
    );
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

