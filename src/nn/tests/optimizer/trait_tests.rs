/*
 * @Author       : 老董
 * @Date         : 2025-12-20
 * @Description  : Optimizer trait 通用行为测试
 *
 * 这些测试验证所有优化器实现的共同行为和 trait 约束
 */

use crate::nn::Graph;
use crate::nn::optimizer::{Adam, Optimizer, SGD};
use crate::tensor::Tensor;

#[test]
fn test_optimizer_trait_implementations() {
    // 验证 SGD 和 Adam 都正确实现了 Optimizer trait
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

    // 测试 SGD 作为 trait object
    let mut sgd: Box<dyn Optimizer> = Box::new(SGD::new(&graph, 0.01).unwrap());
    assert_eq!(sgd.learning_rate(), 0.01);
    sgd.set_learning_rate(0.02);
    assert_eq!(sgd.learning_rate(), 0.02);

    // 测试 Adam 作为 trait object
    let mut adam: Box<dyn Optimizer> = Box::new(Adam::new_default(&graph, 0.001).unwrap());
    assert_eq!(adam.learning_rate(), 0.001);
    adam.set_learning_rate(0.002);
    assert_eq!(adam.learning_rate(), 0.002);
}

#[test]
fn test_optimizer_with_multiple_parameters() {
    // 测试优化器能正确处理多个参数
    // 使用 ADALINE 结构确保梯度不为0
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

    // 显式设置参数值，确保 output > 0，使得 loss_input = label * output < 0
    // 这样 PerceptionLoss 的梯度才不为0
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

    // 使用新 API：forward -> backward -> step
    let mut sgd = SGD::new(&graph, 0.01).unwrap();
    graph.zero_grad().unwrap();
    graph.forward(loss).unwrap();
    graph.backward(loss).unwrap();
    sgd.step(&mut graph).unwrap();

    // 两个参数都应该更新
    let new_w = graph.get_node_value(w).unwrap().unwrap();
    let new_b = graph.get_node_value(b).unwrap().unwrap();

    // 两个参数都应该变化
    let w_changed = new_w != &initial_w;
    let b_changed = new_b != &initial_b;
    assert!(w_changed, "参数 w 应该在优化后改变");
    assert!(b_changed, "参数 b 应该在优化后改变");
}

#[test]
fn test_optimizer_step_without_backward() {
    // 测试在没有调用 backward 的情况下调用 step
    let mut graph = Graph::new();
    let param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();

    // 记录更新前的参数值
    let param_before = graph.get_node_value(param).unwrap().unwrap().clone();

    let mut sgd = SGD::new(&graph, 0.01).unwrap();

    // 直接调用 step（没有梯度）
    // 应该不报错，参数保持不变（因为没有梯度）
    let result = sgd.step(&mut graph);
    assert!(result.is_ok());

    // 验证参数值未改变
    let param_after = graph.get_node_value(param).unwrap().unwrap();
    assert_eq!(
        &param_before, param_after,
        "未调用 backward 时，step 不应改变参数值"
    );
}
