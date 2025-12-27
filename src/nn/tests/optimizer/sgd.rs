/*
 * @Author       : 老董
 * @Date         : 2025-12-20
 * @Description  : SGD (随机梯度下降) 优化器测试
 */

use crate::nn::Graph;
use crate::nn::optimizer::{Optimizer, SGD};
use crate::tensor::Tensor;

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
    // 预期值通过 PyTorch 验证，见 tests/python/calc_jacobi_by_pytorch/optimizer_test_values.py
    //
    // 计算图: output = w @ x, loss_input = label @ output, loss = perception_loss(loss_input)
    // 初始值: w=1, x=1, label=-1 => output=1, loss_input=-1, loss=1 (因为-1<0)
    // 梯度推导:
    //   d(loss)/d(loss_input) = -1 (perception_loss在x<0时)
    //   d(loss_input)/d(output) = label = -1
    //   d(output)/d(w) = x = 1
    //   d(loss)/d(w) = (-1) * (-1) * 1 = 1
    // SGD更新: w_new = w_old - lr * grad = 1.0 - 0.1 * 1.0 = 0.9
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
        .set_node_value(w, Some(&Tensor::new(&[1.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(x, Some(&Tensor::new(&[1.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(label, Some(&Tensor::new(&[-1.0], &[1, 1])))
        .unwrap();

    let mut sgd = SGD::new(&graph, 0.1).unwrap();
    sgd.one_step(&mut graph, loss).unwrap();
    sgd.update(&mut graph).unwrap();

    // 验证：PyTorch 计算结果 w_new = 0.9
    let new_w = graph.get_node_value(w).unwrap().unwrap();
    let new_w_value = new_w.get(&[0, 0]).get_data_number().unwrap();
    assert_eq!(new_w_value, 0.9, "SGD更新后w应该等于0.9（PyTorch验证）");
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
    // 预期值通过 PyTorch 验证，见 tests/python/calc_jacobi_by_pytorch/optimizer_test_values.py
    //
    // 计算图: output = w @ x, loss_input = label @ output, loss = perception_loss(loss_input)
    // 初始值: w=1, x=2, label=-1 => output=2, loss_input=-2, loss=2
    // 每次迭代的梯度:
    //   d(loss)/d(w) = d(loss)/d(loss_input) * d(loss_input)/d(output) * d(output)/d(w)
    //                = (-1) * (-1) * 2 = 2
    // 3次累积后平均梯度 = (2+2+2)/3 = 2
    // SGD更新: w_new = w_old - lr * avg_grad = 1.0 - 0.1 * 2.0 = 0.8
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
        .set_node_value(w, Some(&Tensor::new(&[1.0], &[1, 1])))
        .unwrap();
    let x_value = Tensor::new(&[2.0], &[1, 1]);
    let label_value = Tensor::new(&[-1.0], &[1, 1]);

    let mut sgd = SGD::new(&graph, 0.1).unwrap();

    // 执行3次one_step（梯度累积）
    for _ in 0..3 {
        graph.set_node_value(x, Some(&x_value)).unwrap();
        graph.set_node_value(label, Some(&label_value)).unwrap();
        sgd.one_step(&mut graph, loss).unwrap();
    }

    sgd.update(&mut graph).unwrap();

    // 验证：PyTorch 计算结果 w_new = 0.8
    let new_w = graph.get_node_value(w).unwrap().unwrap();
    let new_w_value = new_w.get(&[0, 0]).get_data_number().unwrap();
    assert_eq!(new_w_value, 0.8, "SGD梯度累积后w应该等于0.8（PyTorch验证）");
}

#[test]
fn test_sgd_with_params() {
    // 测试 with_params 创建优化器：只更新指定参数
    let mut graph = Graph::new();
    let x = graph.new_input_node(&[1, 1], Some("x")).unwrap();
    let label = graph.new_input_node(&[1, 1], Some("label")).unwrap();

    // 创建两个可训练参数
    let w1 = graph.new_parameter_node(&[1, 1], Some("w1")).unwrap();
    let w2 = graph.new_parameter_node(&[1, 1], Some("w2")).unwrap();

    // 构建: output = w2 @ (w1 @ x)
    let hidden = graph.new_mat_mul_node(w1, x, Some("hidden")).unwrap();
    let output = graph.new_mat_mul_node(w2, hidden, Some("output")).unwrap();
    let loss_input = graph
        .new_mat_mul_node(label, output, Some("loss_input"))
        .unwrap();
    let loss = graph
        .new_perception_loss_node(loss_input, Some("loss"))
        .unwrap();

    // 设置初始值
    graph
        .set_node_value(w1, Some(&Tensor::new(&[2.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(w2, Some(&Tensor::new(&[3.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(x, Some(&Tensor::new(&[1.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(label, Some(&Tensor::new(&[-1.0], &[1, 1])))
        .unwrap();

    // 只优化 w1
    let mut sgd = SGD::with_params(&[w1], 0.1);

    let w1_before = graph.get_node_value(w1).unwrap().unwrap().clone();
    let w2_before = graph.get_node_value(w2).unwrap().unwrap().clone();

    sgd.one_step(&mut graph, loss).unwrap();
    sgd.update(&mut graph).unwrap();

    let w1_after = graph.get_node_value(w1).unwrap().unwrap();
    let w2_after = graph.get_node_value(w2).unwrap().unwrap();

    // w1 应该被更新
    assert_ne!(
        &w1_before, w1_after,
        "w1 应该被优化器更新（在 with_params 列表中）"
    );
    // w2 应该保持不变
    assert_eq!(
        &w2_before, w2_after,
        "w2 不应该被更新（不在 with_params 列表中）"
    );
}

#[test]
fn test_sgd_with_params_different_learning_rates() {
    // 测试为不同参数使用不同学习率（迁移学习场景）
    let mut graph = Graph::new();
    let x = graph.new_input_node(&[1, 1], Some("x")).unwrap();
    let label = graph.new_input_node(&[1, 1], Some("label")).unwrap();

    // 预训练层参数（小学习率）
    let w_pretrained = graph
        .new_parameter_node(&[1, 1], Some("w_pretrained"))
        .unwrap();
    // 新层参数（大学习率）
    let w_new = graph.new_parameter_node(&[1, 1], Some("w_new")).unwrap();

    // 构建: output = w_new @ (w_pretrained @ x)
    let hidden = graph
        .new_mat_mul_node(w_pretrained, x, Some("hidden"))
        .unwrap();
    let output = graph
        .new_mat_mul_node(w_new, hidden, Some("output"))
        .unwrap();
    let loss_input = graph
        .new_mat_mul_node(label, output, Some("loss_input"))
        .unwrap();
    let loss = graph
        .new_perception_loss_node(loss_input, Some("loss"))
        .unwrap();

    // 设置初始值
    graph
        .set_node_value(w_pretrained, Some(&Tensor::new(&[2.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(w_new, Some(&Tensor::new(&[3.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(x, Some(&Tensor::new(&[1.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(label, Some(&Tensor::new(&[-1.0], &[1, 1])))
        .unwrap();

    // 预训练层使用小学习率
    let mut sgd_pretrained = SGD::with_params(&[w_pretrained], 0.001);
    // 新层使用大学习率
    let mut sgd_new = SGD::with_params(&[w_new], 0.1);

    let w_pretrained_before = graph.get_node_value(w_pretrained).unwrap().unwrap().clone();
    let w_new_before = graph.get_node_value(w_new).unwrap().unwrap().clone();

    // 同时更新两个优化器
    sgd_pretrained.one_step(&mut graph, loss).unwrap();
    sgd_new.one_step(&mut graph, loss).unwrap();
    sgd_pretrained.update(&mut graph).unwrap();
    sgd_new.update(&mut graph).unwrap();

    let w_pretrained_after = graph.get_node_value(w_pretrained).unwrap().unwrap();
    let w_new_after = graph.get_node_value(w_new).unwrap().unwrap();

    // 两个参数都应该被更新
    assert_ne!(
        &w_pretrained_before, w_pretrained_after,
        "预训练层参数应该被更新"
    );
    assert_ne!(&w_new_before, w_new_after, "新层参数应该被更新");

    // 新层的变化幅度应该更大（因为学习率是 100 倍）
    let pretrained_change = (w_pretrained_after.get(&[0, 0]).get_data_number().unwrap()
        - w_pretrained_before.get(&[0, 0]).get_data_number().unwrap())
    .abs();
    let new_change = (w_new_after.get(&[0, 0]).get_data_number().unwrap()
        - w_new_before.get(&[0, 0]).get_data_number().unwrap())
    .abs();

    assert!(
        new_change > pretrained_change,
        "新层变化幅度（{}）应该大于预训练层（{}）",
        new_change,
        pretrained_change
    );
}
