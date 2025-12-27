/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : 简单回归任务集成测试
 *
 * 演示使用 MSELoss 进行简单的线性回归任务
 */

use approx::assert_abs_diff_eq;
use only_torch::nn::Graph;
use only_torch::nn::optimizer::{Optimizer, SGD};
use only_torch::tensor::Tensor;
use std::fs;

/// 简单线性回归：学习 y = 2x + 1
///
/// 这个测试展示了如何使用 MSELoss 进行回归任务：
/// 1. 创建简单的线性模型 y_pred = w * x + b
/// 2. 使用 MSELoss 计算预测值和真实值之间的误差
/// 3. 使用 SGD 优化器更新参数
/// 4. 验证学习到的参数接近真实值 (w=2, b=1)
#[test]
fn test_simple_linear_regression() {
    let mut graph = Graph::new_with_seed(42);

    // 创建网络结构: y_pred = x @ w + b
    // x: [batch, 1], w: [1, 1], b: [1, 1]
    let x_id = graph.new_input_node(&[1, 1], Some("x")).unwrap();
    let w_id = graph.new_parameter_node(&[1, 1], Some("w")).unwrap();
    let b_id = graph.new_parameter_node(&[1, 1], Some("b")).unwrap();

    // y_pred = x @ w
    let xw_id = graph.new_mat_mul_node(x_id, w_id, Some("xw")).unwrap();

    // ones 节点用于广播 bias
    let ones_id = graph.new_input_node(&[1, 1], Some("ones")).unwrap();

    // bias_broadcast = ones @ b
    let bias_broadcast_id = graph
        .new_mat_mul_node(ones_id, b_id, Some("bias_broadcast"))
        .unwrap();

    // y_pred = xw + bias_broadcast
    let y_pred_id = graph
        .new_add_node(&[xw_id, bias_broadcast_id], Some("y_pred"))
        .unwrap();

    // 目标值
    let y_true_id = graph.new_input_node(&[1, 1], Some("y_true")).unwrap();

    // MSE 损失
    let loss_id = graph
        .new_mse_loss_node(y_pred_id, y_true_id, Some("loss"))
        .unwrap();

    // 初始化参数（随机值，远离真实值）
    graph
        .set_node_value(w_id, Some(&Tensor::new(&[0.5], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(b_id, Some(&Tensor::new(&[0.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(ones_id, Some(&Tensor::new(&[1.0], &[1, 1])))
        .unwrap();

    // 保存网络结构可视化（训练前）
    let output_dir = "tests/outputs";
    fs::create_dir_all(output_dir).ok();
    graph
        .save_visualization(&format!("{output_dir}/simple_regression"), None)
        .unwrap();
    graph
        .save_summary(&format!("{output_dir}/simple_regression_summary.md"))
        .unwrap();
    println!("网络结构已保存: {}/simple_regression.png", output_dir);

    // 创建优化器
    let mut optimizer = SGD::new(&graph, 0.1).unwrap();

    // 训练数据: y = 2x + 1
    let training_data: Vec<(f32, f32)> = vec![
        (0.0, 1.0),   // 2*0 + 1 = 1
        (1.0, 3.0),   // 2*1 + 1 = 3
        (2.0, 5.0),   // 2*2 + 1 = 5
        (3.0, 7.0),   // 2*3 + 1 = 7
        (4.0, 9.0),   // 2*4 + 1 = 9
        (-1.0, -1.0), // 2*(-1) + 1 = -1
    ];

    // 训练 100 个 epoch
    for epoch in 0..100 {
        let mut total_loss = 0.0;

        for &(x_val, y_val) in &training_data {
            graph
                .set_node_value(x_id, Some(&Tensor::new(&[x_val], &[1, 1])))
                .unwrap();
            graph
                .set_node_value(y_true_id, Some(&Tensor::new(&[y_val], &[1, 1])))
                .unwrap();

            optimizer.one_step(&mut graph, loss_id).unwrap();

            let loss = graph.get_node_value(loss_id).unwrap().unwrap();
            total_loss += loss[[0, 0]];
        }

        // 更新参数
        optimizer.update(&mut graph).unwrap();

        // 打印每 20 个 epoch 的平均损失
        if (epoch + 1) % 20 == 0 {
            let avg_loss = total_loss / training_data.len() as f32;
            let w = graph.get_node_value(w_id).unwrap().unwrap()[[0, 0]];
            let b = graph.get_node_value(b_id).unwrap().unwrap()[[0, 0]];
            println!(
                "Epoch {}: avg_loss = {:.6}, w = {:.4}, b = {:.4}",
                epoch + 1,
                avg_loss,
                w,
                b
            );
        }
    }

    // 验证学习到的参数
    let learned_w = graph.get_node_value(w_id).unwrap().unwrap()[[0, 0]];
    let learned_b = graph.get_node_value(b_id).unwrap().unwrap()[[0, 0]];

    println!("\n最终结果:");
    println!("  真实参数: w = 2.0, b = 1.0");
    println!("  学习参数: w = {:.4}, b = {:.4}", learned_w, learned_b);

    // 验证参数接近真实值
    assert_abs_diff_eq!(learned_w, 2.0, epsilon = 0.1);
    assert_abs_diff_eq!(learned_b, 1.0, epsilon = 0.1);
}

/// 多输入线性回归：学习 y = x1 + 2*x2 + 3
///
/// 测试多个输入特征的回归任务
#[test]
fn test_multi_input_linear_regression() {
    let mut graph = Graph::new_with_seed(123);

    // 创建网络结构: y_pred = x @ w + b
    // x: [1, 2], w: [2, 1], b: [1, 1]
    let x_id = graph.new_input_node(&[1, 2], Some("x")).unwrap();
    let w_id = graph.new_parameter_node(&[2, 1], Some("w")).unwrap();
    let b_id = graph.new_parameter_node(&[1, 1], Some("b")).unwrap();

    // y_pred = x @ w
    let xw_id = graph.new_mat_mul_node(x_id, w_id, Some("xw")).unwrap();

    // ones 节点用于广播 bias
    let ones_id = graph.new_input_node(&[1, 1], Some("ones")).unwrap();

    // bias_broadcast = ones @ b
    let bias_broadcast_id = graph
        .new_mat_mul_node(ones_id, b_id, Some("bias_broadcast"))
        .unwrap();

    // y_pred = xw + bias_broadcast
    let y_pred_id = graph
        .new_add_node(&[xw_id, bias_broadcast_id], Some("y_pred"))
        .unwrap();

    // 目标值
    let y_true_id = graph.new_input_node(&[1, 1], Some("y_true")).unwrap();

    // MSE 损失
    let loss_id = graph
        .new_mse_loss_node(y_pred_id, y_true_id, Some("loss"))
        .unwrap();

    // 初始化参数
    graph
        .set_node_value(w_id, Some(&Tensor::new(&[0.0, 0.0], &[2, 1])))
        .unwrap();
    graph
        .set_node_value(b_id, Some(&Tensor::new(&[0.0], &[1, 1])))
        .unwrap();
    graph
        .set_node_value(ones_id, Some(&Tensor::new(&[1.0], &[1, 1])))
        .unwrap();

    let mut optimizer = SGD::new(&graph, 0.05).unwrap();

    // 训练数据: y = x1 + 2*x2 + 3
    let training_data: Vec<([f32; 2], f32)> = vec![
        ([0.0, 0.0], 3.0),  // 0 + 0 + 3 = 3
        ([1.0, 0.0], 4.0),  // 1 + 0 + 3 = 4
        ([0.0, 1.0], 5.0),  // 0 + 2 + 3 = 5
        ([1.0, 1.0], 6.0),  // 1 + 2 + 3 = 6
        ([2.0, 1.0], 7.0),  // 2 + 2 + 3 = 7
        ([1.0, 2.0], 8.0),  // 1 + 4 + 3 = 8
        ([-1.0, 1.0], 4.0), // -1 + 2 + 3 = 4
    ];

    // 训练 200 个 epoch
    for _ in 0..200 {
        for &(x_val, y_val) in &training_data {
            graph
                .set_node_value(x_id, Some(&Tensor::new(&x_val, &[1, 2])))
                .unwrap();
            graph
                .set_node_value(y_true_id, Some(&Tensor::new(&[y_val], &[1, 1])))
                .unwrap();

            optimizer.one_step(&mut graph, loss_id).unwrap();
        }
        optimizer.update(&mut graph).unwrap();
    }

    // 验证学习到的参数
    let learned_w = graph.get_node_value(w_id).unwrap().unwrap();
    let learned_b = graph.get_node_value(b_id).unwrap().unwrap()[[0, 0]];

    println!("\n多输入回归最终结果:");
    println!("  真实参数: w = [1.0, 2.0], b = 3.0");
    println!(
        "  学习参数: w = [{:.4}, {:.4}], b = {:.4}",
        learned_w[[0, 0]],
        learned_w[[1, 0]],
        learned_b
    );

    // 验证参数接近真实值
    assert_abs_diff_eq!(learned_w[[0, 0]], 1.0, epsilon = 0.2);
    assert_abs_diff_eq!(learned_w[[1, 0]], 2.0, epsilon = 0.2);
    assert_abs_diff_eq!(learned_b, 3.0, epsilon = 0.2);
}
