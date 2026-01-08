/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : 简单回归任务集成测试（全批量梯度下降）
 *
 * 演示使用 MSELoss 进行多输入线性回归任务。
 * 采用全批量梯度下降（Full Batch Gradient Descent）训练方式：
 * 每个 epoch 使用全部样本一次性计算梯度并更新参数。
 */

use approx::assert_abs_diff_eq;
use only_torch::nn::Graph;
use only_torch::nn::optimizer::{Optimizer, SGD};
use only_torch::tensor::Tensor;
use std::fs;

/// 简单线性回归：学习 y = x1 + 2*x2 + 3
///
/// 这个测试展示了如何使用 `MSELoss` 进行回归任务：
/// 1. 创建简单的线性模型 `y_pred` = x @ w + b
/// 2. 使用 `MSELoss` 计算预测值和真实值之间的误差
/// 3. 使用 SGD 优化器更新参数（全批量训练）
/// 4. 验证学习到的参数接近真实值 (w=[1, 2], b=3)
#[test]
fn test_simple_regression_full_batch() {
    // 训练数据: y = x1 + 2*x2 + 3
    let x_data: Vec<f32> = vec![
        0.0, 0.0, // sample 0
        1.0, 0.0, // sample 1
        0.0, 1.0, // sample 2
        1.0, 1.0, // sample 3
        2.0, 1.0, // sample 4
        1.0, 2.0, // sample 5
        -1.0, 1.0, // sample 6
    ];
    let y_data: Vec<f32> = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 4.0];
    let batch_size = 7;

    let mut graph = Graph::new_with_seed(123);

    // 创建网络结构: y_pred = x @ w + b
    // x: [batch, 2], w: [2, 1], b: [1, 1]
    let x_id = graph.new_input_node(&[batch_size, 2], Some("x")).unwrap();
    let w_id = graph.new_parameter_node(&[2, 1], Some("w")).unwrap();
    let b_id = graph.new_parameter_node(&[1, 1], Some("b")).unwrap();

    // y_pred = x @ w
    let xw_id = graph.new_mat_mul_node(x_id, w_id, Some("xw")).unwrap();

    // ones 节点用于广播 bias
    let ones_id = graph
        .new_input_node(&[batch_size, 1], Some("ones"))
        .unwrap();

    // bias_broadcast = ones @ b
    let bias_broadcast_id = graph
        .new_mat_mul_node(ones_id, b_id, Some("bias_broadcast"))
        .unwrap();

    // y_pred = xw + bias_broadcast
    let y_pred_id = graph
        .new_add_node(&[xw_id, bias_broadcast_id], Some("y_pred"))
        .unwrap();

    // 目标值
    let y_true_id = graph
        .new_input_node(&[batch_size, 1], Some("y_true"))
        .unwrap();

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
        .set_node_value(ones_id, Some(&Tensor::ones(&[batch_size, 1])))
        .unwrap();

    // 设置训练数据
    graph
        .set_node_value(x_id, Some(&Tensor::new(&x_data, &[batch_size, 2])))
        .unwrap();
    graph
        .set_node_value(y_true_id, Some(&Tensor::new(&y_data, &[batch_size, 1])))
        .unwrap();

    // 保存网络结构可视化（训练前）
    let output_dir = "tests/outputs";
    fs::create_dir_all(output_dir).ok();
    graph
        .save_visualization(format!("{output_dir}/simple_regression_full_batch"), None)
        .unwrap();
    graph
        .save_summary(format!(
            "{output_dir}/simple_regression_full_batch_summary.md"
        ))
        .unwrap();
    println!("网络结构已保存: {output_dir}/simple_regression_full_batch.png");

    let mut optimizer = SGD::new(&graph, 0.05).unwrap();

    // 训练 200 个 epoch（全批量训练）
    for epoch in 0..200 {
        graph.zero_grad().unwrap();
        graph.forward(loss_id).unwrap();
        let loss = graph.backward(loss_id).unwrap();
        optimizer.step(&mut graph).unwrap();

        // 打印每 50 个 epoch 的损失
        if (epoch + 1) % 50 == 0 {
            let w = graph.get_node_value(w_id).unwrap().unwrap();
            let b = graph.get_node_value(b_id).unwrap().unwrap()[[0, 0]];
            println!(
                "Epoch {}: loss = {:.6}, w = [{:.4}, {:.4}], b = {:.4}",
                epoch + 1,
                loss,
                w[[0, 0]],
                w[[1, 0]],
                b
            );
        }
    }

    // 验证学习到的参数
    let learned_w = graph.get_node_value(w_id).unwrap().unwrap();
    let learned_b = graph.get_node_value(b_id).unwrap().unwrap()[[0, 0]];

    println!("\n最终结果:");
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
