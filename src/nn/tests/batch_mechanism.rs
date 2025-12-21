/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : Batch 机制单元测试
 *                 验证 batch forward/backward 与累加单样本的结果一致
 */

use approx::assert_abs_diff_eq;

use crate::nn::optimizer::{Optimizer, SGD};
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use crate::tensor_slice;

/// 测试 batch forward 与单样本 forward 的一致性
#[test]
fn test_batch_forward_equals_single() -> Result<(), GraphError> {
    let seed = 42u64;
    let batch_size = 3;
    let input_dim = 4;
    let output_dim = 2;

    // 创建测试数据
    let samples = vec![
        Tensor::normal_seeded(0.0, 1.0, &[1, input_dim], seed),
        Tensor::normal_seeded(0.0, 1.0, &[1, input_dim], seed + 1),
        Tensor::normal_seeded(0.0, 1.0, &[1, input_dim], seed + 2),
    ];

    // 合并成 batch（使用 stack 沿第一维度连接）
    let sample_refs: Vec<&Tensor> = samples.iter().collect();
    let batch_input = Tensor::stack(&sample_refs, false);
    assert_eq!(batch_input.shape(), &[batch_size, input_dim]);

    // 创建图和网络
    let mut graph = Graph::new_with_seed(seed);
    let x = graph.new_input_node(&[batch_size, input_dim], Some("x"))?;
    let w = graph.new_parameter_node_seeded(&[input_dim, output_dim], Some("w"), seed + 100)?;
    let z = graph.new_mat_mul_node(x, w, Some("z"))?;
    let out = graph.new_sigmoid_node(z, Some("out"))?;

    // Batch forward
    graph.set_node_value(x, Some(&batch_input))?;
    graph.forward_batch(out)?;
    let batch_output = graph.get_node_value(out)?.unwrap().clone();

    // 单样本 forward 并收集结果
    let mut single_outputs = Vec::new();
    for (_i, sample) in samples.iter().enumerate() {
        let mut graph_single = Graph::new_with_seed(seed);
        let x_single = graph_single.new_input_node(&[1, input_dim], Some("x"))?;
        let w_single = graph_single.new_parameter_node_seeded(
            &[input_dim, output_dim],
            Some("w"),
            seed + 100,
        )?;
        let z_single = graph_single.new_mat_mul_node(x_single, w_single, Some("z"))?;
        let out_single = graph_single.new_sigmoid_node(z_single, Some("out"))?;

        graph_single.set_node_value(x_single, Some(sample))?;
        graph_single.forward_node(out_single)?;
        single_outputs.push(graph_single.get_node_value(out_single)?.unwrap().clone());
    }

    // 验证 batch 输出的每一行等于对应的单样本输出
    for (i, single_out) in single_outputs.iter().enumerate() {
        for j in 0..output_dim {
            let batch_val = batch_output[[i, j]];
            let single_val = single_out[[0, j]];
            assert_abs_diff_eq!(batch_val, single_val, epsilon = 1e-5);
        }
    }

    println!("✅ test_batch_forward_equals_single 通过");
    Ok(())
}

/// 测试 batch backward 梯度与累加单样本梯度的一致性
#[test]
fn test_batch_gradient_equals_accumulated_single() -> Result<(), GraphError> {
    let seed = 42u64;
    let batch_size = 3;
    let input_dim = 4;
    let hidden_dim = 3;
    let output_dim = 2;

    // 创建测试数据
    let samples = vec![
        Tensor::normal_seeded(0.0, 1.0, &[1, input_dim], seed),
        Tensor::normal_seeded(0.0, 1.0, &[1, input_dim], seed + 1),
        Tensor::normal_seeded(0.0, 1.0, &[1, input_dim], seed + 2),
    ];
    let labels = vec![
        Tensor::new(&[1.0, 0.0], &[1, output_dim]),
        Tensor::new(&[0.0, 1.0], &[1, output_dim]),
        Tensor::new(&[1.0, 0.0], &[1, output_dim]),
    ];

    // 合并成 batch
    let sample_refs: Vec<&Tensor> = samples.iter().collect();
    let label_refs: Vec<&Tensor> = labels.iter().collect();
    let batch_input = Tensor::stack(&sample_refs, false);
    let batch_labels = Tensor::stack(&label_refs, false);

    // ========== Batch 模式 ==========
    let mut graph_batch = Graph::new_with_seed(seed);
    let x_b = graph_batch.new_input_node(&[batch_size, input_dim], Some("x"))?;
    let y_b = graph_batch.new_input_node(&[batch_size, output_dim], Some("y"))?;
    let w1_b =
        graph_batch.new_parameter_node_seeded(&[input_dim, hidden_dim], Some("w1"), seed + 100)?;
    let w2_b =
        graph_batch.new_parameter_node_seeded(&[hidden_dim, output_dim], Some("w2"), seed + 101)?;

    let z1_b = graph_batch.new_mat_mul_node(x_b, w1_b, None)?;
    let a1_b = graph_batch.new_sigmoid_node(z1_b, None)?;
    let z2_b = graph_batch.new_mat_mul_node(a1_b, w2_b, None)?;
    let loss_b = graph_batch.new_softmax_cross_entropy_node(z2_b, y_b, Some("loss"))?;

    // Batch forward + backward
    graph_batch.set_node_value(x_b, Some(&batch_input))?;
    graph_batch.set_node_value(y_b, Some(&batch_labels))?;
    graph_batch.forward_batch(loss_b)?;
    graph_batch.backward_batch(loss_b)?;

    let batch_grad_w1 = graph_batch.get_node_grad_batch(w1_b)?.unwrap().clone();
    let batch_grad_w2 = graph_batch.get_node_grad_batch(w2_b)?.unwrap().clone();

    // ========== 单样本累加模式 ==========
    let mut accumulated_grad_w1 = Tensor::zeros(&[input_dim, hidden_dim]);
    let mut accumulated_grad_w2 = Tensor::zeros(&[hidden_dim, output_dim]);

    for i in 0..batch_size {
        let mut graph_single = Graph::new_with_seed(seed);
        let x_s = graph_single.new_input_node(&[1, input_dim], Some("x"))?;
        let y_s = graph_single.new_input_node(&[1, output_dim], Some("y"))?;
        let w1_s = graph_single.new_parameter_node_seeded(
            &[input_dim, hidden_dim],
            Some("w1"),
            seed + 100,
        )?;
        let w2_s = graph_single.new_parameter_node_seeded(
            &[hidden_dim, output_dim],
            Some("w2"),
            seed + 101,
        )?;

        let z1_s = graph_single.new_mat_mul_node(x_s, w1_s, None)?;
        let a1_s = graph_single.new_sigmoid_node(z1_s, None)?;
        let z2_s = graph_single.new_mat_mul_node(a1_s, w2_s, None)?;
        let loss_s = graph_single.new_softmax_cross_entropy_node(z2_s, y_s, Some("loss"))?;

        graph_single.set_node_value(x_s, Some(&samples[i]))?;
        graph_single.set_node_value(y_s, Some(&labels[i]))?;
        graph_single.forward_node(loss_s)?;
        graph_single.backward_nodes(&[w1_s, w2_s], loss_s)?;

        // 累加梯度
        if let Some(grad) = graph_single.get_node_grad(w1_s)? {
            accumulated_grad_w1 = accumulated_grad_w1 + grad;
        }
        if let Some(grad) = graph_single.get_node_grad(w2_s)? {
            accumulated_grad_w2 = accumulated_grad_w2 + grad;
        }
    }

    // 求平均
    let avg_grad_w1 = accumulated_grad_w1 / (batch_size as f32);
    let avg_grad_w2 = accumulated_grad_w2 / (batch_size as f32);

    // ========== 验证 ==========
    let tolerance = 1e-4;

    // 验证 w1 梯度
    assert_abs_diff_eq!(batch_grad_w1, avg_grad_w1, epsilon = tolerance);

    // 验证 w2 梯度
    assert_abs_diff_eq!(batch_grad_w2, avg_grad_w2, epsilon = tolerance);

    println!("✅ test_batch_gradient_equals_accumulated_single 通过");
    Ok(())
}

/// 测试 batch 优化器更新与单样本累加更新的一致性
#[test]
fn test_batch_optimizer_update() -> Result<(), GraphError> {
    let seed = 42u64;
    let batch_size = 4;
    let input_dim = 3;
    let output_dim = 2;
    let learning_rate = 0.1;

    // 创建测试数据
    let batch_input = Tensor::normal_seeded(0.0, 1.0, &[batch_size, input_dim], seed);
    let batch_labels = Tensor::new(
        &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        &[batch_size, output_dim],
    );

    // ========== Batch 模式 ==========
    let mut graph_batch = Graph::new_with_seed(seed);
    let x_b = graph_batch.new_input_node(&[batch_size, input_dim], Some("x"))?;
    let y_b = graph_batch.new_input_node(&[batch_size, output_dim], Some("y"))?;
    let w_b =
        graph_batch.new_parameter_node_seeded(&[input_dim, output_dim], Some("w"), seed + 100)?;
    let z_b = graph_batch.new_mat_mul_node(x_b, w_b, None)?;
    let loss_b = graph_batch.new_softmax_cross_entropy_node(z_b, y_b, Some("loss"))?;

    let w_init = graph_batch.get_node_value(w_b)?.unwrap().clone();

    let mut optimizer_batch = SGD::new(&graph_batch, learning_rate)?;

    graph_batch.set_node_value(x_b, Some(&batch_input))?;
    graph_batch.set_node_value(y_b, Some(&batch_labels))?;
    optimizer_batch.one_step_batch(&mut graph_batch, loss_b)?;
    optimizer_batch.update_batch(&mut graph_batch)?;

    let w_after_batch = graph_batch.get_node_value(w_b)?.unwrap().clone();

    // ========== 单样本累加模式 ==========
    let mut graph_single = Graph::new_with_seed(seed);
    let x_s = graph_single.new_input_node(&[1, input_dim], Some("x"))?;
    let y_s = graph_single.new_input_node(&[1, output_dim], Some("y"))?;
    let w_s =
        graph_single.new_parameter_node_seeded(&[input_dim, output_dim], Some("w"), seed + 100)?;
    let z_s = graph_single.new_mat_mul_node(x_s, w_s, None)?;
    let loss_s = graph_single.new_softmax_cross_entropy_node(z_s, y_s, Some("loss"))?;

    let mut optimizer_single = SGD::new(&graph_single, learning_rate)?;

    for i in 0..batch_size {
        // 切片获取单个样本（保持 [1, dim] 形状）
        let sample = tensor_slice!(batch_input, i, ..);
        let label = tensor_slice!(batch_labels, i, ..);
        graph_single.set_node_value(x_s, Some(&sample))?;
        graph_single.set_node_value(y_s, Some(&label))?;
        optimizer_single.one_step(&mut graph_single, loss_s)?;
    }
    optimizer_single.update(&mut graph_single)?;

    let w_after_single = graph_single.get_node_value(w_s)?.unwrap().clone();

    // ========== 验证 ==========
    let tolerance = 1e-4;

    // 验证初始参数相同
    assert_eq!(
        w_init,
        graph_single
            .get_node_value(w_s)?
            .map(|_| w_init.clone())
            .unwrap()
    );

    // 验证更新后的参数
    assert_abs_diff_eq!(w_after_batch, w_after_single, epsilon = tolerance);

    println!("✅ test_batch_optimizer_update 通过");
    Ok(())
}
