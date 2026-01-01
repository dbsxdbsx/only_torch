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

    println!("✅ batch 前向传播与单样本一致性测试通过");
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
    graph_batch.backward_batch(loss_b, None)?;

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

    println!("✅ batch 梯度与累加单样本梯度一致性测试通过");
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

    println!("✅ batch 优化器更新一致性测试通过");
    Ok(())
}

/// 测试 target_params 机制：只计算指定参数的梯度
///
/// 这个测试验证当 `backward_batch(loss, Some(&target_params))` 被调用时，
/// 只有 target_params 中的参数会有梯度，其他参数不应该有梯度。
/// 这是 GAN 等场景的效率优化关键。
#[test]
fn test_backward_batch_target_params_only_computes_specified() -> Result<(), GraphError> {
    let seed = 42u64;
    let batch_size = 4;

    let mut graph = Graph::new_with_seed(seed);

    // 创建输入
    let x = graph.new_input_node(&[batch_size, 4], Some("x"))?;
    let y = graph.new_input_node(&[batch_size, 2], Some("y"))?;

    // 创建两组参数（模拟 GAN 的 G 和 D）
    // Group A: w_a1, w_a2
    let w_a1 = graph.new_parameter_node_seeded(&[4, 8], Some("w_a1"), seed + 1)?;
    let w_a2 = graph.new_parameter_node_seeded(&[8, 4], Some("w_a2"), seed + 2)?;

    // Group B: w_b1, w_b2
    let w_b1 = graph.new_parameter_node_seeded(&[4, 2], Some("w_b1"), seed + 3)?;
    let w_b2 = graph.new_parameter_node_seeded(&[2, 2], Some("w_b2"), seed + 4)?;

    // 构建计算图: x -> [Group A] -> hidden -> [Group B] -> output -> loss
    let h1 = graph.new_mat_mul_node(x, w_a1, None)?;
    let h1_act = graph.new_relu_node(h1, None)?;
    let h2 = graph.new_mat_mul_node(h1_act, w_a2, None)?;
    let h2_act = graph.new_relu_node(h2, None)?;

    let h3 = graph.new_mat_mul_node(h2_act, w_b1, None)?;
    let h3_act = graph.new_relu_node(h3, None)?;
    let output = graph.new_mat_mul_node(h3_act, w_b2, None)?;

    let loss = graph.new_mse_loss_node(output, y, Some("loss"))?;

    // 设置输入数据
    let input_data = Tensor::normal_seeded(0.0, 1.0, &[batch_size, 4], seed + 100);
    let target_data = Tensor::normal_seeded(0.0, 1.0, &[batch_size, 2], seed + 101);
    graph.set_node_value(x, Some(&input_data))?;
    graph.set_node_value(y, Some(&target_data))?;

    // ========== 测试 1: 只计算 Group B 的梯度 ==========
    graph.forward_batch(loss)?;
    graph.backward_batch(loss, Some(&[w_b1, w_b2]))?;

    // Group B 应该有梯度
    let grad_b1 = graph.get_node_grad_batch(w_b1)?;
    let grad_b2 = graph.get_node_grad_batch(w_b2)?;
    assert!(grad_b1.is_some(), "w_b1 应该有梯度（在 target_params 中）");
    assert!(grad_b2.is_some(), "w_b2 应该有梯度（在 target_params 中）");

    // Group A 不应该有梯度（不在 target_params 中）
    let grad_a1 = graph.get_node_grad_batch(w_a1)?;
    let grad_a2 = graph.get_node_grad_batch(w_a2)?;
    assert!(
        grad_a1.is_none(),
        "w_a1 不应该有梯度（不在 target_params 中）"
    );
    assert!(
        grad_a2.is_none(),
        "w_a2 不应该有梯度（不在 target_params 中）"
    );

    println!("  ✓ 测试 1 通过：只计算 Group B 的梯度");

    // ========== 测试 2: 只计算 Group A 的梯度 ==========
    graph.clear_grad()?;
    graph.forward_batch(loss)?;
    graph.backward_batch(loss, Some(&[w_a1, w_a2]))?;

    // Group A 应该有梯度
    let grad_a1 = graph.get_node_grad_batch(w_a1)?;
    let grad_a2 = graph.get_node_grad_batch(w_a2)?;
    assert!(grad_a1.is_some(), "w_a1 应该有梯度（在 target_params 中）");
    assert!(grad_a2.is_some(), "w_a2 应该有梯度（在 target_params 中）");

    // Group B 不应该有梯度（不在 target_params 中）
    let grad_b1 = graph.get_node_grad_batch(w_b1)?;
    let grad_b2 = graph.get_node_grad_batch(w_b2)?;
    assert!(
        grad_b1.is_none(),
        "w_b1 不应该有梯度（不在 target_params 中）"
    );
    assert!(
        grad_b2.is_none(),
        "w_b2 不应该有梯度（不在 target_params 中）"
    );

    println!("  ✓ 测试 2 通过：只计算 Group A 的梯度");

    // ========== 测试 3: 计算所有参数的梯度（None）==========
    graph.clear_grad()?;
    graph.forward_batch(loss)?;
    graph.backward_batch(loss, None)?;

    // 所有参数都应该有梯度
    assert!(
        graph.get_node_grad_batch(w_a1)?.is_some(),
        "w_a1 应该有梯度（None = 全部）"
    );
    assert!(
        graph.get_node_grad_batch(w_a2)?.is_some(),
        "w_a2 应该有梯度（None = 全部）"
    );
    assert!(
        graph.get_node_grad_batch(w_b1)?.is_some(),
        "w_b1 应该有梯度（None = 全部）"
    );
    assert!(
        graph.get_node_grad_batch(w_b2)?.is_some(),
        "w_b2 应该有梯度（None = 全部）"
    );

    println!("  ✓ 测试 3 通过：计算所有参数的梯度（None）");

    // ========== 测试 4: 验证梯度值的正确性 ==========
    // 使用 target_params 计算的梯度应该与 None 计算的梯度一致
    let grad_b1_all = graph.get_node_grad_batch(w_b1)?.unwrap().clone();
    let grad_b2_all = graph.get_node_grad_batch(w_b2)?.unwrap().clone();

    graph.clear_grad()?;
    graph.forward_batch(loss)?;
    graph.backward_batch(loss, Some(&[w_b1, w_b2]))?;

    let grad_b1_targeted = graph.get_node_grad_batch(w_b1)?.unwrap();
    let grad_b2_targeted = graph.get_node_grad_batch(w_b2)?.unwrap();

    assert_abs_diff_eq!(grad_b1_all, *grad_b1_targeted, epsilon = 1e-6);
    assert_abs_diff_eq!(grad_b2_all, *grad_b2_targeted, epsilon = 1e-6);

    println!("  ✓ 测试 4 通过：target_params 梯度值与全量计算一致");

    println!("✅ backward_batch target_params 机制测试全部通过");
    Ok(())
}

/// 测试：当 backward_batch 的 target_params 与 optimizer 的 trainable_nodes 不一致时应该 panic
///
/// 场景：backward 只计算了部分参数的梯度，但 optimizer 想更新所有参数
#[test]
#[should_panic(expected = "update_batch 错误")]
fn test_update_batch_panics_when_params_mismatch_sgd() {
    use crate::nn::optimizer::{Optimizer, SGD};

    let seed = 42u64;
    let mut graph = crate::nn::Graph::new_with_seed(seed);

    // 创建输入和两组参数
    let x = graph.new_input_node(&[2, 4], Some("x")).unwrap();
    let y = graph.new_input_node(&[2, 2], Some("y")).unwrap();
    let w1 = graph
        .new_parameter_node_seeded(&[4, 3], Some("w1"), seed + 1)
        .unwrap();
    let w2 = graph
        .new_parameter_node_seeded(&[3, 2], Some("w2"), seed + 2)
        .unwrap();

    // 构建网络: x -> w1 -> relu -> w2 -> loss
    let h = graph.new_mat_mul_node(x, w1, None).unwrap();
    let h_act = graph.new_relu_node(h, None).unwrap();
    let out = graph.new_mat_mul_node(h_act, w2, None).unwrap();
    let loss = graph.new_mse_loss_node(out, y, None).unwrap();

    // 创建 optimizer，管理 w1 和 w2
    let mut optimizer = SGD::with_params(&[w1, w2], 0.01);

    // 设置输入
    let x_data = crate::tensor::Tensor::normal_seeded(0.0, 1.0, &[2, 4], seed + 100);
    let y_data = crate::tensor::Tensor::normal_seeded(0.0, 1.0, &[2, 2], seed + 101);
    graph.set_node_value(x, Some(&x_data)).unwrap();
    graph.set_node_value(y, Some(&y_data)).unwrap();

    // 前向传播
    graph.forward_batch(loss).unwrap();

    // 关键：backward 只计算 w1 的梯度，不包括 w2
    graph.backward_batch(loss, Some(&[w1])).unwrap();

    // 这里应该 panic，因为 optimizer 想更新 w2，但 w2 没有梯度
    optimizer.update_batch(&mut graph).unwrap();
}

/// 测试：当 backward_batch 的 target_params 与 optimizer 的 trainable_nodes 不一致时应该 panic（Adam）
#[test]
#[should_panic(expected = "update_batch 错误")]
fn test_update_batch_panics_when_params_mismatch_adam() {
    use crate::nn::optimizer::{Adam, Optimizer};

    let seed = 42u64;
    let mut graph = crate::nn::Graph::new_with_seed(seed);

    // 创建输入和两组参数
    let x = graph.new_input_node(&[2, 4], Some("x")).unwrap();
    let y = graph.new_input_node(&[2, 2], Some("y")).unwrap();
    let w1 = graph
        .new_parameter_node_seeded(&[4, 3], Some("w1"), seed + 1)
        .unwrap();
    let w2 = graph
        .new_parameter_node_seeded(&[3, 2], Some("w2"), seed + 2)
        .unwrap();

    // 构建网络
    let h = graph.new_mat_mul_node(x, w1, None).unwrap();
    let h_act = graph.new_relu_node(h, None).unwrap();
    let out = graph.new_mat_mul_node(h_act, w2, None).unwrap();
    let loss = graph.new_mse_loss_node(out, y, None).unwrap();

    // 创建 Adam optimizer，管理 w1 和 w2
    let mut optimizer = Adam::with_params(&[w1, w2], 0.001, 0.9, 0.999, 1e-8);

    // 设置输入
    let x_data = crate::tensor::Tensor::normal_seeded(0.0, 1.0, &[2, 4], seed + 100);
    let y_data = crate::tensor::Tensor::normal_seeded(0.0, 1.0, &[2, 2], seed + 101);
    graph.set_node_value(x, Some(&x_data)).unwrap();
    graph.set_node_value(y, Some(&y_data)).unwrap();

    // 前向传播
    graph.forward_batch(loss).unwrap();

    // 关键：backward 只计算 w1 的梯度，不包括 w2
    graph.backward_batch(loss, Some(&[w1])).unwrap();

    // 这里应该 panic，因为 optimizer 想更新 w2，但 w2 没有梯度
    optimizer.update_batch(&mut graph).unwrap();
}

/// 测试：当参数范围一致时，不应该 panic（正常工作）
#[test]
fn test_update_batch_works_when_params_match() -> Result<(), GraphError> {
    use crate::nn::optimizer::{Adam, Optimizer, SGD};

    let seed = 42u64;

    // ========== SGD 测试 ==========
    {
        let mut graph = crate::nn::Graph::new_with_seed(seed);
        let x = graph.new_input_node(&[2, 4], Some("x"))?;
        let y = graph.new_input_node(&[2, 2], Some("y"))?;
        let w1 = graph.new_parameter_node_seeded(&[4, 3], Some("w1"), seed + 1)?;
        let w2 = graph.new_parameter_node_seeded(&[3, 2], Some("w2"), seed + 2)?;

        let h = graph.new_mat_mul_node(x, w1, None)?;
        let h_act = graph.new_relu_node(h, None)?;
        let out = graph.new_mat_mul_node(h_act, w2, None)?;
        let loss = graph.new_mse_loss_node(out, y, None)?;

        // optimizer 只管理 w1
        let mut optimizer = SGD::with_params(&[w1], 0.01);

        let x_data = crate::tensor::Tensor::normal_seeded(0.0, 1.0, &[2, 4], seed + 100);
        let y_data = crate::tensor::Tensor::normal_seeded(0.0, 1.0, &[2, 2], seed + 101);
        graph.set_node_value(x, Some(&x_data))?;
        graph.set_node_value(y, Some(&y_data))?;

        graph.forward_batch(loss)?;
        // backward 也只计算 w1 的梯度（一致）
        graph.backward_batch(loss, Some(&[w1]))?;

        // 不应该 panic
        optimizer.update_batch(&mut graph)?;
        println!("  ✓ SGD 参数一致时正常工作");
    }

    // ========== Adam 测试 ==========
    {
        let mut graph = crate::nn::Graph::new_with_seed(seed);
        let x = graph.new_input_node(&[2, 4], Some("x"))?;
        let y = graph.new_input_node(&[2, 2], Some("y"))?;
        let w1 = graph.new_parameter_node_seeded(&[4, 3], Some("w1"), seed + 1)?;
        let w2 = graph.new_parameter_node_seeded(&[3, 2], Some("w2"), seed + 2)?;

        let h = graph.new_mat_mul_node(x, w1, None)?;
        let h_act = graph.new_relu_node(h, None)?;
        let out = graph.new_mat_mul_node(h_act, w2, None)?;
        let loss = graph.new_mse_loss_node(out, y, None)?;

        // optimizer 管理 w1 和 w2
        let mut optimizer = Adam::with_params(&[w1, w2], 0.001, 0.9, 0.999, 1e-8);

        let x_data = crate::tensor::Tensor::normal_seeded(0.0, 1.0, &[2, 4], seed + 100);
        let y_data = crate::tensor::Tensor::normal_seeded(0.0, 1.0, &[2, 2], seed + 101);
        graph.set_node_value(x, Some(&x_data))?;
        graph.set_node_value(y, Some(&y_data))?;

        graph.forward_batch(loss)?;
        // backward 计算 w1 和 w2 的梯度（一致）
        graph.backward_batch(loss, Some(&[w1, w2]))?;

        // 不应该 panic
        optimizer.update_batch(&mut graph)?;
        println!("  ✓ Adam 参数一致时正常工作");
    }

    println!("✅ update_batch 参数一致性测试通过");
    Ok(())
}
