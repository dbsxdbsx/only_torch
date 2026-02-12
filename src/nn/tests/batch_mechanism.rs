/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : 批量处理数学一致性测试
 *
 *                 验证批量输入与逐样本处理的结果一致：
 *                 - forward([batch, ...]) 的每行 == 逐样本 forward 的结果
 *                 - backward(batch_loss) 的梯度 == 逐样本梯度的平均
 *
 *                 设计理念：单样本是 batch_size=1 的特例，使用统一的 API。
 *
 *                 使用底层 Graph + inner_rc() API（graph_dynamic 模式）。
 */

use approx::assert_abs_diff_eq;

use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

/// 测试 batch forward 与单样本 forward 的一致性
///
/// 验证：forward([batch, ...]) 的每一行 == 单独 forward 每个样本的结果
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
    let batch_input = Tensor::stack(&sample_refs, 0, false);
    assert_eq!(batch_input.shape(), &[batch_size, input_dim]);

    // ========== 批量处理 ==========
    let graph_batch = Graph::new_with_seed(seed);
    let inner_batch = graph_batch.inner_rc();
    let mut gi_batch = inner_batch.borrow_mut();

    let x_b = gi_batch.create_basic_input_node(&[batch_size, input_dim], Some("x"))?;
    let w_b = gi_batch
        .create_parameter_node_seeded(&[input_dim, output_dim], Some("w"), seed + 100)?;
    let z_b = gi_batch.create_mat_mul_node(vec![x_b.clone(), w_b.clone()], Some("z"))?;
    let out_b = gi_batch.create_sigmoid_node(z_b.clone(), Some("out"))?;

    x_b.set_value(Some(&batch_input))?;
    gi_batch.forward_via_node_inner(&out_b)?;
    let batch_output = out_b.value().unwrap().clone();

    // ========== 逐样本处理 ==========
    let mut single_outputs = Vec::new();
    for sample in samples.iter() {
        let graph_single = Graph::new_with_seed(seed);
        let inner_single = graph_single.inner_rc();
        let mut gi_single = inner_single.borrow_mut();

        let x_s = gi_single.create_basic_input_node(&[1, input_dim], Some("x"))?;
        let w_s = gi_single
            .create_parameter_node_seeded(&[input_dim, output_dim], Some("w"), seed + 100)?;
        let z_s = gi_single.create_mat_mul_node(vec![x_s.clone(), w_s.clone()], Some("z"))?;
        let out_s = gi_single.create_sigmoid_node(z_s.clone(), Some("out"))?;

        x_s.set_value(Some(sample))?;
        gi_single.forward_via_node_inner(&out_s)?;
        single_outputs.push(out_s.value().unwrap().clone());
    }

    // ========== 验证 ==========
    // batch 输出的每一行应该等于对应的单样本输出
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
///
/// 验证：backward(batch_loss) 的梯度 == Σ backward(single_loss) / batch_size
///
/// 这是 VJP 模式下 batch 处理的**核心数学等价性**验证
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
    let batch_input = Tensor::stack(&sample_refs, 0, false);
    let batch_labels = Tensor::stack(&label_refs, 0, false);

    // ========== 批量处理（统一 API）==========
    let graph_batch = Graph::new_with_seed(seed);
    let inner_batch = graph_batch.inner_rc();
    let mut gi_batch = inner_batch.borrow_mut();

    let x_b = gi_batch.create_basic_input_node(&[batch_size, input_dim], Some("x"))?;
    let y_b = gi_batch.create_basic_input_node(&[batch_size, output_dim], Some("y"))?;
    let w1_b = gi_batch
        .create_parameter_node_seeded(&[input_dim, hidden_dim], Some("w1"), seed + 100)?;
    let w2_b = gi_batch
        .create_parameter_node_seeded(&[hidden_dim, output_dim], Some("w2"), seed + 101)?;

    let z1_b = gi_batch.create_mat_mul_node(vec![x_b.clone(), w1_b.clone()], None)?;
    let a1_b = gi_batch.create_sigmoid_node(z1_b.clone(), None)?;
    let z2_b = gi_batch.create_mat_mul_node(vec![a1_b.clone(), w2_b.clone()], None)?;
    // 使用 MSE loss（输出标量 [1,1]，内部自动对 batch 求平均）
    let loss_b = gi_batch.create_mse_mean_node(z2_b.clone(), y_b.clone(), Some("loss"))?;

    // Batch forward + backward（统一 API）
    x_b.set_value(Some(&batch_input))?;
    y_b.set_value(Some(&batch_labels))?;
    gi_batch.forward_via_node_inner(&loss_b)?;
    gi_batch.backward_via_node_inner(&loss_b)?;

    let batch_grad_w1 = w1_b.grad().unwrap().clone();
    let batch_grad_w2 = w2_b.grad().unwrap().clone();

    // ========== 逐样本累加 ==========
    let mut accumulated_grad_w1 = Tensor::zeros(&[input_dim, hidden_dim]);
    let mut accumulated_grad_w2 = Tensor::zeros(&[hidden_dim, output_dim]);

    for i in 0..batch_size {
        let graph_single = Graph::new_with_seed(seed);
        let inner_single = graph_single.inner_rc();
        let mut gi_single = inner_single.borrow_mut();

        let x_s = gi_single.create_basic_input_node(&[1, input_dim], Some("x"))?;
        let y_s = gi_single.create_basic_input_node(&[1, output_dim], Some("y"))?;
        let w1_s = gi_single
            .create_parameter_node_seeded(&[input_dim, hidden_dim], Some("w1"), seed + 100)?;
        let w2_s = gi_single
            .create_parameter_node_seeded(&[hidden_dim, output_dim], Some("w2"), seed + 101)?;

        let z1_s = gi_single.create_mat_mul_node(vec![x_s.clone(), w1_s.clone()], None)?;
        let a1_s = gi_single.create_sigmoid_node(z1_s.clone(), None)?;
        let z2_s = gi_single.create_mat_mul_node(vec![a1_s.clone(), w2_s.clone()], None)?;
        let loss_s = gi_single.create_mse_mean_node(z2_s.clone(), y_s.clone(), Some("loss"))?;

        x_s.set_value(Some(&samples[i]))?;
        y_s.set_value(Some(&labels[i]))?;
        gi_single.forward_via_node_inner(&loss_s)?;
        gi_single.backward_via_node_inner(&loss_s)?;

        // 累加梯度
        if let Some(grad) = w1_s.grad() {
            accumulated_grad_w1 = accumulated_grad_w1 + grad;
        }
        if let Some(grad) = w2_s.grad() {
            accumulated_grad_w2 = accumulated_grad_w2 + grad;
        }
    }

    // 求平均（与批量处理的隐式平均对齐）
    let avg_grad_w1 = accumulated_grad_w1 / (batch_size as f32);
    let avg_grad_w2 = accumulated_grad_w2 / (batch_size as f32);

    // ========== 验证 ==========
    let tolerance = 1e-4;

    assert_abs_diff_eq!(batch_grad_w1, avg_grad_w1, epsilon = tolerance);
    assert_abs_diff_eq!(batch_grad_w2, avg_grad_w2, epsilon = tolerance);

    println!("✅ batch 梯度与累加单样本梯度一致性测试通过");
    Ok(())
}

/// 测试批量处理的参数更新与逐样本累加更新的一致性
///
/// 验证：使用批量计算的梯度更新参数 == 使用逐样本累加梯度更新参数
///
/// 注意：此测试直接使用 SGD 公式 θ = θ - lr * grad，不依赖 optimizer API
#[test]
fn test_batch_parameter_update_equals_accumulated_single() -> Result<(), GraphError> {
    let seed = 42u64;
    let batch_size = 4;
    let input_dim = 3;
    let output_dim = 2;
    let learning_rate = 0.1;

    // 创建测试数据
    let samples = vec![
        Tensor::normal_seeded(0.0, 1.0, &[1, input_dim], seed),
        Tensor::normal_seeded(0.0, 1.0, &[1, input_dim], seed + 1),
        Tensor::normal_seeded(0.0, 1.0, &[1, input_dim], seed + 2),
        Tensor::normal_seeded(0.0, 1.0, &[1, input_dim], seed + 3),
    ];
    let labels = vec![
        Tensor::new(&[1.0, 0.0], &[1, output_dim]),
        Tensor::new(&[0.0, 1.0], &[1, output_dim]),
        Tensor::new(&[1.0, 0.0], &[1, output_dim]),
        Tensor::new(&[0.0, 1.0], &[1, output_dim]),
    ];

    // 合并成 batch
    let sample_refs: Vec<&Tensor> = samples.iter().collect();
    let label_refs: Vec<&Tensor> = labels.iter().collect();
    let batch_input = Tensor::stack(&sample_refs, 0, false);
    let batch_labels = Tensor::stack(&label_refs, 0, false);

    // ========== 批量处理 ==========
    let graph_batch = Graph::new_with_seed(seed);
    let inner_batch = graph_batch.inner_rc();
    let mut gi_batch = inner_batch.borrow_mut();

    let x_b = gi_batch.create_basic_input_node(&[batch_size, input_dim], Some("x"))?;
    let y_b = gi_batch.create_basic_input_node(&[batch_size, output_dim], Some("y"))?;
    let w_b = gi_batch
        .create_parameter_node_seeded(&[input_dim, output_dim], Some("w"), seed + 100)?;
    let z_b = gi_batch.create_mat_mul_node(vec![x_b.clone(), w_b.clone()], None)?;
    let loss_b = gi_batch.create_mse_mean_node(z_b.clone(), y_b.clone(), Some("loss"))?;

    let w_init = w_b.value().unwrap().clone();

    // Batch forward + backward
    x_b.set_value(Some(&batch_input))?;
    y_b.set_value(Some(&batch_labels))?;
    gi_batch.forward_via_node_inner(&loss_b)?;
    gi_batch.backward_via_node_inner(&loss_b)?;

    // 手动应用 SGD 更新：θ = θ - lr * grad
    let batch_grad = w_b.grad().unwrap().clone();
    let w_after_batch = &w_init - learning_rate * &batch_grad;

    // ========== 逐样本累加 ==========
    let mut accumulated_grad = Tensor::zeros(&[input_dim, output_dim]);

    for i in 0..batch_size {
        let graph_single = Graph::new_with_seed(seed);
        let inner_single = graph_single.inner_rc();
        let mut gi_single = inner_single.borrow_mut();

        let x_s = gi_single.create_basic_input_node(&[1, input_dim], Some("x"))?;
        let y_s = gi_single.create_basic_input_node(&[1, output_dim], Some("y"))?;
        let w_s = gi_single
            .create_parameter_node_seeded(&[input_dim, output_dim], Some("w"), seed + 100)?;
        let z_s = gi_single.create_mat_mul_node(vec![x_s.clone(), w_s.clone()], None)?;
        let loss_s = gi_single.create_mse_mean_node(z_s.clone(), y_s.clone(), Some("loss"))?;

        x_s.set_value(Some(&samples[i]))?;
        y_s.set_value(Some(&labels[i]))?;
        gi_single.forward_via_node_inner(&loss_s)?;
        gi_single.backward_via_node_inner(&loss_s)?;

        // 累加梯度
        if let Some(grad) = w_s.grad() {
            accumulated_grad = accumulated_grad + grad;
        }
    }

    // 求平均并应用 SGD 更新
    let avg_grad = accumulated_grad / (batch_size as f32);
    let w_after_single = &w_init - learning_rate * &avg_grad;

    // ========== 验证 ==========
    let tolerance = 1e-4;

    assert_abs_diff_eq!(w_after_batch, w_after_single, epsilon = tolerance);

    println!("✅ batch 参数更新与累加单样本更新一致性测试通过");
    Ok(())
}
