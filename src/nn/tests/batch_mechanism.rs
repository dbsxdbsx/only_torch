/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : Batch 机制单元测试
 *                 验证 batch forward/backward 与累加单样本的结果一致
 *
 *                 设计理念（参考 autodiff_unification_design.md）：
 *                 - 单样本是 batch_size=1 的特例
 *                 - 统一的 forward/backward API 自动处理 batch 维度
 *                 - 这些测试验证底层 batch 实现的**数学正确性**
 */

use approx::assert_abs_diff_eq;

use crate::nn::{GraphError, GraphInner};
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
    let batch_input = Tensor::stack(&sample_refs, false);
    assert_eq!(batch_input.shape(), &[batch_size, input_dim]);

    // ========== Batch 模式 ==========
    let mut graph_batch = GraphInner::new_with_seed(seed);
    let x_b = graph_batch.new_basic_input_node(&[batch_size, input_dim], Some("x"))?;
    let w_b =
        graph_batch.new_parameter_node_seeded(&[input_dim, output_dim], Some("w"), seed + 100)?;
    let z_b = graph_batch.new_mat_mul_node(x_b, w_b, Some("z"))?;
    let out_b = graph_batch.new_sigmoid_node(z_b, Some("out"))?;

    graph_batch.set_node_value(x_b, Some(&batch_input))?;
    graph_batch.forward(out_b)?; // 统一 API
    let batch_output = graph_batch.get_node_value(out_b)?.unwrap().clone();

    // ========== 单样本模式 ==========
    let mut single_outputs = Vec::new();
    for sample in samples.iter() {
        let mut graph_single = GraphInner::new_with_seed(seed);
        let x_s = graph_single.new_basic_input_node(&[1, input_dim], Some("x"))?;
        let w_s = graph_single.new_parameter_node_seeded(
            &[input_dim, output_dim],
            Some("w"),
            seed + 100,
        )?;
        let z_s = graph_single.new_mat_mul_node(x_s, w_s, Some("z"))?;
        let out_s = graph_single.new_sigmoid_node(z_s, Some("out"))?;

        graph_single.set_node_value(x_s, Some(sample))?;
        graph_single.forward(out_s)?;
        single_outputs.push(graph_single.get_node_value(out_s)?.unwrap().clone());
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
    let batch_input = Tensor::stack(&sample_refs, false);
    let batch_labels = Tensor::stack(&label_refs, false);

    // ========== Batch 模式（统一 API）==========
    let mut graph_batch = GraphInner::new_with_seed(seed);
    let x_b = graph_batch.new_basic_input_node(&[batch_size, input_dim], Some("x"))?;
    let y_b = graph_batch.new_basic_input_node(&[batch_size, output_dim], Some("y"))?;
    let w1_b =
        graph_batch.new_parameter_node_seeded(&[input_dim, hidden_dim], Some("w1"), seed + 100)?;
    let w2_b =
        graph_batch.new_parameter_node_seeded(&[hidden_dim, output_dim], Some("w2"), seed + 101)?;

    let z1_b = graph_batch.new_mat_mul_node(x_b, w1_b, None)?;
    let a1_b = graph_batch.new_sigmoid_node(z1_b, None)?;
    let z2_b = graph_batch.new_mat_mul_node(a1_b, w2_b, None)?;
    // 使用 MSE loss（输出标量 [1,1]，内部自动对 batch 求平均）
    let loss_b = graph_batch.new_mse_loss_node(z2_b, y_b, Some("loss"))?;

    // Batch forward + backward（统一 API）
    graph_batch.set_node_value(x_b, Some(&batch_input))?;
    graph_batch.set_node_value(y_b, Some(&batch_labels))?;
    graph_batch.forward(loss_b)?;
    graph_batch.backward(loss_b)?;

    let batch_grad_w1 = graph_batch.get_node_grad(w1_b)?.unwrap().clone();
    let batch_grad_w2 = graph_batch.get_node_grad(w2_b)?.unwrap().clone();

    // ========== 单样本累加模式 ==========
    let mut accumulated_grad_w1 = Tensor::zeros(&[input_dim, hidden_dim]);
    let mut accumulated_grad_w2 = Tensor::zeros(&[hidden_dim, output_dim]);

    for i in 0..batch_size {
        let mut graph_single = GraphInner::new_with_seed(seed);
        let x_s = graph_single.new_basic_input_node(&[1, input_dim], Some("x"))?;
        let y_s = graph_single.new_basic_input_node(&[1, output_dim], Some("y"))?;
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
        let loss_s = graph_single.new_mse_loss_node(z2_s, y_s, Some("loss"))?;

        graph_single.set_node_value(x_s, Some(&samples[i]))?;
        graph_single.set_node_value(y_s, Some(&labels[i]))?;
        graph_single.forward(loss_s)?;
        graph_single.backward(loss_s)?;

        // 累加梯度
        if let Some(grad) = graph_single.get_node_grad(w1_s)? {
            accumulated_grad_w1 = accumulated_grad_w1 + grad;
        }
        if let Some(grad) = graph_single.get_node_grad(w2_s)? {
            accumulated_grad_w2 = accumulated_grad_w2 + grad;
        }
    }

    // 求平均（与 batch 模式的隐式平均对齐）
    let avg_grad_w1 = accumulated_grad_w1 / (batch_size as f32);
    let avg_grad_w2 = accumulated_grad_w2 / (batch_size as f32);

    // ========== 验证 ==========
    let tolerance = 1e-4;

    assert_abs_diff_eq!(batch_grad_w1, avg_grad_w1, epsilon = tolerance);
    assert_abs_diff_eq!(batch_grad_w2, avg_grad_w2, epsilon = tolerance);

    println!("✅ batch 梯度与累加单样本梯度一致性测试通过");
    Ok(())
}

/// 测试 batch 模式下的参数更新与单样本累加更新的一致性
///
/// 验证：使用 batch 计算的梯度更新参数 == 使用累加单样本梯度更新参数
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
    let batch_input = Tensor::stack(&sample_refs, false);
    let batch_labels = Tensor::stack(&label_refs, false);

    // ========== Batch 模式 ==========
    let mut graph_batch = GraphInner::new_with_seed(seed);
    let x_b = graph_batch.new_basic_input_node(&[batch_size, input_dim], Some("x"))?;
    let y_b = graph_batch.new_basic_input_node(&[batch_size, output_dim], Some("y"))?;
    let w_b =
        graph_batch.new_parameter_node_seeded(&[input_dim, output_dim], Some("w"), seed + 100)?;
    let z_b = graph_batch.new_mat_mul_node(x_b, w_b, None)?;
    let loss_b = graph_batch.new_mse_loss_node(z_b, y_b, Some("loss"))?;

    let w_init = graph_batch.get_node_value(w_b)?.unwrap().clone();

    // Batch forward + backward
    graph_batch.set_node_value(x_b, Some(&batch_input))?;
    graph_batch.set_node_value(y_b, Some(&batch_labels))?;
    graph_batch.forward(loss_b)?;
    graph_batch.backward(loss_b)?;

    // 手动应用 SGD 更新：θ = θ - lr * grad
    let batch_grad = graph_batch.get_node_grad(w_b)?.unwrap().clone();
    let w_after_batch = &w_init - learning_rate * &batch_grad;

    // ========== 单样本累加模式 ==========
    let mut accumulated_grad = Tensor::zeros(&[input_dim, output_dim]);

    for i in 0..batch_size {
        let mut graph_single = GraphInner::new_with_seed(seed);
        let x_s = graph_single.new_basic_input_node(&[1, input_dim], Some("x"))?;
        let y_s = graph_single.new_basic_input_node(&[1, output_dim], Some("y"))?;
        let w_s = graph_single.new_parameter_node_seeded(
            &[input_dim, output_dim],
            Some("w"),
            seed + 100,
        )?;
        let z_s = graph_single.new_mat_mul_node(x_s, w_s, None)?;
        let loss_s = graph_single.new_mse_loss_node(z_s, y_s, Some("loss"))?;

        graph_single.set_node_value(x_s, Some(&samples[i]))?;
        graph_single.set_node_value(y_s, Some(&labels[i]))?;
        graph_single.forward(loss_s)?;
        graph_single.backward(loss_s)?;

        // 累加梯度
        if let Some(grad) = graph_single.get_node_grad(w_s)? {
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
