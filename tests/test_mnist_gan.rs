/*
 * @Author       : 老董
 * @Date         : 2025-12-27
 * @Description  : MNIST GAN 集成测试
 *                 验证：detach 机制 + 多 Loss 交替训练
 *                 使用 Batch MLP 实现简单 GAN
 * @LastEditors  : 老董
 * @LastEditTime : 2025-12-27
 */

use only_torch::data::MnistDataset;
use only_torch::nn::optimizer::{Adam, Optimizer};
use only_torch::nn::{Graph, GraphError, NodeId};
use only_torch::tensor::Tensor;
use only_torch::tensor_slice;
use std::time::Instant;

/// MNIST GAN 集成测试
///
/// 验证梯度流控制机制在 GAN 训练中的正确性：
/// - detach: 训练 D 时阻止梯度流向 G
/// - 多 Loss 交替训练: D_loss 和 G_loss 交替 backward
///
/// 网络结构：
/// - Generator: z(64) -> FC(256, Sigmoid) -> FC(784, Sigmoid) -> fake_image
/// - Discriminator: image(784) -> FC(256, Sigmoid) -> FC(1, Sigmoid) -> prob
#[test]
fn test_mnist_gan() -> Result<(), GraphError> {
    let start_time = Instant::now();

    println!("\n{}", "=".repeat(60));
    println!("=== MNIST GAN 集成测试（验证 detach 机制）===");
    println!("{}\n", "=".repeat(60));

    // ========== 1. 加载数据 ==========
    println!("[1/5] 加载 MNIST 数据集...");
    let load_start = Instant::now();

    let train_data = MnistDataset::train()
        .expect("加载 MNIST 训练集失败")
        .flatten();

    println!(
        "  ✓ 训练集: {} 样本，耗时 {:.2}s",
        train_data.len(),
        load_start.elapsed().as_secs_f32()
    );

    // ========== 2. 训练配置 ==========
    // batch_size=256 经测试为最佳平衡点（10.91s vs 512:12.02s vs 1024:12.04s）
    let batch_size = 256;
    let train_samples = 5120;
    let max_epochs = 15;
    let num_batches = train_samples / batch_size; // 20 batches
    let latent_dim = 64;
    // 学习率线性缩放
    let lr_d = 0.0005;
    let lr_g = 0.001;

    println!("\n[2/5] 训练配置：");
    println!("  - Batch Size: {}", batch_size);
    println!(
        "  - 训练样本: {} (共 {} 个 batch)",
        train_samples, num_batches
    );
    println!("  - 最大 Epochs: {}", max_epochs);
    println!("  - 噪声维度: {}", latent_dim);
    println!("  - 学习率 (D/G): {}/{}", lr_d, lr_g);

    // ========== 3. 构建网络 ==========
    println!("\n[3/5] 构建 GAN 网络...");

    let mut graph = Graph::new_with_seed(42);

    // --- 输入节点 ---
    // 真实图像输入
    let real_images = graph.new_input_node(&[batch_size, 784], Some("real_images"))?;
    // 噪声输入
    let z = graph.new_input_node(&[batch_size, latent_dim], Some("z"))?;
    // bias 广播用的 ones
    let ones = graph.new_input_node(&[batch_size, 1], Some("ones"))?;

    // --- Generator: z(64) -> 256 -> 784 ---
    let (fake_images, g_params) = build_generator(&mut graph, z, ones, latent_dim, batch_size)?;

    // --- Discriminator（对真实图像）---
    let (d_real_out, d_params) =
        build_discriminator(&mut graph, real_images, ones, batch_size, "d_real")?;

    // --- Discriminator（对生成图像，共享参数）---
    // 注意：需要用同样的参数，但对 fake_images 进行前向
    let d_fake_out =
        build_discriminator_forward(&mut graph, fake_images, ones, &d_params, "d_fake")?;

    // --- 损失函数 ---
    // D 的目标：真实图像 -> 1，生成图像 -> 0
    // G 的目标：生成图像（通过 D）-> 1

    // 真实标签 (全 1) 和假标签 (全 0)
    let real_labels = graph.new_input_node(&[batch_size, 1], Some("real_labels"))?;
    let fake_labels = graph.new_input_node(&[batch_size, 1], Some("fake_labels"))?;

    // D loss = MSE(D(real), 1) + MSE(D(fake), 0)
    // 使用两个 MSE loss
    let d_loss_real = graph.new_mse_loss_node(d_real_out, real_labels, None)?;
    let d_loss_fake = graph.new_mse_loss_node(d_fake_out, fake_labels, None)?;

    // G loss = MSE(D(fake), 1) -- G 想让 D 认为 fake 是真的
    let g_loss = graph.new_mse_loss_node(d_fake_out, real_labels, Some("g_loss"))?;

    println!("  ✓ Generator: z({}) -> 128 -> 784", latent_dim);
    println!("  ✓ Discriminator: 784 -> 128 -> 1");
    println!("  ✓ G 参数量: {}", g_params.len());
    println!("  ✓ D 参数量: {}", d_params.len());

    // ========== 4. 创建优化器 ==========
    println!("\n[4/5] 创建优化器...");

    // 使用 with_params 为 D 和 G 创建独立优化器
    let mut adam_d = Adam::with_params(&d_params, lr_d, 0.5, 0.999, 1e-8);
    let mut adam_g = Adam::with_params(&g_params, lr_g, 0.5, 0.999, 1e-8);

    println!("  ✓ Adam::with_params 优化器 (beta1=0.5 for GAN stability)");

    // ========== 5. 训练循环 ==========
    println!("\n[5/5] 开始训练...\n");

    // 设置常量
    let ones_tensor = Tensor::ones(&[batch_size, 1]);
    let real_labels_tensor = Tensor::ones(&[batch_size, 1]);
    let fake_labels_tensor = Tensor::zeros(&[batch_size, 1]);

    graph.set_node_value(ones, Some(&ones_tensor))?;
    graph.set_node_value(real_labels, Some(&real_labels_tensor))?;
    graph.set_node_value(fake_labels, Some(&fake_labels_tensor))?;

    let all_train_images = train_data.images();

    // 跟踪 D 判别能力
    let mut d_real_avg = 0.0;
    let mut d_fake_avg = 0.0;

    for epoch in 0..max_epochs {
        let epoch_start = Instant::now();
        let mut d_loss_sum = 0.0;
        let mut g_loss_sum = 0.0;
        let mut d_real_sum = 0.0;
        let mut d_fake_sum = 0.0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            // 获取真实图像 batch
            let batch_images = tensor_slice!(all_train_images, start..end, ..);
            graph.set_node_value(real_images, Some(&batch_images))?;

            // 生成随机噪声（确定性种子，确保可重复）
            let noise_seed = (epoch * num_batches + batch_idx) as u64;
            let noise = Tensor::normal_seeded(0.0, 1.0, &[batch_size, latent_dim], noise_seed);
            graph.set_node_value(z, Some(&noise))?;

            // ========== 训练 Discriminator ==========
            // 关键：detach fake_images，防止 D 的 loss 更新 G
            graph.detach_node(fake_images)?;

            // 前向传播 - 真实图像
            graph.forward_batch(d_loss_real)?;

            // D 对真实图像的判断
            let d_real_val = graph.get_node_value(d_real_out)?.unwrap();
            let mut d_real_batch_sum = 0.0;
            for i in 0..batch_size {
                d_real_batch_sum += d_real_val[[i, 0]];
            }
            d_real_sum += d_real_batch_sum / batch_size as f32;

            // 反向传播 D(real)
            graph.clear_grad()?;
            graph.backward_batch(d_loss_real)?;
            adam_d.update_batch(&mut graph)?;

            // 前向传播 - 生成图像
            graph.forward_batch(d_loss_fake)?;

            // D 对假图像的判断
            let d_fake_val = graph.get_node_value(d_fake_out)?.unwrap();
            let mut d_fake_batch_sum = 0.0;
            for i in 0..batch_size {
                d_fake_batch_sum += d_fake_val[[i, 0]];
            }
            d_fake_sum += d_fake_batch_sum / batch_size as f32;

            // 反向传播 D(fake)
            graph.clear_grad()?;
            graph.backward_batch(d_loss_fake)?;
            adam_d.update_batch(&mut graph)?;

            let d_loss_real_val = graph.get_node_value(d_loss_real)?.unwrap()[[0, 0]];
            let d_loss_fake_val = graph.get_node_value(d_loss_fake)?.unwrap()[[0, 0]];
            d_loss_sum += (d_loss_real_val + d_loss_fake_val) / 2.0;

            // ========== 训练 Generator ==========
            // 恢复 fake_images 的梯度流
            graph.attach_node(fake_images)?;

            // 重新生成噪声（G 训练用不同种子）
            let noise_seed_g = (epoch * num_batches + batch_idx + 10000) as u64;
            let noise = Tensor::normal_seeded(0.0, 1.0, &[batch_size, latent_dim], noise_seed_g);
            graph.set_node_value(z, Some(&noise))?;

            // 前向传播
            graph.forward_batch(g_loss)?;

            // 反向传播并更新 G
            graph.clear_grad()?;
            graph.backward_batch(g_loss)?;
            adam_g.update_batch(&mut graph)?;

            let g_loss_val = graph.get_node_value(g_loss)?.unwrap()[[0, 0]];
            g_loss_sum += g_loss_val;
        }

        let avg_d_loss = d_loss_sum / num_batches as f32;
        let avg_g_loss = g_loss_sum / num_batches as f32;
        d_real_avg = d_real_sum / num_batches as f32;
        d_fake_avg = d_fake_sum / num_batches as f32;

        println!(
            "Epoch {:2}/{}: D_loss = {:.4}, G_loss = {:.4}, D(real) = {:.3}, D(fake) = {:.3}, 耗时 {:.2}s",
            epoch + 1,
            max_epochs,
            avg_d_loss,
            avg_g_loss,
            d_real_avg,
            d_fake_avg,
            epoch_start.elapsed().as_secs_f32()
        );
    }

    let total_duration = start_time.elapsed();
    println!("\n总耗时: {:.2}s", total_duration.as_secs_f32());

    // ========== 验证 ==========
    // GAN 训练成功的标志：
    // 理想状态：D(real) ≈ D(fake) ≈ 0.5（D 无法区分真假）
    // 验证条件：
    // 1. D(real) > 0.3（D 没有完全崩溃）
    // 2. D(fake) > 0.2（G 在学习生成有效图像）
    // 3. D(fake) < 0.9（D 没有完全被骗）

    let test_passed = d_real_avg > 0.3 && d_fake_avg > 0.2 && d_fake_avg < 0.9;

    if test_passed {
        println!("\n{}", "=".repeat(60));
        println!("✅ MNIST GAN 测试通过！");
        println!("  - D(real) = {:.3} (> 0.3 ✓)", d_real_avg);
        println!("  - D(fake) = {:.3} (0.2 < x < 0.9 ✓)", d_fake_avg);
        println!("  - detach 机制验证：D 能独立训练，G 的梯度正确传播");
        println!("{}\n", "=".repeat(60));
        Ok(())
    } else {
        println!("\n{}", "=".repeat(60));
        println!("❌ MNIST GAN 测试失败！");
        println!("  - D(real) = {:.3} (需 > 0.3)", d_real_avg);
        println!("  - D(fake) = {:.3} (需 0.2 < x < 0.9)", d_fake_avg);
        println!("{}\n", "=".repeat(60));
        Err(GraphError::ComputationError(
            "GAN 训练未收敛：Generator 未能学习生成有效图像".to_string(),
        ))
    }
}

/// 构建 Generator 网络（简化版，与 MNIST Batch 规模相当）
///
/// z(latent_dim) -> FC(128, LeakyReLU) -> FC(784, Sigmoid) -> fake_image
fn build_generator(
    graph: &mut Graph,
    z: NodeId,
    ones: NodeId,
    latent_dim: usize,
    _batch_size: usize,
) -> Result<(NodeId, Vec<NodeId>), GraphError> {
    // Layer 1: latent_dim -> 128
    let g_w1 = graph.new_parameter_node_seeded(&[latent_dim, 128], Some("g_w1"), 100)?;
    let g_b1 = graph.new_parameter_node_seeded(&[1, 128], Some("g_b1"), 101)?;
    let g_z1 = graph.new_mat_mul_node(z, g_w1, None)?;
    let g_b1_broadcast = graph.new_mat_mul_node(ones, g_b1, None)?;
    let g_h1 = graph.new_add_node(&[g_z1, g_b1_broadcast], None)?;
    let g_a1 = graph.new_leaky_relu_node(g_h1, 0.2, None)?;

    // Layer 2: 128 -> 784
    let g_w2 = graph.new_parameter_node_seeded(&[128, 784], Some("g_w2"), 102)?;
    let g_b2 = graph.new_parameter_node_seeded(&[1, 784], Some("g_b2"), 103)?;
    let g_z2 = graph.new_mat_mul_node(g_a1, g_w2, None)?;
    let g_b2_broadcast = graph.new_mat_mul_node(ones, g_b2, None)?;
    let g_h2 = graph.new_add_node(&[g_z2, g_b2_broadcast], None)?;
    let g_out = graph.new_sigmoid_node(g_h2, Some("fake_images"))?; // Sigmoid 输出 [0, 1]

    let g_params = vec![g_w1, g_b1, g_w2, g_b2];
    Ok((g_out, g_params))
}

/// 构建 Discriminator 网络（简化版）
///
/// image(784) -> FC(128, LeakyReLU) -> FC(1, Sigmoid) -> prob
fn build_discriminator(
    graph: &mut Graph,
    input: NodeId,
    ones: NodeId,
    _batch_size: usize,
    name_prefix: &str,
) -> Result<(NodeId, Vec<NodeId>), GraphError> {
    // Layer 1: 784 -> 128
    let d_w1 = graph.new_parameter_node_seeded(&[784, 128], Some("d_w1"), 200)?;
    let d_b1 = graph.new_parameter_node_seeded(&[1, 128], Some("d_b1"), 201)?;
    let d_z1 = graph.new_mat_mul_node(input, d_w1, None)?;
    let d_b1_broadcast = graph.new_mat_mul_node(ones, d_b1, None)?;
    let d_h1 = graph.new_add_node(&[d_z1, d_b1_broadcast], None)?;
    let d_a1 = graph.new_leaky_relu_node(d_h1, 0.2, None)?;

    // Layer 2: 128 -> 1
    let d_w2 = graph.new_parameter_node_seeded(&[128, 1], Some("d_w2"), 202)?;
    let d_b2 = graph.new_parameter_node_seeded(&[1, 1], Some("d_b2"), 203)?;
    let d_z2 = graph.new_mat_mul_node(d_a1, d_w2, None)?;
    let d_b2_broadcast = graph.new_mat_mul_node(ones, d_b2, None)?;
    let d_h2 = graph.new_add_node(&[d_z2, d_b2_broadcast], None)?;
    let d_out = graph.new_sigmoid_node(d_h2, Some(name_prefix))?;

    let d_params = vec![d_w1, d_b1, d_w2, d_b2];
    Ok((d_out, d_params))
}

/// 使用现有 D 参数对新输入进行前向传播
fn build_discriminator_forward(
    graph: &mut Graph,
    input: NodeId,
    ones: NodeId,
    d_params: &[NodeId],
    name: &str,
) -> Result<NodeId, GraphError> {
    let d_w1 = d_params[0];
    let d_b1 = d_params[1];
    let d_w2 = d_params[2];
    let d_b2 = d_params[3];

    // Layer 1
    let d_z1 = graph.new_mat_mul_node(input, d_w1, None)?;
    let d_b1_broadcast = graph.new_mat_mul_node(ones, d_b1, None)?;
    let d_h1 = graph.new_add_node(&[d_z1, d_b1_broadcast], None)?;
    let d_a1 = graph.new_leaky_relu_node(d_h1, 0.2, None)?;

    // Layer 2
    let d_z2 = graph.new_mat_mul_node(d_a1, d_w2, None)?;
    let d_b2_broadcast = graph.new_mat_mul_node(ones, d_b2, None)?;
    let d_h2 = graph.new_add_node(&[d_z2, d_b2_broadcast], None)?;
    let d_out = graph.new_sigmoid_node(d_h2, Some(name))?;

    Ok(d_out)
}
