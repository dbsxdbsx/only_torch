/*
 * MNIST GAN 训练示例
 *
 * 使用 PyTorch 风格的 Only-Torch API 训练 GAN。
 *
 * # 运行
 * ```bash
 * cargo run --example mnist_gan
 * # 或
 * just run mnist_gan
 * ```
 *
 * # 关键特性演示
 * 1. `ForwardInput` trait: D 可以接收 Tensor（真实图像）或 Var（生成图像）
 * 2. 函数式 `detach()`: 训练 D 时阻止梯度流向 G
 * 3. `ModelState` 智能缓存: Tensor 输入缓存，Var 输入不缓存
 */

mod model;

use model::{Discriminator, Generator, LATENT_DIM};
use only_torch::data::MnistDataset;
use only_torch::nn::{Adam, Graph, GraphError, Module, MseLoss, Optimizer};
use only_torch::tensor::Tensor;
use only_torch::tensor_slice;
use std::time::Instant;

/// 训练配置（与 archive 版本一致）
const BATCH_SIZE: usize = 256;
const EPOCHS: usize = 15;
const LR_D: f32 = 0.0005; // D 学习率
const LR_G: f32 = 0.001; // G 学习率（通常 G 需要更高学习率）
const TRAIN_SAMPLES: usize = 5120; // 使用部分数据加快演示

fn main() -> Result<(), GraphError> {
    println!("=== MNIST GAN 训练示例 ===\n");

    // 1. 加载 MNIST 数据集
    println!("[1/4] 加载 MNIST 数据集...");
    let load_start = Instant::now();

    let train_data = MnistDataset::train()
        .expect("加载 MNIST 训练集失败")
        .flatten();

    // 提取部分数据用于训练
    // 注：MNIST 图像已经是 [0, 1] 范围，与 sigmoid 输出匹配
    let all_images = train_data.images();
    let train_images = tensor_slice!(all_images, 0usize..TRAIN_SAMPLES, ..);

    println!(
        "  ✓ 训练样本: {} ({:.1}s)",
        TRAIN_SAMPLES,
        load_start.elapsed().as_secs_f32()
    );

    // 2. 创建模型
    println!("\n[2/4] 创建模型...");
    let graph = Graph::new_with_seed(42);
    let generator = Generator::new(&graph)?;
    let discriminator = Discriminator::new(&graph)?;

    // 损失函数（使用 MSE 作为简化的 GAN 损失）
    let criterion = MseLoss::new();

    // 优化器（beta1=0.5 对 GAN 更稳定）
    let mut g_optimizer =
        Adam::new_with_config(&graph, &generator.parameters(), LR_G, 0.5, 0.999, 1e-8);
    let mut d_optimizer =
        Adam::new_with_config(&graph, &discriminator.parameters(), LR_D, 0.5, 0.999, 1e-8);

    println!("  Generator: {LATENT_DIM} -> 128 -> 784");
    println!("  Discriminator: 784 -> 128 -> 1");
    println!("  学习率: D={LR_D}, G={LR_G}");

    // 3. 训练循环
    println!("\n[3/4] 开始训练...\n");
    let train_start = Instant::now();

    let num_batches = TRAIN_SAMPLES / BATCH_SIZE;

    // 追踪 D 的判别能力
    let mut d_real_avg = 0.0;
    let mut d_fake_avg = 0.0;

    for epoch in 0..EPOCHS {
        let epoch_start = Instant::now();
        let mut d_loss_sum = 0.0;
        let mut g_loss_sum = 0.0;
        let mut d_real_sum = 0.0;
        let mut d_fake_sum = 0.0;
        let mut batch_count = 0;

        for batch_idx in 0..num_batches {
            // === 准备数据 ===
            let start = batch_idx * BATCH_SIZE;
            let end = start + BATCH_SIZE;

            // 获取真实图像
            let real_images = tensor_slice!(train_images, start..end, ..);

            // 生成随机噪声（使用确定性种子保证可重复）
            let noise_seed = (epoch * num_batches + batch_idx) as u64;
            let z = Tensor::normal_seeded(0.0, 1.0, &[BATCH_SIZE, LATENT_DIM], noise_seed);

            // === 训练 Discriminator ===
            d_optimizer.zero_grad()?;

            // D 对真实图像的判别
            let real_out = discriminator.forward(&real_images)?;
            let real_labels = Tensor::ones(&[BATCH_SIZE, 1]);
            let d_real_loss = criterion.forward(&real_out, &real_labels)?;

            // 记录 D(real) 平均值
            let real_out_val = real_out.value()?.unwrap();
            let mut d_real_batch_sum = 0.0;
            for i in 0..BATCH_SIZE {
                d_real_batch_sum += real_out_val[[i, 0]];
            }
            d_real_sum += d_real_batch_sum / BATCH_SIZE as f32;

            // 反向传播 D(real)
            d_real_loss.backward()?;
            d_optimizer.step()?;

            // G 生成假图像
            let fake_images = generator.forward(&z)?;

            // D 对假图像的判别（使用 detach 阻止梯度流向 G）
            d_optimizer.zero_grad()?;
            let fake_detached = fake_images.detach();
            let fake_out = discriminator.forward(&fake_detached)?;
            let fake_labels = Tensor::zeros(&[BATCH_SIZE, 1]);
            let d_fake_loss = criterion.forward(&fake_out, &fake_labels)?;

            // 记录 D(fake) 平均值
            let fake_out_val = fake_out.value()?.unwrap();
            let mut d_fake_batch_sum = 0.0;
            for i in 0..BATCH_SIZE {
                d_fake_batch_sum += fake_out_val[[i, 0]];
            }
            d_fake_sum += d_fake_batch_sum / BATCH_SIZE as f32;

            // 反向传播 D(fake)
            let d_fake_loss_val = d_fake_loss.backward()?;
            d_optimizer.step()?;

            d_loss_sum += d_fake_loss_val;

            // === 训练 Generator ===
            g_optimizer.zero_grad()?;

            // 使用不同的噪声训练 G
            let noise_seed_g = (epoch * num_batches + batch_idx + 10000) as u64;
            let z_g = Tensor::normal_seeded(0.0, 1.0, &[BATCH_SIZE, LATENT_DIM], noise_seed_g);
            let fake_images_g = generator.forward(&z_g)?;

            // G 希望 D 把假图像判别为真
            let fake_out_for_g = discriminator.forward(&fake_images_g)?;
            let g_loss = criterion.forward(&fake_out_for_g, &real_labels)?;
            let g_loss_val = g_loss.backward()?;
            g_optimizer.step()?;

            d_loss_sum += d_fake_loss_val;
            g_loss_sum += g_loss_val;
            batch_count += 1;
        }

        let avg_d_loss = d_loss_sum / batch_count as f32;
        let avg_g_loss = g_loss_sum / batch_count as f32;
        d_real_avg = d_real_sum / batch_count as f32;
        d_fake_avg = d_fake_sum / batch_count as f32;

        println!(
            "Epoch {:2}/{}: D_loss={:.4}, G_loss={:.4}, D(real)={:.3}, D(fake)={:.3} ({:.1}s)",
            epoch + 1,
            EPOCHS,
            avg_d_loss,
            avg_g_loss,
            d_real_avg,
            d_fake_avg,
            epoch_start.elapsed().as_secs_f32()
        );
    }

    // 验证训练效果
    println!("\n训练结果验证:");
    println!("  D(real) = {d_real_avg:.3} (理想值接近 0.5-1.0)");
    println!("  D(fake) = {d_fake_avg:.3} (理想值接近 0.3-0.7)");

    println!(
        "\n训练完成！总用时: {:.1}s",
        train_start.elapsed().as_secs_f32()
    );

    // 4. 生成样本并保存可视化
    println!("\n[4/4] 生成样本并保存可视化...");

    // 生成一张图像用于验证
    let z_vis = Tensor::normal(0.0, 1.0, &[1, LATENT_DIM]);
    let fake_vis = generator.forward(&z_vis)?;

    // 获取生成的图像值
    let fake_val = fake_vis.value()?.unwrap();
    println!("  生成图像形状: {:?}", fake_val.shape());

    // 通过 D 判别，构建完整的 GAN 计算图
    let d_out_vis = discriminator.forward(&fake_vis)?;
    d_out_vis.forward()?; // 确保前向传播完成

    // 保存计算图可视化（Graphviz，启用模型分组）
    let vis_result = graph.save_visualization_grouped("examples/mnist_gan/mnist_gan", None)?;
    println!("  计算图已保存: {}", vis_result.dot_path.display());
    if let Some(img_path) = &vis_result.image_path {
        println!("  可视化图像: {}", img_path.display());
    }

    println!("\n=== GAN 训练示例完成 ===");
    println!("✅ 演示了 ForwardInput trait 和函数式 detach() 的使用");

    Ok(())
}
