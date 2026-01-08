/*
 * @Author       : 老董
 * @Date         : 2025-07-24 16:30:00
 * @LastEditors  : 老董
 * @LastEditTime : 2025-12-21 16:30:00
 * @Description  : 批量ADALINE示例测试，参考自：https://github.com/zc911/MatrixSlow/blob/master/example/ch03/adaline_batch.py
 */

use only_torch::nn::optimizer::{Adam, Optimizer};
use only_torch::nn::{Graph, GraphError};
use only_torch::tensor::Tensor;
use only_torch::{tensor_slice, tensor_where};

#[test]
fn test_adaline_batch_with_optimizer() -> Result<(), GraphError> {
    let start_time = std::time::Instant::now();

    // 构造训练数据（使用固定种子确保测试可重复性）
    let seed_base: u64 = 42;
    let male_heights = Tensor::normal_seeded(171.0, 6.0, &[500], seed_base);
    let female_heights = Tensor::normal_seeded(158.0, 5.0, &[500], seed_base + 1);

    let male_weights = Tensor::normal_seeded(70.0, 10.0, &[500], seed_base + 2);
    let female_weights = Tensor::normal_seeded(57.0, 8.0, &[500], seed_base + 3);

    let male_bfrs = Tensor::normal_seeded(16.0, 2.0, &[500], seed_base + 4);
    let female_bfrs = Tensor::normal_seeded(22.0, 2.0, &[500], seed_base + 5);

    let male_labels = Tensor::new(&[1.0; 500], &[500]);
    let female_labels = Tensor::new(&[-1.0; 500], &[500]);

    let mut train_set = Tensor::stack(
        &[
            &Tensor::stack(&[&male_heights, &female_heights], false),
            &Tensor::stack(&[&male_weights, &female_weights], false),
            &Tensor::stack(&[&male_bfrs, &female_bfrs], false),
            &Tensor::stack(&[&male_labels, &female_labels], false),
        ],
        true,
    );
    train_set.permute_mut(&[1, 0]);
    train_set.shuffle_mut_seeded(Some(0), seed_base + 6); // 使用固定种子打乱样本顺序
    println!("训练集形状: {:?}", train_set.shape());

    // 批大小
    let batch_size = 10;
    println!("批大小: {batch_size}");

    // 创建计算图
    let mut graph = Graph::new();

    // batch_size x 3矩阵，每行保存一个样本，整个节点保存一个mini batch的样本
    let x = graph.new_input_node(&[batch_size, 3], Some("X"))?;

    // 保存一个mini batch的样本的类别标签
    let label = graph.new_input_node(&[batch_size, 1], Some("label"))?;

    // 权值向量，3x1矩阵（使用固定种子）
    let w = graph.new_parameter_node_seeded(&[3, 1], Some("w"), seed_base + 7)?;

    // 阈值，1x1矩阵（使用固定种子）
    let b = graph.new_parameter_node_seeded(&[1, 1], Some("b"), seed_base + 8)?;

    // 对一个mini batch的样本计算输出
    let xw = graph.new_mat_mul_node(x, w, Some("xw"))?;

    // 创建全1向量节点，用于偏置广播
    let ones = graph.new_input_node(&[batch_size, 1], Some("ones"))?;

    // 使用ScalarMultiply节点实现偏置广播：bias = b * ones
    let bias_broadcasted = graph.new_scalar_multiply_node(b, ones, Some("bias_broadcasted"))?;

    // 输出 = xw + bias_broadcasted
    let output = graph.new_add_node(&[xw, bias_broadcasted], Some("output"))?;
    let predict = graph.new_sign_node(output, None)?; // Sign 直接输出 {-1, 0, 1}

    // 一个mini batch的样本的损失函数
    // 使用Multiply节点计算逐元素乘法：label * output
    let label_output = graph.new_multiply_node(label, output, Some("label_output"))?;
    // PerceptionLoss 已经输出标量 [1,1]（对整个 batch 自动取平均）
    let loss = graph.new_perception_loss_node(label_output, Some("loss"))?;

    // 学习率
    let learning_rate = 0.0001;

    // 创建Adam优化器
    let mut optimizer = Adam::new_default(&graph, learning_rate)?;

    // 测试参数
    let max_epochs = 100;
    let target_accuracy = 0.95; // 95%准确率目标
    let consecutive_success_required = 3;
    let mut consecutive_success_count = 0;
    let mut test_passed = false;

    // 设置全1向量的值（用于偏置广播）
    let ones_data = vec![1.0; batch_size];
    let ones_tensor = Tensor::new(&ones_data, &[batch_size, 1]);
    graph.set_node_value(ones, Some(&ones_tensor))?;

    // 训练执行最多max_epochs个epoch
    for epoch in 0..max_epochs {
        // 遍历训练集中的批次
        let num_batches = train_set.shape()[0].div_ceil(batch_size); // 向上取整

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, train_set.shape()[0]);
            let actual_batch_size = end_idx - start_idx;

            // 如果最后一个批次大小不足，跳过（简化处理）
            if actual_batch_size < batch_size {
                continue;
            }

            // 取一个mini batch的样本的特征 (batch_size x 3)
            let mut features_data = Vec::with_capacity(batch_size * 3);
            for i in start_idx..end_idx {
                for j in 0..3 {
                    features_data.push(train_set.get(&[i, j]).get_data_number().unwrap());
                }
            }
            let features = Tensor::new(&features_data, &[batch_size, 3]);

            // 取一个mini batch的样本的标签 (batch_size x 1)
            let mut labels_data = Vec::with_capacity(batch_size);
            for i in start_idx..end_idx {
                labels_data.push(train_set.get(&[i, 3]).get_data_number().unwrap());
            }
            let labels = Tensor::new(&labels_data, &[batch_size, 1]);

            // 将特征赋给X节点，将标签赋给label节点
            graph.set_node_value(x, Some(&features))?;
            graph.set_node_value(label, Some(&labels))?;

            // 使用新 API 执行一步训练（PyTorch 风格）
            graph.zero_grad()?;
            graph.forward(loss)?;
            graph.backward(loss)?;
            optimizer.step(&mut graph)?;
        }

        // 每个epoch结束后评价模型的正确率
        let mut pred_vec = Vec::new();

        // 遍历训练集，计算当前模型对每个样本的预测值
        let num_batches = train_set.shape()[0].div_ceil(batch_size);

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, train_set.shape()[0]);
            let actual_batch_size = end_idx - start_idx;

            // 如果最后一个批次大小不足，跳过（简化处理）
            if actual_batch_size < batch_size {
                continue;
            }

            // 取一个mini batch的样本的特征
            let mut features_data = Vec::with_capacity(batch_size * 3);
            for i in start_idx..end_idx {
                for j in 0..3 {
                    features_data.push(train_set.get(&[i, j]).get_data_number().unwrap());
                }
            }
            let features = Tensor::new(&features_data, &[batch_size, 3]);
            graph.set_node_value(x, Some(&features))?;

            // 前向传播计算output（bias_broadcasted通过ScalarMultiply节点自动计算）
            graph.forward(output)?;

            // 在模型的predict节点上执行前向传播
            graph.forward(predict)?;
            let predict_value = graph.get_node_value(predict)?.unwrap();

            // 收集当前批次的预测结果
            for i in 0..batch_size {
                pred_vec.push(predict_value.get(&[i, 0]).get_data_number().unwrap());
            }
        }

        // Sign 已直接输出 {-1, 1}，无需转换
        let pred_tensor = Tensor::new(&pred_vec, &[pred_vec.len(), 1]);

        // 计算准确率（只考虑完整批次的样本）
        let valid_samples = (train_set.shape()[0] / batch_size) * batch_size;
        let train_set_labels = tensor_slice!(train_set, 0..valid_samples, 3);
        let pred_subset = tensor_slice!(pred_tensor, 0..valid_samples, 0);

        let filtered_sum = tensor_where!(train_set_labels == pred_subset, 1.0, 0.0).sum();
        let accuracy = filtered_sum.get_data_number().unwrap() / valid_samples as f32;
        let accuracy_percent = accuracy * 100.0;

        // 打印当前epoch数和模型在训练集上的正确率
        println!(
            "训练回合: {}, 正确率: {:.1}% (有效样本: {})",
            epoch + 1,
            accuracy_percent,
            valid_samples
        );

        // 检查是否达到目标准确率
        if accuracy >= target_accuracy {
            consecutive_success_count += 1;

            // 检查是否连续达到目标准确率足够次数
            if consecutive_success_count >= consecutive_success_required {
                test_passed = true;
                println!(
                    "🎉 测试通过！连续{}次达到{:.1}%以上准确率",
                    consecutive_success_required,
                    target_accuracy * 100.0
                );
                break;
            }
        } else {
            consecutive_success_count = 0; // 重置连续成功计数
        }
    }

    let duration = start_time.elapsed();
    println!("总耗时: {duration:.2?}");

    // 检查测试是否通过
    if test_passed {
        println!("✅ 批量ADALINE优化器测试成功通过！");
        Ok(())
    } else {
        println!(
            "❌ 批量ADALINE优化器测试失败：在{}个epoch内未能连续{}次达到{:.1}%以上准确率",
            max_epochs,
            consecutive_success_required,
            target_accuracy * 100.0
        );
        Err(GraphError::ComputationError(format!(
            "批量ADALINE优化器测试失败：在{}个epoch内未能连续{}次达到{:.1}%以上准确率",
            max_epochs,
            consecutive_success_required,
            target_accuracy * 100.0
        )))
    }
}
