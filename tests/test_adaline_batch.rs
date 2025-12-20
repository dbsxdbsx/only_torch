/*
 * @Author       : 老董
 * @Date         : 2025-07-24 16:30:00
 * @LastEditors  : 老董
 * @LastEditTime : 2025-07-24 16:30:00
 * @Description  : 批量ADALINE示例测试，参考自：https://github.com/zc911/MatrixSlow/blob/master/example/ch03/adaline_batch.py
 */

use only_torch::nn::optimizer::{Adam, Optimizer};
use only_torch::nn::{Graph, GraphError};
use only_torch::tensor::Tensor;
use only_torch::tensor_where;

#[test]
fn test_adaline_batch_with_optimizer() -> Result<(), GraphError> {
    let start_time = std::time::Instant::now();

    // 构造训练数据（与Python版本相同）
    let male_heights = Tensor::normal(171.0, 6.0, &[500]);
    let female_heights = Tensor::normal(158.0, 5.0, &[500]);

    let male_weights = Tensor::normal(70.0, 10.0, &[500]);
    let female_weights = Tensor::normal(57.0, 8.0, &[500]);

    let male_bfrs = Tensor::normal(16.0, 2.0, &[500]);
    let female_bfrs = Tensor::normal(22.0, 2.0, &[500]);

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
    train_set.shuffle_mut(Some(0)); // 随机打乱样本顺序
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

    // 权值向量，3x1矩阵
    let w = graph.new_parameter_node(&[3, 1], Some("w"))?;

    // 阈值，1x1矩阵
    let b = graph.new_parameter_node(&[1, 1], Some("b"))?;

    // 对一个mini batch的样本计算输出
    let xw = graph.new_mat_mul_node(x, w, Some("xw"))?;

    // 创建偏置广播节点（手动设置值）
    let bias_broadcasted = graph.new_input_node(&[batch_size, 1], Some("bias_broadcasted"))?;

    // 输出 = xw + bias_broadcasted
    let output = graph.new_add_node(&[xw, bias_broadcasted], Some("output"))?;
    let predict = graph.new_step_node(output, None)?;

    // 一个mini batch的样本的损失函数
    // 使用逐元素乘法：label * output
    let loss_input = graph.new_input_node(&[batch_size, 1], Some("loss_input"))?;
    let loss = graph.new_perception_loss_node(loss_input, Some("loss"))?;

    // 学习率
    let learning_rate = 0.0001;

    // 创建Adam优化器
    let mut optimizer = Adam::new_default(&graph, learning_rate)?;

    // 测试参数
    let max_epochs = 50;
    let target_accuracy = 0.95; // 95%
    let consecutive_success_required = 3;
    let mut consecutive_success_count = 0;
    let mut test_passed = false;

    // 设置全1向量的值（用于偏置广播）
    let ones_data = vec![1.0; batch_size];
    let ones_tensor = Tensor::new(&ones_data, &[batch_size, 1]);

    // 训练执行最多50个epoch，或直到达到成功条件
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

            // 手动计算偏置广播：b_broadcasted = ones * b
            let b_value = graph.get_node_value(b)?.unwrap();
            let b_broadcasted_value =
                &ones_tensor * b_value.get(&[0, 0]).get_data_number().unwrap();
            graph.set_node_value(bias_broadcasted, Some(&b_broadcasted_value))?;

            // 前向传播计算output = xw + bias_broadcasted
            graph.forward_node(output)?;

            // 计算损失输入：label * output (逐元素乘法)
            let output_value = graph.get_node_value(output)?.unwrap();
            let loss_input_value = &labels * output_value;
            graph.set_node_value(loss_input, Some(&loss_input_value))?;

            // 使用优化器执行一步训练
            optimizer.one_step(&mut graph, loss)?;
        }

        // 每个batch结束后更新参数
        optimizer.update(&mut graph)?;

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

            // 手动计算偏置广播
            let b_value = graph.get_node_value(b)?.unwrap();
            let b_broadcasted_value =
                &ones_tensor * b_value.get(&[0, 0]).get_data_number().unwrap();
            graph.set_node_value(bias_broadcasted, Some(&b_broadcasted_value))?;

            // 前向传播计算output
            graph.forward_node(output)?;

            // 在模型的predict节点上执行前向传播
            graph.forward_node(predict)?;
            let predict_value = graph.get_node_value(predict)?.unwrap();

            // 收集当前批次的预测结果
            for i in 0..batch_size {
                pred_vec.push(predict_value.get(&[i, 0]).get_data_number().unwrap());
            }
        }

        // 将1/0结果转化成1/-1结果，好与训练标签的约定一致
        let pred_tensor = Tensor::new(&pred_vec, &[pred_vec.len(), 1]) * 2.0 - 1.0;

        // 计算准确率（只考虑完整批次的样本）
        let valid_samples = (train_set.shape()[0] / batch_size) * batch_size;
        let train_set_labels = train_set.slice(&[&(0..valid_samples), &3]);
        let pred_subset = pred_tensor.slice(&[&(0..valid_samples), &0]);

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
