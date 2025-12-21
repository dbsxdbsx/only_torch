/*
 * @Author       : 老董
 * @Date         : 2025-07-24 16:00:00
 * @LastEditors  : 老董
 * @LastEditTime : 2025-07-24 16:00:00
 * @Description  : 优化器示例测试，参考自：https://github.com/zc911/MatrixSlow/blob/master/example/ch03/optimizer_example.py
 */

use only_torch::nn::optimizer::{Optimizer, SGD};
use only_torch::nn::{Graph, GraphError};
use only_torch::tensor::Tensor;
use only_torch::{tensor_slice, tensor_where};

#[test]
fn test_optimizer_example() -> Result<(), GraphError> {
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

    // 创建计算图
    // 方式1: Granular API（当前方式，精确控制每个参数的种子）
    let mut graph = Graph::new();
    // 方式2: Graph级别种子（更简洁，推荐用于一般训练场景）
    // let mut graph = Graph::new_with_seed(seed_base);

    // 构造计算图：输入向量，是一个3x1矩阵，不需要初始化，不参与训练
    let x = graph.new_input_node(&[3, 1], Some("x"))?;

    // 类别标签，1男，-1女
    let label = graph.new_input_node(&[1, 1], Some("label"))?;

    // 权重向量，是一个1x3矩阵，需要初始化，参与训练
    // Granular: 使用显式种子
    let w = graph.new_parameter_node_seeded(&[1, 3], Some("w"), seed_base + 7)?;
    // Graph种子: let w = graph.new_parameter_node(&[1, 3], Some("w"))?;

    // 阈值，是一个1x1矩阵，需要初始化，参与训练
    // Granular: 使用显式种子
    let b = graph.new_parameter_node_seeded(&[1, 1], Some("b"), seed_base + 8)?;
    // Graph种子: let b = graph.new_parameter_node(&[1, 1], Some("b"))?;

    // ADALINE的预测输出
    let wx = graph.new_mat_mul_node(w, x, None)?;
    let output = graph.new_add_node(&[wx, b], None)?;
    let predict = graph.new_step_node(output, None)?;

    // 损失函数：使用MatMul节点连接label和output，保持梯度链完整
    let loss_input = graph.new_mat_mul_node(label, output, Some("loss_input"))?;
    let loss = graph.new_perception_loss_node(loss_input, Some("loss"))?;

    // 学习率（与Python版本一致）
    let learning_rate = 0.0001;

    // 创建SGD优化器
    let mut optimizer = SGD::new(&graph, learning_rate)?;

    // mini batch参数
    let mini_batch_size = 8;
    let mut cur_batch_size = 0;

    // 测试参数（与test_adaline.rs一致）
    let max_epochs = 100;
    let target_accuracy = 0.95; // 95%
    let consecutive_success_required = 3;
    let mut consecutive_success_count = 0;
    let mut test_passed = false;

    // 训练执行最多50个epoch，或直到达到成功条件
    for epoch in 0..max_epochs {
        // 遍历训练集中的样本
        for i in 0..train_set.shape()[0] {
            // 取第i个样本的前3列（除最后一列的所有列），构造3x1矩阵对象
            let features = tensor_slice!(train_set, i, 0..3).transpose();

            // 取第i个样本的最后一列，是该样本的性别标签（1男，-1女），构造1x1矩阵对象
            let l = tensor_slice!(train_set, i, 3);

            // 将特征赋给x节点，将标签赋给label节点
            graph.set_node_value(x, Some(&features))?;
            graph.set_node_value(label, Some(&l))?;

            // 优化器执行一次前向传播和一次后向传播
            // loss_input = label * output 会在前向传播时自动计算
            optimizer.one_step(&mut graph, loss)?;
            cur_batch_size += 1;

            // 当积累到一个mini batch的时候，完成一次参数更新
            if cur_batch_size == mini_batch_size {
                // 在第一个epoch的第一个batch打印调试信息
                if epoch == 0 && i < mini_batch_size {
                    println!(
                        "更新前 w: {:?}",
                        graph.get_node_value(w)?.unwrap().get(&[0, 0])
                    );
                    println!(
                        "更新前 b: {:?}",
                        graph.get_node_value(b)?.unwrap().get(&[0, 0])
                    );
                }

                optimizer.update(&mut graph)?;

                if epoch == 0 && i < mini_batch_size {
                    println!(
                        "更新后 w: {:?}",
                        graph.get_node_value(w)?.unwrap().get(&[0, 0])
                    );
                    println!(
                        "更新后 b: {:?}",
                        graph.get_node_value(b)?.unwrap().get(&[0, 0])
                    );
                }

                cur_batch_size = 0;
            }
        }

        // 处理最后不完整的batch
        if cur_batch_size > 0 {
            optimizer.update(&mut graph)?;
            cur_batch_size = 0;
        }

        // 每个epoch结束后评价模型的正确率
        let mut pred_vec = Vec::new();

        // 遍历训练集，计算当前模型对每个样本的预测值
        for i in 0..train_set.shape()[0] {
            let features = tensor_slice!(train_set, i, 0..3).transpose();
            graph.set_node_value(x, Some(&features))?;

            // 在模型的predict节点上执行前向传播
            graph.forward_node(predict)?;
            let predict_value = graph.get_node_value(predict)?.unwrap();
            pred_vec.push(predict_value.get(&[0, 0]).get_data_number().unwrap());
        }

        // 将1/0结果转化成1/-1结果，好与训练标签的约定一致
        let pred_tensor = Tensor::new(&pred_vec, &[pred_vec.len(), 1]) * 2.0 - 1.0;

        // 计算准确率
        let train_set_labels = tensor_slice!(train_set, 0..train_set.shape()[0], 3);
        let pred_subset = tensor_slice!(pred_tensor, 0..pred_vec.len(), 0);

        let filtered_sum = tensor_where!(train_set_labels == pred_subset, 1.0, 0.0).sum();
        let accuracy = filtered_sum.get_data_number().unwrap() / train_set.shape()[0] as f32;
        let accuracy_percent = accuracy * 100.0;

        // 打印当前epoch数和模型在训练集上的正确率
        println!("训练回合: {}, 正确率: {:.1}%", epoch + 1, accuracy_percent);

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
        println!("✅ 优化器示例测试成功通过！");
        Ok(())
    } else {
        println!(
            "❌ 优化器示例测试失败：在{}个epoch内未能连续{}次达到{:.1}%以上准确率",
            max_epochs,
            consecutive_success_required,
            target_accuracy * 100.0
        );
        Err(GraphError::ComputationError(format!(
            "优化器示例测试失败：在{}个epoch内未能连续{}次达到{:.1}%以上准确率",
            max_epochs,
            consecutive_success_required,
            target_accuracy * 100.0
        )))
    }
}
