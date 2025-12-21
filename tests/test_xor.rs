/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : XOR（异或）问题测试 - 经典的非线性分类问题，需要隐藏层+非线性激活才能解决
 *                 网络结构：Input(2) -> Hidden(Tanh) -> Output -> PerceptionLoss
 * @LastEditors  : 老董
 * @LastEditTime : 2025-12-21
 */
use only_torch::nn::{Graph, GraphError};
use only_torch::tensor::Tensor;

/// XOR问题训练数据
/// 输入: (0,0), (0,1), (1,0), (1,1)
/// 输出: 0, 1, 1, 0
/// 转换为-1/+1标签用于PerceptionLoss: -1, +1, +1, -1
fn get_xor_data() -> (Vec<Tensor>, Vec<Tensor>) {
    let inputs = vec![
        Tensor::new(&[0.0, 0.0], &[2, 1]),
        Tensor::new(&[0.0, 1.0], &[2, 1]),
        Tensor::new(&[1.0, 0.0], &[2, 1]),
        Tensor::new(&[1.0, 1.0], &[2, 1]),
    ];
    // 使用-1/+1标签
    let labels = vec![
        Tensor::new(&[-1.0], &[1, 1]), // XOR(0,0) = 0 -> -1
        Tensor::new(&[1.0], &[1, 1]),  // XOR(0,1) = 1 -> +1
        Tensor::new(&[1.0], &[1, 1]),  // XOR(1,0) = 1 -> +1
        Tensor::new(&[-1.0], &[1, 1]), // XOR(1,1) = 0 -> -1
    ];
    (inputs, labels)
}

#[test]
fn test_xor() -> Result<(), GraphError> {
    let start_time = std::time::Instant::now();

    // 使用固定种子确保测试可重复性
    let seed_base: u64 = 42;

    // 创建计算图
    let mut graph = Graph::new();

    // ========== 网络结构 ==========
    // 输入层: 2个特征
    let x = graph.new_input_node(&[2, 1], Some("x"))?;
    // 标签: 1x1
    let label = graph.new_input_node(&[1, 1], Some("label"))?;

    // 隐藏层权重和偏置 (使用固定种子初始化)
    // 隐藏层有4个神经元，输入2个特征，所以权重形状为[4, 2]
    let w1 = graph.new_parameter_node_seeded(&[4, 2], Some("w1"), seed_base)?;
    let b1 = graph.new_parameter_node_seeded(&[4, 1], Some("b1"), seed_base + 1)?;

    // 输出层权重和偏置
    // 输出1个值，隐藏层4个神经元，所以权重形状为[1, 4]
    let w2 = graph.new_parameter_node_seeded(&[1, 4], Some("w2"), seed_base + 2)?;
    let b2 = graph.new_parameter_node_seeded(&[1, 1], Some("b2"), seed_base + 3)?;

    // 隐藏层: h = tanh(w1 * x + b1)
    let wx1 = graph.new_mat_mul_node(w1, x, None)?;
    let z1 = graph.new_add_node(&[wx1, b1], None)?;
    let h = graph.new_tanh_node(z1, Some("hidden"))?;

    // 输出层: output = w2 * h + b2
    let wx2 = graph.new_mat_mul_node(w2, h, None)?;
    let output = graph.new_add_node(&[wx2, b2], Some("output"))?;

    // 预测: step函数将输出转换为0/1
    let predict = graph.new_step_node(output, Some("predict"))?;

    // 损失函数: PerceptionLoss(label * output)
    let loss_input = graph.new_mat_mul_node(label, output, Some("loss_input"))?;
    let loss = graph.new_perception_loss_node(loss_input, Some("loss"))?;

    // 获取训练数据
    let (inputs, labels) = get_xor_data();

    // 学习率
    let learning_rate = 1.0;

    // 测试参数
    let max_epochs = 500; // 通常30-50个epoch就能收敛
    let target_accuracy = 1.0; // 100% (XOR只有4个样本，应该能完全学会)
    let consecutive_success_required = 10;
    let mut consecutive_success_count = 0;
    let mut test_passed = false;

    // 训练循环
    for epoch in 0..max_epochs {
        // 遍历所有4个XOR样本
        for (input, lbl) in inputs.iter().zip(labels.iter()) {
            // 设置输入和标签
            graph.set_node_value(x, Some(input))?;
            graph.set_node_value(label, Some(lbl))?;

            // 前向传播
            graph.forward_node(loss)?;

            // 反向传播
            graph.backward_nodes(&[w1, b1, w2, b2], loss)?;

            // 更新参数
            // w1
            let w1_value = graph.get_node_value(w1)?.unwrap();
            let w1_grad = graph.get_node_grad(w1)?.unwrap();
            graph.set_node_value(w1, Some(&(w1_value - learning_rate * w1_grad)))?;

            // b1
            let b1_value = graph.get_node_value(b1)?.unwrap();
            let b1_grad = graph.get_node_grad(b1)?.unwrap();
            graph.set_node_value(b1, Some(&(b1_value - learning_rate * b1_grad)))?;

            // w2
            let w2_value = graph.get_node_value(w2)?.unwrap();
            let w2_grad = graph.get_node_grad(w2)?.unwrap();
            graph.set_node_value(w2, Some(&(w2_value - learning_rate * w2_grad)))?;

            // b2
            let b2_value = graph.get_node_value(b2)?.unwrap();
            let b2_grad = graph.get_node_grad(b2)?.unwrap();
            graph.set_node_value(b2, Some(&(b2_value - learning_rate * b2_grad)))?;

            // 清除梯度
            graph.clear_jacobi()?;
        }

        // 评估准确率
        let mut correct = 0;
        for (input, lbl) in inputs.iter().zip(labels.iter()) {
            graph.set_node_value(x, Some(input))?;
            graph.forward_node(predict)?;

            let pred_value = graph.get_node_value(predict)?.unwrap().get(&[0, 0]);
            let pred = pred_value.get_data_number().unwrap();
            // 将预测的0/1转换为-1/+1
            let pred_label = pred * 2.0 - 1.0;

            let expected_label = lbl.get(&[0, 0]).get_data_number().unwrap();
            if pred_label == expected_label {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / 4.0;

        // 每50个epoch打印一次进度
        if (epoch + 1) % 50 == 0 || epoch == 0 || accuracy == 1.0 {
            println!(
                "训练回合: {}, 正确率: {}/{} ({:.1}%)",
                epoch + 1,
                correct,
                4,
                accuracy * 100.0
            );
        }

        // 检查是否达到目标准确率
        if accuracy >= target_accuracy {
            consecutive_success_count += 1;
            if consecutive_success_count >= consecutive_success_required {
                test_passed = true;
                println!(
                    "🎉 测试通过！连续{}次达到{:.1}%准确率",
                    consecutive_success_required,
                    target_accuracy * 100.0
                );
                break;
            }
        } else {
            consecutive_success_count = 0;
        }
    }

    let duration = start_time.elapsed();
    println!("总耗时: {duration:.2?}");

    // 打印最终的预测结果
    println!("\n=== 最终预测结果 ===");
    for (input, lbl) in inputs.iter().zip(labels.iter()) {
        graph.set_node_value(x, Some(input))?;
        graph.forward_node(output)?;
        graph.forward_node(predict)?;

        let raw_output = graph.get_node_value(output)?.unwrap().get(&[0, 0]);
        let pred_value = graph.get_node_value(predict)?.unwrap().get(&[0, 0]);

        let x1 = input.get(&[0, 0]).get_data_number().unwrap() as i32;
        let x2 = input.get(&[1, 0]).get_data_number().unwrap() as i32;
        let expected = if lbl.get(&[0, 0]).get_data_number().unwrap() > 0.0 {
            1
        } else {
            0
        };
        let predicted = pred_value.get_data_number().unwrap() as i32;

        println!(
            "XOR({}, {}) = {} (预测: {}, 原始输出: {:.4})",
            x1,
            x2,
            expected,
            predicted,
            raw_output.get_data_number().unwrap()
        );
    }

    if test_passed {
        println!("\n✅ XOR测试成功通过！证明网络能学习非线性函数");
        Ok(())
    } else {
        println!(
            "\n❌ XOR测试失败：在{}个epoch内未能连续{}次达到{:.1}%准确率",
            max_epochs,
            consecutive_success_required,
            target_accuracy * 100.0
        );
        Err(GraphError::ComputationError(format!(
            "XOR测试失败：在{}个epoch内未能连续{}次达到{:.1}%准确率",
            max_epochs,
            consecutive_success_required,
            target_accuracy * 100.0
        )))
    }
}

