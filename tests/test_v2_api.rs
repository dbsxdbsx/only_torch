/*
 * V2 API 集成测试
 *
 * 测试 GraphHandle + Var 的基本功能：
 * - 创建节点
 * - 链式调用
 * - 算子重载
 * - 前向/反向传播
 */

use only_torch::nn::{GraphHandle, Init, VarActivationOps, VarLossOps, VarMatrixOps};
use only_torch::tensor::Tensor;

/// 测试基本的 V2 API 创建和前向传播
#[test]
fn test_v2_basic_forward() {
    let graph = GraphHandle::new();

    // 创建输入
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]))
        .unwrap();

    // 链式激活函数
    let h = x.relu();

    // 前向传播
    h.forward().unwrap();

    // 验证结果
    let result = h.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[1, 4]);
}

/// 测试算子重载（加法）
#[test]
fn test_v2_operator_add() {
    let graph = GraphHandle::new();

    let a = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let b = graph.input(&Tensor::new(&[3.0, 4.0], &[1, 2])).unwrap();

    // 使用算子重载
    let c = &a + &b;

    c.forward().unwrap();

    let result = c.value().unwrap().unwrap();
    let data = result.data_as_slice();
    assert_eq!(data, &[4.0, 6.0]);
}

/// 测试算子重载（减法）
#[test]
fn test_v2_operator_sub() {
    let graph = GraphHandle::new();

    let a = graph.input(&Tensor::new(&[5.0, 6.0], &[1, 2])).unwrap();
    let b = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();

    // 使用算子重载
    let c = &a - &b;

    c.forward().unwrap();

    let result = c.value().unwrap().unwrap();
    let data = result.data_as_slice();
    assert_eq!(data, &[4.0, 4.0]);
}

/// 测试算子重载（乘法 - 元素级）
#[test]
fn test_v2_operator_mul() {
    let graph = GraphHandle::new();

    let a = graph.input(&Tensor::new(&[2.0, 3.0], &[1, 2])).unwrap();
    let b = graph.input(&Tensor::new(&[4.0, 5.0], &[1, 2])).unwrap();

    // 使用算子重载
    let c = &a * &b;

    c.forward().unwrap();

    let result = c.value().unwrap().unwrap();
    let data = result.data_as_slice();
    assert_eq!(data, &[8.0, 15.0]);
}

/// 测试链式调用
#[test]
fn test_v2_chain_calls() {
    let graph = GraphHandle::new();

    let x = graph
        .input(&Tensor::new(&[-1.0, 2.0, -3.0, 4.0], &[1, 4]))
        .unwrap();

    // 链式调用：ReLU -> Sigmoid
    let y = x.relu().sigmoid();

    y.forward().unwrap();

    let result = y.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[1, 4]);

    // ReLU 后：[0, 2, 0, 4]
    // Sigmoid 后：约 [0.5, 0.88, 0.5, 0.98]
    let data = result.data_as_slice();
    assert!((data[0] - 0.5).abs() < 0.01); // sigmoid(0) = 0.5
    assert!(data[1] > 0.8); // sigmoid(2) ≈ 0.88
}

/// 测试参数初始化
#[test]
fn test_v2_parameter_init() {
    let graph = GraphHandle::new();

    // 使用 Xavier 初始化创建参数
    let w = graph.parameter(&[10, 5], Init::Xavier, "weight").unwrap();

    let val = w.value().unwrap().unwrap();
    assert_eq!(val.shape(), &[10, 5]);

    // 验证初始化的统计特性
    let data = val.data_as_slice();
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 0.1); // 均值接近 0
}

/// 测试 MSE Loss 和反向传播
#[test]
fn test_v2_mse_backward() {
    let graph = GraphHandle::new();

    // 简单线性模型：y = w * x
    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let target = graph.input(&Tensor::new(&[2.0, 4.0], &[1, 2])).unwrap();

    // 计算 MSE loss
    let loss = x.mse_loss(&target).unwrap();

    // 反向传播
    let loss_val = loss.backward().unwrap();

    // 验证 loss 值
    // MSE = mean((x - target)^2) = mean((1-2)^2 + (2-4)^2) = mean(1 + 4) = 2.5
    assert!((loss_val - 2.5).abs() < 0.01);
}

/// 测试 detach 功能
#[test]
fn test_v2_detach() {
    let graph = GraphHandle::new();

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let y = x.relu();

    // Detach 后的节点不参与梯度计算
    let z = y.detach().unwrap();

    // 验证 detach 返回的是同一个节点
    assert_eq!(y.node_id(), z.node_id());
}

/// 测试矩阵乘法
#[test]
fn test_v2_matmul() {
    let graph = GraphHandle::new();

    // [1, 2] @ [[1], [2]] = [5]
    let a = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let b = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();

    let c = a.matmul(&b).unwrap();
    c.forward().unwrap();

    let result = c.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[1, 1]);

    let val = result.get_data_number().unwrap();
    assert!((val - 5.0).abs() < 0.001);
}

/// 测试 GraphHandle 的 Clone 语义
#[test]
fn test_v2_graph_clone() {
    let graph1 = GraphHandle::new();
    let graph2 = graph1.clone();

    // 在 graph1 上创建节点
    let x = graph1.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    // graph2 应该能看到这个节点（因为它们共享同一个 GraphInner）
    let node_count = graph2.inner().nodes_count();
    assert!(node_count >= 1);
}

/// 测试负号运算符
#[test]
fn test_v2_operator_neg() {
    let graph = GraphHandle::new();

    let a = graph
        .input(&Tensor::new(&[1.0, -2.0, 3.0], &[1, 3]))
        .unwrap();
    let b = -&a;

    b.forward().unwrap();

    let result = b.value().unwrap().unwrap();
    let data = result.data_as_slice();
    assert_eq!(data, &[-1.0, 2.0, -3.0]);
}

// ==================== XOR 完整训练测试 ====================

/// XOR问题 - 使用 V2 API 的完整训练流程
///
/// 网络结构：Input(2) -> Hidden(4, Tanh) -> Output(1) -> PerceptionLoss
/// 这是 Phase 1b 的验收测试，证明 V2 API 能完成端到端训练
#[test]
fn test_v2_xor_training() {
    let start_time = std::time::Instant::now();

    // ========== 创建计算图 ==========
    let graph = GraphHandle::new();

    // 输入变量（Plan A：建图一次，训练循环中通过 set_value 喂新数据）
    let x = graph.zeros(&[2, 1]).unwrap();
    let label = graph.zeros(&[1, 1]).unwrap();

    // 隐藏层参数（使用种子初始化确保可重复）
    let w1 = graph.parameter_seeded(&[4, 2], "w1", 42).unwrap();
    let b1 = graph.parameter_seeded(&[4, 1], "b1", 43).unwrap();

    // 输出层参数
    let w2 = graph.parameter_seeded(&[1, 4], "w2", 44).unwrap();
    let b2 = graph.parameter_seeded(&[1, 1], "b2", 45).unwrap();

    // ========== 构建网络（使用 V2 链式调用和算子重载）==========
    // 隐藏层: h = tanh(w1 @ x + b1)
    let wx1 = w1.matmul(&x).unwrap();
    let z1 = &wx1 + &b1;
    let h = z1.tanh();

    // 输出层: output = w2 @ h + b2
    let wx2 = w2.matmul(&h).unwrap();
    let output = &wx2 + &b2;

    // 预测: step函数
    let predict = output.step();

    // 损失: PerceptionLoss(label * output)
    let loss_input = label.matmul(&output).unwrap();
    let loss = loss_input.perception_loss();

    // ========== 训练数据 ==========
    let inputs = vec![
        Tensor::new(&[0.0, 0.0], &[2, 1]),
        Tensor::new(&[0.0, 1.0], &[2, 1]),
        Tensor::new(&[1.0, 0.0], &[2, 1]),
        Tensor::new(&[1.0, 1.0], &[2, 1]),
    ];
    let labels = vec![
        Tensor::new(&[-1.0], &[1, 1]), // XOR(0,0) = 0 -> -1
        Tensor::new(&[1.0], &[1, 1]),  // XOR(0,1) = 1 -> +1
        Tensor::new(&[1.0], &[1, 1]),  // XOR(1,0) = 1 -> +1
        Tensor::new(&[-1.0], &[1, 1]), // XOR(1,1) = 0 -> -1
    ];

    // 学习率
    let learning_rate = 1.0;
    let max_epochs = 500;
    let target_accuracy = 1.0;
    let consecutive_success_required = 10;
    let mut consecutive_success_count = 0;
    let mut test_passed = false;

    // ========== 训练循环 ==========
    for epoch in 0..max_epochs {
        // 遍历所有样本
        for (input, lbl) in inputs.iter().zip(labels.iter()) {
            // 设置输入（使用 V2 API）
            x.set_value(input).unwrap();
            label.set_value(lbl).unwrap();

            // 前向 + 反向传播（V2 的 ensure-forward 语义）
            loss.backward().unwrap();

            // 手动更新参数（Phase 2 会有 Optimizer）
            // w1 -= lr * grad
            let w1_val = w1.value().unwrap().unwrap();
            let w1_grad = w1.grad().unwrap().unwrap();
            w1.set_value(&(&w1_val - learning_rate * &w1_grad)).unwrap();

            // b1 -= lr * grad
            let b1_val = b1.value().unwrap().unwrap();
            let b1_grad = b1.grad().unwrap().unwrap();
            b1.set_value(&(&b1_val - learning_rate * &b1_grad)).unwrap();

            // w2 -= lr * grad
            let w2_val = w2.value().unwrap().unwrap();
            let w2_grad = w2.grad().unwrap().unwrap();
            w2.set_value(&(&w2_val - learning_rate * &w2_grad)).unwrap();

            // b2 -= lr * grad
            let b2_val = b2.value().unwrap().unwrap();
            let b2_grad = b2.grad().unwrap().unwrap();
            b2.set_value(&(&b2_val - learning_rate * &b2_grad)).unwrap();

            // 清除梯度
            graph.zero_grad().unwrap();
        }

        // 评估准确率
        let mut correct = 0;
        for (input, lbl) in inputs.iter().zip(labels.iter()) {
            x.set_value(input).unwrap();
            predict.forward().unwrap();

            let pred = predict.value().unwrap().unwrap();
            let pred_val = pred.get_data_number().unwrap();
            // 预测的 0/1 转换为 -1/+1
            let pred_label = pred_val * 2.0 - 1.0;

            let expected = lbl.get_data_number().unwrap();
            if pred_label == expected {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / 4.0;

        // 检查是否达到目标
        if accuracy >= target_accuracy {
            consecutive_success_count += 1;
            if consecutive_success_count >= consecutive_success_required {
                test_passed = true;
                println!(
                    "🎉 V2 XOR 测试通过！第 {} 轮，连续 {} 次达到 100% 准确率",
                    epoch + 1,
                    consecutive_success_required
                );
                break;
            }
        } else {
            consecutive_success_count = 0;
        }
    }

    let duration = start_time.elapsed();
    println!("V2 XOR 训练耗时: {duration:.2?}");

    assert!(
        test_passed,
        "V2 XOR 测试失败：未能在 {} 轮内达到目标准确率",
        max_epochs
    );
}
