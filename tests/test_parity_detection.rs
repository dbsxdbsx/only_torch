/*
 * IT-1: 奇偶性检测集成测试（固定长度 + 单序列）
 *
 * 任务：判断 0/1 序列中 1 的个数是奇数还是偶数
 * 目的：验证循环机制 + BPTT 能协同工作
 *
 * 网络结构（使用原子节点手工搭建）：
 *   input_t ──┐
 *             ├──→ [Add] ──→ [Tanh] ──→ hidden_t
 *   h_prev_t ─┘                           │
 *       ↑                                 │
 *       └─────────────────────────────────┘ (循环连接)
 *
 *   hidden_T ──→ [MatMul] ──→ [Sigmoid] ──→ output
 *                   ↑
 *               w_out (参数)
 *
 * 训练目标：
 *   序列 [1,0,1,1,0] → 1的个数=3 → 奇数 → 输出接近 1
 *   序列 [1,1,0,0,0] → 1的个数=2 → 偶数 → 输出接近 0
 */

use only_torch::nn::{Graph, GraphError, NodeId};
use only_torch::tensor::Tensor;

/// 生成奇偶性检测数据
///
/// 返回: (sequences, labels)
/// - sequences: Vec<Vec<f32>>，每个内层 Vec 是一个 0/1 序列
/// - labels: Vec<f32>，1.0 表示奇数个 1，0.0 表示偶数个 1
fn generate_parity_data(
    num_samples: usize,
    seq_len: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut sequences = Vec::with_capacity(num_samples);
    let mut labels = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        // 简单的伪随机生成
        let mut hasher = DefaultHasher::new();
        (seed, i as u64).hash(&mut hasher);
        let mut hash = hasher.finish();

        let mut seq = Vec::with_capacity(seq_len);
        let mut count_ones = 0u32;

        for _ in 0..seq_len {
            let bit = (hash & 1) as f32;
            seq.push(bit);
            count_ones += bit as u32;
            hash >>= 1;
            if hash == 0 {
                hasher = DefaultHasher::new();
                (seed, i as u64, seq.len()).hash(&mut hasher);
                hash = hasher.finish();
            }
        }

        sequences.push(seq);
        labels.push(if count_ones % 2 == 1 { 1.0 } else { 0.0 });
    }

    (sequences, labels)
}

/// 创建奇偶性检测 RNN 网络（完整版）
///
/// 使用标准 RNN 单元结构：
///   `hidden_t` = `tanh(w_ih` * `input_t` + `w_hh` * `h_prev` + `b_h`)
///   output = `sigmoid(w_ho` * `hidden_T` + `b_o`)
///
/// 返回: (graph, `input_node`, `output_node`, `loss_node`, `target_node`, params)
fn create_parity_rnn(
    seed: u64,
) -> Result<(Graph, NodeId, NodeId, NodeId, NodeId, Vec<NodeId>), GraphError> {
    let mut graph = Graph::new_with_seed(seed);
    graph.set_train_mode();

    let hidden_size = 4; // 增加隐藏层大小以提高表达能力

    // 输入节点：单个时间步的输入（标量）
    let input = graph.new_input_node(&[1, 1], Some("input"))?;

    // 循环状态节点：上一时间步的隐藏状态
    let h_prev = graph.new_state_node(&[1, hidden_size], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, hidden_size])))?;

    // === RNN 参数 ===
    // 输入到隐藏层权重
    let w_ih = graph.new_parameter_node(&[1, hidden_size], Some("w_ih"))?;
    // Xavier 初始化近似
    graph.set_node_value(
        w_ih,
        Some(&Tensor::new(&[0.5, -0.5, 0.3, -0.3], &[1, hidden_size])),
    )?;

    // 隐藏到隐藏权重
    let w_hh = graph.new_parameter_node(&[hidden_size, hidden_size], Some("w_hh"))?;
    let w_hh_init: Vec<f32> = vec![
        0.1, 0.2, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.1,
    ];
    graph.set_node_value(
        w_hh,
        Some(&Tensor::new(&w_hh_init, &[hidden_size, hidden_size])),
    )?;

    // 隐藏层偏置
    let b_h = graph.new_parameter_node(&[1, hidden_size], Some("b_h"))?;
    graph.set_node_value(b_h, Some(&Tensor::zeros(&[1, hidden_size])))?;

    // === 隐藏层计算 ===
    // input_contrib = input * w_ih  (1x1 @ 1xH = 1xH)
    let input_contrib = graph.new_mat_mul_node(input, w_ih, Some("input_contrib"))?;

    // hidden_contrib = h_prev * w_hh  (1xH @ HxH = 1xH)
    let hidden_contrib = graph.new_mat_mul_node(h_prev, w_hh, Some("hidden_contrib"))?;

    // pre_hidden = input_contrib + hidden_contrib + b_h
    let sum1 = graph.new_add_node(&[input_contrib, hidden_contrib], Some("sum1"))?;
    let pre_hidden = graph.new_add_node(&[sum1, b_h], Some("pre_hidden"))?;

    // hidden = tanh(pre_hidden)
    let hidden = graph.new_tanh_node(pre_hidden, Some("hidden"))?;

    // 循环连接：hidden -> h_prev
    graph.connect_recurrent(hidden, h_prev)?;

    // === 输出层 ===
    // 隐藏到输出权重
    let w_ho = graph.new_parameter_node(&[hidden_size, 1], Some("w_ho"))?;
    graph.set_node_value(
        w_ho,
        Some(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[hidden_size, 1])),
    )?;

    // 输出偏置
    let b_o = graph.new_parameter_node(&[1, 1], Some("b_o"))?;
    graph.set_node_value(b_o, Some(&Tensor::zeros(&[1, 1])))?;

    // pre_output = hidden * w_ho + b_o
    let hidden_out = graph.new_mat_mul_node(hidden, w_ho, Some("hidden_out"))?;
    let pre_output = graph.new_add_node(&[hidden_out, b_o], Some("pre_output"))?;

    // output = sigmoid(pre_output)
    let output = graph.new_sigmoid_node(pre_output, Some("output"))?;

    // 目标节点
    let target = graph.new_input_node(&[1, 1], Some("target"))?;

    // MSE Loss
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    let params = vec![w_ih, w_hh, b_h, w_ho, b_o];

    Ok((graph, input, output, loss, target, params))
}

/// 手动 SGD 更新
fn sgd_update(graph: &mut Graph, params: &[NodeId], lr: f32) -> Result<(), GraphError> {
    for &param in params {
        if let Some(param_grad) = graph.get_node_grad(param)? {
            // 获取当前参数值
            let current_value = graph.get_node_value(param)?.unwrap().clone();

            // 梯度下降：θ = θ - lr * grad
            // 梯度形状可能需要 reshape 为参数的形状
            let param_shape = current_value.shape();
            let grad = if param_grad.shape() == param_shape {
                param_grad.clone()
            } else {
                param_grad.reshape(param_shape)
            };

            let new_value = &current_value - &(&grad * lr);
            graph.set_node_value(param, Some(&new_value))?;
        }
    }
    Ok(())
}

/// 训练一个序列（带可选调试输出）
fn train_sequence(
    graph: &mut Graph,
    input_node: NodeId,
    loss_node: NodeId,
    target_node: NodeId,
    params: &[NodeId],
    sequence: &[f32],
    label: f32,
    lr: f32,
    debug: bool,
) -> Result<f32, GraphError> {
    // 重置循环状态
    graph.reset();

    // 设置目标值（BPTT 需要 loss 在每个时间步都被计算，所以提前设置）
    let target_tensor = Tensor::new(&[label], &[1, 1]);
    graph.set_node_value(target_node, Some(&target_tensor))?;

    // 前向传播整个序列（通过 loss_node，这样历史快照会包含 loss 值）
    for &bit in sequence {
        let input_tensor = Tensor::new(&[bit], &[1, 1]);
        graph.set_node_value(input_node, Some(&input_tensor))?;
        graph.step(loss_node)?;
    }

    // 获取最终 loss 值
    let loss_value = graph
        .get_node_value(loss_node)?
        .ok_or_else(|| GraphError::InvalidOperation("Loss 节点没有值".to_string()))?
        .data_as_slice()[0];

    if debug {
        println!("    序列: {sequence:?}, 标签: {label}");
        println!("    损失值: {loss_value:.6}");
        for (i, &param) in params.iter().enumerate() {
            let val = graph.get_node_value(param)?.map(|v| v.data_as_slice()[0]);
            println!("    参数[{i}] BPTT 前: {val:?}");
        }
    }

    // BPTT
    graph.backward_through_time(params, loss_node)?;

    if debug {
        for (i, &param) in params.iter().enumerate() {
            let grad = graph.get_node_grad(param)?.map(|j| j.data_as_slice()[0]);
            println!("    参数[{i}] 梯度: {grad:?}");
        }
    }

    // 手动 SGD 更新
    sgd_update(graph, params, lr)?;

    // 清零梯度
    graph.zero_grad()?;

    Ok(loss_value)
}

/// 评估准确率
fn evaluate(
    graph: &mut Graph,
    input_node: NodeId,
    output_node: NodeId,
    sequences: &[Vec<f32>],
    labels: &[f32],
) -> Result<f32, GraphError> {
    let mut correct = 0;

    // 切换到评估模式
    graph.set_eval_mode();

    for (seq, &label) in sequences.iter().zip(labels.iter()) {
        graph.reset();

        // 前向传播
        for &bit in seq {
            let input_tensor = Tensor::new(&[bit], &[1, 1]);
            graph.set_node_value(input_node, Some(&input_tensor))?;
            graph.step(output_node)?;
        }

        // 获取输出
        let output = graph
            .get_node_value(output_node)?
            .ok_or_else(|| GraphError::InvalidOperation("Output 节点没有值".to_string()))?
            .data_as_slice()[0];

        // 二分类：output > 0.5 预测为 1，否则为 0
        let prediction = if output > 0.5 { 1.0 } else { 0.0 };
        if (prediction - label).abs() < 0.1 {
            correct += 1;
        }
    }

    // 恢复训练模式
    graph.set_train_mode();

    Ok(correct as f32 / sequences.len() as f32)
}

/// IT-1: 奇偶性检测集成测试
///
/// 验证 RNN 网络能够学习判断序列中 1 的奇偶性
#[test]
fn test_parity_detection_can_learn() {
    println!("\n========== IT-1: 奇偶性检测集成测试 ==========\n");

    // 超参数
    let seq_len = 5;
    let num_train = 100;
    let num_test = 50;
    let epochs = 100;
    let lr = 0.1;
    let seed = 42u64;

    // 生成数据
    let (train_seqs, train_labels) = generate_parity_data(num_train, seq_len, seed);
    let (test_seqs, test_labels) = generate_parity_data(num_test, seq_len, seed + 1000);

    // 验证数据生成正确性（抽样检查）
    println!("[数据验证] 抽样检查前 3 个训练样本:");
    for i in 0..3 {
        let count_ones: u32 = train_seqs[i].iter().map(|&x| x as u32).sum();
        let expected = if count_ones % 2 == 1 { 1.0 } else { 0.0 };
        println!(
            "  seq={:?}, ones={}, label={}, expected={}",
            train_seqs[i], count_ones, train_labels[i], expected
        );
        assert_eq!(train_labels[i], expected, "样本 {i} 数据生成错误");
    }

    // 创建网络
    let (mut graph, input, output, loss, target, params) =
        create_parity_rnn(seed).expect("创建网络失败");

    println!("\n[网络结构] {} 个参数节点", params.len());

    // 保存网络拓扑可视化
    let viz_result = graph.save_visualization("tests/outputs/it1_parity_rnn", None);
    match &viz_result {
        Ok(output) => {
            println!("[可视化] DOT 文件: {}", output.dot_path.display());
            if let Some(img) = &output.image_path {
                println!("[可视化] 图像文件: {}", img.display());
            }
        }
        Err(e) => println!("[可视化] 跳过（{e:?}）"),
    }

    // 初始准确率
    let initial_acc =
        evaluate(&mut graph, input, output, &test_seqs, &test_labels).expect("评估失败");
    println!("\n[训练前] 初始准确率: {:.1}%", initial_acc * 100.0);

    // 前向/反向传播健壮性检查（第一个样本）
    println!("\n[健壮性检查] 单样本前向/反向传播:");
    let first_loss = train_sequence(
        &mut graph,
        input,
        loss,
        target,
        &params,
        &train_seqs[0],
        train_labels[0],
        lr,
        true,
    )
    .expect("第一个样本训练失败");
    assert!(first_loss >= 0.0, "损失值应非负");
    assert!(first_loss < 10.0, "损失值应在合理范围内");
    println!("  ✓ 前向/反向传播正常");

    // 训练循环
    println!("\n[训练过程]");
    let mut best_acc = initial_acc;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (seq, &label) in train_seqs.iter().zip(train_labels.iter()) {
            let loss_val = train_sequence(
                &mut graph, input, loss, target, &params, seq, label, lr, false,
            )
            .expect("训练失败");
            total_loss += loss_val;
        }

        // 每 20 轮评估一次
        if (epoch + 1) % 20 == 0 {
            let acc =
                evaluate(&mut graph, input, output, &test_seqs, &test_labels).expect("评估失败");
            let avg_loss = total_loss / num_train as f32;

            println!(
                "  Epoch {:3}: avg_loss={:.4}, test_acc={:.1}%",
                epoch + 1,
                avg_loss,
                acc * 100.0,
            );

            if acc > best_acc {
                best_acc = acc;
            }
        }
    }

    // 最终评估
    let final_acc =
        evaluate(&mut graph, input, output, &test_seqs, &test_labels).expect("评估失败");

    println!("\n[结果]");
    println!("  初始准确率: {:.1}%", initial_acc * 100.0);
    println!("  最终准确率: {:.1}%", final_acc * 100.0);
    println!("  最佳准确率: {:.1}%", best_acc * 100.0);

    // 验收标准
    assert!(
        best_acc > 0.55,
        "网络应学习到优于随机猜测的能力。最佳准确率: {:.1}%",
        best_acc * 100.0
    );

    assert!(
        final_acc > initial_acc || best_acc > initial_acc,
        "网络应在训练中有所改进。初始: {:.1}%, 最终: {:.1}%, 最佳: {:.1}%",
        initial_acc * 100.0,
        final_acc * 100.0,
        best_acc * 100.0
    );

    println!("\n✅ IT-1 测试通过！\n");
}
