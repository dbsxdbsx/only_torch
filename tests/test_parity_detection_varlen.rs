//! IT-3a: 变长奇偶性检测集成测试（暂时禁用）
//!
//! ## 当前状态
//!
//! 此测试文件暂时被完全注释，原因如下：
//!
//! 1. **API 重构**：项目正在进行 Architecture V2 重构，所有 Layer（包括 Rnn、Lstm、Gru）
//!    已迁移到新的 struct API（PyTorch 风格），旧的函数式 API 已被删除。
//!
//! 2. **非标准机制**：此测试使用「状态冻结 mask」机制处理变长序列：
//!    `h_t = mask * h̃_t + (1-mask) * h_prev`
//!    这不是 PyTorch 的标准做法，PyTorch 推荐使用 `pack_padded_sequence`。
//!
//! 3. **架构决策待定**：是否在新 Rnn struct 中支持 mask 参数，需要进一步讨论。
//!
//! ## 未来修改路线
//!
//! ### 方案 A：为 Rnn/Lstm/Gru 添加 mask 支持
//!
//! ```ignore
//! impl Rnn {
//!     pub fn step_with_mask(&self, x: &Tensor, mask: &Tensor) -> Result<&Var, GraphError>;
//! }
//! ```
//!
//! 优点：保持与现有测试逻辑兼容，支持简单的变长序列处理。
//! 缺点：非 PyTorch 标准模式，可能增加 API 复杂度。
//!
//! ### 方案 B：实现 pack_padded_sequence（推荐）
//!
//! 遵循 PyTorch 标准做法：
//!
//! ```ignore
//! let packed = pack_padded_sequence(&input, &lengths, batch_first=true);
//! let (output, hidden) = rnn.forward(&packed);
//! let unpacked = pad_packed_sequence(&output, batch_first=true);
//! ```
//!
//! 优点：与 PyTorch 完全一致，业界标准做法，计算效率更高（跳过 padding 计算）。
//! 缺点：实现复杂度较高，需要新增 PackedSequence 数据结构。
//!
//! ### 方案 C：删除此测试
//!
//! 如果变长序列支持不是当前优先级，可以将此测试归档或删除，
//! 待项目成熟后再考虑实现 pack_padded_sequence。
//!
//! ## 相关文件
//!
//! - `tests/test_parity_detection.rs`：定长版本，已适配新 API，测试通过
//! - `src/nn/layer/rnn.rs`：新版 Rnn struct 实现
//! - `.doc/design/architecture_v2_design.md`：架构设计文档

// =============================================================================
// 以下为原始代码（已注释）
// =============================================================================

/*
 * IT-3a: 奇偶性检测集成测试（变长 + Batch + Padding/Mask）
 *
 * 与 IT-2 相同的任务，但序列长度可变
 * 目的：验证 Padding + Mask 机制在变长序列上的正确性
 *
 * 变长处理方案（Padding + Mask）：
 *   1. 将所有序列填充到 max_len
 *   2. 生成 mask 张量标记有效时间步
 *   3. 使用状态冻结公式：h_t = mask * h̃_t + (1-mask) * h_{t-1}
 *
 * 网络结构（与 IT-2 类似，但添加 mask 门控）：
 *   input_t [B, 1] ──┐
 *                    ├──→ [Add] ──→ [Tanh] ──→ h̃_t [B, H]
 *   h_prev_t [B, H] ─┘                           │
 *                                                ↓
 *   mask_t [B, H] ──→ [状态冻结] ──→ hidden_t [B, H]
 *       ↑                               │
 *       └───────────────────────────────┘ (循环连接)
 *
 *   hidden_T ──→ [MatMul] ──→ [Sigmoid] ──→ output [B, 1]
 *
 * 验收标准：
 *   1. 变长序列能正常训练
 *   2. 短序列的状态在 padding 区间被正确冻结
 *   3. Loss 有下降趋势
 */

/*
use only_torch::nn::{Graph, GraphError, NodeId};
use only_torch::tensor::Tensor;

/// 生成变长奇偶性检测数据
///
/// 返回: (sequences, labels, lengths)
/// - sequences: Vec<Vec<f32>>，每个序列长度在 [`min_len`, `max_len`] 范围内
/// - labels: Vec<f32>，1.0 表示奇数个 1，0.0 表示偶数个 1
/// - lengths: Vec<usize>，每个序列的实际长度
fn generate_varlen_parity_data(
    num_samples: usize,
    min_len: usize,
    max_len: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<usize>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut sequences = Vec::with_capacity(num_samples);
    let mut labels = Vec::with_capacity(num_samples);
    let mut lengths = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        // 生成随机长度
        let mut hasher = DefaultHasher::new();
        (seed, i as u64, "len").hash(&mut hasher);
        let len_hash = hasher.finish();
        let seq_len = min_len + (len_hash as usize % (max_len - min_len + 1));

        // 生成序列内容
        hasher = DefaultHasher::new();
        (seed, i as u64).hash(&mut hasher);
        let mut hash = hasher.finish();

        let mut seq = Vec::with_capacity(seq_len);
        let mut count_ones = 0u32;

        for j in 0..seq_len {
            let bit = (hash & 1) as f32;
            seq.push(bit);
            count_ones += bit as u32;
            hash >>= 1;
            if hash == 0 {
                hasher = DefaultHasher::new();
                (seed, i as u64, j).hash(&mut hasher);
                hash = hasher.finish();
            }
        }

        sequences.push(seq);
        labels.push(if count_ones % 2 == 1 { 1.0 } else { 0.0 });
        lengths.push(seq_len);
    }

    (sequences, labels, lengths)
}

/// 将变长序列填充到 `max_len`
fn pad_sequences(sequences: &[Vec<f32>], max_len: usize) -> Vec<Vec<f32>> {
    sequences
        .iter()
        .map(|seq| {
            let mut padded = seq.clone();
            padded.resize(max_len, 0.0); // padding 用 0 填充
            padded
        })
        .collect()
}

/// 生成 mask 张量（标记有效时间步）
///
/// 返回: Vec<Tensor>，每个元素是 [batch, `hidden_size`] 的 mask
/// mask[t][b, :] = 1.0 if t < lengths[b] else 0.0
fn generate_masks(lengths: &[usize], max_len: usize, hidden_size: usize) -> Vec<Tensor> {
    let batch_size = lengths.len();
    let mut masks = Vec::with_capacity(max_len);

    for t in 0..max_len {
        let mut mask_data = Vec::with_capacity(batch_size * hidden_size);
        for &len in lengths {
            let val = if t < len { 1.0 } else { 0.0 };
            // 复制到所有 hidden 维度
            for _ in 0..hidden_size {
                mask_data.push(val);
            }
        }
        masks.push(Tensor::new(&mask_data, &[batch_size, hidden_size]));
    }

    masks
}

/// 将多个序列的第 t 步组合成 batch tensor
fn get_batch_input(sequences: &[Vec<f32>], t: usize) -> Tensor {
    let batch_size = sequences.len();
    let data: Vec<f32> = sequences.iter().map(|seq| seq[t]).collect();
    Tensor::new(&data, &[batch_size, 1])
}

/// 将多个标签组合成 batch tensor
fn get_batch_labels(labels: &[f32]) -> Tensor {
    let batch_size = labels.len();
    Tensor::new(labels, &[batch_size, 1])
}

/// 创建支持变长 batch 的 RNN 网络（带 mask 状态冻结）
///
/// 关键：添加 mask 输入节点和状态冻结逻辑
/// `h_t` = mask * `h̃_t` + (1 - mask) * `h_prev`
///     = `h_prev` + mask * (`h̃_t` - `h_prev`)
fn create_varlen_parity_rnn(
    batch_size: usize,
    hidden_size: usize,
    seed: u64,
) -> Result<(Graph, NodeId, NodeId, NodeId, NodeId, NodeId, Vec<NodeId>), GraphError> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let graph = Graph::new_with_seed(seed);
    graph.train();

    // === 输入节点 ===
    let input = graph.inner_mut().new_input_node(&[batch_size, 1], Some("input"))?;
    let mask = graph.inner_mut().new_input_node(&[batch_size, hidden_size], Some("mask"))?;

    // 用于计算 (1 - mask)：先创建一个 -1 标量节点
    let neg_one_param = graph.inner_mut().new_parameter_node(&[1, 1], Some("neg_one"))?;
    graph.inner_mut().set_node_value(neg_one_param, Some(&Tensor::new(&[-1.0], &[1, 1])))?;

    // 循环状态节点
    let h_prev = graph.inner_mut().new_state_node(&[batch_size, hidden_size], Some("h_prev"))?;
    graph.inner_mut().set_node_value(h_prev, Some(&Tensor::zeros(&[batch_size, hidden_size])))?;

    // === RNN 参数（使用 Xavier 风格初始化）===
    // 简单伪随机初始化
    let mut hasher = DefaultHasher::new();

    // W_ih: [1, hidden_size]
    let w_ih = graph.inner_mut().new_parameter_node(&[1, hidden_size], Some("w_ih"))?;
    let scale_ih = (6.0 / (1.0 + hidden_size as f32)).sqrt();
    let w_ih_init: Vec<f32> = (0..hidden_size)
        .map(|i| {
            (seed, "w_ih", i).hash(&mut hasher);
            let h = hasher.finish();
            (h as f32 / u64::MAX as f32).mul_add(2.0, -1.0) * scale_ih
        })
        .collect();
    graph.inner_mut().set_node_value(w_ih, Some(&Tensor::new(&w_ih_init, &[1, hidden_size])))?;

    // W_hh: [hidden_size, hidden_size]
    let w_hh = graph.inner_mut().new_parameter_node(&[hidden_size, hidden_size], Some("w_hh"))?;
    let scale_hh = (6.0 / (hidden_size as f32 * 2.0)).sqrt();
    let w_hh_init: Vec<f32> = (0..hidden_size * hidden_size)
        .map(|i| {
            (seed, "w_hh", i).hash(&mut hasher);
            let h = hasher.finish();
            (h as f32 / u64::MAX as f32).mul_add(2.0, -1.0) * scale_hh
        })
        .collect();
    graph.inner_mut().set_node_value(
        w_hh,
        Some(&Tensor::new(&w_hh_init, &[hidden_size, hidden_size])),
    )?;

    // === 隐藏层计算（h̃_t = tanh(input * W_ih + h_prev * W_hh)）===
    let input_contrib = graph.inner_mut().new_mat_mul_node(input, w_ih, Some("input_contrib"))?;
    let hidden_contrib = graph.inner_mut().new_mat_mul_node(h_prev, w_hh, Some("hidden_contrib"))?;
    let pre_hidden = graph.inner_mut().new_add_node(&[input_contrib, hidden_contrib], Some("pre_hidden"))?;
    let h_tilde = graph.inner_mut().new_tanh_node(pre_hidden, Some("h_tilde"))?;

    // === 状态冻结：h_t = h_prev + mask * (h̃_t - h_prev) ===
    // 计算 -h_prev
    let neg_h_prev = graph.inner_mut().new_scalar_multiply_node(neg_one_param, h_prev, Some("neg_h_prev"))?;
    // 计算 delta = h̃_t + (-h_prev) = h̃_t - h_prev
    let delta = graph.inner_mut().new_add_node(&[h_tilde, neg_h_prev], Some("delta"))?;
    // 计算 mask * delta
    let masked_delta = graph.inner_mut().new_multiply_node(mask, delta, Some("masked_delta"))?;
    // 计算 hidden = h_prev + masked_delta
    let hidden = graph.inner_mut().new_add_node(&[h_prev, masked_delta], Some("hidden"))?;

    // 循环连接
    graph.inner_mut().connect_recurrent(hidden, h_prev)?;

    // === 输出层 ===
    let w_ho = graph.inner_mut().new_parameter_node(&[hidden_size, 1], Some("w_ho"))?;
    let scale_ho = (6.0 / (hidden_size as f32 + 1.0)).sqrt();
    let w_ho_init: Vec<f32> = (0..hidden_size)
        .map(|i| {
            (seed, "w_ho", i).hash(&mut hasher);
            let h = hasher.finish();
            (h as f32 / u64::MAX as f32).mul_add(2.0, -1.0) * scale_ho
        })
        .collect();
    graph.inner_mut().set_node_value(w_ho, Some(&Tensor::new(&w_ho_init, &[hidden_size, 1])))?;

    let pre_output = graph.inner_mut().new_mat_mul_node(hidden, w_ho, Some("pre_output"))?;
    let output = graph.inner_mut().new_sigmoid_node(pre_output, Some("output"))?;

    // 目标节点
    let target = graph.inner_mut().new_input_node(&[batch_size, 1], Some("target"))?;

    // MSE Loss
    let loss = graph.inner_mut().new_mse_loss_node(output, target, Some("loss"))?;

    // 可训练参数（不包含 neg_one）
    let params = vec![w_ih, w_hh, w_ho];

    Ok((graph, input, mask, output, loss, target, params))
}

/// 手动 SGD 更新
fn sgd_update(graph: &Graph, params: &[NodeId], lr: f32) -> Result<(), GraphError> {
    for &param in params {
        if let Some(param_grad) = graph.inner().get_node_grad(param)? {
            let current_value = graph.inner().get_node_value(param)?.unwrap().clone();
            let param_shape = current_value.shape();
            let grad = if param_grad.shape() == param_shape {
                param_grad.clone()
            } else {
                param_grad.reshape(param_shape)
            };
            let new_value = &current_value - &(&grad * lr);
            graph.inner_mut().set_node_value(param, Some(&new_value))?;
        }
    }
    Ok(())
}

/// 训练一个变长 batch
fn train_varlen_batch(
    graph: &Graph,
    input_node: NodeId,
    mask_node: NodeId,
    loss_node: NodeId,
    target_node: NodeId,
    params: &[NodeId],
    padded_sequences: &[Vec<f32>],
    labels: &[f32],
    masks: &[Tensor],
    lr: f32,
    debug: bool,
) -> Result<f32, GraphError> {
    let max_len = padded_sequences[0].len();

    // 重置循环状态
    graph.inner_mut().reset();

    // 设置目标值
    let batch_labels = get_batch_labels(labels);
    graph.inner_mut().set_node_value(target_node, Some(&batch_labels))?;

    // 前向传播整个序列（带 mask）
    for t in 0..max_len {
        let batch_input = get_batch_input(padded_sequences, t);
        graph.inner_mut().set_node_value(input_node, Some(&batch_input))?;
        graph.inner_mut().set_node_value(mask_node, Some(&masks[t]))?;
        graph.inner_mut().step(loss_node)?;
    }

    // 获取最终 loss 值
    let loss_value = graph
        .inner()
        .get_node_value(loss_node)?
        .ok_or_else(|| GraphError::InvalidOperation("Loss 节点没有值".to_string()))?
        .data_as_slice()[0];

    if debug {
        println!(
            "    批大小: {}, 最大长度: {}",
            padded_sequences.len(),
            max_len
        );
        println!("    损失值: {loss_value:.6}");
    }

    // BPTT
    graph.inner_mut().backward_through_time(params, loss_node)?;

    if debug {
        for (i, &param) in params.iter().enumerate() {
            let grad = graph
                .inner()
                .get_node_grad(param)?
                .map(|j| format!("{:.6}", j.data_as_slice()[0]));
            println!("    参数[{i}] 梯度: {grad:?}");
        }
    }

    // SGD 更新
    sgd_update(graph, params, lr)?;

    // 清零梯度
    graph.inner_mut().zero_grad()?;

    Ok(loss_value)
}

/// 评估准确率（变长版本）
fn evaluate_varlen_batch(
    graph: &Graph,
    input_node: NodeId,
    mask_node: NodeId,
    output_node: NodeId,
    padded_sequences: &[Vec<f32>],
    labels: &[f32],
    hidden_size: usize,
    batch_size: usize,
) -> Result<f32, GraphError> {
    let mut correct = 0;
    let total = padded_sequences.len();
    let max_len = padded_sequences[0].len();

    graph.eval();

    // 分 batch 评估
    for chunk_start in (0..total).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(total);
        let actual_batch_size = chunk_end - chunk_start;

        // 只处理完整的 batch
        if actual_batch_size != batch_size {
            continue;
        }

        let batch_seqs: Vec<Vec<f32>> = padded_sequences[chunk_start..chunk_end].to_vec();
        let batch_labels: Vec<f32> = labels[chunk_start..chunk_end].to_vec();

        // 评估时使用全 1 mask（不需要严格冻结）
        let all_ones_mask = Tensor::ones(&[batch_size, hidden_size]);

        graph.inner_mut().reset();

        // 前向传播
        for t in 0..max_len {
            let batch_input = get_batch_input(&batch_seqs, t);
            graph.inner_mut().set_node_value(input_node, Some(&batch_input))?;
            graph.inner_mut().set_node_value(mask_node, Some(&all_ones_mask))?;
            graph.inner_mut().step(output_node)?;
        }

        // 获取输出
        let output = graph.inner().get_node_value(output_node)?.unwrap();

        // 统计正确数
        for (i, &label) in batch_labels.iter().enumerate() {
            let pred_val = output[[i, 0]];
            let pred_label = if pred_val > 0.5 { 1.0 } else { 0.0 };
            if (pred_label - label).abs() < 0.01 {
                correct += 1;
            }
        }
    }

    graph.train();

    Ok(correct as f32 / (total - (total % batch_size)) as f32)
}

/// 打印变长序列的输出分布（用于调试）
fn print_varlen_output_distribution(
    graph: &Graph,
    input_node: NodeId,
    mask_node: NodeId,
    output_node: NodeId,
    padded_sequences: &[Vec<f32>],
    labels: &[f32],
    lengths: &[usize],
    hidden_size: usize,
    batch_size: usize,
) {
    graph.eval();

    let chunk_seqs: Vec<Vec<f32>> = padded_sequences[0..batch_size].to_vec();
    let chunk_labels: Vec<f32> = labels[0..batch_size].to_vec();
    let chunk_lengths: Vec<usize> = lengths[0..batch_size].to_vec();
    let max_len = chunk_seqs[0].len();

    // 生成 masks
    let masks = generate_masks(&chunk_lengths, max_len, hidden_size);

    graph.inner_mut().reset();

    for t in 0..max_len {
        let batch_input = get_batch_input(&chunk_seqs, t);
        graph
            .inner_mut()
            .set_node_value(input_node, Some(&batch_input))
            .unwrap();
        graph.inner_mut().set_node_value(mask_node, Some(&masks[t])).unwrap();
        let _ = graph.inner_mut().step(output_node);
    }

    let output = graph.inner().get_node_value(output_node).unwrap().unwrap();

    println!("  === 输出分布（前 {} 个样本）===", batch_size.min(8));
    for (i, &label) in chunk_labels.iter().take(8).enumerate() {
        let pred_val = output[[i, 0]];
        let actual_len = chunk_lengths[i];
        // 只显示实际序列内容（不含 padding）
        let seq_str: String = chunk_seqs[i][..actual_len]
            .iter()
            .map(|&x| if x > 0.5 { '1' } else { '0' })
            .collect();
        println!(
            "    序列={} (len={}), 标签={}, 输出={:.4}, 预测={}",
            seq_str,
            actual_len,
            label,
            pred_val,
            i32::from(pred_val > 0.5)
        );
    }

    graph.train();
}

/// IT-3a: 变长奇偶性检测集成测试
///
/// 验证 Padding + Mask 机制在变长序列上的正确性
/// 验收标准：达到 90% 以上测试准确率
#[test]
fn test_varlen_parity_detection_can_learn() {
    println!("\n========== IT-3a: 变长奇偶性检测集成测试 ==========\n");

    // 超参数（调优后）
    let batch_size = 16;
    let min_len = 3;
    let max_len = 8;
    let num_train = 256; // 增加训练数据
    let num_test = 64;
    let epochs = 500; // 增加训练轮数
    let lr = 0.5;
    let seed = 42u64;
    let hidden_size = 8; // 增加隐藏层大小

    // 生成变长数据
    let (train_seqs, train_labels, train_lengths) =
        generate_varlen_parity_data(num_train, min_len, max_len, seed);
    let (test_seqs, test_labels, test_lengths) =
        generate_varlen_parity_data(num_test, min_len, max_len, seed + 1000);

    // 验证数据生成正确性
    println!("[数据验证] 抽样检查前 5 个训练样本:");
    for i in 0..5 {
        let count_ones: u32 = train_seqs[i].iter().map(|&x| x as u32).sum();
        let expected = if count_ones % 2 == 1 { 1.0 } else { 0.0 };
        println!(
            "  seq={:?} (len={}), ones={}, label={}, expected={}",
            train_seqs[i], train_lengths[i], count_ones, train_labels[i], expected
        );
        assert_eq!(train_labels[i], expected, "样本 {i} 数据生成错误");
    }

    // 统计长度分布
    let mut len_counts = vec![0usize; max_len + 1];
    for &len in &train_lengths {
        len_counts[len] += 1;
    }
    println!("\n[长度分布] 训练集:");
    for len in min_len..=max_len {
        println!("  长度 {}: {} 个样本", len, len_counts[len]);
    }

    // 填充序列到 max_len
    let padded_train = pad_sequences(&train_seqs, max_len);
    let padded_test = pad_sequences(&test_seqs, max_len);

    // 创建网络
    let (graph, input, mask, output, loss, target, params) =
        create_varlen_parity_rnn(batch_size, hidden_size, seed).expect("创建网络失败");

    println!(
        "\n[网络结构] {} 个参数节点, batch_size={}, max_len={}",
        params.len(),
        batch_size,
        max_len
    );

    // 保存网络拓扑可视化
    let viz_result = graph.save_visualization("tests/outputs/it3a_parity_rnn_varlen", None);
    match &viz_result {
        Ok(output) => {
            println!("[可视化] DOT 文件: {}", output.dot_path.display());
            if let Some(img) = &output.image_path {
                println!("[可视化] 图像文件: {}", img.display());
            }
        }
        Err(e) => println!("[可视化] 跳过（{e:?}）"),
    }

    // 健壮性检查：第一个 batch 的前向/反向传播
    println!("\n[健壮性检查] 第一个 batch 前向/反向传播:");
    let first_batch_seqs: Vec<Vec<f32>> = padded_train[0..batch_size].to_vec();
    let first_batch_labels: Vec<f32> = train_labels[0..batch_size].to_vec();
    let first_batch_lengths: Vec<usize> = train_lengths[0..batch_size].to_vec();
    let first_batch_masks = generate_masks(&first_batch_lengths, max_len, hidden_size);

    let first_loss = train_varlen_batch(
        &graph,
        input,
        mask,
        loss,
        target,
        &params,
        &first_batch_seqs,
        &first_batch_labels,
        &first_batch_masks,
        lr,
        true,
    )
    .expect("第一个 batch 训练失败");
    assert!(first_loss >= 0.0, "损失值应非负");
    assert!(first_loss < 10.0, "损失值应在合理范围内");
    println!("  ✓ 变长 Batch 前向/反向传播正常");

    // 验证 mask 状态冻结机制（使用同一个网络）
    println!("\n[健壮性检查] Mask 状态冻结验证:");
    graph.inner_mut().reset();

    // 第一步：所有样本都有效 (mask=1)
    // 使用 batch 中前两个样本的简化输入来验证
    let input_t1 = Tensor::new(
        &vec![1.0; batch_size]
            .into_iter()
            .chain(std::iter::repeat_n(0.0, batch_size * 0))
            .collect::<Vec<_>>(),
        &[batch_size, 1],
    );
    let mask_t1 = Tensor::ones(&[batch_size, hidden_size]);
    graph.inner_mut().set_node_value(input, Some(&input_t1)).unwrap();
    graph.inner_mut().set_node_value(mask, Some(&mask_t1)).unwrap();
    graph.inner_mut().step(output).unwrap();
    let out_t1 = graph.inner().get_node_value(output).unwrap().unwrap().clone();

    // 第二步：样本 0 继续 (mask=1)，样本 1 结束 (mask=0)，其他样本也继续
    let input_t2 = Tensor::ones(&[batch_size, 1]);
    let mut mask_data = vec![1.0; batch_size * hidden_size]; // 默认全部有效
    // 只将样本 1 的 mask 设为 0（冻结）
    for h in 0..hidden_size {
        mask_data[hidden_size + h] = 0.0;
    }
    let mask_t2 = Tensor::new(&mask_data, &[batch_size, hidden_size]);
    graph.inner_mut().set_node_value(input, Some(&input_t2)).unwrap();
    graph.inner_mut().set_node_value(mask, Some(&mask_t2)).unwrap();
    graph.inner_mut().step(output).unwrap();
    let out_t2 = graph.inner().get_node_value(output).unwrap().unwrap().clone();

    let sample0_changed = (out_t2[[0, 0]] - out_t1[[0, 0]]).abs() > 0.001;
    let sample1_frozen = (out_t2[[1, 0]] - out_t1[[1, 0]]).abs() < 0.001;

    println!(
        "    样本 0 (mask=1): {:.4} -> {:.4} (changed: {})",
        out_t1[[0, 0]],
        out_t2[[0, 0]],
        sample0_changed
    );
    println!(
        "    样本 1 (mask=0): {:.4} -> {:.4} (frozen: {})",
        out_t1[[1, 0]],
        out_t2[[1, 0]],
        sample1_frozen
    );

    assert!(sample0_changed, "样本 0 (mask=1) 的输出应该变化");
    assert!(sample1_frozen, "样本 1 (mask=0) 的输出应该被冻结");
    println!("  ✓ Mask 状态冻结机制正常");

    // 重置网络状态，准备训练
    graph.inner_mut().reset();

    // 训练循环
    println!("\n[训练过程]");
    let mut first_epoch_loss = 0.0;
    let mut last_epoch_loss = 0.0;
    let mut best_acc = 0.0f32;
    let num_batches = num_train / batch_size;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            let batch_seqs: Vec<Vec<f32>> = padded_train[start..end].to_vec();
            let batch_labels: Vec<f32> = train_labels[start..end].to_vec();
            let batch_lengths: Vec<usize> = train_lengths[start..end].to_vec();
            let batch_masks = generate_masks(&batch_lengths, max_len, hidden_size);

            let loss_val = train_varlen_batch(
                &graph,
                input,
                mask,
                loss,
                target,
                &params,
                &batch_seqs,
                &batch_labels,
                &batch_masks,
                lr,
                false,
            )
            .expect("训练失败");

            total_loss += loss_val;
        }

        let avg_loss = total_loss / num_batches as f32;

        if epoch == 0 {
            first_epoch_loss = avg_loss;
        }
        if epoch == epochs - 1 {
            last_epoch_loss = avg_loss;
        }

        // 每 50 轮评估一次
        if (epoch + 1) % 50 == 0 {
            let acc = evaluate_varlen_batch(
                &graph,
                input,
                mask,
                output,
                &padded_test,
                &test_labels,
                hidden_size,
                batch_size,
            )
            .expect("评估失败");

            if acc > best_acc {
                best_acc = acc;
            }

            println!(
                "  Epoch {:3}: avg_loss={:.4}, test_acc={:.1}%, best={:.1}%",
                epoch + 1,
                avg_loss,
                acc * 100.0,
                best_acc * 100.0,
            );

            // 在关键时间点打印输出分布
            if epoch + 1 == 50 || epoch + 1 == epochs {
                print_varlen_output_distribution(
                    &graph,
                    input,
                    mask,
                    output,
                    &padded_test,
                    &test_labels,
                    &test_lengths,
                    hidden_size,
                    batch_size,
                );
            }
        }
    }

    // 最终评估
    let final_acc = evaluate_varlen_batch(
        &graph,
        input,
        mask,
        output,
        &padded_test,
        &test_labels,
        hidden_size,
        batch_size,
    )
    .expect("评估失败");
    if final_acc > best_acc {
        best_acc = final_acc;
    }

    // 结果汇总
    println!("\n[结果]");
    println!("  首 epoch Loss: {first_epoch_loss:.4}");
    println!("  末 epoch Loss: {last_epoch_loss:.4}");
    println!("  Loss 变化: {:.4}", first_epoch_loss - last_epoch_loss);
    println!("  最终准确率: {:.1}%", final_acc * 100.0);
    println!("  最佳准确率: {:.1}%", best_acc * 100.0);

    // 验收标准：必须达到 90% 以上准确率
    assert!(
        best_acc >= 0.90,
        "网络应达到 90% 以上准确率。最佳准确率: {:.1}%",
        best_acc * 100.0
    );

    println!(
        "\n✅ IT-3a 测试通过！变长序列达到 {:.1}% 准确率\n",
        best_acc * 100.0
    );
}
*/
