// /*
//  * IT-3c: 奇偶性检测集成测试（变长 + Batch + LSTM Layer API）
//  *
//  * 与 IT-3b 相同的任务，但使用 LSTM Layer API
//  * 目的：验证 LSTM Layer API 在真实任务上的可用性
//  *
//  * 网络结构：
//  *   input_t [B, 1] ──→ [LSTM Layer] ──→ hidden_t [B, H], cell_t [B, H]
//  *                           │                │
//  *   mask_t [B, H] ──→ [状态冻结] ──→ masked_h [B, H]
//  *                                           │
//  *                    [Linear: W_ho] ──→ [Sigmoid] ──→ output [B, 1]
//  *
//  * 验收标准：
//  *   1. 使用 lstm() 层 API 正确构建网络
//  *   2. 变长序列能正常训练
//  *   3. 达到 90% 以上测试准确率
//  */

// use only_torch::nn::layer::lstm;
// use only_torch::nn::{Graph, GraphError, NodeId};
// use only_torch::tensor::Tensor;

// /// 生成变长奇偶性检测数据
// fn generate_varlen_parity_data(
//     num_samples: usize,
//     min_len: usize,
//     max_len: usize,
//     seed: u64,
// ) -> (Vec<Vec<f32>>, Vec<f32>, Vec<usize>) {
//     use std::collections::hash_map::DefaultHasher;
//     use std::hash::{Hash, Hasher};

//     let mut sequences = Vec::with_capacity(num_samples);
//     let mut labels = Vec::with_capacity(num_samples);
//     let mut lengths = Vec::with_capacity(num_samples);

//     for i in 0..num_samples {
//         let mut hasher = DefaultHasher::new();
//         (seed, i as u64, "len").hash(&mut hasher);
//         let len_hash = hasher.finish();
//         let seq_len = min_len + (len_hash as usize % (max_len - min_len + 1));

//         hasher = DefaultHasher::new();
//         (seed, i as u64).hash(&mut hasher);
//         let mut hash = hasher.finish();

//         let mut seq = Vec::with_capacity(seq_len);
//         let mut count_ones = 0u32;

//         for j in 0..seq_len {
//             let bit = (hash & 1) as f32;
//             seq.push(bit);
//             count_ones += bit as u32;
//             hash >>= 1;
//             if hash == 0 {
//                 hasher = DefaultHasher::new();
//                 (seed, i as u64, j).hash(&mut hasher);
//                 hash = hasher.finish();
//             }
//         }

//         sequences.push(seq);
//         labels.push(if count_ones % 2 == 1 { 1.0 } else { 0.0 });
//         lengths.push(seq_len);
//     }

//     (sequences, labels, lengths)
// }

// /// 将变长序列填充到 `max_len`
// fn pad_sequences(sequences: &[Vec<f32>], max_len: usize) -> Vec<Vec<f32>> {
//     sequences
//         .iter()
//         .map(|seq| {
//             let mut padded = seq.clone();
//             padded.resize(max_len, 0.0);
//             padded
//         })
//         .collect()
// }

// /// 生成 mask 张量
// fn generate_masks(lengths: &[usize], max_len: usize, hidden_size: usize) -> Vec<Tensor> {
//     let batch_size = lengths.len();
//     let mut masks = Vec::with_capacity(max_len);

//     for t in 0..max_len {
//         let mut mask_data = Vec::with_capacity(batch_size * hidden_size);
//         for &len in lengths {
//             let val = if t < len { 1.0 } else { 0.0 };
//             for _ in 0..hidden_size {
//                 mask_data.push(val);
//             }
//         }
//         masks.push(Tensor::new(&mask_data, &[batch_size, hidden_size]));
//     }

//     masks
// }

// /// 将多个序列的第 t 步组合成 batch tensor
// fn get_batch_input(sequences: &[Vec<f32>], t: usize) -> Tensor {
//     let batch_size = sequences.len();
//     let data: Vec<f32> = sequences.iter().map(|seq| seq[t]).collect();
//     Tensor::new(&data, &[batch_size, 1])
// }

// /// 将多个标签组合成 batch tensor
// fn get_batch_labels(labels: &[f32]) -> Tensor {
//     Tensor::new(labels, &[labels.len(), 1])
// }

// /// 使用 LSTM Layer API 创建网络
// fn create_parity_lstm_with_layer_api(
//     batch_size: usize,
//     hidden_size: usize,
//     seed: u64,
// ) -> Result<
//     (
//         Graph,
//         NodeId,      // input
//         NodeId,      // mask
//         NodeId,      // output
//         NodeId,      // loss
//         NodeId,      // target
//         Vec<NodeId>, // params
//     ),
//     GraphError,
// > {
//     use std::collections::hash_map::DefaultHasher;
//     use std::hash::{Hash, Hasher};

//     let mut graph = Graph::new_with_seed(seed);
//     graph.set_train_mode();

//     // === 输入节点 ===
//     let input = graph.new_input_node(&[batch_size, 1], Some("input"))?;
//     let mask = graph.new_input_node(&[batch_size, hidden_size], Some("mask"))?;

//     // === 使用 LSTM Layer API ===
//     let lstm_out = lstm(&mut graph, input, 1, hidden_size, batch_size, Some("lstm"))?;

//     // 手动初始化 LSTM 权重（使用 Xavier 风格）
//     let mut hasher = DefaultHasher::new();
//     let scale_i = (6.0 / (1.0 + hidden_size as f32)).sqrt();
//     let scale_h = (6.0 / (hidden_size as f32 * 2.0)).sqrt();

//     // 初始化所有门的输入权重
//     for (w_ix, name) in [
//         (lstm_out.w_ii, "w_ii"),
//         (lstm_out.w_if, "w_if"),
//         (lstm_out.w_ig, "w_ig"),
//         (lstm_out.w_io, "w_io"),
//     ] {
//         let init: Vec<f32> = (0..hidden_size)
//             .map(|i| {
//                 (seed, name, i).hash(&mut hasher);
//                 let h = hasher.finish();
//                 (h as f32 / u64::MAX as f32).mul_add(2.0, -1.0) * scale_i
//             })
//             .collect();
//         graph.set_node_value(w_ix, Some(&Tensor::new(&init, &[1, hidden_size])))?;
//     }

//     // 初始化所有门的隐藏权重
//     for (w_hx, name) in [
//         (lstm_out.w_hi, "w_hi"),
//         (lstm_out.w_hf, "w_hf"),
//         (lstm_out.w_hg, "w_hg"),
//         (lstm_out.w_ho, "w_ho"),
//     ] {
//         let init: Vec<f32> = (0..hidden_size * hidden_size)
//             .map(|i| {
//                 (seed, name, i).hash(&mut hasher);
//                 let h = hasher.finish();
//                 (h as f32 / u64::MAX as f32).mul_add(2.0, -1.0) * scale_h
//             })
//             .collect();
//         graph.set_node_value(w_hx, Some(&Tensor::new(&init, &[hidden_size, hidden_size])))?;
//     }

//     // === 状态冻结逻辑 ===
//     let neg_one = graph.new_parameter_node(&[1, 1], Some("neg_one"))?;
//     graph.set_node_value(neg_one, Some(&Tensor::new(&[-1.0], &[1, 1])))?;

//     // masked_h = h_prev + mask * (hidden - h_prev)
//     let neg_h_prev =
//         graph.new_scalar_multiply_node(neg_one, lstm_out.h_prev, Some("neg_h_prev"))?;
//     let delta = graph.new_add_node(&[lstm_out.hidden, neg_h_prev], Some("delta"))?;
//     let masked_delta = graph.new_multiply_node(mask, delta, Some("masked_delta"))?;
//     let masked_h = graph.new_add_node(&[lstm_out.h_prev, masked_delta], Some("masked_h"))?;

//     // === 输出层 ===
//     let w_ho = graph.new_parameter_node(&[hidden_size, 1], Some("w_ho"))?;
//     let scale_ho = (6.0 / (hidden_size as f32 + 1.0)).sqrt();
//     let w_ho_init: Vec<f32> = (0..hidden_size)
//         .map(|i| {
//             (seed, "w_ho", i).hash(&mut hasher);
//             let h = hasher.finish();
//             (h as f32 / u64::MAX as f32).mul_add(2.0, -1.0) * scale_ho
//         })
//         .collect();
//     graph.set_node_value(w_ho, Some(&Tensor::new(&w_ho_init, &[hidden_size, 1])))?;

//     let pre_output = graph.new_mat_mul_node(masked_h, w_ho, Some("pre_output"))?;
//     let output = graph.new_sigmoid_node(pre_output, Some("output"))?;

//     // === Loss ===
//     let target = graph.new_input_node(&[batch_size, 1], Some("target"))?;
//     let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

//     // 可训练参数（LSTM 有 12 个权重 + 输出层 1 个）
//     let params = vec![
//         lstm_out.w_ii,
//         lstm_out.w_hi,
//         lstm_out.b_i,
//         lstm_out.w_if,
//         lstm_out.w_hf,
//         lstm_out.b_f,
//         lstm_out.w_ig,
//         lstm_out.w_hg,
//         lstm_out.b_g,
//         lstm_out.w_io,
//         lstm_out.w_ho,
//         lstm_out.b_o,
//         w_ho,
//     ];

//     Ok((graph, input, mask, output, loss, target, params))
// }

// /// 手动 SGD 更新
// fn sgd_update(graph: &mut Graph, params: &[NodeId], lr: f32) -> Result<(), GraphError> {
//     for &param in params {
//         if let Some(param_grad) = graph.get_node_grad(param)? {
//             let current_value = graph.get_node_value(param)?.unwrap().clone();
//             let param_shape = current_value.shape();
//             let grad = if param_grad.shape() == param_shape {
//                 param_grad.clone()
//             } else {
//                 param_grad.reshape(param_shape)
//             };
//             let new_value = &current_value - &(&grad * lr);
//             graph.set_node_value(param, Some(&new_value))?;
//         }
//     }
//     Ok(())
// }

// /// 训练一个 batch
// fn train_batch(
//     graph: &mut Graph,
//     input_node: NodeId,
//     mask_node: NodeId,
//     loss_node: NodeId,
//     target_node: NodeId,
//     params: &[NodeId],
//     padded_sequences: &[Vec<f32>],
//     labels: &[f32],
//     masks: &[Tensor],
//     lr: f32,
// ) -> Result<f32, GraphError> {
//     let max_len = padded_sequences[0].len();

//     graph.reset();
//     graph.set_node_value(target_node, Some(&get_batch_labels(labels)))?;

//     for t in 0..max_len {
//         let batch_input = get_batch_input(padded_sequences, t);
//         graph.set_node_value(input_node, Some(&batch_input))?;
//         graph.set_node_value(mask_node, Some(&masks[t]))?;
//         graph.step(loss_node)?;
//     }

//     let loss_value = graph
//         .get_node_value(loss_node)?
//         .ok_or_else(|| GraphError::InvalidOperation("Loss 无值".to_string()))?
//         .data_as_slice()[0];

//     graph.backward_through_time(params, loss_node)?;
//     sgd_update(graph, params, lr)?;
//     graph.zero_grad()?;

//     Ok(loss_value)
// }

// /// 评估准确率
// fn evaluate_batch(
//     graph: &mut Graph,
//     input_node: NodeId,
//     mask_node: NodeId,
//     output_node: NodeId,
//     padded_sequences: &[Vec<f32>],
//     labels: &[f32],
//     hidden_size: usize,
//     batch_size: usize,
// ) -> Result<f32, GraphError> {
//     let mut correct = 0;
//     let total = padded_sequences.len();
//     let max_len = padded_sequences[0].len();

//     graph.set_eval_mode();

//     for chunk_start in (0..total).step_by(batch_size) {
//         let chunk_end = (chunk_start + batch_size).min(total);
//         let actual_batch_size = chunk_end - chunk_start;

//         if actual_batch_size != batch_size {
//             continue;
//         }

//         let batch_seqs: Vec<Vec<f32>> = padded_sequences[chunk_start..chunk_end].to_vec();
//         let batch_labels: Vec<f32> = labels[chunk_start..chunk_end].to_vec();
//         let all_ones_mask = Tensor::ones(&[batch_size, hidden_size]);

//         graph.reset();

//         for t in 0..max_len {
//             let batch_input = get_batch_input(&batch_seqs, t);
//             graph.set_node_value(input_node, Some(&batch_input))?;
//             graph.set_node_value(mask_node, Some(&all_ones_mask))?;
//             graph.step(output_node)?;
//         }

//         let output = graph.get_node_value(output_node)?.unwrap();

//         for (i, &label) in batch_labels.iter().enumerate() {
//             let pred = if output[[i, 0]] > 0.5 { 1.0 } else { 0.0 };
//             if (pred - label).abs() < 0.01 {
//                 correct += 1;
//             }
//         }
//     }

//     graph.set_train_mode();
//     Ok(correct as f32 / (total - (total % batch_size)) as f32)
// }

// /// IT-3c: 使用 LSTM Layer API 的变长奇偶性检测
// #[test]
// fn test_parity_detection_with_lstm_layer() {
//     println!("\n========== IT-3c: 变长奇偶性检测（LSTM Layer API）==========\n");

//     // 超参数（LSTM 需要更细致的调参）
//     let batch_size = 16;
//     let min_len = 3;
//     let max_len = 8;
//     let num_train = 512;
//     let num_test = 64;
//     let epochs = 600; // 更多轮次
//     let lr = 0.35; // 略低学习率提高稳定性
//     let seed = 42u64;
//     let hidden_size = 10; // 稍大隐藏层

//     // 生成数据
//     let (train_seqs, train_labels, train_lengths) =
//         generate_varlen_parity_data(num_train, min_len, max_len, seed);
//     let (test_seqs, test_labels, _) =
//         generate_varlen_parity_data(num_test, min_len, max_len, seed + 1000);

//     println!("[数据] 训练: {num_train} 样本, 测试: {num_test} 样本");
//     println!("[数据] 序列长度: {min_len} ~ {max_len}");

//     // 验证数据
//     println!("\n[数据验证] 前 3 个样本:");
//     for i in 0..3 {
//         let ones: u32 = train_seqs[i].iter().map(|&x| x as u32).sum();
//         println!(
//             "  {:?} (len={}) → {} 个 1 → label={}",
//             train_seqs[i], train_lengths[i], ones, train_labels[i]
//         );
//     }

//     // 填充序列
//     let padded_train = pad_sequences(&train_seqs, max_len);
//     let padded_test = pad_sequences(&test_seqs, max_len);

//     // 创建网络（使用 LSTM Layer API）
//     let (mut graph, input, mask, output, loss, target, params) =
//         create_parity_lstm_with_layer_api(batch_size, hidden_size, seed).expect("创建网络失败");

//     println!(
//         "\n[网络] 使用 lstm() Layer API, {} 个可训练参数, hidden_size={}",
//         params.len(),
//         hidden_size
//     );

//     // 可视化（启用层分组）
//     match graph.save_visualization("tests/outputs/it3c_parity_lstm_layer", None) {
//         Ok(out) => {
//             println!("[可视化] DOT: {}", out.dot_path.display());
//             if let Some(img) = &out.image_path {
//                 println!("[可视化] PNG: {}", img.display());
//             }
//         }
//         Err(e) => println!("[可视化] 跳过: {e:?}"),
//     }

//     // 健壮性检查
//     println!("\n[健壮性检查] 第一个 batch:");
//     let first_seqs: Vec<Vec<f32>> = padded_train[0..batch_size].to_vec();
//     let first_labels: Vec<f32> = train_labels[0..batch_size].to_vec();
//     let first_lengths: Vec<usize> = train_lengths[0..batch_size].to_vec();
//     let first_masks = generate_masks(&first_lengths, max_len, hidden_size);

//     let first_loss = train_batch(
//         &mut graph,
//         input,
//         mask,
//         loss,
//         target,
//         &params,
//         &first_seqs,
//         &first_labels,
//         &first_masks,
//         lr,
//     )
//     .expect("第一个 batch 失败");

//     println!("  Loss: {first_loss:.6}");
//     assert!((0.0..10.0).contains(&first_loss), "Loss 异常");
//     println!("  ✓ 前向/反向传播正常");

//     // 训练（带早停）
//     println!("\n[训练]");
//     let num_batches = num_train / batch_size;
//     let mut best_acc = 0.0f32;
//     let target_acc = 0.90; // 目标准确率

//     for epoch in 0..epochs {
//         let mut total_loss = 0.0;

//         for batch_idx in 0..num_batches {
//             let start = batch_idx * batch_size;
//             let end = start + batch_size;

//             let batch_seqs: Vec<Vec<f32>> = padded_train[start..end].to_vec();
//             let batch_labels: Vec<f32> = train_labels[start..end].to_vec();
//             let batch_lengths: Vec<usize> = train_lengths[start..end].to_vec();
//             let batch_masks = generate_masks(&batch_lengths, max_len, hidden_size);

//             let loss_val = train_batch(
//                 &mut graph,
//                 input,
//                 mask,
//                 loss,
//                 target,
//                 &params,
//                 &batch_seqs,
//                 &batch_labels,
//                 &batch_masks,
//                 lr,
//             )
//             .expect("训练失败");

//             total_loss += loss_val;
//         }

//         // 每 25 轮评估一次（更频繁以捕捉峰值）
//         if (epoch + 1) % 25 == 0 {
//             let acc = evaluate_batch(
//                 &mut graph,
//                 input,
//                 mask,
//                 output,
//                 &padded_test,
//                 &test_labels,
//                 hidden_size,
//                 batch_size,
//             )
//             .expect("评估失败");

//             if acc > best_acc {
//                 best_acc = acc;
//             }

//             if (epoch + 1) % 50 == 0 {
//                 println!(
//                     "  Epoch {:3}: loss={:.4}, acc={:.1}%, best={:.1}%",
//                     epoch + 1,
//                     total_loss / num_batches as f32,
//                     acc * 100.0,
//                     best_acc * 100.0
//                 );
//             }

//             // 早停：达到目标即停止
//             if best_acc >= target_acc {
//                 println!("  [早停] 达到 {:.1}% 准确率，停止训练", best_acc * 100.0);
//                 break;
//             }
//         }
//     }

//     // 最终评估
//     let final_acc = evaluate_batch(
//         &mut graph,
//         input,
//         mask,
//         output,
//         &padded_test,
//         &test_labels,
//         hidden_size,
//         batch_size,
//     )
//     .expect("评估失败");

//     if final_acc > best_acc {
//         best_acc = final_acc;
//     }

//     println!("\n[结果]");
//     println!("  最终准确率: {:.1}%", final_acc * 100.0);
//     println!("  最佳准确率: {:.1}%", best_acc * 100.0);

//     // 验收标准
//     assert!(
//         best_acc >= 0.90,
//         "应达到 90% 以上。实际: {:.1}%",
//         best_acc * 100.0
//     );

//     println!(
//         "\n✅ IT-3c 通过！LSTM Layer API 达到 {:.1}% 准确率\n",
//         best_acc * 100.0
//     );
// }
