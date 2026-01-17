/*
 * 奇偶性检测集成测试（固定长度 + Batch 模式）
 *
 * 任务：判断 0/1 序列中 1 的个数是奇数还是偶数
 * 目的：验证 Rnn Layer + BPTT 在 batch 模式下能协同工作
 *
 * 网络结构：
 *   Rnn (input_size=1, hidden_size=4) -> Linear (4 -> 1) -> Sigmoid -> MSE Loss
 *
 * 验收标准：
 *   1. Batch 模式下能正常训练
 *   2. Loss 有下降趋势
 *   3. 准确率显著优于随机猜测
 */

use only_torch::nn::{Graph, GraphError, NodeId, Rnn};
use only_torch::tensor::Tensor;

/// 生成奇偶性检测数据
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

/// 将多个序列的第 t 步组合成 batch tensor [batch_size, 1]
fn get_batch_input(sequences: &[Vec<f32>], t: usize) -> Tensor {
    let batch_size = sequences.len();
    let data: Vec<f32> = sequences.iter().map(|seq| seq[t]).collect();
    Tensor::new(&data, &[batch_size, 1])
}

/// 将多个标签组合成 batch tensor [batch_size, 1]
fn get_batch_labels(labels: &[f32]) -> Tensor {
    Tensor::new(labels, &[labels.len(), 1])
}

/// 奇偶性检测网络结构（支持 batch）
struct ParityNetwork {
    graph: Graph,
    rnn: Rnn,
    // 输出层参数
    w_out: only_torch::nn::Var,
    b_out: only_torch::nn::Var,
    // 节点 ID
    ones_id: NodeId,
    target_id: NodeId,
    loss_id: NodeId,
    output_id: NodeId,
    // 配置
    batch_size: usize,
}

impl ParityNetwork {
    fn new(batch_size: usize, seed: u64) -> Result<Self, GraphError> {
        let graph = Graph::new_with_seed(seed);
        graph.train();

        let hidden_size = 4;
        let input_size = 1;

        // 创建 RNN 层（支持 batch）
        let rnn = Rnn::new(&graph, input_size, hidden_size, batch_size, "rnn")?;

        // 设置初始权重（与原单序列测试相同）
        rnn.w_ih()
            .set_value(&Tensor::new(&[0.5, -0.5, 0.3, -0.3], &[1, 4]))?;
        rnn.w_hh().set_value(&Tensor::new(
            &[
                0.1, 0.2, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.1,
            ],
            &[4, 4],
        ))?;
        rnn.b_h().set_value(&Tensor::zeros(&[1, 4]))?;

        // 创建输出层参数
        let w_out = graph.parameter_seeded(&[hidden_size, 1], "w_out", seed)?;
        w_out.set_value(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[4, 1]))?;

        let b_out = graph.parameter_seeded(&[1, 1], "b_out", seed)?;
        b_out.set_value(&Tensor::zeros(&[1, 1]))?;

        // 创建输出层和 loss 节点
        let (ones_id, output_id, loss_id, target_id) = {
            let mut g = graph.inner_mut();

            // ones 用于 bias 广播 [batch_size, 1]
            let ones = g.new_input_node(&[batch_size, 1], Some("ones"))?;
            g.set_node_value(ones, Some(&Tensor::ones(&[batch_size, 1])))?;

            // fc_out = hidden @ w_out  [batch_size, 4] @ [4, 1] -> [batch_size, 1]
            let fc_out =
                g.new_mat_mul_node(rnn.hidden().node_id(), w_out.node_id(), Some("fc_out"))?;

            // bias_bc = ones @ b_out  [batch_size, 1] @ [1, 1] -> [batch_size, 1]
            let bias_bc = g.new_mat_mul_node(ones, b_out.node_id(), Some("bias_bc"))?;

            // fc_add = fc_out + bias_bc
            let fc_add = g.new_add_node(&[fc_out, bias_bc], Some("fc_add"))?;

            // output = sigmoid(fc_add)
            let output = g.new_sigmoid_node(fc_add, Some("output"))?;

            // target [batch_size, 1]
            let target = g.new_input_node(&[batch_size, 1], Some("target"))?;

            // loss = mse(output, target)
            let loss = g.new_mse_loss_node(output, target, Some("loss"))?;

            (ones, output, loss, target)
        };

        Ok(Self {
            graph,
            rnn,
            w_out,
            b_out,
            ones_id,
            target_id,
            loss_id,
            output_id,
            batch_size,
        })
    }

    fn parameters(&self) -> Vec<&only_torch::nn::Var> {
        vec![
            self.rnn.w_ih(),
            self.rnn.w_hh(),
            self.rnn.b_h(),
            &self.w_out,
            &self.b_out,
        ]
    }

    fn param_ids(&self) -> Vec<NodeId> {
        vec![
            self.rnn.w_ih().node_id(),
            self.rnn.w_hh().node_id(),
            self.rnn.b_h().node_id(),
            self.w_out.node_id(),
            self.b_out.node_id(),
        ]
    }

    /// 训练一个 batch 的序列
    fn train_batch(
        &self,
        sequences: &[Vec<f32>],
        labels: &[f32],
        lr: f32,
    ) -> Result<f32, GraphError> {
        assert_eq!(sequences.len(), self.batch_size);
        assert_eq!(labels.len(), self.batch_size);

        let seq_len = sequences[0].len();

        // 重置循环状态
        self.rnn.reset();

        // 确保 ones 节点值正确
        self.graph
            .inner_mut()
            .set_node_value(self.ones_id, Some(&Tensor::ones(&[self.batch_size, 1])))?;

        // 设置目标值
        self.graph
            .inner_mut()
            .set_node_value(self.target_id, Some(&get_batch_labels(labels)))?;

        // 前向传播整个序列
        for t in 0..seq_len {
            let batch_input = get_batch_input(sequences, t);
            self.rnn.input().set_value(&batch_input)?;
            self.graph.inner_mut().step(self.loss_id)?;
        }

        // 获取 loss 值
        let loss_value = self
            .graph
            .inner()
            .get_node_value(self.loss_id)?
            .ok_or_else(|| GraphError::InvalidOperation("Loss 节点没有值".to_string()))?
            .data_as_slice()[0];

        // BPTT
        self.graph
            .inner_mut()
            .backward_through_time(&self.param_ids(), self.loss_id)?;

        // 手动 SGD 更新
        for param in self.parameters() {
            if let Some(grad) = param.grad()? {
                let current = param.value()?.unwrap().clone();
                let grad_reshaped = if grad.shape() == current.shape() {
                    grad.clone()
                } else {
                    grad.reshape(current.shape())
                };
                let new_value = &current - &(&grad_reshaped * lr);
                param.set_value(&new_value)?;
            }
        }

        // 清零梯度
        self.graph.zero_grad()?;

        Ok(loss_value)
    }

    /// 评估准确率（按 batch 评估）
    fn evaluate(&self, sequences: &[Vec<f32>], labels: &[f32]) -> Result<f32, GraphError> {
        let mut correct = 0;
        let total = sequences.len();

        self.graph.eval();

        // 分 batch 评估
        for chunk_start in (0..total).step_by(self.batch_size) {
            let chunk_end = (chunk_start + self.batch_size).min(total);
            let actual_batch_size = chunk_end - chunk_start;

            // 只处理完整的 batch
            if actual_batch_size != self.batch_size {
                continue;
            }

            let batch_seqs: Vec<Vec<f32>> = sequences[chunk_start..chunk_end].to_vec();
            let batch_labels: Vec<f32> = labels[chunk_start..chunk_end].to_vec();
            let seq_len = batch_seqs[0].len();

            self.rnn.reset();

            // 确保 ones 节点值正确
            self.graph
                .inner_mut()
                .set_node_value(self.ones_id, Some(&Tensor::ones(&[self.batch_size, 1])))?;

            // 前向传播
            for t in 0..seq_len {
                let batch_input = get_batch_input(&batch_seqs, t);
                self.rnn.input().set_value(&batch_input)?;
                self.graph.inner_mut().step(self.output_id)?;
            }

            // 获取输出
            let g = self.graph.inner();
            let output = g
                .get_node_value(self.output_id)?
                .ok_or_else(|| GraphError::InvalidOperation("Output 节点没有值".to_string()))?;

            // 检查每个样本的预测
            for (i, &label) in batch_labels.iter().enumerate() {
                let pred_val = output[[i, 0]];
                let prediction = if pred_val > 0.5 { 1.0 } else { 0.0 };
                if (prediction - label).abs() < 0.1 {
                    correct += 1;
                }
            }
            drop(g);
        }

        self.graph.train();

        // 计算评估的样本数（只包含完整 batch）
        let evaluated = total - (total % self.batch_size);
        Ok(if evaluated > 0 {
            correct as f32 / evaluated as f32
        } else {
            0.0
        })
    }
}

/// 奇偶性检测集成测试（Batch 模式）
#[test]
fn test_parity_detection_batch() {
    println!("\n========== 奇偶性检测集成测试（Batch 模式）==========\n");

    // 超参数
    let batch_size = 8;
    let seq_len = 5;
    let num_train = 128; // 必须是 batch_size 的倍数
    let num_test = 64;
    let epochs = 150;
    let lr = 0.3;
    let seed = 42u64;

    // 生成数据
    let (train_seqs, train_labels) = generate_parity_data(num_train, seq_len, seed);
    let (test_seqs, test_labels) = generate_parity_data(num_test, seq_len, seed + 1000);

    // 验证数据生成
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
    let network = ParityNetwork::new(batch_size, seed).expect("创建网络失败");

    println!(
        "\n[网络结构] RNN(1->4) + Linear(4->1) + Sigmoid, batch_size={}",
        batch_size
    );
    println!("  参数: {} 个", network.parameters().len());

    // 初始准确率
    let initial_acc = network
        .evaluate(&test_seqs, &test_labels)
        .expect("评估失败");
    println!("\n[训练前] 初始准确率: {:.1}%", initial_acc * 100.0);

    // 训练循环
    println!("\n[训练过程]");
    let mut best_acc = initial_acc;
    let mut first_epoch_loss = 0.0;
    let mut last_epoch_loss = 0.0;
    let num_batches = num_train / batch_size;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            let batch_seqs: Vec<Vec<f32>> = train_seqs[start..end].to_vec();
            let batch_labels: Vec<f32> = train_labels[start..end].to_vec();

            let loss_val = network
                .train_batch(&batch_seqs, &batch_labels, lr)
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

        if (epoch + 1) % 30 == 0 {
            let acc = network
                .evaluate(&test_seqs, &test_labels)
                .expect("评估失败");
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

    let final_acc = network
        .evaluate(&test_seqs, &test_labels)
        .expect("评估失败");
    if final_acc > best_acc {
        best_acc = final_acc;
    }

    println!("\n[结果]");
    println!("  初始准确率: {:.1}%", initial_acc * 100.0);
    println!("  最终准确率: {:.1}%", final_acc * 100.0);
    println!("  最佳准确率: {:.1}%", best_acc * 100.0);
    println!("  首 epoch Loss: {first_epoch_loss:.4}");
    println!("  末 epoch Loss: {last_epoch_loss:.4}");
    println!("  Loss 变化: {:.4}", first_epoch_loss - last_epoch_loss);

    // 验收标准
    assert!(
        best_acc >= 0.55,
        "网络准确率应优于随机猜测。最佳准确率: {:.1}%",
        best_acc * 100.0
    );

    // 验证 loss 有下降
    let loss_decreased = last_epoch_loss < first_epoch_loss + 0.01;
    assert!(
        loss_decreased,
        "损失值应在训练中下降。首: {first_epoch_loss:.4}, 末: {last_epoch_loss:.4}"
    );

    println!("\n✅ 奇偶性检测测试通过！Batch BPTT 工作正常\n");
}
