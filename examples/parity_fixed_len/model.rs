//! 奇偶性检测模型定义（固定长度输入）
//!
//! 使用 RNN 层判断二进制序列中 1 的个数是奇数还是偶数。
//!
//! ## 任务说明
//! - 输入：长度为 N 的 0/1 序列
//! - 输出：2 类分类（偶数=类0，奇数=类1）
//!
//! ## 网络结构
//! ```text
//! x_1 → RNN → h_1
//! x_2 → RNN → h_2
//! ...
//! x_N → RNN → h_N → Linear(2) → Softmax → output
//! ```
//!
//! ## 数据格式
//! - 输入序列：`[batch, seq_len, input_size]`（batch_first=True，与 PyTorch 一致）
//! - 输出：`[batch, 2]`（2 类 logits）

use only_torch::nn::{Graph, GraphError, Linear, Module, Rnn, Var};
use only_torch::tensor::Tensor;

/// 奇偶性检测 RNN 模型
pub struct ParityRNN {
    rnn: Rnn,
    fc: Linear,
    /// 输出节点（2 类 logits）
    output: Var,
    batch_size: usize,
}

impl ParityRNN {
    /// 创建奇偶性检测模型
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `hidden_size`: RNN 隐藏层大小
    /// - `batch_size`: 批大小
    pub fn new(graph: &Graph, hidden_size: usize, batch_size: usize) -> Result<Self, GraphError> {
        // RNN: input_size=1 (单个 bit), hidden_size 由参数指定
        let rnn = Rnn::new(graph, 1, hidden_size, batch_size, "rnn")?;

        // Linear: hidden_size -> 2 (二分类：偶数/奇数)
        let fc = Linear::new(graph, hidden_size, 2, true, "fc")?;

        // 构建输出计算图: fc(rnn.hidden()) → 2 类 logits
        let output = fc.forward(rnn.hidden());

        Ok(Self {
            rnn,
            fc,
            output,
            batch_size,
        })
    }

    /// 前向传播（PyTorch 风格）
    ///
    /// 一次性处理整个序列，用户无需手动迭代时间步。
    /// 使用 `Graph::set_training_target()` 设置的目标节点。
    ///
    /// # 参数
    /// - `x`: 输入张量 `[batch, seq_len, input_size]`
    pub fn forward(&self, x: &Tensor) -> Result<&Var, GraphError> {
        self.rnn.forward(x)?;
        Ok(&self.output)
    }

    /// 获取输出节点
    pub fn output(&self) -> &Var {
        &self.output
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl Module for ParityRNN {
    fn parameters(&self) -> Vec<Var> {
        let mut params = vec![
            self.rnn.w_ih().clone(),
            self.rnn.w_hh().clone(),
            self.rnn.b_h().clone(),
        ];
        params.extend(self.fc.parameters());
        params
    }
}

/// 生成奇偶性检测数据（Tensor 格式）
///
/// # 参数
/// - `num_samples`: 样本数量
/// - `seq_len`: 序列长度
/// - `seed`: 随机种子
///
/// # 返回
/// - `sequences`: `[num_samples, seq_len, 1]` 的输入序列 Tensor
/// - `labels`: `[num_samples, 2]` 的 one-hot 标签 Tensor
pub fn generate_parity_data(num_samples: usize, seq_len: usize, seed: u64) -> (Tensor, Tensor) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut seq_data = Vec::with_capacity(num_samples * seq_len);
    let mut label_data = Vec::with_capacity(num_samples * 2);

    for i in 0..num_samples {
        // 伪随机生成
        let mut hasher = DefaultHasher::new();
        (seed, i as u64).hash(&mut hasher);
        let mut hash = hasher.finish();

        let mut count_ones = 0u32;

        for j in 0..seq_len {
            if hash == 0 {
                hasher = DefaultHasher::new();
                (seed, i as u64, j).hash(&mut hasher);
                hash = hasher.finish();
            }
            let bit = (hash & 1) as f32;
            seq_data.push(bit);
            count_ones += bit as u32;
            hash >>= 1;
        }

        // one-hot 标签: 偶数=[1,0], 奇数=[0,1]
        let is_odd = count_ones % 2 == 1;
        if is_odd {
            label_data.push(0.0);
            label_data.push(1.0);
        } else {
            label_data.push(1.0);
            label_data.push(0.0);
        }
    }

    let sequences = Tensor::new(&seq_data, &[num_samples, seq_len, 1]);
    let labels = Tensor::new(&label_data, &[num_samples, 2]);

    (sequences, labels)
}
