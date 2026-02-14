/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : Embedding 层 - 词嵌入查找表
 *
 * 本质是查表操作：根据整数索引从权重矩阵中选取对应行。
 * 内部使用 Gather 节点实现，自动支持梯度回传。
 *
 * 权重矩阵形状：[vocab_size, embed_dim]
 * 输入索引：[N] 或 [N, T]（整数值，0-based）
 * 输出：[N, embed_dim] 或 [N, T, embed_dim]
 */

use crate::nn::graph::NodeGroupContext;
use crate::nn::{Graph, GraphError, Init, Module, Var, VarShapeOps};
use crate::tensor::Tensor;

/// 词嵌入层
///
/// 将整数索引映射为稠密向量表示。
///
/// # 使用示例
/// ```ignore
/// let emb = Embedding::new(&graph, 10000, 256, "word_emb")?;
/// let indices = graph.input(&Tensor::new(&[0., 5., 3.], &[1, 3]))?;
/// let vectors = emb.forward(&indices);
/// // vectors: [1, 3, 256]
/// ```
pub struct Embedding {
    /// 嵌入权重矩阵 [vocab_size, embed_dim]
    weight: Var,
    /// 词表大小
    vocab_size: usize,
    /// 嵌入维度
    embed_dim: usize,
    /// 层名称
    name: String,
    /// 分组实例 ID
    instance_id: usize,
}

impl Embedding {
    /// 创建新的 Embedding 层
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `vocab_size`: 词表大小（索引范围 0..vocab_size）
    /// - `embed_dim`: 嵌入向量维度
    /// - `name`: 层名称前缀
    pub fn new(
        graph: &Graph,
        vocab_size: usize,
        embed_dim: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        assert!(vocab_size > 0, "Embedding: vocab_size 必须 > 0");
        assert!(embed_dim > 0, "Embedding: embed_dim 必须 > 0");

        // 权重初始化为标准正态分布
        let weight = graph.parameter(
            &[vocab_size, embed_dim],
            Init::Normal { mean: 0.0, std: 1.0 },
            &format!("{name}_weight"),
        )?;

        let instance_id = graph.inner_mut().next_node_group_instance_id();

        Ok(Self {
            weight,
            vocab_size,
            embed_dim,
            name: name.to_string(),
            instance_id,
        })
    }

    /// 前向传播 — 查找嵌入向量
    ///
    /// # 参数
    /// - `indices`: 整数索引 Var/Tensor，形状 [N] 或 [N, T]，值域 [0, vocab_size)
    ///
    /// # 返回
    /// - [N, embed_dim]（1D 索引）或 [N, T, embed_dim]（2D 索引）
    pub fn forward(&self, indices: &Var) -> Var {
        let _graph = self.weight.get_graph();

        // 分组上下文
        let desc = format!("V={}, D={}", self.vocab_size, self.embed_dim);
        let _guard = NodeGroupContext::for_layer(
            indices,
            "Embedding",
            self.instance_id,
            &self.name,
            &desc,
        );
        _guard.tag_existing(&self.weight);

        // 获取索引数据（整数值）
        let idx_tensor = indices.value().expect("Embedding: 索引尚未计算").unwrap();
        let idx_shape = idx_tensor.shape();

        // 将索引展平为 [total_indices]
        let flat_idx = idx_tensor.flatten();
        let total = flat_idx.size();

        // 构建 Gather 用的 2D 索引 [total_indices, embed_dim]
        // 每行重复相同的索引值
        let flat_data = flat_idx.flatten_view();
        let mut gather_idx_data = Vec::with_capacity(total * self.embed_dim);
        for i in 0..total {
            let idx_val = flat_data[i] as usize;
            assert!(
                idx_val < self.vocab_size,
                "Embedding: 索引 {idx_val} 超出 vocab_size {}",
                self.vocab_size
            );
            for _ in 0..self.embed_dim {
                gather_idx_data.push(idx_val as f32);
            }
        }
        let gather_idx = Tensor::new(&gather_idx_data, &[total, self.embed_dim]);

        // 使用 Gather(dim=0) 从 weight[V, D] 中选取行
        let gathered = self
            .weight
            .gather(0, &gather_idx)
            .expect("Embedding: gather 失败");

        // 将输出 reshape 为目标形状
        match idx_shape.len() {
            1 => {
                // [N] → gathered 已是 [N, embed_dim]
                gathered
            }
            2 => {
                // [N, T] → [total, D] → [N, T, D]
                let n = idx_shape[0];
                let t = idx_shape[1];
                gathered
                    .reshape(&[n, t, self.embed_dim])
                    .expect("Embedding: reshape 失败")
            }
            _ => panic!(
                "Embedding: 索引应为 1D [N] 或 2D [N, T]，得到 {}D",
                idx_shape.len()
            ),
        }
    }

    /// 获取嵌入权重
    pub const fn weight(&self) -> &Var {
        &self.weight
    }

    /// 获取词表大小
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// 获取嵌入维度
    pub const fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

impl Module for Embedding {
    fn parameters(&self) -> Vec<Var> {
        vec![self.weight.clone()]
    }
}
