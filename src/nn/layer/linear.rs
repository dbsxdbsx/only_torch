/*
 * @Author       : 老董
 * @Date         : 2026-01-17
 * @Description  : Linear (全连接) 层 - PyTorch 风格 API
 *
 * 输入/输出形状：
 * - 输入：[batch_size, in_features]
 * - 输出：[batch_size, out_features]
 *
 * 计算：output = x @ W + b
 */

use crate::nn::{Graph, GraphError, Init, Module, Var, VarMatrixOps};

// ==================== 新版 Linear 结构体（推荐）====================

/// Linear (全连接) 层
///
/// PyTorch 风格的全连接层：`output = x @ W + b`
///
/// # 输入/输出形状
/// - 输入：[batch_size, in_features]
/// - 输出：[batch_size, out_features]
///
/// # 使用示例
/// ```ignore
/// let fc = Linear::new(&graph, 784, 128, true, "fc1")?;
/// let h = fc.forward(&x).relu();  // 链式调用
/// ```
pub struct Linear {
    /// 权重参数 [in_features, out_features]
    weights: Var,
    /// 偏置参数 [1, out_features]（可选）
    bias: Option<Var>,
    /// 输入特征维度
    in_features: usize,
    /// 输出特征维度
    out_features: usize,
    /// 层名称（用于可视化分组）
    name: String,
}

impl Linear {
    /// 创建新的 Linear 层
    ///
    /// # 参数
    /// - `graph`: 计算图句柄
    /// - `in_features`: 输入特征维度
    /// - `out_features`: 输出特征维度
    /// - `use_bias`: 是否使用偏置
    /// - `name`: 层名称前缀
    ///
    /// # 返回
    /// Linear 层实例
    pub fn new(
        graph: &Graph,
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        name: &str,
    ) -> Result<Self, GraphError> {
        // 创建权重参数：Kaiming 初始化适合 ReLU
        let weights = graph.parameter(
            &[in_features, out_features],
            Init::Kaiming,
            &format!("{name}_W"),
        )?;

        // 创建偏置参数（可选）：零初始化
        let bias = if use_bias {
            Some(graph.parameter(&[1, out_features], Init::Zeros, &format!("{name}_b"))?)
        } else {
            None
        };

        Ok(Self {
            weights,
            bias,
            in_features,
            out_features,
            name: name.to_string(),
        })
    }

    /// 创建新的 Linear 层（带种子，确保可重复性）
    ///
    /// # 参数
    /// - `graph`: 计算图句柄
    /// - `in_features`: 输入特征维度
    /// - `out_features`: 输出特征维度
    /// - `use_bias`: 是否使用偏置
    /// - `name`: 层名称前缀
    /// - `seed`: 随机种子
    ///
    /// # 返回
    /// Linear 层实例
    pub fn new_seeded(
        graph: &Graph,
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        name: &str,
        seed: u64,
    ) -> Result<Self, GraphError> {
        // 创建权重参数：使用固定种子初始化
        let weights =
            graph.parameter_seeded(&[in_features, out_features], &format!("{name}_W"), seed)?;

        // 创建偏置参数（可选）：零初始化（无需种子）
        let bias = if use_bias {
            Some(graph.parameter(&[1, out_features], Init::Zeros, &format!("{name}_b"))?)
        } else {
            None
        };

        Ok(Self {
            weights,
            bias,
            in_features,
            out_features,
            name: name.to_string(),
        })
    }

    /// 前向传播
    ///
    /// 计算 `x @ W + b`
    ///
    /// # 参数
    /// - `x`: 输入 Var，形状 [batch_size, in_features]
    ///
    /// # 返回
    /// 输出 Var，形状 [batch_size, out_features]
    ///
    /// # Panics
    /// 如果输入形状不匹配
    pub fn forward(&self, x: &Var) -> Var {
        // x @ W: [batch, in] @ [in, out] = [batch, out]
        let xw = x.matmul(&self.weights).expect("Linear matmul 失败");

        // 如果有 bias，需要广播加法
        if let Some(ref bias) = self.bias {
            // 获取 batch_size
            let x_shape = x.value_expected_shape();
            let batch_size = x_shape[0];

            // 使用 Graph handle 创建 ones 矩阵用于 bias 广播
            let graph = x.get_graph();
            let ones = graph.ones(&[batch_size, 1]).expect("创建 ones 节点失败");

            // ones @ b: [batch, 1] @ [1, out] = [batch, out]
            let bias_broadcast = ones.matmul(bias).expect("bias broadcast matmul 失败");

            // output + bias
            let output = &xw + &bias_broadcast;

            // 注册层分组（用于可视化）
            graph.inner_mut().register_layer_group(
                &self.name,
                "Linear",
                &format!("{}→{}", self.in_features, self.out_features),
                vec![
                    self.weights.node_id(),
                    bias.node_id(),
                    ones.node_id(),
                    xw.node_id(),
                    bias_broadcast.node_id(),
                    output.node_id(),
                ],
            );

            output
        } else {
            // 无 bias 时，直接返回 xw
            let graph = x.get_graph();
            graph.inner_mut().register_layer_group(
                &self.name,
                "Linear",
                &format!("{}→{} (no bias)", self.in_features, self.out_features),
                vec![self.weights.node_id(), xw.node_id()],
            );
            xw
        }
    }

    /// 获取输入特征维度
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// 获取输出特征维度
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// 获取权重 Var
    pub fn weights(&self) -> &Var {
        &self.weights
    }

    /// 获取偏置 Var（如果有）
    pub fn bias(&self) -> Option<&Var> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Var> {
        let mut params = vec![self.weights.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}
