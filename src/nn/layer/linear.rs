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

use crate::nn::graph::NodeGroupContext;
use crate::nn::{Graph, GraphError, Init, IntoVar, Module, Var, VarMatrixOps};

// ==================== 新版 Linear 结构体（推荐）====================

/// Linear (全连接) 层
///
/// `PyTorch` 风格的全连接层：`output = x @ W + b`
///
/// # 输入/输出形状
/// - 输入：[`batch_size`, `in_features`]
/// - 输出：[`batch_size`, `out_features`]
///
/// # 使用示例
/// ```ignore
/// let fc = Linear::new(&graph, 784, 128, true, "fc1")?;
/// let h = fc.forward(&x).relu();  // 链式调用
/// ```
pub struct Linear {
    /// 权重参数 [`in_features`, `out_features`]
    weights: Var,
    /// 偏置参数 [1, `out_features`]（可选）
    bias: Option<Var>,
    /// 输入特征维度
    in_features: usize,
    /// 输出特征维度
    out_features: usize,
    /// 层名称（用于可视化分组）
    name: String,
    /// 分组实例 ID（用于可视化 cluster）
    instance_id: usize,
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
        // 如果 graph 有 model_name，自动拼接为 "ModelName/layer_name" 格式
        let full_name = match graph.model_name() {
            Some(model) => format!("{model}/{name}"),
            None => name.to_string(),
        };

        // 创建权重参数：Kaiming 初始化适合 ReLU
        let weights = graph.parameter(
            &[in_features, out_features],
            Init::Kaiming,
            &format!("{full_name}_W"),
        )?;

        // 创建偏置参数（可选）：零初始化
        let bias = if use_bias {
            Some(graph.parameter(&[1, out_features], Init::Zeros, &format!("{full_name}_b"))?)
        } else {
            None
        };

        let instance_id = graph.inner_mut().next_node_group_instance_id();

        Ok(Self {
            weights,
            bias,
            in_features,
            out_features,
            name: full_name,
            instance_id,
        })
    }

    /// 前向传播
    ///
    /// 计算 `x @ W + b`
    ///
    /// # 参数
    /// - `x`: 输入，支持 `&Tensor`、`Tensor`、`&Var`、`Var`（自动转换）
    ///
    /// # 返回
    /// 输出 Var，形状 [`batch_size`, `out_features`]
    ///
    /// # Panics
    /// 如果输入形状不匹配
    pub fn forward(&self, x: impl IntoVar) -> Var {
        // 自动将 Tensor 转为 Var（从 weights 获取 Graph）
        let x = x
            .into_var(&self.weights.get_graph())
            .expect("Linear 输入转换失败");

        // 分组上下文：自动标记 forward 期间创建的节点 + Parameter 节点
        let desc = if self.bias.is_some() {
            format!("[?, {}] → [?, {}]", self.in_features, self.out_features)
        } else {
            format!(
                "[?, {}] → [?, {}] (no bias)",
                self.in_features, self.out_features
            )
        };
        let _guard = NodeGroupContext::for_layer(&x, "Linear", self.instance_id, &self.name, &desc);
        _guard.tag_existing(&self.weights);
        if let Some(ref bias) = self.bias {
            _guard.tag_existing(bias);
        }

        // x @ W: [batch, in] @ [in, out] = [batch, out]
        let xw = x.matmul(&self.weights).expect("Linear matmul 失败");

        // 如果有 bias，直接加法（Add 支持广播：[batch, out] + [1, out]）
        if let Some(ref bias) = self.bias {
            &xw + bias
        } else {
            xw
        }
    }

    /// 获取输入特征维度
    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    /// 获取输出特征维度
    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    /// 获取权重 Var
    pub const fn weights(&self) -> &Var {
        &self.weights
    }

    /// 获取偏置 Var（如果有）
    pub const fn bias(&self) -> Option<&Var> {
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
