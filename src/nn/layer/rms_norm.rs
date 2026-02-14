/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : RMSNorm 层 - LayerNorm 的简化版
 *
 * 完整 RMS 归一化：y = gamma * x / sqrt(mean(x^2) + eps)
 *
 * 与 LayerNorm 的区别：
 * - 不减均值（no mean centering）
 * - 不含 beta（无偏移参数）
 * - 计算更快
 *
 * 常用于 LLaMA、Gemma 等现代 Transformer 架构。
 */

use crate::nn::graph::NodeGroupContext;
use crate::nn::{Graph, GraphError, Init, IntoVar, Module, Var};

/// RMS 归一化层
///
/// `y = gamma * x / sqrt(mean(x^2) + eps)`
///
/// # 使用示例
/// ```ignore
/// let rn = RMSNorm::new(&graph, &[64], 1e-5, "rn1")?;
/// let h = rn.forward(&x);  // x: [N, T, 64]
/// ```
pub struct RMSNorm {
    /// 缩放参数 gamma
    gamma: Var,
    /// 归一化形状
    normalized_shape: Vec<usize>,
    /// 层名称
    name: String,
    /// 分组实例 ID
    instance_id: usize,
    /// eps
    eps: f32,
}

impl RMSNorm {
    /// 创建新的 RMSNorm 层
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `normalized_shape`: 归一化的形状（输入的最后几个维度）
    /// - `eps`: 数值稳定性常数（典型值 1e-5）
    /// - `name`: 层名称前缀
    pub fn new(
        graph: &Graph,
        normalized_shape: &[usize],
        eps: f32,
        name: &str,
    ) -> Result<Self, GraphError> {
        assert!(
            !normalized_shape.is_empty(),
            "RMSNorm: normalized_shape 不能为空"
        );

        let param_shape = if normalized_shape.len() == 1 {
            vec![1, normalized_shape[0]]
        } else {
            normalized_shape.to_vec()
        };

        let gamma =
            graph.parameter(&param_shape, Init::Ones, &format!("{name}_gamma"))?;

        let instance_id = graph.inner_mut().next_node_group_instance_id();

        Ok(Self {
            gamma,
            normalized_shape: normalized_shape.to_vec(),
            name: name.to_string(),
            instance_id,
            eps,
        })
    }

    /// 前向传播
    pub fn forward(&self, x: impl IntoVar) -> Var {
        use std::rc::Rc;
        let x = x
            .into_var(&self.gamma.get_graph())
            .expect("RMSNorm 输入转换失败");
        let graph = x.get_graph();

        let desc = format!("shape={:?}", self.normalized_shape);
        let _guard =
            NodeGroupContext::for_layer(&x, "RMSNorm", self.instance_id, &self.name, &desc);
        _guard.tag_existing(&self.gamma);

        // RMSNormOp: x → x_hat
        let x_hat = {
            let rn_node = graph
                .inner_mut()
                .create_rms_norm_op_node(
                    Rc::clone(x.node()),
                    self.normalized_shape.len(),
                    self.eps,
                    Some(&format!("{}_norm", self.name)),
                )
                .expect("RMSNorm 归一化失败");
            Var::new_with_rc_graph(rn_node, &graph.inner_rc())
        };

        // gamma * x_hat
        &x_hat * &self.gamma
    }

    /// 获取 gamma 参数
    pub const fn gamma(&self) -> &Var {
        &self.gamma
    }
}

impl Module for RMSNorm {
    fn parameters(&self) -> Vec<Var> {
        vec![self.gamma.clone()]
    }
}
