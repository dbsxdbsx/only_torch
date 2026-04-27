/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : LayerNorm 层 - PyTorch 风格 API
 *
 * 完整层归一化：y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * 内部组合：
 * - LayerNormOp 节点：归一化运算
 * - gamma (scale) 参数节点
 * - beta (shift) 参数节点
 * - Mul + Add 节点：gamma * x_hat + beta
 *
 * 输入形状任意 [*, normalized_shape]，
 * 沿最后 len(normalized_shape) 个维度归一化。
 */

use crate::nn::graph::NodeGroupContext;
use crate::nn::{Graph, GraphError, Init, IntoVar, Module, Var};

/// 层归一化
///
/// `PyTorch` 风格的层归一化：`y = gamma * (x - mean) / sqrt(var + eps) + beta`
///
/// 与 `BatchNorm` 不同，`LayerNorm` 没有 running stats，
/// 训练和推理模式行为完全一致。
///
/// # 使用示例
/// ```ignore
/// // 对最后一维 (D=64) 归一化
/// let ln = LayerNorm::new(&graph, &[64], 1e-5, "ln1")?;
/// let h = ln.forward(&x);  // x: [N, T, 64]
/// ```
pub struct LayerNorm {
    /// 缩放参数 gamma
    gamma: Var,
    /// 偏移参数 beta
    beta: Var,
    /// 归一化形状
    normalized_shape: Vec<usize>,
    /// 层名称
    name: String,
    /// 分组实例 ID
    instance_id: usize,
    /// eps
    eps: f32,
}

impl LayerNorm {
    /// 创建新的 LayerNorm 层
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
            "LayerNorm: normalized_shape 不能为空"
        );

        // 参数系统要求 2D+，1D 时前置一维
        let param_shape = if normalized_shape.len() == 1 {
            vec![1, normalized_shape[0]]
        } else {
            normalized_shape.to_vec()
        };

        // gamma 初始化为 1，beta 初始化为 0
        let gamma = graph.parameter(&param_shape, Init::Ones, &format!("{name}_gamma"))?;
        let beta = graph.parameter(&param_shape, Init::Zeros, &format!("{name}_beta"))?;

        let instance_id = graph.inner_mut().next_node_group_instance_id();

        Ok(Self {
            gamma,
            beta,
            normalized_shape: normalized_shape.to_vec(),
            name: name.to_string(),
            instance_id,
            eps,
        })
    }

    /// 前向传播
    ///
    /// 计算 `gamma * layer_norm(x) + beta`
    ///
    /// # 参数
    /// - `x`: 输入 Var，最后几维大小须与 `normalized_shape` 一致
    pub fn forward(&self, x: impl IntoVar) -> Var {
        use std::rc::Rc;
        let x = x
            .into_var(&self.gamma.get_graph())
            .expect("LayerNorm 输入转换失败");
        let graph = x.get_graph();

        // 分组上下文
        let desc = format!("shape={:?}", self.normalized_shape);
        let _guard =
            NodeGroupContext::for_layer(&x, "LayerNorm", self.instance_id, &self.name, &desc);
        _guard.tag_existing(&self.gamma);
        _guard.tag_existing(&self.beta);

        // 1. LayerNormOp: x → x_hat
        let x_hat = {
            let ln_node = graph
                .inner_mut()
                .create_layer_norm_op_node(
                    Rc::clone(x.node()),
                    self.normalized_shape.len(),
                    self.eps,
                    Some(&format!("{}_norm", self.name)),
                )
                .expect("LayerNorm 归一化失败");
            Var::new_with_rc_graph(ln_node, &graph.inner_rc())
        };

        // 2. gamma * x_hat（广播乘法）
        let scaled = &x_hat * &self.gamma;

        // 3. + beta
        &scaled + &self.beta
    }

    /// 获取归一化形状
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// 获取 gamma 参数
    pub const fn gamma(&self) -> &Var {
        &self.gamma
    }

    /// 获取 beta 参数
    pub const fn beta(&self) -> &Var {
        &self.beta
    }
}

impl Module for LayerNorm {
    fn parameters(&self) -> Vec<Var> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}
