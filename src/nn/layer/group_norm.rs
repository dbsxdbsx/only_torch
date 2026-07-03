/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : GroupNorm 层 - 分组归一化
 *
 * 将通道分为 num_groups 组，每组独立归一化。
 * GroupNorm(1, C) ≈ LayerNorm（沿通道归一化）
 * GroupNorm(C, C) = InstanceNorm（每通道独立归一化）
 *
 * 输入 [N, C, ...], 输出同形。
 */

use crate::nn::graph::NodeGroupContext;
use crate::nn::{
    Graph, GraphError, Init, IntoVar, Module, Var, VarActivationOps, VarReduceOps, VarShapeOps,
};

/// 分组归一化层
///
/// # 使用示例
/// ```ignore
/// let gn = GroupNorm::new(&graph, 8, 32, 1e-5, "gn")?;
/// let h = gn.forward(&x);  // x: [N, 32, H, W]
/// ```
pub struct GroupNorm {
    gamma: Var,
    beta: Var,
    num_groups: usize,
    num_channels: usize,
    eps: f32,
    name: String,
    instance_id: usize,
}

impl GroupNorm {
    /// 创建 `GroupNorm` 层
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `num_groups`: `分组数（num_channels` 必须能被 `num_groups` 整除）
    /// - `num_channels`: 通道数
    /// - `eps`: 数值稳定性常数
    /// - `name`: 层名称
    pub fn new(
        graph: &Graph,
        num_groups: usize,
        num_channels: usize,
        eps: f32,
        name: &str,
    ) -> Result<Self, GraphError> {
        assert!(
            num_channels.is_multiple_of(num_groups),
            "GroupNorm: num_channels={num_channels} 必须能被 num_groups={num_groups} 整除"
        );

        let gamma = graph.parameter(&[1, num_channels], Init::Ones, &format!("{name}_gamma"))?;
        let beta = graph.parameter(&[1, num_channels], Init::Zeros, &format!("{name}_beta"))?;

        let instance_id = graph.inner_mut().next_node_group_instance_id();

        Ok(Self {
            gamma,
            beta,
            num_groups,
            num_channels,
            eps,
            name: name.to_string(),
            instance_id,
        })
    }

    /// 前向传播
    ///
    /// 输入 [N, C, ...] → 分组归一化 → gamma * `x_hat` + beta
    pub fn forward(&self, x: impl IntoVar) -> Var {
        let x = x
            .into_var(&self.gamma.get_graph())
            .expect("GroupNorm 输入转换失败");

        let desc = format!("G={}, C={}", self.num_groups, self.num_channels);
        let _guard =
            NodeGroupContext::for_layer(&x, "GroupNorm", self.instance_id, &self.name, &desc);
        _guard.tag_existing(&self.gamma);
        _guard.tag_existing(&self.beta);

        // 可微分组合实现（全部走图内节点，梯度正确回传到 x）：
        // [N, C, ...] → reshape [N·G, group_size] → 逐行 (x−mean)/sqrt(var+eps)
        // → reshape 回原形 → gamma·x_hat + beta。
        // 旧实现曾在图外用纯 Tensor 计算 x_hat 再包回 input 节点，导致输入侧梯度
        // 被整体截断（gamma/beta 有梯度、上游层永远学不到）——已重写修复。
        let shape = x.value_expected_shape();
        let ndim = shape.len();
        assert!(ndim >= 2, "GroupNorm: 输入至少 2D [N, C, ...]");
        let n = shape[0];
        let c = shape[1];
        assert_eq!(c, self.num_channels);

        let channels_per_group = c / self.num_groups;
        let spatial_size: usize = shape[2..].iter().product::<usize>().max(1);
        let group_size = channels_per_group * spatial_size;

        // [N·G, group_size]：每行 = 一个 (样本, 组)
        let x2 = x
            .reshape(&[n * self.num_groups, group_size])
            .expect("GroupNorm: 分组 reshape 失败");
        let mean = x2.mean_axis(1); // [N·G, 1]（keepdims）
        let mean_b = mean
            .repeat(&[1, group_size])
            .expect("GroupNorm: mean 广播失败");
        let centered = &x2 - &mean_b;
        // biased 方差（与 PyTorch GroupNorm 一致）
        let var = centered.square().mean_axis(1); // [N·G, 1]
        let std = (var + self.eps).sqrt();
        let std_b = std
            .repeat(&[1, group_size])
            .expect("GroupNorm: std 广播失败");
        let x_hat = (&centered / &std_b)
            .reshape(&shape)
            .expect("GroupNorm: 还原 reshape 失败");

        // gamma/beta 形状 [1, C]，需要 reshape 以匹配输入维度
        // [1, C] → [1, C, 1, 1, ...] 用于广播（使用 Var.reshape 保持梯度链）
        let (gamma, beta) = if ndim > 2 {
            let mut param_shape = vec![1usize; ndim];
            param_shape[1] = c;
            (
                self.gamma
                    .reshape(&param_shape)
                    .expect("GroupNorm gamma reshape 失败"),
                self.beta
                    .reshape(&param_shape)
                    .expect("GroupNorm beta reshape 失败"),
            )
        } else {
            (self.gamma.clone(), self.beta.clone())
        };
        &(&x_hat * &gamma) + &beta
    }
}

impl Module for GroupNorm {
    fn parameters(&self) -> Vec<Var> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}
