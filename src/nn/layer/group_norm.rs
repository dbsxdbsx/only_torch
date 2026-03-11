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
use crate::nn::{Graph, GraphError, Init, IntoVar, Module, Var, VarShapeOps};
use crate::tensor::Tensor;

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
    /// 创建 GroupNorm 层
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `num_groups`: 分组数（num_channels 必须能被 num_groups 整除）
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
            num_channels % num_groups == 0,
            "GroupNorm: num_channels={num_channels} 必须能被 num_groups={num_groups} 整除"
        );

        let gamma = graph.parameter(
            &[1, num_channels],
            Init::Ones,
            &format!("{name}_gamma"),
        )?;
        let beta = graph.parameter(
            &[1, num_channels],
            Init::Zeros,
            &format!("{name}_beta"),
        )?;

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
    /// 输入 [N, C, ...] → 分组归一化 → gamma * x_hat + beta
    pub fn forward(&self, x: impl IntoVar) -> Var {
        let x = x
            .into_var(&self.gamma.get_graph())
            .expect("GroupNorm 输入转换失败");

        let desc = format!("G={}, C={}", self.num_groups, self.num_channels);
        let _guard = NodeGroupContext::for_layer(
            &x,
            "GroupNorm",
            self.instance_id,
            &self.name,
            &desc,
        );
        _guard.tag_existing(&self.gamma);
        _guard.tag_existing(&self.beta);

        // 手动实现 GroupNorm：
        // 1. 获取输入 Tensor
        // 2. 纯 Tensor 计算均值/方差/归一化
        // 3. 结果包装回 Var 用于 gamma/beta 的乘加
        let x_tensor = x.value().expect("GroupNorm: 输入尚未计算").unwrap();
        let shape = x_tensor.shape();
        let ndim = shape.len();
        assert!(ndim >= 2, "GroupNorm: 输入至少 2D [N, C, ...]");
        let n = shape[0];
        let c = shape[1];
        assert_eq!(c, self.num_channels);

        let channels_per_group = c / self.num_groups;
        let spatial_size: usize = shape[2..].iter().product::<usize>().max(1);
        let group_size = channels_per_group * spatial_size;

        let flat = x_tensor.flatten_view();
        let mut x_hat_data = vec![0.0f32; x_tensor.size()];

        for b in 0..n {
            for g in 0..self.num_groups {
                // 计算均值
                let mut mean = 0.0f32;
                for ch_in_g in 0..channels_per_group {
                    let ch = g * channels_per_group + ch_in_g;
                    for s in 0..spatial_size {
                        let idx = b * c * spatial_size + ch * spatial_size + s;
                        mean += flat[idx];
                    }
                }
                mean /= group_size as f32;

                // 计算方差
                let mut var = 0.0f32;
                for ch_in_g in 0..channels_per_group {
                    let ch = g * channels_per_group + ch_in_g;
                    for s in 0..spatial_size {
                        let idx = b * c * spatial_size + ch * spatial_size + s;
                        let diff = flat[idx] - mean;
                        var += diff * diff;
                    }
                }
                var /= group_size as f32;

                let inv_std = 1.0 / (var + self.eps).sqrt();

                // 归一化
                for ch_in_g in 0..channels_per_group {
                    let ch = g * channels_per_group + ch_in_g;
                    for s in 0..spatial_size {
                        let idx = b * c * spatial_size + ch * spatial_size + s;
                        x_hat_data[idx] = (flat[idx] - mean) * inv_std;
                    }
                }
            }
        }

        let x_hat_tensor = Tensor::new(&x_hat_data, shape);
        let graph_rc = self.gamma.get_graph();
        let x_hat = graph_rc.input(&x_hat_tensor).expect("GroupNorm: 创建 x_hat 节点失败");

        // gamma/beta 形状 [1, C]，需要 reshape 以匹配输入维度
        // [1, C] → [1, C, 1, 1, ...] 用于广播（使用 Var.reshape 保持梯度链）
        let (gamma, beta) = if ndim > 2 {
            let mut param_shape = vec![1usize; ndim];
            param_shape[1] = c;
            (
                self.gamma.reshape(&param_shape).expect("GroupNorm gamma reshape 失败"),
                self.beta.reshape(&param_shape).expect("GroupNorm beta reshape 失败"),
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
