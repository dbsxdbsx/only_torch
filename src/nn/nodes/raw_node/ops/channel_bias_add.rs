/*
 * @Author       : 老董
 * @Date         : 2025-12-27
 * @Description  : ChannelBiasAdd 节点 - 用于卷积层的通道级偏置广播
 *
 * 功能：将形状为 [C] 或 [1, C] 的 bias 广播加到 [batch, C, H, W] 的输入上
 * 数学：output[b, c, h, w] = input[b, c, h, w] + bias[c]
 *
 * 设计理由：
 * - 保持显式节点（NEAT 友好）
 * - 专门处理 Conv2d 的 bias 广播，避免复杂的 reshape 操作
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

#[derive(Clone)]
pub(crate) struct ChannelBiasAdd {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    shape: Vec<usize>, // 输出形状 [batch, C, H, W]
}

impl ChannelBiasAdd {
    /// 创建 `ChannelBiasAdd` 节点
    ///
    /// # 参数
    /// - `parents[0]`: input，形状 [batch, C, H, W] 或 [C, H, W]
    /// - `parents[1]`: bias，形状 [C] 或 [1, C]
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "ChannelBiasAdd 节点需要恰好 2 个父节点（input 和 bias）".to_string(),
            ));
        }

        let input_shape = parents[0].value_expected_shape();
        let bias_shape = parents[1].value_expected_shape();

        // 2. 验证 input 形状（3D 或 4D）
        if input_shape.len() != 3 && input_shape.len() != 4 {
            return Err(GraphError::InvalidOperation(format!(
                "ChannelBiasAdd 的 input 必须是 3D [C, H, W] 或 4D [batch, C, H, W]，\
                 实际为 {input_shape:?}"
            )));
        }

        // 3. 获取通道数
        let channels = if input_shape.len() == 4 {
            input_shape[1] // [batch, C, H, W]
        } else {
            input_shape[0] // [C, H, W]
        };

        // 4. 验证 bias 形状
        let bias_channels = if bias_shape.len() == 1 {
            bias_shape[0] // [C]
        } else if bias_shape.len() == 2 && bias_shape[0] == 1 {
            bias_shape[1] // [1, C]
        } else {
            return Err(GraphError::InvalidOperation(format!(
                "ChannelBiasAdd 的 bias 必须是 [C] 或 [1, C]，实际为 {bias_shape:?}"
            )));
        };

        if channels != bias_channels {
            return Err(GraphError::ShapeMismatch {
                expected: vec![channels],
                got: vec![bias_channels],
                message: format!(
                    "ChannelBiasAdd 的 bias 通道数 ({bias_channels}) 与 input 通道数 ({channels}) 不匹配"
                ),
            });
        }

        // 5. 输出形状与 input 相同
        let shape = input_shape.to_vec();

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            shape,
        })
    }
}

impl TraitNode for ChannelBiasAdd {
    fn id(&self) -> NodeId {
        self.id.unwrap()
    }

    fn set_id(&mut self, id: NodeId) {
        self.id = Some(id);
    }

    fn name(&self) -> &str {
        self.name.as_ref().unwrap()
    }

    fn set_name(&mut self, name: &str) {
        self.name = Some(name.to_string());
    }

    fn value_expected_shape(&self) -> &[usize] {
        &self.shape
    }

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        let input = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!("{} 的 input 父节点没有值", self.display_node()))
        })?;

        let bias = parents[1].value().ok_or_else(|| {
            GraphError::ComputationError(format!("{} 的 bias 父节点没有值", self.display_node()))
        })?;

        // 获取 bias 数据
        // bias 形状为 [C] 或 [1, C]
        let bias_shape = bias.shape();
        let channels = if bias_shape.len() == 1 {
            bias_shape[0]
        } else {
            bias_shape[1]
        };

        // 克隆 input 并添加 bias
        let mut result = input.clone();
        let shape = result.shape().to_vec();

        // 获取 bias 值的辅助函数
        let get_bias = |c: usize| -> f32 {
            if bias_shape.len() == 1 {
                bias[[c]]
            } else {
                bias[[0, c]]
            }
        };

        if shape.len() == 4 {
            // [batch, C, H, W]
            let (batch, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            assert_eq!(c, channels);
            for b in 0..batch {
                for ci in 0..c {
                    let bias_val = get_bias(ci);
                    for hi in 0..h {
                        for wi in 0..w {
                            result[[b, ci, hi, wi]] += bias_val;
                        }
                    }
                }
            }
        } else {
            // [C, H, W]
            let (c, h, w) = (shape[0], shape[1], shape[2]);
            assert_eq!(c, channels);
            for ci in 0..c {
                let bias_val = get_bias(ci);
                for hi in 0..h {
                    for wi in 0..w {
                        result[[ci, hi, wi]] += bias_val;
                    }
                }
            }
        }

        self.value = Some(result);
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    // ========== VJP 模式 ==========

    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let target_shape = target_parent.value_expected_shape();

        if target_shape == self.shape.as_slice() {
            // 对 input: 梯度直接传递
            Ok(upstream_grad.clone())
        } else {
            // 对 bias: 梯度在 (batch, H, W) 维度求和
            // upstream_grad: [batch, C, H, W]，结果: [1, C] 或 [C]
            let upstream_shape = upstream_grad.shape();

            if upstream_shape.len() == 4 {
                let (batch, c, h, w) = (
                    upstream_shape[0],
                    upstream_shape[1],
                    upstream_shape[2],
                    upstream_shape[3],
                );
                let mut grad = Tensor::zeros(target_shape);

                for b in 0..batch {
                    for ci in 0..c {
                        for hi in 0..h {
                            for wi in 0..w {
                                let grad_val = upstream_grad[[b, ci, hi, wi]];
                                if target_shape.len() == 1 {
                                    grad[[ci]] += grad_val;
                                } else {
                                    grad[[0, ci]] += grad_val;
                                }
                            }
                        }
                    }
                }
                Ok(grad)
            } else {
                // 3D: [C, H, W]
                let (c, h, w) = (upstream_shape[0], upstream_shape[1], upstream_shape[2]);
                let mut grad = Tensor::zeros(target_shape);

                for ci in 0..c {
                    for hi in 0..h {
                        for wi in 0..w {
                            let grad_val = upstream_grad[[ci, hi, wi]];
                            if target_shape.len() == 1 {
                                grad[[ci]] += grad_val;
                            } else {
                                grad[[0, ci]] += grad_val;
                            }
                        }
                    }
                }
                Ok(grad)
            }
        }
    }

    fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }

    fn clear_value(&mut self) -> Result<(), GraphError> {
        self.value = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
