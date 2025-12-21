/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : 2D 最大池化节点（PyTorch 风格）
 *
 * 设计决策：
 * - 记录最大值位置用于反向传播（稀疏梯度）
 * - 支持 Jacobi 模式（单样本）和 Batch 模式
 * - 输入格式：[C, H, W] 或 [batch, C, H, W]
 * - 输出格式：[C, H', W'] 或 [batch, C, H', W']
 *
 * 父节点：
 * - parents[0]: 输入数据
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

/// 2D 最大池化节点
#[derive(Clone)]
pub(crate) struct MaxPool2d {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    grad: Option<Tensor>,
    shape: Vec<usize>, // 输出形状

    // 池化参数
    kernel_size: (usize, usize), // (kH, kW)
    stride: (usize, usize),      // (sH, sW)

    // 缓存（用于反向传播）
    // 存储每个输出位置对应的最大值在输入中的索引
    // 形状与输出相同，值为展平后的输入索引
    max_indices: Option<Tensor>,
    input_shape: Vec<usize>, // 原始输入形状
}

impl MaxPool2d {
    /// 创建 MaxPool2d 节点
    ///
    /// # 参数
    /// - `parents`: [输入节点]
    /// - `kernel_size`: 池化窗口大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)，默认等于 kernel_size
    ///
    /// # 输入形状约定
    /// - 输入: [C, H, W] 或 [batch, C, H, W]
    pub(crate) fn new(
        parents: &[&NodeHandle],
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "MaxPool2d 节点需要 1 个父节点".to_string(),
            ));
        }

        let input_shape = parents[0].value_expected_shape();

        // 2. 验证输入形状：3D [C, H, W] 或 4D [batch, C, H, W]
        let (batch_size, channels, input_h, input_w) = match input_shape.len() {
            3 => (None, input_shape[0], input_shape[1], input_shape[2]),
            4 => (
                Some(input_shape[0]),
                input_shape[1],
                input_shape[2],
                input_shape[3],
            ),
            _ => {
                return Err(GraphError::ShapeMismatch {
                    expected: vec![0, 0, 0],
                    got: input_shape.to_vec(),
                    message: format!(
                        "MaxPool2d 输入必须是 3D [C, H, W] 或 4D [batch, C, H, W]，得到 {:?}",
                        input_shape
                    ),
                });
            }
        };

        let (k_h, k_w) = kernel_size;
        let (s_h, s_w) = stride.unwrap_or(kernel_size); // 默认 stride = kernel_size

        // 3. 验证池化窗口不超过输入尺寸
        if k_h > input_h || k_w > input_w {
            return Err(GraphError::InvalidOperation(format!(
                "MaxPool2d 池化窗口 {}x{} 超出输入尺寸 {}x{}",
                k_h, k_w, input_h, input_w
            )));
        }

        // 4. 计算输出尺寸
        let output_h = (input_h - k_h) / s_h + 1;
        let output_w = (input_w - k_w) / s_w + 1;

        if output_h == 0 || output_w == 0 {
            return Err(GraphError::InvalidOperation(format!(
                "MaxPool2d 输出尺寸无效：输入 {}x{}，核 {}x{}，步长 {:?}",
                input_h, input_w, k_h, k_w, (s_h, s_w)
            )));
        }

        // 5. 确定输出形状
        let output_shape = match batch_size {
            Some(b) => vec![b, channels, output_h, output_w],
            None => vec![channels, output_h, output_w],
        };

        Ok(Self {
            id: None,
            name: None,
            value: None,
            jacobi: None,
            grad: None,
            shape: output_shape,
            kernel_size,
            stride: (s_h, s_w),
            max_indices: None,
            input_shape: input_shape.to_vec(),
        })
    }
}

impl TraitNode for MaxPool2d {
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
            GraphError::ComputationError(format!(
                "{}的输入父节点{}没有值",
                self.display_node(),
                parents[0]
            ))
        })?;

        let input_shape = input.shape();
        let is_batch = input_shape.len() == 4;

        let (batch_size, channels, in_h, in_w) = if is_batch {
            (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            (1, input_shape[0], input_shape[1], input_shape[2])
        };

        let (k_h, k_w) = self.kernel_size;
        let (s_h, s_w) = self.stride;
        let out_h = (in_h - k_h) / s_h + 1;
        let out_w = (in_w - k_w) / s_w + 1;

        // 输出形状
        let output_shape = if is_batch {
            vec![batch_size, channels, out_h, out_w]
        } else {
            vec![channels, out_h, out_w]
        };

        let mut output = Tensor::zeros(&output_shape);
        // 存储最大值索引（用于反向传播）
        let mut max_indices = Tensor::zeros(&output_shape);

        // 执行最大池化
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let h_start = oh * s_h;
                        let w_start = ow * s_w;

                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx: usize = 0;

                        // 在池化窗口中找最大值
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = h_start + kh;
                                let iw = w_start + kw;

                                let val = if is_batch {
                                    input[[b, c, ih, iw]]
                                } else {
                                    input[[c, ih, iw]]
                                };

                                if val > max_val {
                                    max_val = val;
                                    // 记录在输入特征图中的位置（相对于当前通道）
                                    max_idx = ih * in_w + iw;
                                }
                            }
                        }

                        if is_batch {
                            output[[b, c, oh, ow]] = max_val;
                            max_indices[[b, c, oh, ow]] = max_idx as f32;
                        } else {
                            output[[c, oh, ow]] = max_val;
                            max_indices[[c, oh, ow]] = max_idx as f32;
                        }
                    }
                }
            }
        }

        self.value = Some(output);
        self.max_indices = Some(max_indices);
        self.input_shape = input_shape.to_vec();

        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// 计算 Jacobi 矩阵（单样本模式）
    ///
    /// MaxPool 的 Jacobi 矩阵非常稀疏：
    /// - 每个输出只依赖一个输入（最大值位置）
    /// - 该位置导数为 1，其他为 0
    fn calc_jacobi_to_a_parent(
        &self,
        _target_parent: &NodeHandle,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let max_indices = self.max_indices.as_ref().ok_or_else(|| {
            GraphError::ComputationError("缺少最大值索引缓存".to_string())
        })?;

        let input_shape = &self.input_shape;
        let is_batch = input_shape.len() == 4;

        // 单样本 Jacobi 只支持非 batch 模式
        if is_batch {
            return Err(GraphError::InvalidOperation(
                "Jacobi 模式不支持 batch 输入，请使用 calc_grad_to_parent".to_string(),
            ));
        }

        let (channels, in_h, in_w) = (input_shape[0], input_shape[1], input_shape[2]);
        let (out_h, out_w) = (self.shape[1], self.shape[2]);

        let output_dim = channels * out_h * out_w;
        let input_dim = channels * in_h * in_w;

        let mut jacobi = Tensor::zeros(&[output_dim, input_dim]);

        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let out_idx = c * out_h * out_w + oh * out_w + ow;
                    let max_pos = max_indices[[c, oh, ow]] as usize;
                    // 输入索引 = channel_offset + max_pos
                    let in_idx = c * in_h * in_w + max_pos;
                    jacobi[[out_idx, in_idx]] = 1.0;
                }
            }
        }

        Ok(jacobi)
    }

    fn jacobi(&self) -> Option<&Tensor> {
        self.jacobi.as_ref()
    }

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.jacobi = jacobi.cloned();
        Ok(())
    }

    // ========== Batch 模式 ==========

    /// 计算 Batch 梯度
    ///
    /// MaxPool 的梯度非常简单：
    /// - 最大值位置：梯度 = upstream_grad
    /// - 其他位置：梯度 = 0
    fn calc_grad_to_parent(
        &self,
        _target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let max_indices = self.max_indices.as_ref().ok_or_else(|| {
            GraphError::ComputationError("缺少最大值索引缓存".to_string())
        })?;

        let input_shape = &self.input_shape;
        let grad_shape = upstream_grad.shape();
        let is_batch = grad_shape.len() == 4;

        let (batch_size, channels, out_h, out_w) = if is_batch {
            (grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3])
        } else {
            (1, grad_shape[0], grad_shape[1], grad_shape[2])
        };

        let (_in_h, in_w) = if is_batch {
            (input_shape[2], input_shape[3])
        } else {
            (input_shape[1], input_shape[2])
        };

        let mut input_grad = Tensor::zeros(input_shape);

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let grad_val = if is_batch {
                            upstream_grad[[b, c, oh, ow]]
                        } else {
                            upstream_grad[[c, oh, ow]]
                        };

                        let max_pos = if is_batch {
                            max_indices[[b, c, oh, ow]] as usize
                        } else {
                            max_indices[[c, oh, ow]] as usize
                        };

                        // 将梯度传递到最大值位置
                        let ih = max_pos / in_w;
                        let iw = max_pos % in_w;

                        if is_batch {
                            input_grad[[b, c, ih, iw]] += grad_val;
                        } else {
                            input_grad[[c, ih, iw]] += grad_val;
                        }
                    }
                }
            }
        }

        Ok(input_grad)
    }

    fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }
}

