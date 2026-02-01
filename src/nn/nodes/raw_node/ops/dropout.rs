/*
 * Dropout 正则化节点
 *
 * 训练时：以概率 p 随机将输入元素置零，并将剩余元素缩放 1/(1-p)
 * 评估时：直接通过（identity）
 *
 * 使用 Inverted Dropout 策略：
 * - 训练时缩放，评估时不变
 * - 这样评估时不需要额外操作
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Dropout 默认丢弃概率
pub const DEFAULT_DROPOUT_P: f32 = 0.5;

/// Dropout 正则化节点
///
/// 训练时随机丢弃部分神经元，防止过拟合。
/// 使用 Inverted Dropout 策略：训练时缩放，评估时直接通过。
///
/// # 公式
/// - 训练：`output = input * mask / (1 - p)`，其中 mask 是随机 0/1 张量
/// - 评估：`output = input`
///
/// # `PyTorch` 对应
/// ```python
/// import torch.nn as nn
/// dropout = nn.Dropout(p=0.5)
/// ```
#[derive(Clone)]
pub(crate) struct Dropout {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    supports_dynamic: bool,
    /// 丢弃概率（0.0 ~ 1.0）
    p: f32,
    /// 当前是否处于训练模式
    is_training: bool,
    /// 随机数生成器（用于生成 mask）
    rng: StdRng,
    /// 缓存的 mask（用于反向传播）
    /// 训练时：mask 中 1 表示保留，0 表示丢弃
    mask: Option<Tensor>,
}

impl Dropout {
    /// 获取丢弃概率
    pub(crate) const fn p(&self) -> f32 {
        self.p
    }

    /// 从父节点形状信息创建 Dropout 节点（核心实现）
    pub(in crate::nn) fn new_from_shapes(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        p: f32,
        seed: u64,
    ) -> Result<Self, GraphError> {
        if !(0.0..1.0).contains(&p) {
            return Err(GraphError::InvalidOperation(format!(
                "Dropout 的 p 必须在 [0, 1) 范围内，但得到: {p}"
            )));
        }

        let supports_dynamic = parent_dynamic_shape.dims().first() == Some(&None);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: parent_shape.to_vec(),
            dynamic_shape: parent_dynamic_shape.clone(),
            supports_dynamic,
            p,
            is_training: true,
            rng: StdRng::seed_from_u64(seed),
            mask: None,
        })
    }

    /// 从 NodeHandle 创建（过渡期 API，委托给 new_from_shapes）
    pub(crate) fn new(parents: &[&NodeHandle], p: f32, seed: u64) -> Result<Self, GraphError> {
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Dropout 节点只需要 1 个父节点".to_string(),
            ));
        }

        Self::new_from_shapes(
            &parents[0].value_expected_shape(),
            &parents[0].dynamic_expected_shape(),
            p,
            seed,
        )
    }

    /// 生成 dropout mask
    ///
    /// mask 中 1 表示保留，0 表示丢弃
    fn generate_mask(&mut self, shape: &[usize]) -> Tensor {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            // 以概率 (1-p) 保留
            let keep = self.rng.r#gen::<f32>() >= self.p;
            data.push(if keep { 1.0 } else { 0.0 });
        }

        Tensor::new(&data, shape)
    }
}

impl TraitNode for Dropout {
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
        &self.fixed_shape
    }

    fn dynamic_expected_shape(&self) -> DynamicShape {
        self.dynamic_shape.clone()
    }

    fn supports_dynamic_batch(&self) -> bool {
        self.supports_dynamic
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let parent_value = parent_values[0];
        if self.is_training && self.p > 0.0 {
            // 训练模式：生成 mask，应用 dropout
            let mask = self.generate_mask(parent_value.shape());
            // Inverted Dropout: output = input * mask / (1 - p)
            let scale = 1.0 / (1.0 - self.p);
            let result = parent_value * &mask * scale;
            self.mask = Some(mask);
            self.value = Some(result);
        } else {
            // 评估模式或 p=0：直接通过
            self.mask = None;
            self.value = Some(parent_value.clone());
        }
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        // Dropout 的梯度：
        // - 训练模式：grad = upstream_grad * mask / (1 - p)
        // - 评估模式：grad = upstream_grad
        if let Some(mask) = &self.mask {
            // 训练模式：梯度也需要乘以 mask 和缩放因子
            let scale = 1.0 / (1.0 - self.p);
            Ok(upstream_grad * mask * scale)
        } else {
            // 评估模式：梯度直接通过
            Ok(upstream_grad.clone())
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
        self.mask = None; // 同时清除 mask
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }

    // ========== 训练模式 ==========

    fn set_training_mode(&mut self, is_training: bool) {
        self.is_training = is_training;
    }
}
