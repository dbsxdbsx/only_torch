/*
 * @Author       : 老董
 * @Date         : 2026-02-12
 * @Description  : Clip（值域裁剪）节点
 *                 实现逐元素裁剪: y = clip(x, min, max)
 *
 * min/max 为标量超参数（f32），存储在节点内部。
 * 等价于 NumPy 的 np.clip() 或 PyTorch 的 torch.clamp()。
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// 值域裁剪节点
///
/// forward: y = clip(x, min, max)
/// backward: dy/dx = 1 if min < x < max, else 0
///
/// ## 输入
/// - 父节点：任意形状的张量
///
/// ## 超参数
/// - min: f32 — 下界
/// - max: f32 — 上界
///
/// ## 输出
/// - 与输入形状相同
#[derive(Clone)]
pub(crate) struct Clip {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状（用于 `value_expected_shape`）
    fixed_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
    /// 裁剪下界
    min: f32,
    /// 裁剪上界
    max: f32,
    /// 缓存输入值，用于反向传播（判断梯度通过与否）
    input_cache: Option<Tensor>,
}

impl Clip {
    /// 从父节点形状信息创建 Clip 节点
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        min: f32,
        max: f32,
    ) -> Result<Self, GraphError> {
        if min > max {
            return Err(GraphError::InvalidOperation(format!(
                "Clip 节点要求 min <= max，但收到 min={}, max={}",
                min, max
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
            min,
            max,
            input_cache: None,
        })
    }
}

impl TraitNode for Clip {
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

    fn dedup_fingerprint(&self) -> Option<u64> {
        use crate::nn::nodes::raw_node::hash_dedup_params;
        Some(hash_dedup_params(&[
            self.min.to_bits() as u64,
            self.max.to_bits() as u64,
        ]))
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        // 缓存输入用于反向传播
        self.input_cache = Some(parent_values[0].clone());
        // 计算 clip(x, min, max)（委托给 Tensor::clip()）
        self.value = Some(parent_values[0].clip(self.min, self.max));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Clip 反向传播的 VJP 计算
    ///
    /// 对于 y = clip(x, min, max)，有：
    /// dy/dx = 1 if min < x < max
    /// dy/dx = 0 if x <= min or x >= max
    ///
    /// VJP: grad_to_parent = upstream_grad * mask
    /// 其中 mask[i] = 1 if min < x[i] < max, else 0
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        let input = self.input_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("Clip 输入缓存为空，需先执行前向传播".to_string())
        })?;

        let min = self.min;
        let max = self.max;

        // mask: 1.0 if min < x < max, else 0.0
        // 边界处（x == min 或 x == max）梯度为 0（与 PyTorch 行为一致）
        let mask = Tensor::new(
            &input
                .data_as_slice()
                .iter()
                .map(|&x| if x > min && x < max { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            input.shape(),
        );

        Ok(upstream_grad * &mask)
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
        self.input_cache = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
