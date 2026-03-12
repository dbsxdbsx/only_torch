/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : Repeat（沿轴重复）节点
 *
 * forward: output = input.repeat(repeats)
 * backward: 将梯度按 repeats 汇聚回原形状（与 repeat 互为逆操作）
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Repeat 节点 — 沿各维度重复张量
#[derive(Clone)]
pub(crate) struct Repeat {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 输出形状（重复后）
    output_shape: Vec<usize>,
    /// 输入形状（重复前）
    input_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 每个维度的重复次数
    repeats: Vec<usize>,
}

impl Repeat {
    pub(crate) fn repeats(&self) -> &[usize] { &self.repeats }

    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        _parent_dynamic_shape: &DynamicShape,
        repeats: Vec<usize>,
    ) -> Result<Self, GraphError> {
        if repeats.len() != parent_shape.len() {
            return Err(GraphError::InvalidOperation(format!(
                "Repeat: repeats 长度 {} 与输入维度数 {} 不一致",
                repeats.len(),
                parent_shape.len()
            )));
        }

        let output_shape: Vec<usize> = parent_shape
            .iter()
            .zip(repeats.iter())
            .map(|(&s, &r)| s * r)
            .collect();

        let dynamic_dims: Vec<Option<usize>> = output_shape.iter().map(|&s| Some(s)).collect();
        let output_dynamic = DynamicShape::new(&dynamic_dims);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            output_shape: output_shape.clone(),
            input_shape: parent_shape.to_vec(),
            dynamic_shape: output_dynamic,
            repeats,
        })
    }
}

impl TraitNode for Repeat {
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
        &self.output_shape
    }

    fn dynamic_expected_shape(&self) -> DynamicShape {
        self.dynamic_shape.clone()
    }

    fn supports_dynamic_batch(&self) -> bool {
        false
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let input = parent_values[0];
        self.value = Some(input.repeat(&self.repeats));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Repeat 反向传播：将梯度汇聚回原形状
    ///
    /// 对每个维度，将重复的部分累加回原始位置
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let ndim = self.input_shape.len();
        let flat_up = upstream_grad.flatten_view();
        let total_in: usize = self.input_shape.iter().product();
        let mut grad_data = vec![0.0f32; total_in];

        let in_strides = Tensor::compute_strides_static(&self.input_shape);
        let out_strides = Tensor::compute_strides_static(&self.output_shape);
        let total_out: usize = self.output_shape.iter().product();

        for i in 0..total_out {
            // 将输出索引映射回输入索引（取模）
            let mut remaining = i;
            let mut in_linear = 0;
            for d in 0..ndim {
                let idx_in_dim = remaining / out_strides[d];
                remaining %= out_strides[d];
                let in_idx = idx_in_dim % self.input_shape[d];
                in_linear += in_idx * in_strides[d];
            }
            grad_data[in_linear] += flat_up[i];
        }

        Ok(GradResult::Computed(Tensor::new(
            &grad_data,
            &self.input_shape,
        )))
    }

    fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    fn grad_mut(&mut self) -> Option<&mut Tensor> {
        self.grad.as_mut()
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
