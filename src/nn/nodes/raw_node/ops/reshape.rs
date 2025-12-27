/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : Reshape 节点 - 改变张量形状而不改变数据
 *                 参考 MatrixSlow/matrixslow/ops/ops.py 中的 Reshape 类
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

/// Reshape 节点 - 改变父节点值的形状
///
/// # 特性
/// - Forward: 将输入张量 reshape 为目标形状
/// - Backward (Jacobi): 单位矩阵（因为只是重新排列元素）
/// - Backward (Gradient): 将上游梯度 reshape 回父节点形状
///
/// # 约束
/// - 目标形状的元素总数必须与输入相同
#[derive(Clone)]
pub(crate) struct Reshape {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    grad: Option<Tensor>,
    /// 目标形状
    target_shape: Vec<usize>,
    /// 父节点的原始形状（用于反向传播）
    parent_shape: Vec<usize>,
}

impl Reshape {
    pub(crate) fn new(parents: &[&NodeHandle], target_shape: &[usize]) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Reshape 节点只需要 1 个父节点".to_string(),
            ));
        }

        // 2. 获取父节点形状
        let parent_shape = parents[0].value_expected_shape().to_vec();
        let parent_size: usize = parent_shape.iter().product();

        // 3. 验证目标形状
        if target_shape.is_empty() {
            return Err(GraphError::InvalidOperation(
                "Reshape 目标形状不能为空".to_string(),
            ));
        }

        let target_size: usize = target_shape.iter().product();
        if parent_size != target_size {
            return Err(GraphError::ShapeMismatch {
                expected: parent_shape.clone(),
                got: target_shape.to_vec(),
                message: format!(
                    "Reshape 目标形状 {:?}（{}个元素）与输入形状 {:?}（{}个元素）的元素总数不匹配",
                    target_shape, target_size, parent_shape, parent_size
                ),
            });
        }

        Ok(Self {
            id: None,
            name: None,
            value: None,
            jacobi: None,
            grad: None,
            target_shape: target_shape.to_vec(),
            parent_shape,
        })
    }

    /// 获取目标形状
    #[allow(dead_code)]
    pub(crate) fn target_shape(&self) -> &[usize] {
        &self.target_shape
    }

    /// 获取父节点原始形状
    #[allow(dead_code)]
    pub(crate) fn parent_shape(&self) -> &[usize] {
        &self.parent_shape
    }
}

impl TraitNode for Reshape {
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
        &self.target_shape
    }

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 1. 获取父节点的值
        let parent_value = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{} 的父节点 {} 没有值。不该触及本错误，否则说明 crate 代码有问题",
                self.display_node(),
                parents[0]
            ))
        })?;

        // 2. Reshape 到目标形状
        self.value = Some(parent_value.reshape(&self.target_shape));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_jacobi_to_a_parent(
        &self,
        _target_parent: &NodeHandle,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // Reshape 的 Jacobi 是单位矩阵
        // 因为每个输出元素正好等于对应的输入元素
        // ∂output[i] / ∂input[i] = 1
        // ∂output[i] / ∂input[j] = 0 (i ≠ j)
        let size = self
            .value()
            .ok_or_else(|| {
                GraphError::ComputationError(format!(
                    "{} 没有值。不该触及本错误，否则说明 crate 代码有问题",
                    self.display_node()
                ))
            })?
            .size();
        Ok(Tensor::eyes(size))
    }

    fn jacobi(&self) -> Option<&Tensor> {
        self.jacobi.as_ref()
    }

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.jacobi = jacobi.cloned();
        Ok(())
    }

    // ========== Batch 模式 ==========

    fn calc_grad_to_parent(
        &self,
        _target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // Reshape 的梯度就是将上游梯度 reshape 回父节点的形状
        // 值不变，只是形状变回去
        Ok(upstream_grad.reshape(&self.parent_shape))
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
}

