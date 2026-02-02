/*
 * @Author       : 老董
 * @Date         : 2026-01-09
 * @Description  : Var 损失函数扩展 trait
 *
 * 提供损失函数的链式调用支持，用户需 import 此 trait 后才能使用。
 * 设计依据：architecture_v2_design.md §4.2.1.3
 */

use crate::nn::{GraphError, Var};
use crate::tensor::Tensor;
use std::rc::Rc;

// ==================== LossTarget Trait ====================

/// Loss 函数的 target 参数类型
///
/// 实现此 trait 的类型可以作为 Loss 函数的 target 参数。
/// 支持 `&Var`、`Var`、`&Tensor`、`Tensor` 四种类型。
///
/// 当传入 Tensor 时，会自动转换为 input 节点。
pub trait LossTarget {
    /// 将 target 转换为 Var（如果已经是 Var 则克隆，如果是 Tensor 则创建 input 节点）
    fn into_var(self, source: &Var) -> Var;
}

impl LossTarget for &Var {
    fn into_var(self, source: &Var) -> Var {
        source.assert_same_graph(self);
        self.clone()
    }
}

impl LossTarget for Var {
    fn into_var(self, source: &Var) -> Var {
        source.assert_same_graph(&self);
        self
    }
}

impl LossTarget for &Tensor {
    fn into_var(self, source: &Var) -> Var {
        // 使用 TargetInput 节点，在可视化中显示为橙色椭圆
        source.tensor_to_target_var(self)
    }
}

impl LossTarget for Tensor {
    fn into_var(self, source: &Var) -> Var {
        // 使用 TargetInput 节点，在可视化中显示为橙色椭圆
        source.tensor_to_target_var(&self)
    }
}

// ==================== VarLossOps Trait ====================

/// 损失函数扩展 trait
///
/// 提供常用损失函数的链式调用，支持 `&Var`、`Var`、`&Tensor`、`Tensor` 作为 target：
/// - `cross_entropy(target)`: 交叉熵损失（含 Softmax）- 用于多分类（互斥类别）
/// - `bce_loss(target)`: 二元交叉熵损失（含 Sigmoid）- 用于二分类/多标签分类
/// - `mse_loss(target)`: 均方误差损失 - 用于回归
/// - `mae_loss(target)`: 平均绝对误差损失 - 用于回归（对异常值更鲁棒）
/// - `huber_loss(target)`: Huber 损失 - 用于强化学习 / 带离群值的回归
///
/// # 使用示例
/// ```ignore
/// use only_torch::nn::var::{Var, VarLossOps};
///
/// let loss = logits.cross_entropy(&labels)?;     // target 是 &Var
/// let loss = output.mse_loss(&target_tensor)?;   // target 是 &Tensor（自动转换）
/// let loss = output.mse_loss(target_tensor)?;    // target 是 Tensor（也可以）
/// let loss = q_values.huber_loss(&target_q)?;    // 强化学习
/// ```
pub trait VarLossOps {
    /// Cross Entropy Loss（含 Softmax）
    ///
    /// 用于多分类任务，各类别互斥（一个样本只属于一个类别）。
    fn cross_entropy<T: LossTarget>(&self, target: T) -> Result<Var, GraphError>;

    /// BCE Loss（二元交叉熵，含 Sigmoid）
    ///
    /// 用于二分类或多标签分类任务（一个样本可同时属于多个类别）。
    fn bce_loss<T: LossTarget>(&self, target: T) -> Result<Var, GraphError>;

    /// MSE Loss（均方误差）
    ///
    /// 用于回归任务。强化学习中常用于 Critic 的 Q 值训练。
    fn mse_loss<T: LossTarget>(&self, target: T) -> Result<Var, GraphError>;

    /// MAE Loss（平均绝对误差）
    ///
    /// 相比 MSE，对异常值更鲁棒，梯度恒定。
    fn mae_loss<T: LossTarget>(&self, target: T) -> Result<Var, GraphError>;

    /// Huber Loss（Smooth L1 Loss）
    ///
    /// 结合 MSE（小误差）和 MAE（大误差）的优点。
    /// 是强化学习（DQN 等）的标准损失函数。
    fn huber_loss<T: LossTarget>(&self, target: T) -> Result<Var, GraphError>;
}

impl VarLossOps for Var {
    fn cross_entropy<T: LossTarget>(&self, target: T) -> Result<Var, GraphError> {
        let target_var = target.into_var(self);
        let graph = self.graph();
        let node = graph.borrow_mut().create_softmax_cross_entropy_node(
            Rc::clone(self.node()),
            Rc::clone(target_var.node()),
            None,
        )?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    fn bce_loss<T: LossTarget>(&self, target: T) -> Result<Var, GraphError> {
        let target_var = target.into_var(self);
        let graph = self.graph();
        let node = graph.borrow_mut().create_bce_mean_node(
            Rc::clone(self.node()),
            Rc::clone(target_var.node()),
            None,
        )?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    fn mse_loss<T: LossTarget>(&self, target: T) -> Result<Var, GraphError> {
        let target_var = target.into_var(self);
        let graph = self.graph();
        let node = graph.borrow_mut().create_mse_mean_node(
            Rc::clone(self.node()),
            Rc::clone(target_var.node()),
            None,
        )?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    fn mae_loss<T: LossTarget>(&self, target: T) -> Result<Var, GraphError> {
        let target_var = target.into_var(self);
        let graph = self.graph();
        let node = graph.borrow_mut().create_mae_mean_node(
            Rc::clone(self.node()),
            Rc::clone(target_var.node()),
            None,
        )?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    fn huber_loss<T: LossTarget>(&self, target: T) -> Result<Var, GraphError> {
        let target_var = target.into_var(self);
        let graph = self.graph();
        let node = graph.borrow_mut().create_huber_default_node(
            Rc::clone(self.node()),
            Rc::clone(target_var.node()),
            None,
        )?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }
}
