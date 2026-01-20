/*
 * @Author       : 老董
 * @Date         : 2026-01-09
 * @Description  : Var 损失函数扩展 trait
 *
 * 提供损失函数的链式调用支持，用户需 import 此 trait 后才能使用。
 * 设计依据：architecture_v2_design.md §4.2.1.3
 */

use crate::nn::{GraphError, Var};
use std::rc::Rc;

/// 损失函数扩展 trait
///
/// 提供常用损失函数的链式调用：
/// - `cross_entropy(target)`: 交叉熵损失（含 Softmax）- 用于分类
/// - `mse_loss(target)`: 均方误差损失 - 用于回归
///
/// # 使用示例
/// ```ignore
/// use only_torch::nn::var::{Var, VarLossOps};
///
/// let loss = logits.cross_entropy(&labels)?;
/// let loss = output.mse_loss(&target)?;
/// ```
pub trait VarLossOps {
    /// Cross Entropy Loss（含 Softmax）
    ///
    /// # 参数
    /// - `target`: 目标标签（one-hot 编码）
    ///
    /// # 返回
    /// 标量损失值节点
    fn cross_entropy(&self, target: &Var) -> Result<Var, GraphError>;

    /// MSE Loss（均方误差）
    ///
    /// # 参数
    /// - `target`: 目标值
    ///
    /// # 返回
    /// 标量损失值节点
    fn mse_loss(&self, target: &Var) -> Result<Var, GraphError>;
}

impl VarLossOps for Var {
    fn cross_entropy(&self, target: &Var) -> Result<Var, GraphError> {
        self.assert_same_graph(target);
        let id = self.graph().borrow_mut().new_softmax_cross_entropy_node(
            self.node_id(),
            target.node_id(),
            None,
        )?;
        Ok(Var::new(id, Rc::clone(self.graph())))
    }

    fn mse_loss(&self, target: &Var) -> Result<Var, GraphError> {
        self.assert_same_graph(target);
        let id =
            self.graph()
                .borrow_mut()
                .new_mse_loss_node(self.node_id(), target.node_id(), None)?;
        Ok(Var::new(id, Rc::clone(self.graph())))
    }
}
