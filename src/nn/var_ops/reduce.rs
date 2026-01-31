/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Var 归约操作扩展 trait
 *
 * 提供归约操作的链式调用接口，包括：
 * - sum / sum_axis: 求和（全局 / 按轴）
 * - mean / mean_axis: 求均值（全局 / 按轴）（TODO）
 */

use crate::nn::var::Var;
use std::rc::Rc;

/// 归约操作扩展 trait
///
/// 提供 Var 的归约操作：
/// - `sum()`: 全局求和，输出 [1, 1]
/// - `sum_axis(axis)`: 沿指定轴求和，保持维度（keepdims=true）
///
/// # 使用示例
/// ```ignore
/// use only_torch::nn::var::{Var, VarReduceOps};
///
/// // 全局求和
/// let total = x.sum();
///
/// // 沿 action 维度求和（SAC Actor Loss: Σ_a π(a|s) * (...)）
/// let action_sum = probs.sum_axis(1);
/// ```
pub trait VarReduceOps {
    /// 全局求和，将所有元素求和为 [1, 1]
    ///
    /// # 示例
    /// ```ignore
    /// let x = graph.input(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])).unwrap();
    /// let total = x.sum();
    /// total.forward().unwrap();
    /// // total = 21.0
    /// ```
    fn sum(&self) -> Var;

    /// 沿指定轴求和（keepdims=true）
    ///
    /// # 参数
    /// - `axis`: 求和轴
    ///
    /// # 输出
    /// 原形状中第 `axis` 维变为 1
    ///
    /// # 示例
    /// ```ignore
    /// let x = graph.input(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])).unwrap();
    /// let row_sum = x.sum_axis(1);  // [2, 3] -> [2, 1]
    /// row_sum.forward().unwrap();
    /// // row_sum = [[6.], [15.]]
    /// ```
    fn sum_axis(&self, axis: usize) -> Var;
}

impl VarReduceOps for Var {
    fn sum(&self) -> Var {
        let id = self
            .graph()
            .borrow_mut()
            .new_sum_node(self.node_id(), None, None)
            .expect("创建 Sum 节点失败");
        Self::new(id, Rc::clone(self.graph()))
    }

    fn sum_axis(&self, axis: usize) -> Var {
        let id = self
            .graph()
            .borrow_mut()
            .new_sum_node(self.node_id(), Some(axis), None)
            .expect("创建 Sum 节点失败");
        Self::new(id, Rc::clone(self.graph()))
    }
}
