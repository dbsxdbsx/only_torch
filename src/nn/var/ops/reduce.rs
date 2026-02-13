/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Var 归约操作扩展 trait
 *
 * 提供归约操作的链式调用接口，包括：
 * - sum / sum_axis: 求和（全局 / 按轴）
 * - mean / mean_axis: 求均值（全局 / 按轴）
 */

use crate::nn::var::Var;
use std::rc::Rc;

/// 归约操作扩展 trait
///
/// 提供 Var 的归约操作：
/// - `sum()`: 全局求和，输出 [1, 1]
/// - `sum_axis(axis)`: 沿指定轴求和，保持维度（keepdims=true）
/// - `mean()`: 全局均值，输出 [1, 1]
/// - `mean_axis(axis)`: 沿指定轴均值，保持维度（keepdims=true）
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
///
/// // 全局均值（常用于 loss 计算）
/// let avg = x.mean();
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

    /// 全局均值，将所有元素求均值为 [1, 1]
    ///
    /// # 示例
    /// ```ignore
    /// let x = graph.input(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])).unwrap();
    /// let avg = x.mean();
    /// avg.forward().unwrap();
    /// // avg = 3.5
    /// ```
    fn mean(&self) -> Var;

    /// 沿指定轴求均值（keepdims=true）
    ///
    /// # 参数
    /// - `axis`: 求均值轴
    ///
    /// # 输出
    /// 原形状中第 `axis` 维变为 1
    ///
    /// # 示例
    /// ```ignore
    /// let x = graph.input(&Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3])).unwrap();
    /// let row_mean = x.mean_axis(1);  // [2, 3] -> [2, 1]
    /// row_mean.forward().unwrap();
    /// // row_mean = [[2.], [5.]]
    /// ```
    fn mean_axis(&self, axis: usize) -> Var;
}

impl VarReduceOps for Var {
    fn sum(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_sum_node(Rc::clone(self.node()), None, None)
            .expect("创建 Sum 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn sum_axis(&self, axis: usize) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_sum_node(Rc::clone(self.node()), Some(axis), None)
            .expect("创建 Sum 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn mean(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_mean_node(Rc::clone(self.node()), None, None)
            .expect("创建 Mean 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn mean_axis(&self, axis: usize) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_mean_node(Rc::clone(self.node()), Some(axis), None)
            .expect("创建 Mean 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }
}
