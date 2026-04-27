/*
 * @Author       : 老董
 * @Date         : 2026-02-14
 * @Description  : Var 选择操作扩展 trait
 *
 * 提供基于值的选择操作（如 TopK），用户需 import 此 trait 后才能使用。
 */

use crate::nn::{GraphError, Var};
use std::rc::Rc;

// ==================== VarSelectionOps Trait ====================

/// 选择操作扩展 trait
///
/// 提供基于值的选择操作的链式调用：
/// - `topk(k, axis, sorted)`: 沿指定轴选取前 k 大值
///
/// # 使用示例
/// ```ignore
/// use only_torch::nn::var::{Var, VarSelectionOps};
///
/// // 选取每行前 2 大的值
/// let top_values = x.topk(2, 1, true)?;
/// ```
pub trait VarSelectionOps {
    /// 沿指定轴选取前 k 大元素
    ///
    /// 返回的 Var 参与梯度计算，梯度仅传递到被选中的位置。
    /// 如需获取索引，可在 Tensor 层面调用 `tensor.topk()`。
    ///
    /// # 参数
    /// - `k`: 选取的元素数量
    /// - `axis`: 操作的轴
    /// - `sorted`: 是否按降序排列结果
    ///
    /// # 返回
    /// 包含前 k 大值的 Var，形状与输入相同但 axis 维度变为 k
    ///
    /// # 示例
    /// ```ignore
    /// // x: [batch, features] = [2, 4]
    /// let top2 = x.topk(2, 1, true)?;  // [2, 2]
    /// ```
    fn topk(&self, k: usize, axis: usize, sorted: bool) -> Result<Var, GraphError>;

    /// 沿指定轴排序，返回排序后的值（可微）
    ///
    /// 排序索引存储在节点内部，用于反向传播时将梯度 scatter 回原始位置。
    ///
    /// # 参数
    /// - `axis`: 排序轴
    /// - `descending`: `true` 为降序，`false` 为升序
    ///
    /// # 返回
    /// 排序后的 Var（形状与输入相同）
    fn sort_values(&self, axis: usize, descending: bool) -> Result<Var, GraphError>;
}

impl VarSelectionOps for Var {
    fn topk(&self, k: usize, axis: usize, sorted: bool) -> Result<Var, GraphError> {
        let graph = self.graph();
        let node =
            graph
                .borrow_mut()
                .create_topk_node(Rc::clone(self.node()), k, axis, sorted, None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    fn sort_values(&self, axis: usize, descending: bool) -> Result<Var, GraphError> {
        let graph = self.graph();
        let node =
            graph
                .borrow_mut()
                .create_sort_node(Rc::clone(self.node()), axis, descending, None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }
}
