/*
 * @Author       : 老董
 * @Date         : 2026-02-14
 * @Description  : Var 条件过滤扩展 trait
 *
 * 提供条件选择的静态方法，用户需 import 此 trait 后才能使用。
 */

use crate::nn::{GraphError, Var};
use crate::tensor::Tensor;
use std::rc::Rc;

/// 条件过滤扩展 trait
///
/// 提供条件选择操作：
/// - `where_cond(condition, if_true, if_false)`: 按掩码选择
///
/// # 使用示例
/// ```ignore
/// use only_torch::nn::{Var, VarFilterOps};
/// use only_torch::Tensor;
///
/// let cond = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
/// let result = Var::where_cond(&cond, &x, &y)?;
/// ```
pub trait VarFilterOps {
    /// 条件选择（类似 `torch.where(condition, x, y)`）
    ///
    /// condition 是布尔掩码张量（非零为 true），不参与梯度。
    /// if_true 和 if_false 是参与计算图的 Var。
    ///
    /// # 参数
    /// - `condition`: 条件张量（非零为 true）
    /// - `if_true`: condition 为 true 时的值
    /// - `if_false`: condition 为 false 时的值
    ///
    /// # 返回
    /// 选择后的 Var
    fn where_cond(condition: &Tensor, if_true: &Var, if_false: &Var) -> Result<Var, GraphError>;
}

impl VarFilterOps for Var {
    fn where_cond(condition: &Tensor, if_true: &Var, if_false: &Var) -> Result<Var, GraphError> {
        // 验证来自同一 Graph
        if !if_true.same_graph(if_false) {
            return Err(GraphError::InvalidOperation(
                "Var::where_cond: if_true 和 if_false 必须来自同一 Graph".to_string(),
            ));
        }

        let graph = if_true.graph();
        let node = graph.borrow_mut().create_where_cond_node(
            Rc::clone(if_true.node()),
            Rc::clone(if_false.node()),
            condition.clone(),
            None,
        )?;

        Ok(Var::new_with_rc_graph(node, &graph))
    }
}
