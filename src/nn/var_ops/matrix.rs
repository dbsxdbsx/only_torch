/*
 * @Author       : 老董
 * @Date         : 2026-01-09
 * @Description  : Var 矩阵运算扩展 trait
 *
 * 提供矩阵运算的链式调用支持，用户需 import 此 trait 后才能使用。
 * 设计依据：architecture_v2_design.md §4.2.1.3
 */

use crate::nn::{GraphError, Var};
use std::rc::Rc;

/// 矩阵运算扩展 trait
///
/// 提供矩阵运算的链式调用：
/// - `matmul(other)`: 矩阵乘法
///
/// # 使用示例
/// ```ignore
/// use only_torch::nn::var::{Var, VarMatrixOps};
///
/// let y = x.matmul(&w)?;
/// ```
///
/// # 未来扩展
/// - `transpose()`: 转置
/// - `reshape(shape)`: 变形
/// - `flatten()`: 展平
pub trait VarMatrixOps {
    /// 矩阵乘法
    ///
    /// # 参数
    /// - `other`: 右侧矩阵
    ///
    /// # 形状要求
    /// - self: [m, k]
    /// - other: [k, n]
    /// - 输出: [m, n]
    fn matmul(&self, other: &Var) -> Result<Var, GraphError>;
}

impl VarMatrixOps for Var {
    fn matmul(&self, other: &Var) -> Result<Var, GraphError> {
        self.assert_same_graph(other);
        let id =
            self.graph()
                .borrow_mut()
                .new_mat_mul_node(self.node_id(), other.node_id(), None)?;
        Ok(Var::new(id, Rc::clone(self.graph())))
    }
}
