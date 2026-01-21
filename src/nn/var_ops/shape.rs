/*
 * @Author       : 老董
 * @Date         : 2026-01-09
 * @Description  : Var 形状变换扩展 trait
 *
 * 提供张量形状变换的链式调用支持，用户需 import 此 trait 后才能使用。
 * 设计依据：architecture_v2_design.md §4.2.1.3
 */

use crate::nn::{GraphError, Var};
use std::rc::Rc;

/// 形状变换扩展 trait
///
/// 提供张量形状变换的链式调用：
/// - `reshape(shape)`: 变形为指定形状
/// - `flatten()`: 展平为一维向量（保留 batch 维度）
/// - `select(axis, index)`: 从指定轴选择一个切片
///
/// # 使用示例
/// ```ignore
/// use only_torch::nn::var::{Var, VarShapeOps};
///
/// let reshaped = x.reshape(&[2, 4])?;
/// let flat = x.flatten()?;
/// let x_t = x_seq.select(1, t)?;  // 从 [batch, seq_len, input] 选择第 t 步
/// ```
pub trait VarShapeOps {
    /// Reshape 变形
    ///
    /// 将张量变形为指定形状，元素总数必须保持一致。
    ///
    /// # 参数
    /// - `shape`: 目标形状
    ///
    /// # 返回
    /// 变形后的 Var
    fn reshape(&self, shape: &[usize]) -> Result<Var, GraphError>;

    /// Flatten 展平
    ///
    /// 将张量展平为 `[batch_size, total_other_elements]`，保留 batch 维度。
    /// 对于输入如：`[batch, C, H, W]`，输出：`[batch, C*H*W]`。
    ///
    /// # 返回
    /// 展平后的 Var
    fn flatten(&self) -> Result<Var, GraphError>;

    /// Select 选择
    ///
    /// 从张量的指定轴选择一个索引，去掉该维度。
    /// 主要用于 RNN 展开式设计：从 `[batch, seq_len, input_size]` 提取单个时间步。
    ///
    /// # 参数
    /// - `axis`: 选择的轴
    /// - `index`: 选择的索引
    ///
    /// # 返回
    /// 选择后的 Var（维度减 1）
    ///
    /// # 示例
    /// ```ignore
    /// // x: [batch, seq_len, input_size]
    /// let x_t = x.select(1, t)?;  // [batch, input_size]
    /// ```
    fn select(&self, axis: usize, index: usize) -> Result<Var, GraphError>;
}

impl VarShapeOps for Var {
    fn reshape(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let id = self
            .graph()
            .borrow_mut()
            .new_reshape_node(self.node_id(), shape, None)?;
        Ok(Self::new(id, Rc::clone(self.graph())))
    }

    fn flatten(&self) -> Result<Var, GraphError> {
        // keep_first_dim = true: 保留 batch 维度，展平为 [batch, other_elements]
        let id = self
            .graph()
            .borrow_mut()
            .new_flatten_node(self.node_id(), true, None)?;
        Ok(Self::new(id, Rc::clone(self.graph())))
    }

    fn select(&self, axis: usize, index: usize) -> Result<Var, GraphError> {
        let id = self
            .graph()
            .borrow_mut()
            .new_select_node(self.node_id(), axis, index, None)?;
        Ok(Self::new(id, Rc::clone(self.graph())))
    }
}
