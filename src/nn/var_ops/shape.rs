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

// ==================== Var 关联函数（多输入操作）====================

impl Var {
    /// 将多个 Var 沿指定轴堆叠/拼接
    ///
    /// 这是一个统一的操作，通过 `new_dim` 参数区分两种模式：
    /// - **Stack 模式**（`new_dim=true`）：在 `axis` 位置插入新维度，类似 `torch.stack`
    /// - **Concat 模式**（`new_dim=false`）：沿现有 `axis` 拼接，类似 `torch.cat`
    ///
    /// # 参数
    /// - `vars`: 要堆叠的 Var 切片（至少 1 个，必须来自同一 Graph）
    /// - `axis`: 堆叠/拼接的轴
    /// - `new_dim`: 是否插入新维度
    ///
    /// # 形状规则
    /// - **Stack 模式**：所有 Var 形状必须相同，输出在 `axis` 位置增加一个维度
    /// - **Concat 模式**：除 `axis` 外其他维度必须相同，输出的 `axis` 维度是各输入之和
    ///
    /// # 示例
    /// ```ignore
    /// use only_torch::nn::Graph;
    ///
    /// let graph = Graph::new();
    /// let a = graph.input(&Tensor::ones(&[2, 3]))?;
    /// let b = graph.input(&Tensor::ones(&[2, 3]))?;
    ///
    /// // Stack 模式：[2,3] + [2,3] -> [2, 2, 3]
    /// let stacked = Var::stack(&[&a, &b], 0, true)?;
    ///
    /// // Concat 模式：[2,3] + [2,3] -> [4, 3]
    /// let concat = Var::stack(&[&a, &b], 0, false)?;
    /// ```
    pub fn stack(vars: &[&Self], axis: usize, new_dim: bool) -> Result<Self, GraphError> {
        // 1. 验证至少有一个 Var
        if vars.is_empty() {
            return Err(GraphError::InvalidOperation(
                "Var::stack 至少需要 1 个 Var".to_string(),
            ));
        }

        // 2. 验证所有 Var 来自同一个 Graph
        let first = vars[0];
        for (i, var) in vars.iter().enumerate().skip(1) {
            if !first.same_graph(var) {
                return Err(GraphError::InvalidOperation(format!(
                    "Var::stack: 第 {i} 个 Var 来自不同的 Graph"
                )));
            }
        }

        // 3. 收集所有 NodeId 并调用 graph 方法
        let node_ids: Vec<_> = vars.iter().map(|v| v.node_id()).collect();
        let graph_rc = Rc::clone(first.graph());
        let new_id = graph_rc
            .borrow_mut()
            .new_stack_node(&node_ids, axis, new_dim, None)?;

        Ok(Self::new(new_id, graph_rc))
    }
}

// ==================== VarShapeOps Trait ====================

/// 形状变换扩展 trait
///
/// 提供张量形状变换的链式调用：
/// - `reshape(shape)`: 变形为指定形状
/// - `flatten()`: 展平为一维向量（保留 batch 维度）
/// - `select(axis, index)`: 从指定轴选择一个切片（固定索引）
/// - `gather(dim, index)`: 按索引张量收集元素（动态索引）
///
/// # 使用示例
/// ```ignore
/// use only_torch::nn::var::{Var, VarShapeOps};
///
/// let reshaped = x.reshape(&[2, 4])?;
/// let flat = x.flatten()?;
/// let x_t = x_seq.select(1, t)?;  // 从 [batch, seq_len, input] 选择第 t 步
/// let q_selected = q_values.gather(1, &actions)?;  // 按动作索引选择 Q 值
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

    /// Select 选择（固定索引）
    ///
    /// 从张量的指定轴选择一个索引，去掉该维度。
    /// 主要用于 RNN 展开式设计：从 `[batch, seq_len, input_size]` 提取单个时间步。
    ///
    /// # 参数
    /// - `axis`: 选择的轴
    /// - `index`: 选择的索引（编译时已知）
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

    /// Gather 收集（动态索引）
    ///
    /// 按索引张量从指定维度收集元素。
    /// 主要用于强化学习：按动作索引选择 Q 值。
    ///
    /// 与 `select` 的区别：
    /// - `select`: 固定索引（编译时已知），去掉该维度
    /// - `gather`: 动态索引（运行时张量），保持维度数不变
    ///
    /// # 参数
    /// - `dim`: gather 的维度
    /// - `index`: 索引 Var（元素为 f32，会被转换为 usize）
    ///
    /// # 返回
    /// 形状与 `index` 相同的 Var
    ///
    /// # 示例
    /// ```ignore
    /// // SAC/DQN 场景：按动作索引选择 Q 值
    /// // q_values: [batch, action_dim]
    /// // actions: [batch, 1]
    /// let q_selected = q_values.gather(1, &actions)?;  // [batch, 1]
    /// ```
    fn gather(&self, dim: usize, index: &Var) -> Result<Var, GraphError>;
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

    fn gather(&self, dim: usize, index: &Var) -> Result<Var, GraphError> {
        self.assert_same_graph(index);
        let id = self
            .graph()
            .borrow_mut()
            .new_gather_node(self.node_id(), index.node_id(), dim, None)?;
        Ok(Self::new(id, Rc::clone(self.graph())))
    }
}
