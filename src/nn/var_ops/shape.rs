/*
 * @Author       : 老董
 * @Date         : 2026-01-09
 * @Description  : Var 形状变换扩展 trait
 *
 * 提供张量形状变换的链式调用支持，用户需 import 此 trait 后才能使用。
 * 设计依据：architecture_v2_design.md §4.2.1.3
 */

use crate::nn::{GraphError, Var};
use crate::tensor::Tensor;
use std::rc::Rc;

// ==================== GatherIndex Trait ====================

/// Gather 操作的 index 参数类型
///
/// 实现此 trait 的类型可以作为 `gather` 的 index 参数。
/// 支持 `&Var`、`Var`、`&Tensor`、`Tensor` 四种类型。
///
/// 当传入 Tensor 时，会自动转换为 input 节点。
///
/// # 示例
/// ```ignore
/// // 两种写法等价：
/// let q_selected = q_values.gather(1, &action_idx_var)?;  // index 是 &Var
/// let q_selected = q_values.gather(1, &action_tensor)?;   // index 是 &Tensor（自动转换）
/// ```
pub trait GatherIndex {
    /// 将 index 转换为 Var
    fn into_var(self, source: &Var) -> Var;
}

impl GatherIndex for &Var {
    fn into_var(self, source: &Var) -> Var {
        source.assert_same_graph(self);
        self.clone()
    }
}

impl GatherIndex for Var {
    fn into_var(self, source: &Var) -> Var {
        source.assert_same_graph(&self);
        self
    }
}

impl GatherIndex for &Tensor {
    fn into_var(self, source: &Var) -> Var {
        source.tensor_to_var(self)
    }
}

impl GatherIndex for Tensor {
    fn into_var(self, source: &Var) -> Var {
        source.tensor_to_var(&self)
    }
}

// ==================== Var 关联函数（多输入操作）====================

impl Var {
    /// 沿新维度堆叠多个 Var（类似 `torch.stack`）
    ///
    /// 在 `axis` 位置插入新维度，所有 Var 形状必须完全相同。
    ///
    /// # 参数
    /// - `vars`: 要堆叠的 Var 切片（至少 1 个，必须来自同一 Graph）
    /// - `axis`: 插入新维度的位置（0 到 ndim）
    ///
    /// # 示例
    /// ```ignore
    /// let graph = Graph::new();
    /// let a = graph.input(&Tensor::ones(&[2, 3]))?;
    /// let b = graph.input(&Tensor::ones(&[2, 3]))?;
    /// let stacked = Var::stack(&[&a, &b], 0)?;  // [2, 2, 3]
    /// ```
    pub fn stack(vars: &[&Self], axis: usize) -> Result<Self, GraphError> {
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

        // 3. 收集所有 NodeInner 并调用 graph 方法
        let nodes: Vec<_> = vars.iter().map(|v| Rc::clone(v.node())).collect();
        let graph = first.graph();
        let node = graph.borrow_mut().create_stack_node(nodes, axis, None)?;

        Ok(Self::new_with_rc_graph(node, &graph))
    }

    /// 沿现有维度拼接多个 Var（类似 `torch.cat` / `tf.concat`）
    ///
    /// 沿 `axis` 轴拼接，该轴大小可以不同，但其他维度必须相同。
    ///
    /// # 参数
    /// - `vars`: 要拼接的 Var 切片（至少 1 个，必须来自同一 Graph）
    /// - `axis`: 拼接的轴（必须是已有维度）
    ///
    /// # 示例
    /// ```ignore
    /// // SAC Critic：拼接 obs 和 action
    /// let input = Var::concat(&[&obs_var, &act_var], 1)?;  // [batch, obs_dim+action_dim]
    /// ```
    pub fn concat(vars: &[&Self], axis: usize) -> Result<Self, GraphError> {
        // 1. 验证至少有一个 Var
        if vars.is_empty() {
            return Err(GraphError::InvalidOperation(
                "Var::concat 至少需要 1 个 Var".to_string(),
            ));
        }

        // 2. 验证所有 Var 来自同一个 Graph
        let first = vars[0];
        for (i, var) in vars.iter().enumerate().skip(1) {
            if !first.same_graph(var) {
                return Err(GraphError::InvalidOperation(format!(
                    "Var::concat: 第 {i} 个 Var 来自不同的 Graph"
                )));
            }
        }

        // 3. 收集所有 NodeInner 并调用 graph 方法
        let nodes: Vec<_> = vars.iter().map(|v| Rc::clone(v.node())).collect();
        let graph = first.graph();
        let node = graph.borrow_mut().create_concat_node(nodes, axis, None)?;

        Ok(Self::new_with_rc_graph(node, &graph))
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
    /// - `index`: 索引（支持 `&Var`、`Var`、`&Tensor`、`Tensor`）
    ///
    /// # 返回
    /// 形状与 `index` 相同的 Var
    ///
    /// # 示例
    /// ```ignore
    /// // SAC/DQN 场景：按动作索引选择 Q 值
    /// // q_values: [batch, action_dim]
    /// // actions: [batch, 1] 可以是 Var 或 Tensor
    /// let q_selected = q_values.gather(1, &actions)?;  // [batch, 1]
    /// ```
    fn gather<I: GatherIndex>(&self, dim: usize, index: I) -> Result<Var, GraphError>;
}

impl VarShapeOps for Var {
    fn reshape(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_reshape_node(Rc::clone(self.node()), shape, None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    fn flatten(&self) -> Result<Var, GraphError> {
        // keep_first_dim = true: 保留 batch 维度，展平为 [batch, other_elements]
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_flatten_node(Rc::clone(self.node()), true, None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    fn select(&self, axis: usize, index: usize) -> Result<Var, GraphError> {
        let graph = self.graph();
        let node =
            graph
                .borrow_mut()
                .create_select_node(Rc::clone(self.node()), axis, index, None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    fn gather<I: GatherIndex>(&self, dim: usize, index: I) -> Result<Var, GraphError> {
        let index_var = index.into_var(self);
        let graph = self.graph();
        let node = graph.borrow_mut().create_gather_node(
            Rc::clone(self.node()),
            Rc::clone(index_var.node()),
            dim,
            None,
        )?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }
}
