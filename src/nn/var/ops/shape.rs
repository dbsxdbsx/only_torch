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

    /// 沿指定轴取连续子范围（不降维）
    ///
    /// 等价于 PyTorch 的 `tensor.narrow(dim, start, length)`。
    ///
    /// # 参数
    /// - `axis`: 操作的轴
    /// - `start`: 起始索引
    /// - `length`: 取的长度
    fn narrow(&self, axis: usize, start: usize, length: usize) -> Result<Var, GraphError>;

    /// 沿指定轴拆分为多段
    ///
    /// 内部创建 N 个 Narrow 节点，返回 `Vec<Var>`。
    /// 梯度通过各 Narrow 节点独立反向传播，自动正确合并。
    ///
    /// # 参数
    /// - `axis`: 拆分的轴
    /// - `sizes`: 各段大小，之和必须等于轴大小
    fn split(&self, axis: usize, sizes: &[usize]) -> Result<Vec<Var>, GraphError>;

    /// Squeeze: 移除指定轴上 size=1 的维度（或所有 size=1 维度）
    ///
    /// 内部通过 `reshape` 实现，不创建新节点类型。
    ///
    /// # 参数
    /// - `axis`: `Some(i)` 移除第 i 维（须为 size=1），`None` 移除所有 size=1 维
    ///
    /// # 示例
    /// ```ignore
    /// // [1, 3, 1, 4] squeeze(Some(0)) → [3, 1, 4]
    /// // [1, 3, 1, 4] squeeze(None)    → [3, 4]
    /// ```
    fn squeeze(&self, axis: Option<usize>) -> Result<Var, GraphError>;

    /// Unsqueeze: 在指定位置插入 size=1 的维度
    ///
    /// 内部通过 `reshape` 实现，不创建新节点类型。
    ///
    /// # 参数
    /// - `axis`: 插入位置（0..=ndim）
    ///
    /// # 示例
    /// ```ignore
    /// // [3, 4] unsqueeze(0) → [1, 3, 4]
    /// // [3, 4] unsqueeze(1) → [3, 1, 4]
    /// // [3, 4] unsqueeze(2) → [3, 4, 1]
    /// ```
    fn unsqueeze(&self, axis: usize) -> Result<Var, GraphError>;

    /// 维度重排（转置的一般形式）
    ///
    /// 等价于 PyTorch 的 `tensor.permute(dims)`。
    ///
    /// # 参数
    /// - `dims`: 维度排列顺序，如 `&[0, 2, 1]` 交换后两维
    ///
    /// # 示例
    /// ```ignore
    /// // x: [batch, seq_len, hidden] → [batch, hidden, seq_len]
    /// let y = x.permute(&[0, 2, 1])?;
    /// ```
    fn permute(&self, dims: &[usize]) -> Result<Var, GraphError>;

    /// 交换两个维度（permute 的便捷接口）
    ///
    /// 等价于 PyTorch 的 `tensor.transpose(dim0, dim1)`。
    /// 内部构造恒等排列并交换 dim0/dim1，然后调用 `permute`。
    ///
    /// # 参数
    /// - `dim0`: 第一个维度
    /// - `dim1`: 第二个维度
    ///
    /// # 示例
    /// ```ignore
    /// // x: [2, 3, 4] → transpose(1, 2) → [2, 4, 3]
    /// let y = x.transpose(1, 2)?;
    /// ```
    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Var, GraphError>;

    /// Pad 常量值填充
    ///
    /// 对张量进行常量值填充，每个维度可指定前后的填充量。
    /// 主要用于 CNN same-padding 和序列对齐。
    ///
    /// # 参数
    /// - `paddings`: 每个维度的填充量 `(before, after)`，长度必须等于维度数
    /// - `value`: 填充值
    ///
    /// # 示例
    /// ```ignore
    /// // 4D CNN padding: 只在 H/W 维度填充
    /// let y = x.pad(&[(0,0), (0,0), (1,1), (1,1)], 0.0)?;
    /// ```
    fn pad(&self, paddings: &[(usize, usize)], value: f32) -> Result<Var, GraphError>;

    /// 将张量沿指定维度等分为 n 块
    ///
    /// 类似 PyTorch 的 `torch.chunk()`。
    /// 如果维度大小不能被 n 整除，最后一块会更小。
    ///
    /// # 参数
    /// - `n`: 块数
    /// - `dim`: 沿哪个维度分割
    ///
    /// # 示例
    /// ```ignore
    /// // x: [6, 4] → chunk(3, 0) → 3 个 [2, 4]
    /// let chunks = x.chunk(3, 0)?;
    /// ```
    fn chunk(&self, n: usize, dim: usize) -> Result<Vec<Var>, GraphError>;

    /// 沿各维度重复张量
    ///
    /// 类似 PyTorch 的 `tensor.repeat(repeats)` / NumPy 的 `np.tile()`。
    ///
    /// # 参数
    /// - `repeats`: 每个维度的重复次数
    ///
    /// # 示例
    /// ```ignore
    /// let y = x.repeat(&[2, 3])?;  // [H, W] → [2*H, 3*W]
    /// ```
    fn repeat(&self, repeats: &[usize]) -> Result<Var, GraphError>;
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

    fn narrow(&self, axis: usize, start: usize, length: usize) -> Result<Var, GraphError> {
        let graph = self.graph();
        let node = graph.borrow_mut().create_narrow_node(
            Rc::clone(self.node()),
            axis,
            start,
            length,
            None,
        )?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    fn split(&self, axis: usize, sizes: &[usize]) -> Result<Vec<Var>, GraphError> {
        // 验证 sizes 之和等于轴大小
        let shape = self.node().shape();
        if axis >= shape.len() {
            return Err(GraphError::InvalidOperation(format!(
                "split: axis {axis} 超出维度 {}",
                shape.len()
            )));
        }
        let total: usize = sizes.iter().sum();
        if total != shape[axis] {
            return Err(GraphError::InvalidOperation(format!(
                "split: sizes 之和 {total} 不等于轴 {axis} 的大小 {}",
                shape[axis]
            )));
        }

        // 创建 N 个 Narrow 节点
        let mut result = Vec::with_capacity(sizes.len());
        let mut start = 0;
        for &size in sizes {
            result.push(self.narrow(axis, start, size)?);
            start += size;
        }
        Ok(result)
    }

    fn squeeze(&self, axis: Option<usize>) -> Result<Var, GraphError> {
        let shape = self.node().shape();
        match axis {
            Some(i) => {
                if i >= shape.len() {
                    return Err(GraphError::InvalidOperation(format!(
                        "squeeze: axis {i} 超出维度 {}",
                        shape.len()
                    )));
                }
                if shape[i] != 1 {
                    return Err(GraphError::InvalidOperation(format!(
                        "squeeze: axis {i} 的大小为 {}（非 1），无法 squeeze",
                        shape[i]
                    )));
                }
                let mut new_shape = shape.clone();
                new_shape.remove(i);
                // 至少保留 1 维
                if new_shape.is_empty() {
                    new_shape.push(1);
                }
                self.reshape(&new_shape)
            }
            None => {
                let new_shape: Vec<usize> = shape.iter().copied().filter(|&d| d != 1).collect();
                // 至少保留 1 维（全 1 的情况）
                if new_shape.is_empty() {
                    self.reshape(&[1])
                } else {
                    self.reshape(&new_shape)
                }
            }
        }
    }

    fn unsqueeze(&self, axis: usize) -> Result<Var, GraphError> {
        let shape = self.node().shape();
        let ndim = shape.len();
        if axis > ndim {
            return Err(GraphError::InvalidOperation(format!(
                "unsqueeze: axis {axis} 超出范围 0..={ndim}"
            )));
        }
        let mut new_shape = shape.clone();
        new_shape.insert(axis, 1);
        self.reshape(&new_shape)
    }

    fn permute(&self, dims: &[usize]) -> Result<Var, GraphError> {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_permute_node(Rc::clone(self.node()), dims, None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Var, GraphError> {
        let ndim = self.node().shape().len();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(GraphError::InvalidOperation(format!(
                "transpose: dim0={dim0} 或 dim1={dim1} 超出维度 {ndim}"
            )));
        }
        // 构造恒等排列并交换 dim0/dim1
        let mut dims: Vec<usize> = (0..ndim).collect();
        dims.swap(dim0, dim1);
        self.permute(&dims)
    }

    fn pad(&self, paddings: &[(usize, usize)], value: f32) -> Result<Var, GraphError> {
        let graph = self.graph();
        let node = graph.borrow_mut().create_pad_node(
            Rc::clone(self.node()),
            paddings.to_vec(),
            value,
            None,
        )?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    fn chunk(&self, n: usize, dim: usize) -> Result<Vec<Var>, GraphError> {
        let shape = self.node().shape();
        let ndim = shape.len();
        if dim >= ndim {
            return Err(GraphError::InvalidOperation(format!(
                "chunk: dim {dim} 超出维度 {ndim}"
            )));
        }
        if n == 0 {
            return Err(GraphError::InvalidOperation(
                "chunk: n 必须 > 0".to_string(),
            ));
        }

        let dim_size = shape[dim];
        let chunk_size = (dim_size + n - 1) / n; // 向上取整
        let mut result = Vec::new();
        let mut start = 0;
        while start < dim_size {
            let len = chunk_size.min(dim_size - start);
            result.push(self.narrow(dim, start, len)?);
            start += len;
        }

        Ok(result)
    }

    fn repeat(&self, repeats: &[usize]) -> Result<Var, GraphError> {
        let graph = self.graph();
        let node = graph.borrow_mut().create_repeat_node(
            Rc::clone(self.node()),
            repeats.to_vec(),
            None,
        )?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }
}
