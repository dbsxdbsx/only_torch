/*
 * @Author       : 老董
 * @Date         : 2026-01-08
 * @Description  : Smart Var - 智能变量句柄，支持算子重载和链式调用
 *
 * 这是架构的核心组件，提供 PyTorch 级用户体验。
 */

mod arithmetic;
mod descriptor;
mod init;
mod into;
pub mod ops;
mod visualization;

pub use init::Init;
pub use into::IntoVar;

use super::graph::GraphInner;
use super::nodes::NodeInner;
use super::{GraphError, NodeId};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::rc::{Rc, Weak};
use std::time::{Duration, Instant};

// ==================== Var 结构 ====================

/// 智能变量句柄 - 携带图引用，支持算子重载和链式调用
///
/// # 设计原则
/// - 持有 `Rc<NodeInner>` 直接控制节点生命周期
/// - 持有 `Weak<RefCell<GraphInner>>` 引用用于全局配置
/// - 用户无需关心内部实现，像 `PyTorch` tensor 一样使用
/// - Clone 语义（非 Copy），但开销极低（Rc clone）
///
/// # 使用示例
/// ```ignore
/// let graph = Graph::new();
/// let x = graph.input(&images)?;      // 返回 Var
/// let h = x.relu();                   // 链式调用
/// let y = h.matmul(&w)?;              // 方法调用
/// let z = &y + &b;                    // 算子重载
/// let loss = z.cross_entropy(&target)?;
/// loss.backward()?;                   // 直接在 Var 上调用
/// ```
#[derive(Clone)]
pub struct Var {
    /// 节点内部结构 - 直接持有节点，控制生命周期
    node: Rc<NodeInner>,
    /// 图引用（Weak 引用，不阻止 Graph 释放）
    graph: Weak<RefCell<GraphInner>>,
}

impl std::fmt::Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Var")
            .field("id", &self.node.id())
            .field("name", &self.node.name())
            .finish()
    }
}

impl Var {
    /// 创建新的 Var
    ///
    /// 直接持有 NodeInner，graph 为 Weak 引用
    #[allow(dead_code)]
    pub(crate) fn new(node: Rc<NodeInner>, graph: Weak<RefCell<GraphInner>>) -> Self {
        Self { node, graph }
    }

    /// 从 Rc<RefCell<GraphInner>> 创建 Var（便捷方法）
    pub(crate) fn new_with_rc_graph(node: Rc<NodeInner>, graph: &Rc<RefCell<GraphInner>>) -> Self {
        Self {
            node,
            graph: Rc::downgrade(graph),
        }
    }

    /// 获取节点 ID
    pub fn node_id(&self) -> NodeId {
        self.node.id()
    }

    /// 获取节点名称
    pub fn name(&self) -> Option<&str> {
        self.node.name()
    }

    /// 获取节点分组标签（用于可视化 cluster）
    pub fn node_group_tag(&self) -> Option<super::graph::NodeGroupTag> {
        self.node.node_group_tag()
    }

    /// 获取 NodeInner 的引用
    pub(crate) fn node(&self) -> &Rc<NodeInner> {
        &self.node
    }

    /// 获取内部图引用（升级 Weak 为 Rc）
    ///
    /// # Panics
    /// 如果 Graph 已被释放，则 panic
    pub(crate) fn graph(&self) -> Rc<RefCell<GraphInner>> {
        self.graph.upgrade().expect("Graph 已被释放，Var 不再有效")
    }

    /// 尝试获取内部图引用（不 panic）
    #[allow(dead_code)]
    pub(crate) fn try_graph(&self) -> Option<Rc<RefCell<GraphInner>>> {
        self.graph.upgrade()
    }

    /// 检查两个 Var 是否来自同一个 Graph
    pub fn same_graph(&self, other: &Self) -> bool {
        match (self.graph.upgrade(), other.graph.upgrade()) {
            (Some(a), Some(b)) => Rc::ptr_eq(&a, &b),
            _ => false, // 任一 Graph 已释放，认为不同
        }
    }

    /// 获取 Var 所属的 Graph handle
    ///
    /// # Panics
    /// 如果 Graph 已被释放，则 panic
    pub fn get_graph(&self) -> super::graph::Graph {
        super::graph::Graph::from_rc(self.graph())
    }

    /// 获取节点的预期输出形状
    pub fn value_expected_shape(&self) -> Vec<usize> {
        self.node.shape()
    }

    /// 获取节点的动态形状
    ///
    /// 返回支持动态维度的形状表示（如 `[?, 128]`）
    pub fn dynamic_expected_shape(&self) -> crate::nn::shape::DynamicShape {
        self.node().dynamic_expected_shape()
    }

    /// 断言两个 Var 来自同一个 Graph，否则 panic（供 trait 使用）
    pub(crate) fn assert_same_graph(&self, other: &Self) {
        assert!(
            self.same_graph(other),
            "不能对来自不同 Graph 的 Var 进行操作"
        );
    }

    // ==================== 梯度流控制 ====================

    /// 创建一个 detached 的 Var（与 PyTorch `tensor.detach()` 语义一致）
    ///
    /// 在图中创建一个新的 Detach 节点。
    /// 返回的 Var：
    /// - 与原 Var 共享前向计算的值
    /// - 反向传播时，梯度不会通过此节点传回原 Var
    /// - 可以继续参与后续的图计算
    ///
    /// # 示例
    /// ```ignore
    /// // GAN 训练：训练 D 时阻止梯度流向 G
    /// let fake_images = G.forward(&noise)?;
    /// let d_fake = D.forward(&fake_images.detach())?;  // 梯度阻断
    /// d_loss.backward()?;  // 梯度不会流向 G
    /// ```
    pub fn detach(&self) -> Self {
        let new_node = self
            .graph()
            .borrow_mut()
            .create_detach_node(Rc::clone(&self.node), None)
            .expect("内部错误：detach 创建 Detach 节点失败");
        Self {
            node: new_node,
            graph: self.graph.clone(),
        }
    }

    /// 检查此 Var 对应的节点是否处于 detached 状态
    ///
    /// detached 节点在反向传播时不会传递梯度给其父节点。
    /// 判断依据：节点类型为 `Detach`，或底层标志位 `is_detached` 为 true。
    ///
    /// # 用途
    /// - `ModelState` 使用此方法判断 Var 输入是否可以缓存
    /// - detached Var 只需要值，不需要梯度流，因此可以像 Tensor 一样缓存
    pub fn is_detached(&self) -> bool {
        use crate::nn::nodes::NodeType;
        self.node
            .with_raw_node(|raw| matches!(raw, NodeType::Detach(_)))
            || self.node.is_detached()
    }

    // ==================== 执行 ====================

    /// 前向传播
    ///
    /// 递归执行从当前节点到所有父节点的前向计算
    pub fn forward(&self) -> Result<(), GraphError> {
        self.graph().borrow_mut().forward_via_node_inner(&self.node)
    }

    /// 反向传播（ensure-forward 语义）
    ///
    /// # 语义
    /// - 自动先执行 forward()，确保 loss 值已计算
    /// - 然后执行反向传播
    /// - 动态图架构下，中间结果由 Rc 引用计数管理，天然支持多次 backward
    ///
    /// # 多任务学习示例
    /// ```ignore
    /// optimizer.zero_grad()?;
    /// let v1 = loss1.backward()?;  // 第一个 loss
    /// let v2 = loss2.backward()?;  // 第二个 loss，梯度自动累积到共享参数
    /// optimizer.step()?;
    /// ```
    ///
    /// # 返回值
    /// 返回 loss 的标量值
    pub fn backward(&self) -> Result<f32, GraphError> {
        let graph_rc = self.graph();
        let mut g = graph_rc.borrow_mut();
        // ensure-forward：先执行前向传播
        g.forward_via_node_inner(&self.node)?;
        // 然后执行反向传播
        g.backward_via_node_inner(&self.node)
    }

    pub(crate) fn backward_timed(&self) -> Result<(f32, Duration, Duration), GraphError> {
        let graph_rc = self.graph();
        let mut g = graph_rc.borrow_mut();

        let forward_start = Instant::now();
        g.forward_via_node_inner(&self.node)?;
        let forward_elapsed = forward_start.elapsed();

        let propagate_start = Instant::now();
        let loss = g.backward_via_node_inner(&self.node)?;
        let propagate_elapsed = propagate_start.elapsed();

        Ok((loss, forward_elapsed, propagate_elapsed))
    }

    // ==================== 值访问和设置 ====================

    /// 获取节点的值（克隆的 Tensor）
    ///
    /// # 自动 forward 语义
    /// 如果节点尚未计算（value 为 None），会自动触发前向传播。
    /// 这让用户无需手动调用 `forward()` 即可直接获取值。
    pub fn value(&self) -> Result<Option<Tensor>, GraphError> {
        // 如果值还没计算，自动触发 forward
        if self.node.value().is_none() {
            self.forward()?;
        }
        Ok(self.node.value())
    }

    /// 设置节点的值
    pub fn set_value(&self, value: &Tensor) -> Result<(), GraphError> {
        self.node.set_value(Some(value))
    }

    /// 获取标量值（假设是 1x1 Tensor）
    pub fn item(&self) -> Result<f32, GraphError> {
        let val = self
            .value()?
            .ok_or(GraphError::NodeNotFound(self.node_id()))?;
        val.get_data_number()
            .ok_or_else(|| GraphError::InvalidOperation("Tensor 不是标量".to_string()))
    }

    /// 获取节点的梯度
    pub fn grad(&self) -> Result<Option<Tensor>, GraphError> {
        Ok(self.node.grad())
    }

    // ==================== 安全版本（返回 Result）====================

    /// 安全的加法（返回 Result）
    pub fn try_add(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行加法".to_string(),
            ));
        }
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_add_node(vec![Rc::clone(&self.node), Rc::clone(&other.node)], None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    /// 安全的减法（返回 Result）
    ///
    /// 使用 Subtract 节点实现，支持广播
    pub fn try_sub(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行减法".to_string(),
            ));
        }
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_subtract_node(vec![Rc::clone(&self.node), Rc::clone(&other.node)], None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    /// 安全的元素级乘法（返回 Result）
    pub fn try_mul(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行乘法".to_string(),
            ));
        }
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_multiply_node(vec![Rc::clone(&self.node), Rc::clone(&other.node)], None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    /// 安全的除法（返回 Result）
    ///
    /// 逐元素除法：`self / other`
    pub fn try_div(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行除法".to_string(),
            ));
        }
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_divide_node(vec![Rc::clone(&self.node), Rc::clone(&other.node)], None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    /// 逐元素取最小值：`min(self, other)`
    ///
    /// 用于 TD3/SAC 的 Twin Q 网络：`min(Q1, Q2)` 减少 Q 值过估计。
    ///
    /// # 示例
    /// ```ignore
    /// let q_min = q1_var.minimum(&q2_var)?;
    /// ```
    pub fn minimum(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行 minimum".to_string(),
            ));
        }
        let graph = self.graph();
        let node = graph.borrow_mut().create_minimum_node(
            Rc::clone(&self.node),
            Rc::clone(&other.node),
            None,
        )?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    /// 逐元素取最大值：`max(self, other)`
    ///
    /// # 示例
    /// ```ignore
    /// let q_max = q1_var.maximum(&q2_var)?;
    /// ```
    pub fn maximum(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行 maximum".to_string(),
            ));
        }
        let graph = self.graph();
        let node = graph.borrow_mut().create_maximum_node(
            Rc::clone(&self.node),
            Rc::clone(&other.node),
            None,
        )?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    /// 创建与当前 Var 相同形状的全零 Var
    ///
    /// 返回一个新的输入节点，值为全零。
    ///
    /// # 示例
    /// ```ignore
    /// let target = prediction.zeros_like()?;
    /// ```
    pub fn zeros_like(&self) -> Result<Self, GraphError> {
        let shape = self.node().shape();
        let graph = self.graph();
        let node = graph.borrow_mut().create_basic_input_node(&shape, None)?;
        node.set_value(Some(&Tensor::zeros(&shape)))?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    /// 创建与当前 Var 相同形状的随机 Var
    ///
    /// 返回一个新的输入节点，值为 U(-1, 1) 均匀分布。
    ///
    /// # 示例
    /// ```ignore
    /// let noise = z.rand_like()?; // GAN 生成器噪声
    /// ```
    pub fn rand_like(&self) -> Result<Self, GraphError> {
        let shape = self.node().shape();
        let graph = self.graph();
        let mut g = graph.borrow_mut();
        let node = g.create_basic_input_node(&shape, None)?;
        let data = if let Some(ref mut rng) = g.rng {
            use rand::distributions::{Distribution, Uniform};
            let dist = Uniform::from(-1.0f32..=1.0f32);
            let values: Vec<f32> = (0..shape.iter().product::<usize>())
                .map(|_| dist.sample(rng))
                .collect();
            Tensor::new(&values, &shape)
        } else {
            Tensor::random(-1.0, 1.0, &shape)
        };
        drop(g);
        node.set_value(Some(&data))?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    /// 创建与当前 Var 相同形状的正态分布随机 Var
    ///
    /// 返回一个新的输入节点，值为 N(0, 1) 标准正态分布。
    /// 当 Graph 有 seed 时使用 Graph RNG（确保确定性）。
    ///
    /// # 示例
    /// ```ignore
    /// let noise = z.randn_like()?; // GAN 生成器标准正态噪声
    /// ```
    pub fn randn_like(&self) -> Result<Self, GraphError> {
        let shape = self.node().shape();
        let graph = self.graph();
        let mut g = graph.borrow_mut();
        let node = g.create_basic_input_node(&shape, None)?;
        let data = if let Some(ref mut rng) = g.rng {
            Tensor::normal_with_rng(0.0, 1.0, &shape, rng)
        } else {
            Tensor::normal(0.0, 1.0, &shape)
        };
        drop(g);
        node.set_value(Some(&data))?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }

    /// 重新激活梯度（与 `detach()` 对称）
    ///
    /// 创建一个新的 Identity 节点，恢复梯度流。
    /// 适用于需要在 detach 后重新参与梯度计算的场景。
    ///
    /// # 示例
    /// ```ignore
    /// let detached = x.detach();  // 梯度阻断
    /// let reattached = detached.attach();  // 恢复梯度
    /// ```
    pub fn attach(&self) -> Self {
        let graph = self.graph();
        let new_node = graph
            .borrow_mut()
            .create_identity_node(Rc::clone(&self.node), None)
            .expect("内部错误：attach 创建 Identity 节点失败");
        Self {
            node: new_node,
            graph: self.graph.clone(),
        }
    }
}
