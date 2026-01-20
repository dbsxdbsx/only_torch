/*
 * @Author       : 老董
 * @Date         : 2026-01-08
 * @Description  : Smart Var - 智能变量句柄，支持算子重载和链式调用
 *
 * 这是 V2 架构的核心组件，提供 PyTorch 级用户体验。
 */

use super::graph::GraphInner;
use super::{GraphError, NodeId};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

// ==================== Init 枚举 ====================

/// 参数初始化策略
#[derive(Debug, Clone)]
pub enum Init {
    /// 常数初始化
    Constant(f32),
    /// 全零
    Zeros,
    /// 全一
    Ones,
    /// 正态分布（使用 Graph 的 RNG）
    Normal { mean: f32, std: f32 },
    /// Kaiming/He 初始化（适用于 `ReLU`）
    Kaiming,
    /// Xavier/Glorot 初始化（适用于 Sigmoid/Tanh）
    Xavier,
}

impl Init {
    /// 生成初始化后的 Tensor（使用全局 RNG）
    pub fn generate(&self, shape: &[usize]) -> Tensor {
        match self {
            Self::Constant(v) => &Tensor::ones(shape) * *v,
            Self::Zeros => Tensor::zeros(shape),
            Self::Ones => Tensor::ones(shape),
            Self::Normal { mean, std } => Tensor::normal(*mean, *std, shape),
            Self::Kaiming => {
                let fan_in = shape[0];
                let std = (2.0 / fan_in as f32).sqrt();
                Tensor::normal(0.0, std, shape)
            }
            Self::Xavier => {
                let (fan_in, fan_out) = (shape[0], shape.get(1).copied().unwrap_or(1));
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                Tensor::normal(0.0, std, shape)
            }
        }
    }

    /// 生成初始化后的 Tensor（使用指定的 RNG）
    pub fn generate_with_rng(&self, shape: &[usize], rng: &mut rand::rngs::StdRng) -> Tensor {
        match self {
            Self::Constant(v) => &Tensor::ones(shape) * *v,
            Self::Zeros => Tensor::zeros(shape),
            Self::Ones => Tensor::ones(shape),
            Self::Normal { mean, std } => Tensor::normal_with_rng(*mean, *std, shape, rng),
            Self::Kaiming => {
                let fan_in = shape[0];
                let std = (2.0 / fan_in as f32).sqrt();
                Tensor::normal_with_rng(0.0, std, shape, rng)
            }
            Self::Xavier => {
                let (fan_in, fan_out) = (shape[0], shape.get(1).copied().unwrap_or(1));
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                Tensor::normal_with_rng(0.0, std, shape, rng)
            }
        }
    }
}

// ==================== Var 结构 ====================

/// 智能变量句柄 - 携带图引用，支持算子重载和链式调用
///
/// # 设计原则
/// - 持有 `Rc<RefCell<GraphInner>>` 引用，实现算子重载
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
    /// 节点 ID
    id: NodeId,
    /// 图引用（用户不可见）
    graph: Rc<RefCell<GraphInner>>,
}

impl std::fmt::Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Var").field("id", &self.id).finish()
    }
}

impl Var {
    /// 创建新的 Var（内部使用）
    pub(crate) const fn new(id: NodeId, graph: Rc<RefCell<GraphInner>>) -> Self {
        Self { id, graph }
    }

    /// 获取节点 ID
    pub const fn node_id(&self) -> NodeId {
        self.id
    }

    /// 获取内部图引用（供 trait 和内部模块使用）
    pub(crate) const fn graph(&self) -> &Rc<RefCell<GraphInner>> {
        &self.graph
    }

    /// 检查两个 Var 是否来自同一个 Graph
    pub fn same_graph(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.graph, &other.graph)
    }

    /// 获取 Var 所属的 Graph handle
    ///
    /// 即使原始 Graph handle 已 drop，此方法仍返回有效的 Graph。
    /// 这是因为 Var 持有 `GraphInner` 的强引用（Rc）。
    pub fn get_graph(&self) -> super::graph::Graph {
        super::graph::Graph::from_rc(Rc::clone(&self.graph))
    }

    /// 获取节点的预期输出形状
    ///
    /// 这个形状在节点创建时就已确定。
    pub fn value_expected_shape(&self) -> Vec<usize> {
        self.graph
            .borrow()
            .get_node_value_expected_shape(self.id)
            .expect("获取形状失败")
            .to_vec()
    }

    /// 断言两个 Var 来自同一个 Graph，否则 panic（供 trait 使用）
    pub(crate) fn assert_same_graph(&self, other: &Self) {
        assert!(
            self.same_graph(other),
            "不能对来自不同 Graph 的 Var 进行操作"
        );
    }

    // ==================== 梯度流控制 ====================

    /// 截断梯度流（返回新的 detached Var）
    pub fn detach(&self) -> Result<Self, GraphError> {
        self.graph.borrow_mut().detach_node(self.id)?;
        Ok(self.clone())
    }

    /// 恢复梯度流
    pub fn attach(&self) -> Result<Self, GraphError> {
        self.graph.borrow_mut().attach_node(self.id)?;
        Ok(self.clone())
    }

    // ==================== 执行 ====================

    /// 前向传播
    pub fn forward(&self) -> Result<(), GraphError> {
        self.graph.borrow_mut().forward(self.id)
    }

    /// 反向传播（ensure-forward 语义）
    ///
    /// # 语义：ensure-forward
    /// - 自动先执行 forward()，确保 loss 值已计算
    /// - 然后执行反向传播
    ///
    /// # 返回值
    /// 返回 loss 的标量值
    pub fn backward(&self) -> Result<f32, GraphError> {
        let mut g = self.graph.borrow_mut();
        // ensure-forward：先执行前向传播
        g.forward(self.id)?;
        // 然后执行反向传播
        g.backward(self.id)
    }

    // ==================== 值访问和设置 ====================

    /// 获取节点的值（克隆的 Tensor）
    pub fn value(&self) -> Result<Option<Tensor>, GraphError> {
        Ok(self.graph.borrow().get_node_value(self.id)?.cloned())
    }

    /// 设置节点的值
    pub fn set_value(&self, value: &Tensor) -> Result<(), GraphError> {
        self.graph.borrow_mut().set_node_value(self.id, Some(value))
    }

    /// 获取标量值（假设是 1x1 Tensor）
    pub fn item(&self) -> Result<f32, GraphError> {
        let val = self.value()?.ok_or(GraphError::NodeNotFound(self.id))?;
        val.get_data_number()
            .ok_or_else(|| GraphError::InvalidOperation("Tensor 不是标量".to_string()))
    }

    /// 获取节点的梯度
    pub fn grad(&self) -> Result<Option<Tensor>, GraphError> {
        self.graph.borrow().get_node_grad(self.id)
    }

    // ==================== 安全版本（返回 Result）====================

    /// 安全的加法（返回 Result）
    pub fn try_add(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行加法".to_string(),
            ));
        }
        let id = self
            .graph
            .borrow_mut()
            .new_add_node(&[self.id, other.id], None)?;
        Ok(Self::new(id, Rc::clone(&self.graph)))
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
        let mut g = self.graph.borrow_mut();
        let id = g.new_subtract_node(self.id, other.id, None)?;
        Ok(Self::new(id, Rc::clone(&self.graph)))
    }

    /// 安全的元素级乘法（返回 Result）
    pub fn try_mul(&self, other: &Self) -> Result<Self, GraphError> {
        if !self.same_graph(other) {
            return Err(GraphError::InvalidOperation(
                "不能对来自不同 Graph 的 Var 进行乘法".to_string(),
            ));
        }
        let id = self
            .graph
            .borrow_mut()
            .new_multiply_node(self.id, other.id, None)?;
        Ok(Self::new(id, Rc::clone(&self.graph)))
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
        let id = self
            .graph
            .borrow_mut()
            .new_divide_node(self.id, other.id, None)?;
        Ok(Self::new(id, Rc::clone(&self.graph)))
    }
}

// ==================== 算子重载 ====================

// Add for &Var
impl Add for &Var {
    type Output = Var;

    fn add(self, other: &Var) -> Var {
        self.try_add(other).expect("Var 加法失败")
    }
}

// Add for Var (consumes self)
impl Add for Var {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        &self + &other
    }
}

// Add<Var> for &Var
impl Add<Var> for &Var {
    type Output = Var;

    fn add(self, other: Var) -> Var {
        self + &other
    }
}

// Add<&Var> for Var
impl Add<&Self> for Var {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        &self + other
    }
}

// Sub for &Var (实现为 self + (-1 * other))
impl Sub for &Var {
    type Output = Var;

    fn sub(self, other: &Var) -> Var {
        self.try_sub(other).expect("Var 减法失败")
    }
}

// Sub for Var
impl Sub for Var {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        &self - &other
    }
}

// Sub<Var> for &Var
impl Sub<Var> for &Var {
    type Output = Var;

    fn sub(self, other: Var) -> Var {
        self - &other
    }
}

// Sub<&Var> for Var
impl Sub<&Self> for Var {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        &self - other
    }
}

// Mul for &Var（逐元素乘法）
impl Mul for &Var {
    type Output = Var;

    fn mul(self, other: &Var) -> Var {
        self.try_mul(other).expect("Var 乘法失败")
    }
}

// Mul for Var
impl Mul for Var {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        &self * &other
    }
}

// Mul<Var> for &Var
impl Mul<Var> for &Var {
    type Output = Var;

    fn mul(self, other: Var) -> Var {
        self * &other
    }
}

// Mul<&Var> for Var
impl Mul<&Self> for Var {
    type Output = Self;

    fn mul(self, other: &Self) -> Self {
        &self * other
    }
}

// Div for &Var（逐元素除法）
impl Div for &Var {
    type Output = Var;

    fn div(self, other: &Var) -> Var {
        self.try_div(other).expect("Var 除法失败")
    }
}

// Div for Var
impl Div for Var {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        &self / &other
    }
}

// Div<Var> for &Var
impl Div<Var> for &Var {
    type Output = Var;

    fn div(self, other: Var) -> Var {
        self / &other
    }
}

// Div<&Var> for Var
impl Div<&Self> for Var {
    type Output = Self;

    fn div(self, other: &Self) -> Self {
        &self / other
    }
}

// Neg for &Var（实现为 -1 * self）
impl Neg for &Var {
    type Output = Var;

    fn neg(self) -> Var {
        let mut g = self.graph.borrow_mut();
        // 创建 -1 常量
        let neg_one_id = g.new_input_node(&[1, 1], None).expect("创建 -1 节点失败");
        g.set_node_value(neg_one_id, Some(&Tensor::new(&[-1.0], &[1, 1])))
            .expect("设置 -1 值失败");
        // -self = -1 * self（Multiply 支持广播）
        let id = g
            .new_multiply_node(neg_one_id, self.id, None)
            .expect("创建取反节点失败");
        Var::new(id, Rc::clone(&self.graph))
    }
}

// Neg for Var
impl Neg for Var {
    type Output = Self;

    fn neg(self) -> Self {
        -&self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_zeros() {
        let tensor = Init::Zeros.generate(&[2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert!(tensor.data_as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_init_ones() {
        let tensor = Init::Ones.generate(&[2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert!(tensor.data_as_slice().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_init_kaiming() {
        let tensor = Init::Kaiming.generate(&[100, 50]);
        assert_eq!(tensor.shape(), &[100, 50]);
        // Kaiming: std = sqrt(2/fan_in) = sqrt(2/100) ≈ 0.1414
        let expected_std = (2.0 / 100.0_f32).sqrt();
        let data = tensor.data_as_slice();
        let actual_std = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
        let actual_std = actual_std.sqrt();
        assert!((actual_std - expected_std).abs() < 0.05);
    }

    #[test]
    fn test_init_xavier() {
        let tensor = Init::Xavier.generate(&[100, 50]);
        assert_eq!(tensor.shape(), &[100, 50]);
        // Xavier: std = sqrt(2/(fan_in + fan_out)) = sqrt(2/150) ≈ 0.1155
        let expected_std = (2.0 / 150.0_f32).sqrt();
        let data = tensor.data_as_slice();
        let actual_std = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
        let actual_std = actual_std.sqrt();
        assert!((actual_std - expected_std).abs() < 0.05);
    }
}
