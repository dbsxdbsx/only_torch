/*
 * @Author       : 老董
 * @Date         : 2026-01-17
 * @LastEditTime : 2026-01-17
 * @Description  : Optimizer API - PyTorch 风格
 *
 * 设计依据：architecture_v2_design.md §4.2.6
 *
 * 核心特性：
 * - Optimizer 持有 Rc<RefCell<GraphInner>> 引用
 * - params 存储 Vec<Var>（保留完整 Var 能力，支持未来 param_groups 等扩展）
 * - zero_grad() 不再需要 &mut Graph 参数
 * - step() 不再需要 &mut Graph 参数
 * - minimize(&self, loss: &Var) 一步完成训练
 */

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::nn::graph::GraphInner;
use crate::nn::{Graph, GraphError, NodeId, Var};
use crate::tensor::Tensor;

/// Optimizer trait（PyTorch 风格）
///
/// # 设计要点
/// - Optimizer 绑定特定参数（通过 Var）
/// - `backward()` 计算所有参数的梯度（由 Var 调用）
/// - `step()` 只更新 Optimizer 绑定的参数
///
/// # 使用示例
/// ```ignore
/// let optimizer = SGD::new(&graph, &model.parameters(), 0.01);
///
/// // 训练循环
/// optimizer.zero_grad()?;
/// let loss = model.forward(x)?.cross_entropy(&y)?;
/// loss.backward()?;
/// optimizer.step()?;
///
/// // 或者一步完成
/// let loss_val = optimizer.minimize(&loss)?;
/// ```
pub trait Optimizer {
    /// 清零所有参数的梯度
    fn zero_grad(&mut self) -> Result<(), GraphError>;

    /// 更新参数（只更新 Optimizer 绑定的参数）
    fn step(&mut self) -> Result<(), GraphError>;

    /// `一步完成训练：zero_grad` → backward(ensure-forward) → step
    ///
    /// # 执行顺序
    /// 1. `zero_grad()` - 清零梯度（必须在前，因为 backward 会累加梯度）
    /// 2. `loss.backward()` - 计算梯度（内部 ensure-forward：必要时先执行前向）
    /// 3. `step()` - 更新参数
    ///
    /// # 参数
    /// - `loss`: loss 节点的 Var
    ///
    /// # 返回
    /// loss 的标量值
    fn minimize(&mut self, loss: &Var) -> Result<f32, GraphError>;

    /// 获取学习率
    fn learning_rate(&self) -> f32;

    /// 设置学习率
    fn set_learning_rate(&mut self, lr: f32);

    /// 重置累积状态（如 Adam 的动量）
    fn reset(&mut self);
}

/// SGD 优化器（PyTorch 风格）
///
/// 随机梯度下降：θ = θ - α * ∇θ
///
/// # 使用示例
/// ```ignore
/// let optimizer = SGD::new(&graph, &model.parameters(), 0.01);
/// optimizer.zero_grad()?;
/// loss.backward()?;
/// optimizer.step()?;
/// ```
pub struct SGD {
    /// 图引用
    graph: Rc<RefCell<GraphInner>>,
    /// 要优化的参数（保留完整 Var，支持未来 `param_groups` 等扩展）
    params: Vec<Var>,
    /// 学习率
    lr: f32,
}

impl SGD {
    /// 创建新的 SGD 优化器
    ///
    /// # 参数
    /// - `graph`: 图句柄
    /// - `params`: 要优化的参数 Var 列表
    /// - `lr`: 学习率
    pub fn new(graph: &Graph, params: &[Var], lr: f32) -> Self {
        Self {
            graph: graph.inner_rc(),
            params: params.to_vec(),
            lr,
        }
    }

    /// 获取优化器绑定的参数列表
    ///
    /// `用于调试、状态查询、param_groups` 等场景
    pub fn params(&self) -> &[Var] {
        &self.params
    }
}

impl Optimizer for SGD {
    fn zero_grad(&mut self) -> Result<(), GraphError> {
        let mut g = self.graph.borrow_mut();
        for param in &self.params {
            g.clear_node_grad(param.node_id())?;
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), GraphError> {
        let mut g = self.graph.borrow_mut();
        for param in &self.params {
            let node_id = param.node_id();
            if let Some(grad) = g.get_node_grad(node_id)? {
                let current = g.get_node_value(node_id)?.ok_or_else(|| {
                    GraphError::ComputationError(format!("参数节点 {node_id:?} 没有值"))
                })?;
                let new_value = current - self.lr * &grad;
                g.set_node_value(node_id, Some(&new_value))?;
            }
        }
        Ok(())
    }

    fn minimize(&mut self, loss: &Var) -> Result<f32, GraphError> {
        self.zero_grad()?;
        let loss_val = loss.backward()?;
        self.step()?;
        Ok(loss_val)
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn reset(&mut self) {
        // SGD 无状态
    }
}

/// Adam 优化器（PyTorch 风格）
///
/// Adam: Adaptive Moment Estimation
/// - m = β1 * m + (1 - β1) * g
/// - v = β2 * v + (1 - β2) * g²
/// - θ = θ - α * `m_hat` / (√`v_hat` + ε)
///
/// # 使用示例
/// ```ignore
/// let optimizer = Adam::new(&graph, &model.parameters(), 0.001);
/// optimizer.zero_grad()?;
/// loss.backward()?;
/// optimizer.step()?;
/// ```
pub struct Adam {
    /// 图引用
    graph: Rc<RefCell<GraphInner>>,
    /// 要优化的参数（保留完整 Var，支持未来 `param_groups` 等扩展）
    params: Vec<Var>,
    /// 学习率
    lr: f32,
    /// β1 (一阶矩衰减)
    beta1: f32,
    /// β2 (二阶矩衰减)
    beta2: f32,
    /// 数值稳定项
    epsilon: f32,
    /// 一阶矩估计（按 `NodeId` 索引，高效查找）
    m: HashMap<NodeId, Tensor>,
    /// 二阶矩估计（按 `NodeId` 索引，高效查找）
    v: HashMap<NodeId, Tensor>,
    /// 时间步
    t: usize,
}

impl Adam {
    /// 创建新的 Adam 优化器
    ///
    /// # 参数
    /// - `graph`: 图句柄
    /// - `params`: 要优化的参数 Var 列表
    /// - `lr`: 学习率
    pub fn new(graph: &Graph, params: &[Var], lr: f32) -> Self {
        Self::new_with_config(graph, params, lr, 0.9, 0.999, 1e-8)
    }

    /// 创建带完整配置的 Adam 优化器
    pub fn new_with_config(
        graph: &Graph,
        params: &[Var],
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Self {
        Self {
            graph: graph.inner_rc(),
            params: params.to_vec(),
            lr,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// 获取优化器绑定的参数列表
    ///
    /// `用于调试、状态查询、param_groups` 等场景
    pub fn params(&self) -> &[Var] {
        &self.params
    }

    /// 获取指定参数的动量状态（一阶矩 m）
    ///
    /// 用于调试和可视化优化过程
    pub fn get_momentum(&self, param: &Var) -> Option<&Tensor> {
        self.m.get(&param.node_id())
    }

    /// 获取指定参数的速度状态（二阶矩 v）
    ///
    /// 用于调试和可视化优化过程
    pub fn get_velocity(&self, param: &Var) -> Option<&Tensor> {
        self.v.get(&param.node_id())
    }

    /// 获取当前时间步
    pub const fn timestep(&self) -> usize {
        self.t
    }
}

impl Optimizer for Adam {
    fn zero_grad(&mut self) -> Result<(), GraphError> {
        let mut g = self.graph.borrow_mut();
        for param in &self.params {
            g.clear_node_grad(param.node_id())?;
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), GraphError> {
        self.t += 1;
        let mut g = self.graph.borrow_mut();

        for param in &self.params {
            let node_id = param.node_id();
            if let Some(grad) = g.get_node_grad(node_id)? {
                let current = g.get_node_value(node_id)?.ok_or_else(|| {
                    GraphError::ComputationError(format!("参数节点 {node_id:?} 没有值"))
                })?;

                // 预计算
                let scaled_grad = &grad * (1.0 - self.beta1);
                let grad_squared = &grad * &grad;
                let scaled_grad_squared = &grad_squared * (1.0 - self.beta2);

                // 更新一阶矩
                let m = self
                    .m
                    .entry(node_id)
                    .or_insert_with(|| Tensor::zeros(grad.shape()));
                *m *= self.beta1;
                *m += &scaled_grad;

                // 更新二阶矩
                let v = self
                    .v
                    .entry(node_id)
                    .or_insert_with(|| Tensor::zeros(grad.shape()));
                *v *= self.beta2;
                *v += &scaled_grad_squared;

                // 偏差修正
                let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));

                // 更新参数
                let v_sqrt = v_hat.sqrt();
                let denom = &v_sqrt + self.epsilon;
                let update = &m_hat / &denom;
                let new_value = current - self.lr * &update;

                g.set_node_value(node_id, Some(&new_value))?;
            }
        }
        Ok(())
    }

    fn minimize(&mut self, loss: &Var) -> Result<f32, GraphError> {
        self.zero_grad()?;
        let loss_val = loss.backward()?;
        self.step()?;
        Ok(loss_val)
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}
