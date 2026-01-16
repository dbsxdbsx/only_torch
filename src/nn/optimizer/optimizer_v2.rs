/*
 * @Author       : 老董
 * @Date         : 2026-01-17
 * @Description  : V2 Optimizer API - PyTorch 风格
 *
 * 设计依据：architecture_v2_design.md §4.2.6
 *
 * 核心改进：
 * - Optimizer 持有 Rc<RefCell<GraphInner>> 引用
 * - zero_grad() 不再需要 &mut Graph 参数
 * - step() 不再需要 &mut Graph 参数
 * - 新增 minimize(&self, loss: &Var) 一步完成
 */

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::nn::graph::GraphInner;
use crate::nn::{GraphError, GraphHandle, NodeId, Var};
use crate::tensor::Tensor;

/// V2 Optimizer trait（PyTorch 风格）
///
/// # 设计要点
/// - Optimizer 绑定特定参数（通过 Var）
/// - `backward()` 计算所有参数的梯度（由 Var 调用）
/// - `step()` 只更新 Optimizer 绑定的参数
///
/// # 使用示例
/// ```ignore
/// let optimizer = SGDv2::new(&graph, &model.parameters(), 0.01);
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
pub trait OptimizerV2 {
    /// 清零所有参数的梯度
    fn zero_grad(&mut self) -> Result<(), GraphError>;

    /// 更新参数（只更新 Optimizer 绑定的参数）
    fn step(&mut self) -> Result<(), GraphError>;

    /// 一步完成：zero_grad + forward + backward + step
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

/// SGD V2 优化器（PyTorch 风格）
///
/// 随机梯度下降：θ = θ - α * ∇θ
///
/// # 使用示例
/// ```ignore
/// let optimizer = SGDv2::new(&graph, &model.parameters(), 0.01);
/// optimizer.zero_grad()?;
/// loss.backward()?;
/// optimizer.step()?;
/// ```
pub struct SGDv2 {
    /// 图引用
    graph: Rc<RefCell<GraphInner>>,
    /// 要优化的参数节点 ID
    params: Vec<NodeId>,
    /// 学习率
    lr: f32,
}

impl SGDv2 {
    /// 创建新的 SGD V2 优化器
    ///
    /// # 参数
    /// - `graph`: 图句柄
    /// - `params`: 要优化的参数 Var 列表
    /// - `lr`: 学习率
    pub fn new(graph: &GraphHandle, params: &[Var], lr: f32) -> Self {
        Self {
            graph: graph.inner_rc(),
            params: params.iter().map(|v| v.node_id()).collect(),
            lr,
        }
    }
}

impl OptimizerV2 for SGDv2 {
    fn zero_grad(&mut self) -> Result<(), GraphError> {
        let mut g = self.graph.borrow_mut();
        for &node_id in &self.params {
            g.clear_node_grad(node_id)?;
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), GraphError> {
        let mut g = self.graph.borrow_mut();
        for &node_id in &self.params {
            if let Some(grad) = g.get_node_grad(node_id)? {
                let current = g.get_node_value(node_id)?.ok_or_else(|| {
                    GraphError::ComputationError(format!("参数节点 {:?} 没有值", node_id))
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

/// Adam V2 优化器（PyTorch 风格）
///
/// Adam: Adaptive Moment Estimation
/// - m = β1 * m + (1 - β1) * g
/// - v = β2 * v + (1 - β2) * g²
/// - θ = θ - α * m_hat / (√v_hat + ε)
///
/// # 使用示例
/// ```ignore
/// let optimizer = Adamv2::new(&graph, &model.parameters(), 0.001);
/// optimizer.zero_grad()?;
/// loss.backward()?;
/// optimizer.step()?;
/// ```
pub struct Adamv2 {
    /// 图引用
    graph: Rc<RefCell<GraphInner>>,
    /// 要优化的参数节点 ID
    params: Vec<NodeId>,
    /// 学习率
    lr: f32,
    /// β1 (一阶矩衰减)
    beta1: f32,
    /// β2 (二阶矩衰减)
    beta2: f32,
    /// 数值稳定项
    epsilon: f32,
    /// 一阶矩估计
    m: HashMap<NodeId, Tensor>,
    /// 二阶矩估计
    v: HashMap<NodeId, Tensor>,
    /// 时间步
    t: usize,
}

impl Adamv2 {
    /// 创建新的 Adam V2 优化器
    ///
    /// # 参数
    /// - `graph`: 图句柄
    /// - `params`: 要优化的参数 Var 列表
    /// - `lr`: 学习率
    pub fn new(graph: &GraphHandle, params: &[Var], lr: f32) -> Self {
        Self::with_config(graph, params, lr, 0.9, 0.999, 1e-8)
    }

    /// 创建带完整配置的 Adam V2 优化器
    pub fn with_config(
        graph: &GraphHandle,
        params: &[Var],
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Self {
        Self {
            graph: graph.inner_rc(),
            params: params.iter().map(|v| v.node_id()).collect(),
            lr,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
}

impl OptimizerV2 for Adamv2 {
    fn zero_grad(&mut self) -> Result<(), GraphError> {
        let mut g = self.graph.borrow_mut();
        for &node_id in &self.params {
            g.clear_node_grad(node_id)?;
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), GraphError> {
        self.t += 1;
        let mut g = self.graph.borrow_mut();

        for &node_id in &self.params {
            if let Some(grad) = g.get_node_grad(node_id)? {
                let current = g.get_node_value(node_id)?.ok_or_else(|| {
                    GraphError::ComputationError(format!("参数节点 {:?} 没有值", node_id))
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
