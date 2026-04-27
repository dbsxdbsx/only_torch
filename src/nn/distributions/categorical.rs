/*
 * @Author       : 老董
 * @Date         : 2026-02-12
 * @LastEditTime : 2026-02-13
 * @Description  : Categorical 分布（离散分类分布）
 *
 * 从 logits 构建离散概率分布，支持计算图内的 log_prob 和 entropy。
 * 用于 SAC-Discrete / Hybrid SAC 的离散动作部分。
 *
 * # 设计原则
 * - **需要梯度 → Var，不需要梯度 → Tensor**
 * - 构造时预计算并缓存 `probs`（softmax）和 `log_probs`（log_softmax）Var，
 *   所有方法共享同一组图节点，避免冗余计算
 * - 唯一返回 Tensor 的方法是 `sample()`（采样不可微）
 *
 * # 参考
 * - PyTorch: `torch.distributions.Categorical`
 */

use crate::nn::Var;
use crate::nn::graph::NodeGroupContext;
use crate::nn::{VarActivationOps, VarReduceOps, VarShapeOps};
use crate::tensor::Tensor;

/// Categorical 分布（离散分类分布）
///
/// 构造时预计算 `probs`（softmax）和 `log_probs`（log_softmax），
/// 所有方法共享同一组图节点。
///
/// # 构造
/// ```ignore
/// let logits = actor.forward(&obs)?;  // Var [batch, num_classes]
/// let dist = Categorical::new(logits);
/// ```
///
/// # 使用模式
/// ```ignore
/// // Var 级（在计算图中，可反向传播）
/// let probs = dist.probs();             // Var [batch, num_classes]
/// let log_probs = dist.log_probs();     // Var [batch, num_classes]
/// let entropy = dist.entropy();         // Var [batch, 1]
/// let lp = dist.log_prob(&action);      // Var [batch, 1]
///
/// // Tensor 级（不参与梯度，仅用于采样）
/// let action = dist.sample();           // Tensor [batch, 1]
/// ```
#[derive(Clone)]
pub struct Categorical {
    /// 未归一化的 logits
    logits: Var,
    /// softmax(logits) — 构造时缓存，所有方法共享
    probs: Var,
    /// log_softmax(logits) — 构造时缓存，所有方法共享
    log_probs: Var,
    /// 分组实例 ID（用于可视化 cluster 标记）
    instance_id: usize,
}

impl Categorical {
    /// 从 logits 创建 Categorical 分布
    ///
    /// 构造时立即创建 softmax 和 log_softmax 节点并缓存，
    /// 后续所有方法复用这两个节点，不会产生冗余图节点。
    ///
    /// # 参数
    /// - `logits` — 未归一化的对数概率，形状 `[batch, num_classes]`
    pub fn new(logits: Var) -> Self {
        let instance_id = logits.graph().borrow_mut().next_node_group_instance_id();
        let _guard = NodeGroupContext::new(&logits, "Categorical", instance_id);
        let probs = logits.softmax();
        let log_probs = logits.log_softmax();
        Self {
            logits,
            probs,
            log_probs,
            instance_id,
        }
    }

    /// 概率向量（Var 级，在计算图中）
    ///
    /// 返回缓存的 softmax(logits)，形状 `[batch, num_classes]`。
    /// 多次调用返回同一图节点的 clone（零开销）。
    pub fn probs(&self) -> Var {
        self.probs.clone()
    }

    /// 对数概率向量（Var 级，在计算图中）
    ///
    /// 返回缓存的 log_softmax(logits)，形状 `[batch, num_classes]`。
    /// 多次调用返回同一图节点的 clone（零开销）。
    pub fn log_probs(&self) -> Var {
        self.log_probs.clone()
    }

    /// 原始 logits（Var 级）
    pub fn logits(&self) -> Var {
        self.logits.clone()
    }

    /// 按概率采样动作索引（Tensor 级，不在计算图中）
    ///
    /// 返回形状 `[batch, 1]`，元素为采样到的类别索引（f32）。
    ///
    /// # 注意
    /// - 采样不可微，因此返回 Tensor 而非 Var
    /// - 使用缓存的 probs Var 的值进行 multinomial 采样
    pub fn sample(&self) -> Tensor {
        let probs_val = self
            .probs
            .value()
            .expect("Categorical sample: forward 失败")
            .expect("Categorical sample: probs 无值");
        let graph = self.probs.graph();
        let mut g = graph.borrow_mut();
        if let Some(ref mut rng) = g.rng {
            probs_val.multinomial_with_rng(1, rng)
        } else {
            drop(g);
            probs_val.multinomial(1)
        }
    }

    /// 指定动作的对数概率（Var 级，可反向传播）
    ///
    /// 等价于 `log_softmax(logits)[action]`，使用缓存的 log_probs。
    ///
    /// # 参数
    /// - `action` — 动作索引 Tensor，形状 `[batch, 1]`
    ///
    /// # 返回
    /// Var，形状 `[batch, 1]`
    pub fn log_prob(&self, action: &Tensor) -> Var {
        let _guard = NodeGroupContext::new(&self.logits, "Categorical", self.instance_id);
        self.log_probs
            .gather(1, action)
            .expect("Categorical log_prob: gather 失败")
    }

    /// 分布熵（Var 级，可反向传播）
    ///
    /// H = -Σ_a p(a) * log p(a)
    ///
    /// 使用缓存的 probs 和 log_probs，不会创建冗余的 softmax/log_softmax 节点。
    ///
    /// # 返回
    /// Var，形状 `[batch, 1]`
    pub fn entropy(&self) -> Var {
        let _guard = NodeGroupContext::new(&self.logits, "Categorical", self.instance_id);
        let weighted = &self.probs * &self.log_probs;
        (-&weighted).sum_axis(1)
    }
}
