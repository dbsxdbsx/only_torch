/*
 * @Author       : 老董
 * @Date         : 2026-01-09
 * @Description  : Module trait 定义
 *
 * 设计依据：architecture_v2_design.md §4.2.4
 */

use super::Var;

/// 模块 trait
///
/// # 设计原则
/// - `forward()` **不是** trait 方法（签名各异）
/// - `new()` **不是** trait 方法（参数各异）
/// - `parameters()` 返回 `Vec<Var>`（签名一致，放入 trait）
/// - 由于 Var 携带图引用，`forward()` 不需要 `&Graph` 参数
///
/// # 使用示例
///
/// ```ignore
/// use only_torch::nn::{Module, Var, Graph, Init};
///
/// struct MLP {
///     fc1: Linear,
///     fc2: Linear,
/// }
///
/// impl MLP {
///     fn new(graph: &Graph, in_dim: usize, hidden: usize, out_dim: usize) -> Self {
///         MLP {
///             fc1: Linear::new(graph, in_dim, hidden, true, "fc1"),
///             fc2: Linear::new(graph, hidden, out_dim, true, "fc2"),
///         }
///     }
///
///     fn forward(&self, x: &Var) -> Var {
///         let h = self.fc1.forward(x).relu();
///         self.fc2.forward(&h)
///     }
/// }
///
/// impl Module for MLP {
///     fn parameters(&self) -> Vec<Var> {
///         [self.fc1.parameters(), self.fc2.parameters()].concat()
///     }
/// }
/// ```
pub trait Module {
    /// 获取所有可训练参数
    ///
    /// 这是 Module trait 的唯一必须实现的方法。
    /// 用于：
    /// - 优化器需要知道要更新哪些参数
    /// - 序列化/保存模型参数
    /// - 统计参数数量
    fn parameters(&self) -> Vec<Var>;

    /// 获取参数数量
    fn num_params(&self) -> usize {
        self.parameters().len()
    }
}

/// 软更新：target = tau * source + (1 - tau) * target
///
/// 用于强化学习中目标网络的平滑更新。
///
/// # 参数
/// - `target`: 目标网络（被更新）
/// - `source`: 源网络（提供新参数）
/// - `tau`: 更新系数，范围 [0, 1]
///   - tau = 0: target 不变
///   - tau = 1: target 完全变为 source（硬更新）
///   - tau ∈ (0, 1): 平滑更新（常用 0.005）
///
/// # 示例
/// ```ignore
/// use only_torch::nn::{soft_update, Module};
///
/// // SAC 目标网络软更新
/// soft_update(&target_critic, &critic, 0.005);
/// ```
pub fn soft_update<T: Module, S: Module>(target: &T, source: &S, tau: f32) {
    for (t_var, s_var) in target.parameters().iter().zip(source.parameters().iter()) {
        if let (Ok(Some(mut t_val)), Ok(Some(s_val))) = (t_var.value(), s_var.value()) {
            t_val.soft_update(&s_val, tau);
            let _ = t_var.set_value(&t_val);
        }
    }
}

/// 硬更新：target = source
///
/// 等价于 `soft_update(target, source, 1.0)`，语义更清晰。
/// 常用于初始化目标网络。
///
/// # 示例
/// ```ignore
/// use only_torch::nn::{hard_update, Module};
///
/// // 初始化目标网络为与主网络相同
/// hard_update(&target_critic, &critic);
/// ```
pub fn hard_update<T: Module, S: Module>(target: &T, source: &S) {
    soft_update(target, source, 1.0);
}
