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
/// use only_torch::nn::{Module, Var, GraphHandle, Init};
///
/// struct MLP {
///     fc1: Linear,
///     fc2: Linear,
/// }
///
/// impl MLP {
///     fn new(graph: &GraphHandle, in_dim: usize, hidden: usize, out_dim: usize) -> Self {
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
