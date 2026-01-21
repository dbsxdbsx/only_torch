/*
 * @Author       : 老董
 * @Date         : 2025-01-20
 * @Description  : MemoryLayer trait - 记忆型层的通用接口
 *
 * 所有具有时间状态的层（RNN, LSTM, GRU 等）都应实现此 trait。
 * 提供统一的 PyTorch 风格 API。
 */

use crate::nn::{GraphError, Module, NodeId, Var};
use crate::tensor::Tensor;

/// 记忆型层的通用接口
///
/// 所有具有时间状态的层（RNN、LSTM、GRU 等）都应实现此 trait。
/// 提供统一的 PyTorch 风格 API，简化用户使用。
///
/// # 核心方法
///
/// | 方法 | 用途 |
/// |------|------|
/// | `forward()` | 处理整个序列（PyTorch 风格，自动检测训练目标） |
/// | `forward_to()` | 处理整个序列（显式指定输出节点，NEAT/高级场景） |
/// | `step()` | 单步前向（手动控制/变长序列） |
/// | `output()` | 获取输出节点（图构建时连接后续层） |
/// | `reset()` | 重置隐藏状态（开始新序列前调用） |
///
/// # 自动检测机制
///
/// `forward()` 会自动从图结构中检测训练目标（loss 节点）：
/// 1. 优先使用 `Graph::set_training_target()` 显式设置的目标
/// 2. 否则自动查找 hidden_output 下游的终端节点
/// 3. 回退到 hidden_output（推理模式）
///
/// # 使用示例
///
/// ```ignore
/// // 完全 PyTorch 风格 - 无需额外设置！
/// let loss = model.output().cross_entropy(&labels)?;
///
/// for epoch in 0..epochs {
///     model.forward(&x_batch)?;  // 自动检测到 loss 节点
///     loss.backward()?;
///     optimizer.step(&mut graph)?;
/// }
/// ```
pub trait MemoryLayer: Module {
    /// 前向传播（PyTorch 风格：自动检测训练目标）
    ///
    /// 自动处理整个序列的所有时间步，用户无需手动迭代。
    /// 会自动从图结构中检测训练目标（loss 节点）。
    ///
    /// # 参数
    /// - `x`: 输入张量，形状 `[batch_size, seq_len, input_size]`
    ///
    /// # 返回
    /// 最终隐藏状态引用
    fn forward(&self, x: &Tensor) -> Result<&Var, GraphError>;

    /// 前向传播（显式指定输出节点）
    ///
    /// 自动处理整个序列的所有时间步，用户无需手动迭代。
    /// 适用于需要精确控制输出节点的场景（如 NEAT、多输出网络）。
    ///
    /// # 参数
    /// - `x`: 输入张量，形状 `[batch_size, seq_len, input_size]`
    /// - `output_node`: 输出节点 ID（如 loss 节点），用于记录完整的计算图历史
    ///
    /// # 返回
    /// 最终隐藏状态引用
    fn forward_to(&self, x: &Tensor, output_node: NodeId) -> Result<&Var, GraphError>;

    /// 单步前向传播
    ///
    /// 设置输入并执行一个时间步的计算。
    /// 用于需要逐时间步控制的场景（如变长序列、手动 BPTT）。
    ///
    /// # 参数
    /// - `x`: 输入张量，形状 `[batch_size, input_size]`
    ///
    /// # 返回
    /// 隐藏状态输出的引用
    fn step(&self, x: &Tensor) -> Result<&Var, GraphError>;

    /// 获取输出节点（用于图构建时连接后续层）
    ///
    /// 返回隐藏状态输出的 Var 引用，可用于连接到其他层。
    fn output(&self) -> &Var;

    /// 重置隐藏状态（开始新序列前调用）
    ///
    /// 清除所有时间步历史和隐藏状态，为处理新序列做准备。
    fn reset(&self);
}
