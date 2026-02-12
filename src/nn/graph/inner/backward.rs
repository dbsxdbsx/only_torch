/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @LastEditTime : 2026-02-02
 * @Description  : GraphInner 反向传播相关方法
 *
 * 动态图架构：已移除旧的 backward/backward_ex/backward_vjp_core 等方法
 * 统一使用 backward_via_node_inner() 通过 NodeInner 进行反向传播
 */

use super::super::error::GraphError;
use super::GraphInner;

impl GraphInner {
    /// 清零梯度（PyTorch 风格）
    ///
    /// 遍历 `parameters` 注册表，清除每个有效参数节点的梯度。
    pub fn zero_grad(&mut self) -> Result<(), GraphError> {
        self.zero_grad_via_parameters()
    }
}
