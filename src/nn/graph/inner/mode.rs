/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner train/eval 模式、detach 机制
 */

use super::super::error::GraphError;
use super::GraphInner;
use crate::nn::NodeId;

impl GraphInner {
    pub const fn set_train_mode(&mut self) {
        self.is_eval_mode = false;
    }

    pub const fn set_eval_mode(&mut self) {
        self.is_eval_mode = true;
    }

    pub const fn is_train_mode(&self) -> bool {
        !self.is_eval_mode
    }

    pub const fn is_grad_enabled(&self) -> bool {
        self.is_train_mode()
    }

    /// 设置 BPTT 调试模式
    #[cfg(test)]
    pub fn set_bptt_debug(&mut self, debug: bool) {
        self.bptt_debug = debug;
    }

    // ========== detach 机制 ==========

    /// 将节点标记为 detached
    pub fn detach_node(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        self.get_node_mut(node_id)?.set_detached(true);
        Ok(())
    }

    /// 取消节点的 detach 状态
    pub fn attach_node(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        self.get_node_mut(node_id)?.set_detached(false);
        Ok(())
    }

    /// 检查节点是否被 detach
    pub fn is_node_detached(&self, node_id: NodeId) -> Result<bool, GraphError> {
        Ok(self.get_node(node_id)?.is_detached())
    }

    /// no_grad 上下文
    pub fn no_grad_scope<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let was_train = self.is_train_mode();
        self.set_eval_mode();
        let result = f(self);
        if was_train {
            self.set_train_mode();
        }
        result
    }
}
