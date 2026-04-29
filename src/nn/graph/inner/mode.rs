/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @LastEditTime : 2026-02-02
 * @Description  : GraphInner train/eval 模式
 *
 * detach 机制通过 Var::detach() 和 NodeInner 实现
 */

use super::GraphInner;
use crate::nn::ExecutionContext;

impl GraphInner {
    pub const fn set_train_mode(&mut self) {
        self.execution_ctx.training = true;
    }

    pub const fn set_eval_mode(&mut self) {
        self.execution_ctx.training = false;
    }

    pub const fn is_train_mode(&self) -> bool {
        self.execution_ctx.training
    }

    pub const fn is_grad_enabled(&self) -> bool {
        self.execution_ctx.grad_enabled
    }

    pub const fn execution_ctx(&self) -> ExecutionContext {
        self.execution_ctx
    }

    pub const fn set_execution_ctx(&mut self, ctx: ExecutionContext) {
        self.execution_ctx = ctx;
    }

    /// `no_grad` 上下文
    pub fn no_grad_scope<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let was_grad_enabled = self.execution_ctx.grad_enabled;
        self.execution_ctx.grad_enabled = false;
        let result = f(self);
        self.execution_ctx.grad_enabled = was_grad_enabled;
        result
    }

    // 注意：detach/attach_node/is_node_detached 已移除
    // 新架构下，detach 通过 Var::detach() 和 NodeInner.set_detached() 实现
}
