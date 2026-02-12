/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @LastEditTime : 2026-02-02
 * @Description  : GraphInner train/eval 模式
 *
 * detach 机制通过 Var::detach() 和 NodeInner 实现
 */

use super::GraphInner;

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

    /// `no_grad` 上下文
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

    // 注意：detach/attach_node/is_node_detached 已移除
    // 新架构下，detach 通过 Var::detach() 和 NodeInner.set_detached() 实现
}
