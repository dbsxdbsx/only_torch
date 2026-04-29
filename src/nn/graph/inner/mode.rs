/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @LastEditTime : 2026-04-29
 * @Description  : GraphInner 执行模式
 *
 * 用单一的 Mode { Train, Inference } 取代旧的 ExecutionContext。
 * detach 机制通过 Var::detach() 和 NodeInner 实现。
 */

use super::GraphInner;
use crate::nn::graph::types::Mode;

impl GraphInner {
    /// 切换到训练模式（forward 写 backward 缓存 + Dropout/BN 训练态 + backward 允许）
    pub const fn train(&mut self) {
        self.mode = Mode::Train;
    }

    /// 切换到推理模式（forward 跳过 backward 缓存 + Dropout/BN 评估态 + backward 报错）
    pub const fn inference(&mut self) {
        self.mode = Mode::Inference;
    }

    /// 是否处于训练模式
    pub const fn is_training(&self) -> bool {
        self.mode.is_training()
    }

    /// 当前执行模式
    pub const fn mode(&self) -> Mode {
        self.mode
    }

    /// 直接设置执行模式
    pub const fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
    }

    /// 临时进入推理模式执行闭包，闭包返回后自动恢复原模式
    pub fn inference_scope<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let prev = self.mode;
        self.mode = Mode::Inference;
        let result = f(self);
        self.mode = prev;
        result
    }
}
