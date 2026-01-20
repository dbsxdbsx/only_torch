/*
 * @Author       : 老董
 * @Date         : 2026-01-09
 * @Description  : Var 激活函数扩展 trait
 *
 * 提供激活函数的链式调用支持，用户需 import 此 trait 后才能使用。
 * 设计依据：architecture_v2_design.md §4.2.1.3
 */

use crate::nn::Var;
use std::rc::Rc;

/// 激活函数扩展 trait
///
/// 提供常用激活函数的链式调用：
/// - `relu()`: `ReLU` 激活
/// - `sigmoid()`: Sigmoid 激活
/// - `tanh()`: Tanh 激活
/// - `leaky_relu(alpha)`: `LeakyReLU` 激活
/// - `softmax()`: Softmax 激活（沿最后一维归一化）
/// - `softplus()`: `SoftPlus` 激活
/// - `step()`: 阶跃函数（用于二分类预测）
/// - `sign()`: 符号函数
///
/// # 使用示例
/// ```ignore
/// use only_torch::nn::var::{Var, VarActivationOps};
///
/// let h = x.relu().sigmoid();
/// let pred = output.step();
/// let probs = logits.softmax();
/// ```
pub trait VarActivationOps {
    /// `ReLU` 激活：max(0, x)
    fn relu(&self) -> Var;

    /// Sigmoid 激活：1 / (1 + exp(-x))
    fn sigmoid(&self) -> Var;

    /// Tanh 激活：(exp(x) - exp(-x)) / (exp(x) + exp(-x))
    fn tanh(&self) -> Var;

    /// `LeakyReLU` 激活：x if x > 0 else alpha * x
    fn leaky_relu(&self, alpha: f64) -> Var;

    /// Softmax `激活：exp(x_i)` / Σ `exp(x_j)`
    ///
    /// 沿最后一维计算 softmax，将 logits 转换为概率分布。
    /// 输入形状 [batch, `num_classes`]，输出形状相同。
    fn softmax(&self) -> Var;

    /// `SoftPlus` 激活：log(1 + exp(x))
    ///
    /// 平滑的 `ReLU` 近似，常用于需要平滑非线性的场景
    fn softplus(&self) -> Var;

    /// Step 函数（阶跃函数）：1 if x >= 0 else 0
    fn step(&self) -> Var;

    /// Sign 函数（符号函数）：1 if x > 0, 0 if x == 0, -1 if x < 0
    fn sign(&self) -> Var;
}

impl VarActivationOps for Var {
    fn relu(&self) -> Var {
        let id = self
            .graph()
            .borrow_mut()
            .new_relu_node(self.node_id(), None)
            .expect("创建 ReLU 节点失败");
        Self::new(id, Rc::clone(self.graph()))
    }

    fn sigmoid(&self) -> Var {
        let id = self
            .graph()
            .borrow_mut()
            .new_sigmoid_node(self.node_id(), None)
            .expect("创建 Sigmoid 节点失败");
        Self::new(id, Rc::clone(self.graph()))
    }

    fn tanh(&self) -> Var {
        let id = self
            .graph()
            .borrow_mut()
            .new_tanh_node(self.node_id(), None)
            .expect("创建 Tanh 节点失败");
        Self::new(id, Rc::clone(self.graph()))
    }

    fn leaky_relu(&self, alpha: f64) -> Var {
        let id = self
            .graph()
            .borrow_mut()
            .new_leaky_relu_node(self.node_id(), alpha, None)
            .expect("创建 LeakyReLU 节点失败");
        Self::new(id, Rc::clone(self.graph()))
    }

    fn softmax(&self) -> Var {
        let id = self
            .graph()
            .borrow_mut()
            .new_softmax_node(self.node_id(), None)
            .expect("创建 Softmax 节点失败");
        Self::new(id, Rc::clone(self.graph()))
    }

    fn softplus(&self) -> Var {
        let id = self
            .graph()
            .borrow_mut()
            .new_softplus_node(self.node_id(), None)
            .expect("创建 SoftPlus 节点失败");
        Self::new(id, Rc::clone(self.graph()))
    }

    fn step(&self) -> Var {
        let id = self
            .graph()
            .borrow_mut()
            .new_step_node(self.node_id(), None)
            .expect("创建 Step 节点失败");
        Self::new(id, Rc::clone(self.graph()))
    }

    fn sign(&self) -> Var {
        let id = self
            .graph()
            .borrow_mut()
            .new_sign_node(self.node_id(), None)
            .expect("创建 Sign 节点失败");
        Self::new(id, Rc::clone(self.graph()))
    }
}
