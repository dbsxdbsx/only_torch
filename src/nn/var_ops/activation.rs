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
/// - `log_softmax()`: LogSoftmax（数值稳定的 log(softmax)）
/// - `softplus()`: `SoftPlus` 激活
/// - `step()`: 阶跃函数（用于二分类预测）
/// - `sign()`: 符号函数
/// - `abs()`: 绝对值函数
/// - `ln()`: 自然对数函数
///
/// # 使用示例
/// ```ignore
/// use only_torch::nn::var::{Var, VarActivationOps};
///
/// let h = x.relu().sigmoid();
/// let pred = output.step();
/// let probs = logits.softmax();
/// let log_probs = logits.log_softmax();  // 数值稳定
/// let magnitude = diff.abs();
/// ```
pub trait VarActivationOps {
    /// `ReLU` 激活：max(0, x)
    fn relu(&self) -> Var;

    /// Sigmoid 激活：1 / (1 + exp(-x))
    fn sigmoid(&self) -> Var;

    /// Tanh 激活：(exp(x) - exp(-x)) / (exp(x) + exp(-x))
    fn tanh(&self) -> Var;

    /// `LeakyReLU` 激活：x if x > 0 else alpha * x
    fn leaky_relu(&self, alpha: f32) -> Var;

    /// Softmax `激活：exp(x_i)` / Σ `exp(x_j)`
    ///
    /// 沿最后一维计算 softmax，将 logits 转换为概率分布。
    /// 输入形状 [batch, `num_classes`]，输出形状相同。
    fn softmax(&self) -> Var;

    /// LogSoftmax：数值稳定的 log(softmax(x))
    ///
    /// 沿最后一维计算 log_softmax。比 `softmax().ln()` 更数值稳定，
    /// 避免 softmax 输出接近 0 时的精度问题。
    ///
    /// 输入形状 [batch, `num_classes`]，输出形状相同。
    /// 常用于计算 log 概率（如 SAC Actor Loss）。
    fn log_softmax(&self) -> Var;

    /// `SoftPlus` 激活：log(1 + exp(x))
    ///
    /// 平滑的 `ReLU` 近似，常用于需要平滑非线性的场景
    fn softplus(&self) -> Var;

    /// Step 函数（阶跃函数）：1 if x >= 0 else 0
    fn step(&self) -> Var;

    /// Sign 函数（符号函数）：1 if x > 0, 0 if x == 0, -1 if x < 0
    fn sign(&self) -> Var;

    /// Abs 函数（绝对值）：|x|
    ///
    /// 梯度为 sign(x)，在 x=0 处为 0（与 `PyTorch` 行为一致）。
    /// 常用于 L1 损失、L1 正则化、距离计算等场景。
    fn abs(&self) -> Var;

    /// Ln 函数（自然对数）：ln(x)
    ///
    /// 梯度为 1/x。输入 x 必须为正数，否则结果为 NaN 或 -Inf。
    /// 常用于计算 log 概率、KL 散度、交叉熵等场景。
    fn ln(&self) -> Var;

    /// Exp 函数（指数）：e^x
    ///
    /// 梯度为 e^x（等于自身前向输出值）。与 Ln 互为反函数。
    /// 常用于 SAC Actor 的 `log_std.exp() → std` 转换、概率密度计算等场景。
    fn exp(&self) -> Var;

    /// Clip 函数（值域裁剪）：clip(x, min, max)
    ///
    /// 将每个元素限制在 `[min, max]` 范围内。边界处梯度为 0。
    /// 等价于 NumPy `np.clip()` / PyTorch `torch.clamp()`。
    /// 常用于 SAC `log_std` 裁剪、PPO ratio clipping、数值稳定等场景。
    fn clip(&self, min: f32, max: f32) -> Var;

    /// Sqrt 函数（平方根）：√x
    ///
    /// 梯度为 0.5/√x。输入 x 应为非负数，否则结果为 NaN。
    /// 常用于 Adam 优化器的 `sqrt(v + eps)`、距离计算等场景。
    fn sqrt(&self) -> Var;
}

impl VarActivationOps for Var {
    fn relu(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_relu_node(Rc::clone(self.node()), None)
            .expect("创建 ReLU 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn sigmoid(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_sigmoid_node(Rc::clone(self.node()), None)
            .expect("创建 Sigmoid 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn tanh(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_tanh_node(Rc::clone(self.node()), None)
            .expect("创建 Tanh 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn leaky_relu(&self, alpha: f32) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_leaky_relu_node(Rc::clone(self.node()), alpha, None)
            .expect("创建 LeakyReLU 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn softmax(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_softmax_node(Rc::clone(self.node()), None)
            .expect("创建 Softmax 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn log_softmax(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_log_softmax_node(Rc::clone(self.node()), None)
            .expect("创建 LogSoftmax 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn softplus(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_softplus_node(Rc::clone(self.node()), None)
            .expect("创建 SoftPlus 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn step(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_step_node(Rc::clone(self.node()), None)
            .expect("创建 Step 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn sign(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_sign_node(Rc::clone(self.node()), None)
            .expect("创建 Sign 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn abs(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_abs_node(Rc::clone(self.node()), None)
            .expect("创建 Abs 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn ln(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_ln_node(Rc::clone(self.node()), None)
            .expect("创建 Ln 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn exp(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_exp_node(Rc::clone(self.node()), None)
            .expect("创建 Exp 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn clip(&self, min: f32, max: f32) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_clip_node(Rc::clone(self.node()), min, max, None)
            .expect("创建 Clip 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }

    fn sqrt(&self) -> Var {
        let graph = self.graph();
        let node = graph
            .borrow_mut()
            .create_sqrt_node(Rc::clone(self.node()), None)
            .expect("创建 Sqrt 节点失败");
        Self::new_with_rc_graph(node, &graph)
    }
}
