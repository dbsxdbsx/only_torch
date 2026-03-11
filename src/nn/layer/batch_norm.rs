/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : BatchNorm 层 - PyTorch 风格 API
 *
 * 完整批归一化：y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * 内部组合：
 * - BatchNormOp 节点：归一化运算（含 train/eval 模式和 running stats）
 * - gamma (scale) 参数节点
 * - beta (shift) 参数节点
 * - Mul + Add 节点：gamma * x_hat + beta
 *
 * BatchNorm1d: 输入 [N, C] 或 [N, C, L]
 * BatchNorm2d: 输入 [N, C, H, W]
 * 两者共用同一结构，仅初始形状不同。
 */

use crate::nn::graph::NodeGroupContext;
use crate::nn::{Graph, GraphError, Init, IntoVar, Module, Var, VarShapeOps};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

/// 批归一化层
///
/// `PyTorch` 风格的批归一化：`y = gamma * (x - mean) / sqrt(var + eps) + beta`
///
/// # 使用示例
/// ```ignore
/// // BatchNorm1d
/// let bn = BatchNorm::new(&graph, 64, 1e-5, 0.1, "bn1")?;
/// let h = bn.forward(&x);
///
/// // BatchNorm2d（API 相同）
/// let bn2d = BatchNorm::new(&graph, 32, 1e-5, 0.1, "bn2d")?;
/// let h = bn2d.forward(&conv_out);
/// ```
pub struct BatchNorm {
    /// 缩放参数 gamma，形状 [1, C, 1, ...] 或 [1, C]
    gamma: Var,
    /// 偏移参数 beta，形状同 gamma
    beta: Var,
    /// 通道数
    num_features: usize,
    /// 层名称
    name: String,
    /// 分组实例 ID
    instance_id: usize,
    /// eps
    eps: f32,
    /// momentum
    momentum: f32,
    /// 运行均值 [C]（跨 forward 调用持久化，与 BatchNormOp 节点共享）
    running_mean: Rc<RefCell<Tensor>>,
    /// 运行方差 [C]（跨 forward 调用持久化，与 BatchNormOp 节点共享）
    running_var: Rc<RefCell<Tensor>>,
}

impl BatchNorm {
    /// 创建新的 BatchNorm 层
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `num_features`: 通道数 C
    /// - `eps`: 数值稳定性常数（典型值 1e-5）
    /// - `momentum`: EMA 动量（典型值 0.1）
    /// - `name`: 层名称前缀
    pub fn new(
        graph: &Graph,
        num_features: usize,
        eps: f32,
        momentum: f32,
        name: &str,
    ) -> Result<Self, GraphError> {
        // gamma 初始化为 1（scale）
        let gamma = graph.parameter(&[1, num_features], Init::Ones, &format!("{name}_gamma"))?;

        // beta 初始化为 0（shift）
        let beta = graph.parameter(&[1, num_features], Init::Zeros, &format!("{name}_beta"))?;

        let instance_id = graph.inner_mut().next_node_group_instance_id();

        // 初始化共享 running stats
        let running_mean = Rc::new(RefCell::new(Tensor::zeros(&[num_features])));
        let running_var = Rc::new(RefCell::new(Tensor::ones(&[num_features])));

        Ok(Self {
            gamma,
            beta,
            num_features,
            name: name.to_string(),
            instance_id,
            eps,
            momentum,
            running_mean,
            running_var,
        })
    }

    /// 前向传播
    ///
    /// 计算 `gamma * batch_norm(x) + beta`
    ///
    /// # 参数
    /// - `x`: 输入 Var，形状 [N, C, ...] 其中 C = `num_features`
    pub fn forward(&self, x: impl IntoVar) -> Var {
        use std::rc::Rc as StdRc;
        let x = x
            .into_var(&self.gamma.get_graph())
            .expect("BatchNorm 输入转换失败");
        let graph = x.get_graph();

        // 分组上下文
        let desc = format!("C={}", self.num_features);
        let _guard =
            NodeGroupContext::for_layer(&x, "BatchNorm", self.instance_id, &self.name, &desc);
        _guard.tag_existing(&self.gamma);
        _guard.tag_existing(&self.beta);

        // 1. BatchNormOp: x → x_hat（归一化）
        //    传入共享的 running stats，确保跨 forward 调用持久化
        let x_hat = {
            let bn_node = graph
                .inner_mut()
                .create_batch_norm_op_node(
                    StdRc::clone(x.node()),
                    self.eps,
                    self.momentum,
                    Rc::clone(&self.running_mean),
                    Rc::clone(&self.running_var),
                    Some(&format!("{}_norm", self.name)),
                )
                .expect("BatchNorm 归一化失败");
            Var::new_with_rc_graph(bn_node, &graph.inner_rc())
        };

        // 2. 对 4D+ 输入，reshape gamma/beta 从 [1, C] 到 [1, C, 1, 1, ...]
        //    使通道维正确对齐（NumPy 广播从右对齐，[1, C] 会对齐到最后一维而非通道维）
        let x_hat_node = x_hat.node();
        let ndim = x_hat_node.value_expected_shape().len();

        let (gamma, beta) = if ndim > 2 {
            let mut param_shape = vec![1usize; ndim];
            param_shape[1] = self.num_features;
            (
                self.gamma.reshape(&param_shape).expect("BatchNorm gamma reshape 失败"),
                self.beta.reshape(&param_shape).expect("BatchNorm beta reshape 失败"),
            )
        } else {
            (self.gamma.clone(), self.beta.clone())
        };

        // 3. gamma * x_hat + beta
        &(&x_hat * &gamma) + &beta
    }

    /// 获取通道数
    pub const fn num_features(&self) -> usize {
        self.num_features
    }

    /// 获取 gamma 参数
    pub const fn gamma(&self) -> &Var {
        &self.gamma
    }

    /// 获取 beta 参数
    pub const fn beta(&self) -> &Var {
        &self.beta
    }
}

impl Module for BatchNorm {
    fn parameters(&self) -> Vec<Var> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}
