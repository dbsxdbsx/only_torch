/*
 * @Author       : 老董
 * @Date         : 2026-01-09
 * @Description  : Var-based Linear (全连接) 层
 *
 * 新版 Linear 层，基于 Var API：
 * - forward() 不需要 &mut Graph 参数
 * - 实现 Module trait
 * - 支持链式调用
 *
 * 设计依据：architecture_v2_design.md §4.2.4
 */

use crate::nn::{GraphError, Graph, Init, Module, Var, VarMatrixOps};

/// Var-based Linear 层
///
/// 全连接层：`output = x @ W + b`
///
/// # 输入/输出形状
/// - 输入：[batch_size, in_features]
/// - 输出：[batch_size, out_features]
///
/// # 使用示例
/// ```ignore
/// let fc = Linear::new(&graph, 784, 128, true, "fc1")?;
/// let h = fc.forward(&x).relu();
/// ```
pub struct Linear {
    /// 权重参数 [in_features, out_features]
    weights: Var,
    /// 偏置参数 [1, out_features]（可选）
    bias: Option<Var>,
    /// 输入特征维度
    in_features: usize,
    /// 输出特征维度
    out_features: usize,
}

impl Linear {
    /// 创建新的 Linear 层
    ///
    /// # 参数
    /// - `graph`: 计算图句柄
    /// - `in_features`: 输入特征维度
    /// - `out_features`: 输出特征维度
    /// - `use_bias`: 是否使用偏置
    /// - `name`: 层名称前缀
    ///
    /// # 返回
    /// Linear 层实例
    pub fn new(
        graph: &Graph,
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        name: &str,
    ) -> Result<Self, GraphError> {
        // 创建权重参数：Kaiming 初始化适合 ReLU
        let weights = graph.parameter(
            &[in_features, out_features],
            Init::Kaiming,
            &format!("{name}_W"),
        )?;

        // 创建偏置参数（可选）：零初始化
        let bias = if use_bias {
            Some(graph.parameter(&[1, out_features], Init::Zeros, &format!("{name}_b"))?)
        } else {
            None
        };

        Ok(Self {
            weights,
            bias,
            in_features,
            out_features,
        })
    }

    /// 前向传播
    ///
    /// 计算 `x @ W + b`
    ///
    /// # 参数
    /// - `x`: 输入 Var，形状 [batch_size, in_features]
    ///
    /// # 返回
    /// 输出 Var，形状 [batch_size, out_features]
    ///
    /// # Panics
    /// 如果输入形状不匹配
    pub fn forward(&self, x: &Var) -> Var {
        // x @ W: [batch, in] @ [in, out] = [batch, out]
        let output = x.matmul(&self.weights).expect("Linear matmul 失败");

        // 如果有 bias，需要广播加法
        if let Some(ref bias) = self.bias {
            // 获取 batch_size
            let x_shape = x.value_expected_shape();
            let batch_size = x_shape[0];

            // 创建 ones 矩阵用于 bias 广播
            // ones: [batch, 1]
            let ones = x
                .graph()
                .borrow_mut()
                .new_input_node(&[batch_size, 1], None)
                .expect("创建 ones 节点失败");
            x.graph()
                .borrow_mut()
                .set_node_value(ones, Some(&crate::tensor::Tensor::ones(&[batch_size, 1])))
                .expect("设置 ones 值失败");
            let ones_var = Var::new(ones, std::rc::Rc::clone(x.graph()));

            // ones @ b: [batch, 1] @ [1, out] = [batch, out]
            let bias_broadcast = ones_var.matmul(bias).expect("bias broadcast matmul 失败");

            // output + bias
            &output + &bias_broadcast
        } else {
            output
        }
    }

    /// 获取输入特征维度
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// 获取输出特征维度
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// 获取权重 Var
    pub fn weights(&self) -> &Var {
        &self.weights
    }

    /// 获取偏置 Var（如果有）
    pub fn bias(&self) -> Option<&Var> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Var> {
        let mut params = vec![self.weights.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}
