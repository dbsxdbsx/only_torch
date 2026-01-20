/*
 * @Author       : 老董
 * @Date         : 2026-01-17
 * @Description  : Rnn (循环神经网络) 层 - PyTorch 风格 API
 *
 * 公式: h_t = tanh(x_t @ W_ih + h_{t-1} @ W_hh + b_h)
 *
 * 与 PyTorch nn.RNNCell 对齐:
 * - input: [batch, input_size]
 * - hidden: [batch, hidden_size]
 * - W_ih: [input_size, hidden_size]
 * - W_hh: [hidden_size, hidden_size]
 * - b_h: [1, hidden_size]
 *
 * 输入/输出形状：
 * - 输入：[batch_size, input_size]
 * - 输出：[batch_size, hidden_size]
 */

use crate::nn::{Graph, GraphError, Init, Module, Var};
use crate::tensor::Tensor;

// ==================== 新版 Rnn 结构体 ====================

/// Rnn (循环神经网络) 层
///
/// PyTorch 风格的 RNN 层：`h_t = tanh(x @ W_ih + h_{t-1} @ W_hh + b_h)`
///
/// # 输入/输出形状
/// - 输入：[batch_size, input_size]
/// - 输出：[batch_size, hidden_size]
///
/// # 使用示例
/// ```ignore
/// let rnn = Rnn::new(&graph, 10, 20, 32, "rnn1")?;
///
/// // 单步前向传播（用于逐时间步处理）
/// input.set_value(&x_t)?;
/// rnn.step()?;
/// let h_t = rnn.hidden().value()?;
///
/// // 或使用 forward 处理单个输入
/// let (output, hidden) = rnn.forward(&x)?;
/// ```
pub struct Rnn {
    /// 输入到隐藏权重 W_ih: [input_size, hidden_size]
    w_ih: Var,
    /// 隐藏到隐藏权重 W_hh: [hidden_size, hidden_size]
    w_hh: Var,
    /// 隐藏层偏置 b_h: [1, hidden_size]
    b_h: Var,
    /// 隐藏状态输出节点（h_t，经过 tanh 激活）
    hidden_output: Var,
    /// 隐藏状态输入节点（h_{t-1}，State 节点）
    hidden_input: Var,
    /// 输入节点
    input_node: Var,
    /// Graph 引用
    graph: Graph,
    /// 配置
    input_size: usize,
    hidden_size: usize,
    batch_size: usize,
    #[allow(dead_code)]
    name: String,
}

impl Rnn {
    /// 创建新的 Rnn 层
    ///
    /// # 参数
    /// - `graph`: 计算图句柄
    /// - `input_size`: 输入特征维度
    /// - `hidden_size`: 隐藏状态维度
    /// - `batch_size`: 批大小
    /// - `name`: 层名称前缀
    ///
    /// # 返回
    /// Rnn 层实例
    pub fn new(
        graph: &Graph,
        input_size: usize,
        hidden_size: usize,
        batch_size: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        // 创建参数节点
        let w_ih = graph.parameter(
            &[input_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_ih"),
        )?;

        let w_hh = graph.parameter(
            &[hidden_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_hh"),
        )?;

        let b_h = graph.parameter(
            &[1, hidden_size],
            Init::Zeros,
            &format!("{name}_b_h"),
        )?;

        // 创建输入节点
        let input_node = graph.zeros(&[batch_size, input_size])?;

        // 创建状态节点和计算图结构
        let (hidden_input, hidden_output) = {
            let mut g = graph.inner_mut();

            // 创建状态节点：h_prev
            let h_prev_id = g.new_state_node(
                &[batch_size, hidden_size],
                Some(&format!("{name}_h_prev")),
            )?;
            g.set_node_value(h_prev_id, Some(&Tensor::zeros(&[batch_size, hidden_size])))?;

            // 计算图结构
            // input_contrib = x @ W_ih
            let input_contrib = g.new_mat_mul_node(
                input_node.node_id(),
                w_ih.node_id(),
                Some(&format!("{name}_input_contrib")),
            )?;

            // hidden_contrib = h_prev @ W_hh
            let hidden_contrib = g.new_mat_mul_node(
                h_prev_id,
                w_hh.node_id(),
                Some(&format!("{name}_hidden_contrib")),
            )?;

            // pre_hidden = input_contrib + hidden_contrib + b_h
            // Add 支持广播：[batch, hidden] + [batch, hidden] + [1, hidden]
            let pre_hidden = g.new_add_node(
                &[input_contrib, hidden_contrib, b_h.node_id()],
                Some(&format!("{name}_pre_h")),
            )?;

            // hidden = tanh(pre_hidden)
            let hidden_id = g.new_tanh_node(pre_hidden, Some(&format!("{name}_h")))?;

            // 建立循环连接
            g.connect_recurrent(hidden_id, h_prev_id)?;

            // 注册层分组
            g.register_layer_group(
                name,
                "Rnn",
                &format!("{input_size}→{hidden_size}"),
                vec![
                    w_ih.node_id(),
                    w_hh.node_id(),
                    b_h.node_id(),
                    input_contrib,
                    hidden_contrib,
                    pre_hidden,
                    hidden_id,
                ],
            );

            // 创建 Var
            let h_prev = Var::new(h_prev_id, graph.inner_rc());
            let hidden = Var::new(hidden_id, graph.inner_rc());

            (h_prev, hidden)
        };

        Ok(Self {
            w_ih,
            w_hh,
            b_h,
            hidden_output,
            hidden_input,
            input_node,
            graph: graph.clone(),
            input_size,
            hidden_size,
            batch_size,
            name: name.to_string(),
        })
    }


    /// 单步前向传播
    ///
    /// 设置输入并执行一个时间步的计算。
    /// 用于需要逐时间步控制的场景（如 BPTT）。
    ///
    /// # 参数
    /// - `x`: 输入张量，形状 [batch_size, input_size]
    ///
    /// # 返回
    /// 隐藏状态输出的引用
    pub fn step(&self, x: &Tensor) -> Result<&Var, GraphError> {
        self.input_node.set_value(x)?;
        self.graph.inner_mut().step(self.hidden_output.node_id())?;
        Ok(&self.hidden_output)
    }

    /// 前向传播（返回 Var 用于链式调用）
    ///
    /// 设置输入并执行计算，返回隐藏状态的 Var。
    /// 适用于构建更复杂的网络结构。
    ///
    /// # 参数
    /// - `x`: 输入 Var，形状 [batch_size, input_size]
    ///
    /// # 返回
    /// 隐藏状态 Var，形状 [batch_size, hidden_size]
    pub fn forward(&self, x: &Var) -> Result<Var, GraphError> {
        // 将输入值复制到 RNN 的输入节点
        if let Some(x_val) = x.value()? {
            self.input_node.set_value(&x_val)?;
        }
        Ok(self.hidden_output.clone())
    }

    /// 重置隐藏状态为零（仅重置状态节点的值）
    pub fn reset_hidden(&self) -> Result<(), GraphError> {
        self.hidden_input.set_value(&Tensor::zeros(&[self.batch_size, self.hidden_size]))?;
        Ok(())
    }

    /// 完整重置（重置隐藏状态 + 清除图的历史快照）
    ///
    /// 用于开始新序列的训练，确保状态完全干净。
    pub fn reset(&self) {
        self.graph.inner_mut().reset();
    }

    /// 设置初始隐藏状态
    pub fn set_hidden(&self, h: &Tensor) -> Result<(), GraphError> {
        self.hidden_input.set_value(h)?;
        Ok(())
    }

    /// 获取当前隐藏状态输出
    pub fn hidden(&self) -> &Var {
        &self.hidden_output
    }

    /// 获取隐藏状态输入节点（State 节点）
    pub fn hidden_input(&self) -> &Var {
        &self.hidden_input
    }

    /// 获取输入节点
    pub fn input(&self) -> &Var {
        &self.input_node
    }

    /// 获取 W_ih 权重
    pub fn w_ih(&self) -> &Var {
        &self.w_ih
    }

    /// 获取 W_hh 权重
    pub fn w_hh(&self) -> &Var {
        &self.w_hh
    }

    /// 获取 b_h 偏置
    pub fn b_h(&self) -> &Var {
        &self.b_h
    }

    /// 获取输入维度
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// 获取隐藏维度
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// 获取批大小
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// 获取 Graph 引用
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
}

impl Module for Rnn {
    fn parameters(&self) -> Vec<Var> {
        vec![self.w_ih.clone(), self.w_hh.clone(), self.b_h.clone()]
    }
}

// 单元测试位于 src/nn/tests/layer_rnn.rs
