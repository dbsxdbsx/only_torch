/*
 * @Description  : BPTT（通过时间反向传播）测试（Phase 2）
 *
 * 测试 backward_through_time() 和 TBPTT 的正确性
 */

use crate::nn::{GraphInner, GraphError, NodeId};
use crate::tensor::Tensor;

// ==================== 辅助函数 ====================

/// 创建标量张量
fn scalar(val: f64) -> Tensor {
    Tensor::new(&[val as f32], &[1, 1])
}

/// 创建一个简单的循环网络用于测试 BPTT
///
/// 结构：
/// ```text
///     input ──┐
///             ├──→ [Add] ──→ hidden ──→ [MatMul] ──→ output
/// prev_hidden┘        │                    ↑
///     ↑               │                 w_out
///     └───────────────┘ (循环连接)
/// ```
///
/// 数学：
///   hidden_t = input_t + hidden_{t-1}
///   output_t = w_out * hidden_t
///
/// 这是一个简单的累加器 + 线性输出
fn create_simple_rnn() -> Result<(GraphInner, NodeId, NodeId, NodeId, NodeId), GraphError> {
    let mut graph = GraphInner::new();
    graph.set_train_mode();

    // 输入节点
    let input = graph.new_input_node(&[1, 1], Some("input"))?;

    // 循环状态节点（接收上一步的 hidden）— 使用 State 节点而非 Input
    let prev_hidden = graph.new_state_node(&[1, 1], Some("prev_hidden"))?;
    graph.set_node_value(prev_hidden, Some(&Tensor::zeros(&[1, 1])))?;

    // hidden = input + prev_hidden
    let hidden = graph.new_add_node(&[input, prev_hidden], Some("hidden"))?;

    // 输出权重
    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&scalar(1.0)))?; // 初始化为 1

    // output = w_out * hidden
    let output = graph.new_mat_mul_node(w_out, hidden, Some("output"))?;

    // 声明循环连接：hidden 的值在下一步传给 prev_hidden
    graph.connect_recurrent(hidden, prev_hidden)?;

    Ok((graph, input, output, w_out, hidden))
}

/// 创建带循环权重的 RNN（用于测试 TBPTT）
///
/// 与 create_simple_rnn 不同，这个网络的循环连接中有一个参数 w_rec
/// 因此 TBPTT 的截断会直接影响 w_rec 的梯度
///
/// 结构：
/// ```text
///     input ──┐
///             ├──→ [Add] ──→ hidden ──→ [MatMul] ──→ output
/// w_rec*prev_hidden┘   │                    ↑
///         ↑            │                 w_out
///         └────────────┘ (循环连接)
/// ```
fn create_rnn_with_recurrent_weight()
-> Result<(GraphInner, NodeId, NodeId, NodeId, NodeId, NodeId), GraphError> {
    let mut graph = GraphInner::new();
    graph.set_train_mode();

    // 输入和目标
    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let target = graph.new_input_node(&[1, 1], Some("target"))?;

    // 循环状态节点
    let prev_hidden = graph.new_state_node(&[1, 1], Some("prev_hidden"))?;
    graph.set_node_value(prev_hidden, Some(&Tensor::zeros(&[1, 1])))?;

    // 循环权重（这是在循环路径中的参数）
    let w_rec = graph.new_parameter_node(&[1, 1], Some("w_rec"))?;
    graph.set_node_value(w_rec, Some(&scalar(0.5)))?;

    // scaled_prev = prev_hidden * w_rec
    let scaled_prev = graph.new_mat_mul_node(prev_hidden, w_rec, Some("scaled_prev"))?;

    // hidden = input + scaled_prev
    let hidden = graph.new_add_node(&[input, scaled_prev], Some("hidden"))?;

    // 输出权重
    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&scalar(1.0)))?;

    // output = hidden * w_out
    let output = graph.new_mat_mul_node(hidden, w_out, Some("output"))?;

    // loss = MSE(output, target)
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 声明循环连接
    graph.connect_recurrent(hidden, prev_hidden)?;

    Ok((graph, input, target, w_rec, w_out, loss))
}

/// 创建带 MSE Loss 的简单 RNN
fn create_rnn_with_loss() -> Result<(GraphInner, NodeId, NodeId, NodeId, NodeId), GraphError> {
    let mut graph = GraphInner::new();
    graph.set_train_mode();

    // 输入和目标
    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let target = graph.new_input_node(&[1, 1], Some("target"))?;

    // 循环状态节点 — 使用 State 节点而非 Input
    let prev_hidden = graph.new_state_node(&[1, 1], Some("prev_hidden"))?;
    graph.set_node_value(prev_hidden, Some(&Tensor::zeros(&[1, 1])))?;

    // hidden = input + prev_hidden
    let hidden = graph.new_add_node(&[input, prev_hidden], Some("hidden"))?;

    // 输出权重
    let w_out = graph.new_parameter_node(&[1, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&scalar(0.5)))?;

    // output = w_out * hidden
    let output = graph.new_mat_mul_node(w_out, hidden, Some("output"))?;

    // loss = MSE(output, target)
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 声明循环连接
    graph.connect_recurrent(hidden, prev_hidden)?;

    Ok((graph, input, target, w_out, loss))
}

// ==================== 基础 BPTT 测试 ====================

#[test]
fn test_bptt_history_recording() {
    let (mut graph, input, output, _, _) = create_simple_rnn().unwrap();

    // 验证初始状态
    assert_eq!(graph.history_len(), 0);

    // 运行几个时间步
    for i in 1..=3 {
        graph
            .set_node_value(input, Some(&scalar(i as f64)))
            .unwrap();
        graph.step(output).unwrap();
    }

    // 验证历史记录
    assert_eq!(graph.history_len(), 3);
    assert_eq!(graph.current_time_step(), 3);

    // reset 应该清除历史
    graph.reset();
    assert_eq!(graph.history_len(), 0);
    assert_eq!(graph.current_time_step(), 0);
}

#[test]
fn test_bptt_no_history_in_eval_mode() {
    let (mut graph, input, output, _, _) = create_simple_rnn().unwrap();

    // 切换到评估模式
    graph.set_eval_mode();

    // 运行几个时间步
    for i in 1..=3 {
        graph
            .set_node_value(input, Some(&scalar(i as f64)))
            .unwrap();
        graph.step(output).unwrap();
    }

    // 评估模式下不记录历史
    assert_eq!(graph.history_len(), 0);
}

#[test]
fn test_bptt_basic_gradient() {
    let (mut graph, input, target, w_out, loss) = create_rnn_with_loss().unwrap();

    // 单步：input=2, target=1
    // hidden = 2 + 0 = 2
    // output = 0.5 * 2 = 1
    // loss = (1 - 1)^2 = 0
    graph.set_node_value(input, Some(&scalar(2.0))).unwrap();
    graph.set_node_value(target, Some(&scalar(1.0))).unwrap();
    graph.step(loss).unwrap();

    // 执行 BPTT
    graph.backward_through_time(&[w_out], loss).unwrap();

    // 获取梯度
    let grad = graph.get_node_grad(w_out).unwrap().unwrap();
    // loss = (w * h - t)^2 = (0.5 * 2 - 1)^2 = 0
    // d(loss)/d(w) = 2 * (w * h - t) * h = 2 * (1 - 1) * 2 = 0
    assert!(
        (grad.data_as_slice()[0] - 0.0).abs() < 1e-5,
        "期望梯度 ≈ 0，实际得到 {}",
        grad.data_as_slice()[0]
    );
}

#[test]
fn test_bptt_nonzero_gradient() {
    let (mut graph, input, target, w_out, loss) = create_rnn_with_loss().unwrap();

    // 单步：input=2, target=2
    // hidden = 2 + 0 = 2
    // output = 0.5 * 2 = 1
    // loss = (1 - 2)^2 = 1
    graph.set_node_value(input, Some(&scalar(2.0))).unwrap();
    graph.set_node_value(target, Some(&scalar(2.0))).unwrap();
    graph.step(loss).unwrap();

    // 执行 BPTT
    graph.backward_through_time(&[w_out], loss).unwrap();

    // 获取梯度
    let grad = graph.get_node_grad(w_out).unwrap().unwrap();
    // loss = (w * h - t)^2 = (0.5 * 2 - 2)^2 = 1
    // d(loss)/d(w) = 2 * (w * h - t) * h = 2 * (1 - 2) * 2 = -4
    assert!(
        (grad.data_as_slice()[0] - (-4.0)).abs() < 1e-5,
        "期望梯度 ≈ -4，实际得到 {}",
        grad.data_as_slice()[0]
    );
}

#[test]
fn test_bptt_multi_step_gradient_accumulation() {
    let (mut graph, input, target, w_out, loss) = create_rnn_with_loss().unwrap();

    // 两个时间步，验证梯度累加机制正常工作
    // 具体数值需要 PyTorch 对照验证（见 IT-1 集成测试）

    // t=0
    graph.set_node_value(input, Some(&scalar(1.0))).unwrap();
    graph.set_node_value(target, Some(&scalar(0.5))).unwrap();
    graph.step(loss).unwrap();

    // t=1
    graph.set_node_value(input, Some(&scalar(1.0))).unwrap();
    graph.set_node_value(target, Some(&scalar(1.5))).unwrap();
    graph.step(loss).unwrap();

    // 执行 BPTT
    graph.backward_through_time(&[w_out], loss).unwrap();

    // 验证梯度存在且非零（具体值在 IT-1 用 PyTorch 验证）
    let grad = graph.get_node_grad(w_out).unwrap().unwrap();
    assert!(grad.data_as_slice()[0].abs() > 1e-6, "梯度应非零");
    assert!(
        grad.data_as_slice()[0] < 0.0,
        "梯度应为负（t=1 时 output < target）"
    );
}

// ==================== TBPTT 测试 ====================

#[test]
fn test_tbptt_truncation() {
    let (mut graph, input, target, w_out, loss) = create_rnn_with_loss().unwrap();

    // 运行 5 个时间步
    for i in 1..=5 {
        graph
            .set_node_value(input, Some(&scalar(i as f64)))
            .unwrap();
        graph.set_node_value(target, Some(&scalar(0.0))).unwrap();
        graph.step(loss).unwrap();
    }

    assert_eq!(graph.history_len(), 5);

    // 只截断最后 2 步
    graph
        .backward_through_time_truncated(&[w_out], loss, Some(2))
        .unwrap();

    // 验证梯度存在（具体值依赖于网络，这里只验证机制工作）
    let grad = graph.get_node_grad(w_out).unwrap();
    assert!(grad.is_some(), "截断 BPTT 后应有梯度");
}

#[test]
fn test_tbptt_full_vs_none() {
    // 验证 truncation_steps = None 等价于 backward_through_time
    let (mut graph1, input1, target1, w_out1, loss1) = create_rnn_with_loss().unwrap();
    let (mut graph2, input2, target2, w_out2, loss2) = create_rnn_with_loss().unwrap();

    // 相同的输入序列
    let inputs = [1.0, 2.0, 3.0];
    let targets = [0.5, 1.0, 1.5];

    for (&inp, &tgt) in inputs.iter().zip(targets.iter()) {
        graph1.set_node_value(input1, Some(&scalar(inp))).unwrap();
        graph1.set_node_value(target1, Some(&scalar(tgt))).unwrap();
        graph1.step(loss1).unwrap();

        graph2.set_node_value(input2, Some(&scalar(inp))).unwrap();
        graph2.set_node_value(target2, Some(&scalar(tgt))).unwrap();
        graph2.step(loss2).unwrap();
    }

    // graph1 使用 backward_through_time
    graph1.backward_through_time(&[w_out1], loss1).unwrap();

    // graph2 使用 truncation_steps = None
    graph2
        .backward_through_time_truncated(&[w_out2], loss2, None)
        .unwrap();

    // 梯度应该相同
    let grad1 = graph1.get_node_grad(w_out1).unwrap().unwrap();
    let grad2 = graph2.get_node_grad(w_out2).unwrap().unwrap();

    assert!(
        (grad1.data_as_slice()[0] - grad2.data_as_slice()[0]).abs() < 1e-6,
        "梯度应相等：{} vs {}",
        grad1.data_as_slice()[0],
        grad2.data_as_slice()[0]
    );
}

// ==================== 错误处理测试 ====================

#[test]
fn test_bptt_empty_history_error() {
    let (mut graph, _, _, w_out, loss) = create_rnn_with_loss().unwrap();

    // 没有调用 step，直接 BPTT 应该报错
    let result = graph.backward_through_time(&[w_out], loss);
    assert!(result.is_err());

    if let Err(e) = result {
        assert!(
            format!("{:?}", e).contains("没有时间步历史"),
            "错误应提及空历史"
        );
    }
}

#[test]
fn test_clear_history_preserves_recurrent_state() {
    let (mut graph, input, output, _, hidden) = create_simple_rnn().unwrap();

    // 运行几步
    for i in 1..=3 {
        graph
            .set_node_value(input, Some(&scalar(i as f64)))
            .unwrap();
        graph.step(output).unwrap();
    }

    // 记录当前 hidden 值（累加后应该是 1+2+3=6）
    let hidden_before = graph.get_node_value(hidden).unwrap().unwrap().clone();

    // 清除历史
    graph.clear_history();

    // 历史应该清空
    assert_eq!(graph.history_len(), 0);

    // 但循环状态应该保留
    assert_eq!(graph.current_time_step(), 3);

    // 继续 step 应该基于之前的状态
    graph.set_node_value(input, Some(&scalar(4.0))).unwrap();
    graph.step(output).unwrap();

    // hidden 应该是 6 + 4 = 10
    let hidden_after = graph.get_node_value(hidden).unwrap().unwrap();
    assert!(
        (hidden_after.data_as_slice()[0] - 10.0).abs() < 1e-6,
        "期望 hidden = 10，实际得到 {}，清除前为 {}",
        hidden_after.data_as_slice()[0],
        hidden_before.data_as_slice()[0]
    );
}

// ==================== 梯度正确性验证 ====================

#[test]
fn test_bptt_gradient_direction_and_magnitude() {
    // 验证 BPTT 梯度的方向和相对大小正确
    // 具体数值与 PyTorch 的对照验证放到 IT-1 集成测试

    let (mut graph, input, target, w_out, loss) = create_rnn_with_loss().unwrap();

    // 场景：output < target，应该增大 w 来减小 loss，所以梯度应该为负

    // t=0: input=2, target=2 → output=1, loss=(1-2)^2=1
    graph.set_node_value(input, Some(&scalar(2.0))).unwrap();
    graph.set_node_value(target, Some(&scalar(2.0))).unwrap();
    graph.step(loss).unwrap();

    // t=1: input=1, target=2 → output=w*(2+1)=1.5, loss=(1.5-2)^2=0.25
    graph.set_node_value(input, Some(&scalar(1.0))).unwrap();
    graph.set_node_value(target, Some(&scalar(2.0))).unwrap();
    graph.step(loss).unwrap();

    // BPTT
    graph.backward_through_time(&[w_out], loss).unwrap();

    let grad = graph.get_node_grad(w_out).unwrap().unwrap();

    // 1. 梯度应该为负（output < target，需要增大 w）
    assert!(
        grad.data_as_slice()[0] < 0.0,
        "梯度应为负，实际：{}",
        grad.data_as_slice()[0]
    );

    // 2. 梯度绝对值应该大于单步的（因为累加了多个时间步）
    // 这验证了梯度累加机制正常工作
    assert!(
        grad.data_as_slice()[0].abs() > 1.0,
        "累加梯度应有显著值，实际：{}",
        grad.data_as_slice()[0]
    );
}

// ==================== 长序列测试 ====================

#[test]
fn test_bptt_long_sequence() {
    let (mut graph, input, target, w_out, loss) = create_rnn_with_loss().unwrap();

    // 20 步序列
    for i in 1..=20 {
        graph
            .set_node_value(input, Some(&scalar(0.1 * i as f64)))
            .unwrap();
        graph.set_node_value(target, Some(&scalar(0.0))).unwrap();
        graph.step(loss).unwrap();
    }

    assert_eq!(graph.history_len(), 20);

    // BPTT 应该能处理
    graph.backward_through_time(&[w_out], loss).unwrap();

    let grad = graph.get_node_grad(w_out).unwrap();
    assert!(grad.is_some(), "长序列 BPTT 后应有梯度");
}

/// 长序列 BPTT 测试（50+ 步），观察梯度范数变化
///
/// 目的：
/// 1. 验证 50+ 步序列的 BPTT 能正常工作
/// 2. 观察不同序列长度下梯度范数的变化趋势
/// 3. 检测潜在的梯度消失/爆炸问题
#[test]
fn test_bptt_very_long_sequence_gradient_norm() {
    use crate::nn::{GraphInner, NodeId};

    /// 创建带 tanh 激活的简单 RNN
    fn create_tanh_rnn() -> (GraphInner, NodeId, NodeId, NodeId, NodeId, NodeId) {
        let mut graph = GraphInner::new_with_seed(42);
        graph.set_train_mode();

        let input = graph.new_input_node(&[1, 1], Some("input")).unwrap();
        let target = graph.new_input_node(&[1, 1], Some("target")).unwrap();

        let h_prev = graph.new_state_node(&[1, 1], Some("h_prev")).unwrap();
        graph
            .set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))
            .unwrap();

        // w_ih: 输入权重
        let w_ih = graph.new_parameter_node(&[1, 1], Some("w_ih")).unwrap();
        graph.set_node_value(w_ih, Some(&scalar(0.5))).unwrap();

        // w_hh: 循环权重（关键：影响梯度传播）
        let w_hh = graph.new_parameter_node(&[1, 1], Some("w_hh")).unwrap();
        graph
            .set_node_value(w_hh, Some(&scalar(0.9))) // 接近 1 但 < 1，避免梯度爆炸
            .unwrap();

        // scaled_input = input * w_ih
        let scaled_input = graph
            .new_mat_mul_node(input, w_ih, Some("scaled_input"))
            .unwrap();

        // scaled_hidden = h_prev * w_hh
        let scaled_hidden = graph
            .new_mat_mul_node(h_prev, w_hh, Some("scaled_hidden"))
            .unwrap();

        // pre_hidden = scaled_input + scaled_hidden
        let pre_hidden = graph
            .new_add_node(&[scaled_input, scaled_hidden], Some("pre_hidden"))
            .unwrap();

        // hidden = tanh(pre_hidden)
        let hidden = graph.new_tanh_node(pre_hidden, Some("hidden")).unwrap();

        graph.connect_recurrent(hidden, h_prev).unwrap();

        // 输出权重
        let w_out = graph.new_parameter_node(&[1, 1], Some("w_out")).unwrap();
        graph.set_node_value(w_out, Some(&scalar(1.0))).unwrap();

        // output = hidden * w_out
        let output = graph
            .new_mat_mul_node(hidden, w_out, Some("output"))
            .unwrap();

        // loss = MSE(output, target)
        let loss = graph
            .new_mse_loss_node(output, target, Some("loss"))
            .unwrap();

        (graph, input, target, w_ih, w_hh, loss)
    }

    /// 计算梯度范数
    fn grad_norm(graph: &GraphInner, params: &[NodeId]) -> f32 {
        let mut sum_sq = 0.0f32;
        for &p in params {
            if let Ok(Some(param_grad)) = graph.get_node_grad(p) {
                for &v in param_grad.data_as_slice() {
                    sum_sq += v * v;
                }
            }
        }
        sum_sq.sqrt()
    }

    println!("\n=== 长序列 BPTT 梯度范数测试 ===");
    println!("{:>10} {:>15} {:>15}", "序列长度", "w_ih 梯度", "w_hh 梯度");
    println!("{}", "-".repeat(45));

    let test_lengths = [10, 20, 50, 100, 200];

    for &seq_len in &test_lengths {
        let (mut graph, input, target, w_ih, w_hh, loss) = create_tanh_rnn();

        // 运行序列
        for i in 0..seq_len {
            // 使用正弦波作为输入，避免 hidden 饱和
            let x = (i as f64 * 0.1).sin() * 0.5;
            graph.set_node_value(input, Some(&scalar(x))).unwrap();
            graph.set_node_value(target, Some(&scalar(0.0))).unwrap();
            graph.step(loss).unwrap();
        }

        // BPTT
        graph.backward_through_time(&[w_ih, w_hh], loss).unwrap();

        // 获取梯度
        let grad_w_ih = graph
            .get_node_grad(w_ih)
            .unwrap()
            .map(|j| j.data_as_slice()[0])
            .unwrap_or(0.0);
        let grad_w_hh = graph
            .get_node_grad(w_hh)
            .unwrap()
            .map(|j| j.data_as_slice()[0])
            .unwrap_or(0.0);

        let current_norm = grad_norm(&graph, &[w_ih, w_hh]);

        println!("{:>10} {:>15.6} {:>15.6}", seq_len, grad_w_ih, grad_w_hh);

        // 验证：梯度应该存在且有限
        assert!(
            grad_w_ih.is_finite(),
            "w_ih 梯度应在 seq_len={} 时有限",
            seq_len
        );
        assert!(
            grad_w_hh.is_finite(),
            "w_hh 梯度应在 seq_len={} 时有限",
            seq_len
        );

        // 验证：梯度范数不应该爆炸（不应该超过 1e6）
        assert!(
            current_norm < 1e6,
            "梯度范数不应爆炸: {} (seq_len={})",
            current_norm,
            seq_len
        );

        // 验证：梯度不应该完全消失（除非 loss 本身很小）
        // 对于长序列，循环权重梯度可能会衰减，但不应该完全为 0
        if seq_len <= 100 {
            assert!(
                grad_w_hh.abs() > 1e-10,
                "w_hh 梯度不应在 seq_len={} 时完全消失",
                seq_len
            );
        }

        let _ = current_norm; // 显式忽略，避免警告
    }

    println!("\n✅ 长序列 BPTT 测试通过：梯度在合理范围内");
}

/// 测试不同 w_hh 值下的梯度行为
///
/// w_hh 接近 1 时梯度传播更远，但也更容易不稳定
/// w_hh 较小时梯度衰减更快，但更稳定
#[test]
fn test_bptt_gradient_stability_vs_w_hh() {
    fn run_bptt_with_w_hh(w_hh_value: f64, seq_len: usize) -> (f32, f32) {
        let mut graph = GraphInner::new_with_seed(42);
        graph.set_train_mode();

        let input = graph.new_input_node(&[1, 1], Some("input")).unwrap();
        let target = graph.new_input_node(&[1, 1], Some("target")).unwrap();

        let h_prev = graph.new_state_node(&[1, 1], Some("h_prev")).unwrap();
        graph
            .set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))
            .unwrap();

        let w_hh = graph.new_parameter_node(&[1, 1], Some("w_hh")).unwrap();
        graph
            .set_node_value(w_hh, Some(&scalar(w_hh_value)))
            .unwrap();

        // hidden = tanh(h_prev * w_hh + input)
        let scaled_hidden = graph.new_mat_mul_node(h_prev, w_hh, None).unwrap();
        let pre_hidden = graph.new_add_node(&[scaled_hidden, input], None).unwrap();
        let hidden = graph.new_tanh_node(pre_hidden, Some("hidden")).unwrap();

        graph.connect_recurrent(hidden, h_prev).unwrap();

        let loss = graph
            .new_mse_loss_node(hidden, target, Some("loss"))
            .unwrap();

        // 运行序列
        for i in 0..seq_len {
            let x = if i == 0 { 1.0 } else { 0.0 }; // 只有第一步有输入
            graph.set_node_value(input, Some(&scalar(x))).unwrap();
            graph.set_node_value(target, Some(&scalar(0.0))).unwrap();
            graph.step(loss).unwrap();
        }

        graph.backward_through_time(&[w_hh], loss).unwrap();

        let grad = graph
            .get_node_grad(w_hh)
            .unwrap()
            .map(|j| j.data_as_slice()[0])
            .unwrap_or(0.0);

        let loss_val = graph
            .get_node_value(loss)
            .unwrap()
            .map(|v| v.data_as_slice()[0])
            .unwrap_or(0.0);

        (grad, loss_val)
    }

    println!("\n=== w_hh 对梯度稳定性的影响 ===");
    println!("{:>8} {:>15} {:>15}", "w_hh", "梯度(50步)", "梯度(100步)");
    println!("{}", "-".repeat(40));

    let w_hh_values = [0.3, 0.5, 0.7, 0.9, 0.95];

    for &w_hh in &w_hh_values {
        let (grad_50, _) = run_bptt_with_w_hh(w_hh, 50);
        let (grad_100, _) = run_bptt_with_w_hh(w_hh, 100);

        println!("{:>8.2} {:>15.6} {:>15.6}", w_hh, grad_50, grad_100);

        // 验证梯度有限
        assert!(grad_50.is_finite(), "梯度应有限");
        assert!(grad_100.is_finite(), "梯度应有限");
    }

    println!("\n✅ 梯度稳定性测试通过");
}

#[test]
fn test_tbptt_vs_full_bptt_different_gradients() {
    // 验证 TBPTT 截断确实产生不同的梯度（当序列足够长时）
    // 注意：必须使用有循环权重的网络，因为 TBPTT 只影响循环路径中的参数梯度
    let (mut graph1, input1, target1, w_rec1, _, loss1) =
        create_rnn_with_recurrent_weight().unwrap();
    let (mut graph2, input2, target2, w_rec2, _, loss2) =
        create_rnn_with_recurrent_weight().unwrap();

    // 10 步序列
    for i in 1..=10 {
        let inp = 0.1 * i as f64;
        let tgt = 0.0;

        graph1.set_node_value(input1, Some(&scalar(inp))).unwrap();
        graph1.set_node_value(target1, Some(&scalar(tgt))).unwrap();
        graph1.step(loss1).unwrap();

        graph2.set_node_value(input2, Some(&scalar(inp))).unwrap();
        graph2.set_node_value(target2, Some(&scalar(tgt))).unwrap();
        graph2.step(loss2).unwrap();
    }

    // graph1: 完整 BPTT（10 步）
    graph1.backward_through_time(&[w_rec1], loss1).unwrap();

    // graph2: TBPTT 截断到 3 步
    graph2
        .backward_through_time_truncated(&[w_rec2], loss2, Some(3))
        .unwrap();

    let grad1 = graph1.get_node_grad(w_rec1).unwrap().unwrap();
    let grad2 = graph2.get_node_grad(w_rec2).unwrap().unwrap();

    // 循环权重的梯度应该不同（TBPTT 只反向传播了部分时间步的贡献）
    // 完整 BPTT 会累积来自所有 10 个时间步的梯度贡献
    // TBPTT 截断到 3 步只会累积最近 3 个时间步的贡献
    assert!(
        (grad1.data_as_slice()[0] - grad2.data_as_slice()[0]).abs() > 1e-6,
        "截断 BPTT 的循环权重梯度应不同于完整 BPTT：{} vs {}",
        grad1.data_as_slice()[0],
        grad2.data_as_slice()[0]
    );
}

/// 大 batch/hidden 尺寸下的 VJP BPTT 验收测试
///
/// 目标：
/// - 确保 VJP 模式在大 batch (64) 和大 hidden (256) 下能正常运行
/// - 确保不会 OOM（VJP 避免了 O(N²) 的 Jacobian 矩阵）
/// - 确保训练速度合理（在测试超时内完成）
#[test]
fn test_vjp_large_batch_hidden() -> Result<(), GraphError> {
    use std::time::Instant;

    // 大参数配置
    let batch_size = 64;
    let hidden_size = 256;
    let seq_len = 20;
    let num_epochs = 5;

    // 创建网络
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    // 输入和状态
    let input = graph.new_input_node(&[batch_size, 1], Some("input"))?;
    let h_prev = graph.new_state_node(&[batch_size, hidden_size], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[batch_size, hidden_size])))?;

    // 权重（使用小范围均匀分布初始化）
    let w_ih = graph.new_parameter_node(&[1, hidden_size], Some("w_ih"))?;
    graph.set_node_value(w_ih, Some(&Tensor::random(-0.1, 0.1, &[1, hidden_size])))?;

    let w_hh = graph.new_parameter_node(&[hidden_size, hidden_size], Some("w_hh"))?;
    graph.set_node_value(
        w_hh,
        Some(&Tensor::random(-0.1, 0.1, &[hidden_size, hidden_size])),
    )?;

    let w_out = graph.new_parameter_node(&[hidden_size, 1], Some("w_out"))?;
    graph.set_node_value(w_out, Some(&Tensor::random(-0.1, 0.1, &[hidden_size, 1])))?;

    // 前向计算：hidden = tanh(input @ w_ih + h_prev @ w_hh)
    let input_contrib = graph.new_mat_mul_node(input, w_ih, Some("input_contrib"))?;
    let hidden_contrib = graph.new_mat_mul_node(h_prev, w_hh, Some("hidden_contrib"))?;
    let pre_hidden = graph.new_add_node(&[input_contrib, hidden_contrib], Some("pre_hidden"))?;
    let hidden = graph.new_tanh_node(pre_hidden, Some("hidden"))?;
    graph.connect_recurrent(hidden, h_prev)?;

    // 输出：output = hidden @ w_out
    let output = graph.new_mat_mul_node(hidden, w_out, Some("output"))?;

    // 损失
    let target = graph.new_input_node(&[batch_size, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    let params = vec![w_ih, w_hh, w_out];

    // 训练循环
    let start = Instant::now();
    let mut total_loss = 0.0;

    for _epoch in 0..num_epochs {
        graph.reset();

        // 生成随机输入序列
        for _t in 0..seq_len {
            let input_data = Tensor::random(-1.0, 1.0, &[batch_size, 1]);
            graph.set_node_value(input, Some(&input_data))?;
            graph.set_node_value(target, Some(&Tensor::zeros(&[batch_size, 1])))?;
            graph.step(loss)?;
        }

        // BPTT
        graph.backward_through_time(&params, loss)?;

        // 记录 loss
        let loss_val = graph.get_node_value(loss)?.unwrap().data_as_slice()[0];
        total_loss += loss_val;

        // 简单 SGD 更新
        let lr = 0.01;
        for &param in &params {
            if let Some(grad) = graph.get_node_grad(param)? {
                let value = graph.get_node_value(param)?.unwrap();
                let grad_reshaped = grad.reshape(value.shape());
                let new_value = value - &grad_reshaped * lr;
                graph.set_node_value(param, Some(&new_value))?;
            }
        }

        // 清理
        graph.zero_grad()?;
    }

    let elapsed = start.elapsed();
    let avg_loss = total_loss / num_epochs as f32;

    println!("\n=== VJP 大 batch/hidden 验收测试 ===");
    println!("  batch_size: {}, hidden_size: {}", batch_size, hidden_size);
    println!("  seq_len: {}, epochs: {}", seq_len, num_epochs);
    println!("  耗时: {:?}", elapsed);
    println!("  平均 loss: {:.6}", avg_loss);

    // 验证：
    // 1. 测试完成（没有 OOM 或 panic）
    // 2. 时间合理（应该在几秒内完成）
    assert!(elapsed.as_secs() < 60, "VJP 测试超时：{:?} >= 60s", elapsed);

    // 3. loss 是有效值（不是 NaN 或 Inf）
    assert!(avg_loss.is_finite(), "loss 不是有效数值：{}", avg_loss);

    println!("✅ VJP 大 batch/hidden 验收测试通过");
    Ok(())
}
