/*
 * State 节点单元测试
 *
 * TODO(RNN 重构): State 节点是旧的"显式时间步"设计的一部分。
 * 待 LSTM/GRU 完成展开式重构后，可能需要评估 State 节点的保留价值。
 * 新的 Rnn 层使用"展开式设计"，不依赖 State 节点。
 *
 * State 节点用于 RNN 的时间状态（如隐藏状态 h、LSTM 的 c）。
 *
 * 与 Input 节点的关键区别：
 *   - State 可以接收并存储梯度（用于 BPTT 梯度传递）
 *   - Input 不能接收梯度（是"梯度汇点"）
 *
 * 与 Parameter 节点的关键区别：
 *   - State 不被优化器更新（不在 get_trainable_nodes() 中）
 *   - Parameter 被优化器更新
 *
 * VJP 迁移说明：
 *   - 统一 API: backward + get_node_grad + zero_grad
 *   - 新 API: backward + get_node_grad + zero_grad
 */

use crate::assert_err;
use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;

/// 测试 State 节点的基本创建
#[test]
fn test_state_node_creation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    let state = graph.new_state_node(&[1, 64], Some("hidden"))?;

    // 验证节点存在且可获取
    assert!(graph.get_node_value(state)?.is_none()); // 初始值为 None
    Ok(())
}

/// 测试 State 节点的值设置
#[test]
fn test_state_node_set_value() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    let state = graph.new_state_node(&[1, 4], Some("hidden"))?;

    // 设置值
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    graph.set_node_value(state, Some(&tensor))?;

    // 验证值
    let value = graph.get_node_value(state)?.unwrap();
    assert_eq!(value.data_as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

/// 测试 State 节点不在 trainable nodes 中
#[test]
fn test_state_not_trainable() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建各种节点
    let input = graph.new_input_node(&[1, 4], Some("input"))?;
    let state = graph.new_state_node(&[1, 4], Some("hidden"))?;
    let param = graph.new_parameter_node(&[4, 4], Some("weight"))?;

    // 获取可训练节点
    let trainable = graph.get_trainable_nodes();

    // 只有 Parameter 在 trainable 中
    assert!(!trainable.contains(&input), "Input 不应可训练");
    assert!(!trainable.contains(&state), "State 不应可训练");
    assert!(trainable.contains(&param), "Parameter 应可训练");
    assert_eq!(trainable.len(), 1, "只有 Parameter 应可训练");

    Ok(())
}

/// 测试 State 节点可以接收梯度（与 Input 的关键区别）
///
/// 在 VJP 模式下，State 节点在 backward 后应该有 grad
#[test]
fn test_state_accepts_grad() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    graph.set_train_mode();

    // 创建网络：state -> add -> loss
    let state = graph.new_state_node(&[1, 2], Some("state"))?;
    graph.set_node_value(state, Some(&Tensor::new(&[1.0, 2.0], &[1, 2])))?;

    let param = graph.new_parameter_node(&[1, 2], Some("param"))?;
    graph.set_node_value(param, Some(&Tensor::new(&[0.5, 0.5], &[1, 2])))?;

    // state + param -> add
    let add = graph.new_add_node(&[state, param], Some("add"))?;

    // 添加 target 和 loss 节点（VJP 模式需要标量 loss）
    let target = graph.new_input_node(&[1, 2], Some("target"))?;
    graph.set_node_value(target, Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;
    let loss = graph.new_mse_loss_node(add, target, Some("loss"))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播（VJP 模式）
    graph.backward(loss)?;

    // State 应该能接收 grad（与 Input 的关键区别）
    let state_grad = graph.get_node_grad(state)?;
    assert!(state_grad.is_some(), "State 节点在 backward 后应有 grad");

    // 验证 grad 形状
    let grad = state_grad.unwrap();
    assert_eq!(grad.shape(), &[1, 2]); // [输入维度]

    Ok(())
}

/// 测试 Input 节点不能有梯度（对照测试）
///
/// 在 VJP 模式下，调用 get_node_grad(input) 应返回错误
#[test]
fn test_input_has_no_grad() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    graph.set_train_mode();

    // 创建网络：input -> add -> loss
    let input = graph.new_input_node(&[1, 2], Some("input"))?;
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 2.0], &[1, 2])))?;

    let param = graph.new_parameter_node(&[1, 2], Some("param"))?;
    graph.set_node_value(param, Some(&Tensor::new(&[0.5, 0.5], &[1, 2])))?;

    let add = graph.new_add_node(&[input, param], Some("add"))?;

    // 添加 target 和 loss 节点
    let target = graph.new_input_node(&[1, 2], Some("target"))?;
    graph.set_node_value(target, Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;
    let loss = graph.new_mse_loss_node(add, target, Some("loss"))?;

    graph.forward(loss)?;
    graph.backward(loss)?;

    // Input 节点查询 grad 应该返回错误
    let grad_result = graph.get_node_grad(input);
    assert_err!(
        grad_result,
        GraphError::InvalidOperation(msg) if msg.contains("不应该有梯度")
    );

    Ok(())
}

/// 测试 State 节点在 forward 中的行为
#[test]
fn test_state_forward_behavior() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let state = graph.new_state_node(&[1, 4], Some("state"))?;

    // 尝试对未设值的 State 进行前向传播应该失败
    let result = graph.forward(state);
    assert_err!(
        result,
        GraphError::InvalidOperation(msg) if msg.contains("是输入/参数/状态节点")
    );

    // 设置值后，State 不应该被 forward 直接调用（它的值由外部设置）
    graph.set_node_value(state, Some(&Tensor::zeros(&[1, 4])))?;
    let result = graph.forward(state);
    assert_err!(
        result,
        GraphError::InvalidOperation(msg) if msg.contains("是输入/参数/状态节点")
    );

    Ok(())
}

/// 测试 State 节点在简单 RNN 结构中的使用
///
/// 验证 State 节点能正确接收和传递梯度
#[test]
fn test_state_in_rnn_structure() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    graph.set_train_mode();

    // 简单 RNN: hidden_t = tanh(h_prev + input * W)
    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let h_prev = graph.new_state_node(&[1, 1], Some("h_prev"))?;
    let w = graph.new_parameter_node(&[1, 1], Some("W"))?;

    // 设置初始值
    graph.set_node_value(input, Some(&Tensor::new(&[1.0], &[1, 1])))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))?;
    graph.set_node_value(w, Some(&Tensor::new(&[0.5], &[1, 1])))?;

    // input * W
    let scaled = graph.new_mat_mul_node(input, w, Some("scaled"))?;

    // h_prev + scaled
    let pre_hidden = graph.new_add_node(&[h_prev, scaled], Some("pre_hidden"))?;

    // tanh
    let hidden = graph.new_tanh_node(pre_hidden, Some("hidden"))?;

    // 添加 target 和 loss（VJP 模式需要标量 loss）
    let target = graph.new_input_node(&[1, 1], Some("target"))?;
    graph.set_node_value(target, Some(&Tensor::new(&[0.5], &[1, 1])))?;
    let loss = graph.new_mse_loss_node(hidden, target, Some("loss"))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.backward(loss)?;

    // 验证 W 有梯度
    let w_grad = graph.get_node_grad(w)?;
    assert!(w_grad.is_some(), "W 应有 grad");

    // 验证 h_prev 也有梯度（这是 State 与 Input 的关键区别）
    let h_prev_grad = graph.get_node_grad(h_prev)?;
    assert!(h_prev_grad.is_some(), "h_prev (State) 应有 grad");

    Ok(())
}

/// 测试 State 节点与循环连接的配合
#[test]
fn test_state_with_recurrent_connection() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    graph.set_train_mode();

    // 创建循环网络
    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let h_prev = graph.new_state_node(&[1, 1], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))?;

    let add = graph.new_add_node(&[input, h_prev], Some("add"))?;
    let hidden = graph.new_tanh_node(add, Some("hidden"))?;

    // 建立循环连接：hidden -> h_prev
    graph.connect_recurrent(hidden, h_prev)?;

    // 第一步
    graph.set_node_value(input, Some(&Tensor::new(&[1.0], &[1, 1])))?;
    graph.step(hidden)?;
    let h1 = graph.get_node_value(hidden)?.unwrap().data_as_slice()[0];

    // 第二步（h_prev 应该自动被更新为上一步的 hidden）
    graph.set_node_value(input, Some(&Tensor::new(&[0.5], &[1, 1])))?;
    graph.step(hidden)?;
    let h2 = graph.get_node_value(hidden)?.unwrap().data_as_slice()[0];

    // h2 应该大于 h1（因为累加了 h_prev）
    assert!(h2 > h1, "h2 ({}) 应大于 h1 ({})，因为循环累加", h2, h1);

    Ok(())
}

/// 测试 State 节点的 zero_grad
#[test]
fn test_state_zero_grad() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    graph.set_train_mode();

    let state = graph.new_state_node(&[1, 2], Some("state"))?;
    graph.set_node_value(state, Some(&Tensor::new(&[1.0, 2.0], &[1, 2])))?;

    let param = graph.new_parameter_node(&[1, 2], Some("param"))?;
    graph.set_node_value(param, Some(&Tensor::new(&[0.5, 0.5], &[1, 2])))?;

    let add = graph.new_add_node(&[state, param], Some("add"))?;

    // 添加 target 和 loss
    let target = graph.new_input_node(&[1, 2], Some("target"))?;
    graph.set_node_value(target, Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;
    let loss = graph.new_mse_loss_node(add, target, Some("loss"))?;

    graph.forward(loss)?;
    graph.backward(loss)?;

    // 验证有 grad
    assert!(graph.get_node_grad(state)?.is_some());

    // 清除 grad
    graph.zero_grad()?;

    // State 的 grad 应该被清除
    // 注意：zero_grad 清除所有非 Parameter 节点的 grad
    let state_grad_after = graph.get_node_grad(state)?;
    assert!(
        state_grad_after.is_none(),
        "State grad 应被 zero_grad() 清除"
    );

    Ok(())
}

/// 测试 State 节点的 reset 行为
#[test]
fn test_state_reset_behavior() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let h_prev = graph.new_state_node(&[1, 1], Some("h_prev"))?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[1, 1])))?;

    let add = graph.new_add_node(&[input, h_prev], Some("add"))?;
    let hidden = graph.new_tanh_node(add, Some("hidden"))?;

    graph.connect_recurrent(hidden, h_prev)?;

    // 执行几步
    graph.set_node_value(input, Some(&Tensor::new(&[1.0], &[1, 1])))?;
    graph.step(hidden)?;
    graph.step(hidden)?;

    let h_before_reset = graph.get_node_value(h_prev)?.unwrap().data_as_slice()[0];
    assert!(h_before_reset != 0.0, "h_prev 在多步后应非零");

    // Reset
    graph.reset();

    // h_prev 应该被重置为零
    let h_after_reset = graph.get_node_value(h_prev)?.unwrap().data_as_slice()[0];
    assert_eq!(
        h_after_reset, 0.0,
        "h_prev 在 reset 后应为 0（实际得到 {}）",
        h_after_reset
    );

    Ok(())
}

/// 测试多个 State 节点（如 LSTM 的 h 和 c）
#[test]
fn test_multiple_state_nodes() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let h = graph.new_state_node(&[1, 64], Some("hidden"))?;
    let c = graph.new_state_node(&[1, 64], Some("cell"))?;

    graph.set_node_value(h, Some(&Tensor::zeros(&[1, 64])))?;
    graph.set_node_value(c, Some(&Tensor::zeros(&[1, 64])))?;

    // 验证两个 State 节点都存在且独立
    assert!(graph.get_node_value(h)?.is_some());
    assert!(graph.get_node_value(c)?.is_some());

    // 两个都不应该在 trainable 中
    let trainable = graph.get_trainable_nodes();
    assert!(!trainable.contains(&h));
    assert!(!trainable.contains(&c));

    Ok(())
}

/// 测试 State 节点的维度验证
#[test]
fn test_state_dimension_validation() {
    let mut graph = GraphInner::new();

    // 2D 应该成功
    assert!(graph.new_state_node(&[1, 64], None).is_ok());

    // 3D 应该成功
    assert!(graph.new_state_node(&[2, 10, 64], None).is_ok());

    // 4D 应该成功（ConvLSTM）
    assert!(graph.new_state_node(&[2, 32, 7, 7], None).is_ok());

    // 1D 应该失败
    assert_err!(
        graph.new_state_node(&[64], None),
        GraphError::DimensionMismatch { expected, got, .. } if *expected == 2 && *got == 1
    );

    // 5D 应该失败
    assert_err!(
        graph.new_state_node(&[1, 2, 3, 4, 5], None),
        GraphError::DimensionMismatch { expected, got, .. } if *expected == 2 && *got == 5
    );
}

// ==================== 误用场景测试 ====================

/// 测试 State 节点未初始化值时的行为
#[test]
fn test_state_used_without_value() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let state = graph.new_state_node(&[1, 1], Some("state"))?;
    // 故意不设置 state 的值

    // 创建使用 state 的计算节点
    let add = graph.new_add_node(&[input, state], Some("add"))?;

    // 设置 input 值
    graph.set_node_value(input, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // forward 时，state 没有值应该会导致错误
    let result = graph.forward(add);

    // 预期：应该报错或返回合理的默认值
    // 当前实现：State 节点没有值时不能前向传播
    assert_err!(
        result,
        GraphError::InvalidOperation(msg) if msg.contains("不能直接前向传播")
    );

    Ok(())
}

/// 测试 State 节点作为普通计算节点使用（无循环连接）
#[test]
fn test_state_without_recurrent_connection() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    graph.set_train_mode();

    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let state = graph.new_state_node(&[1, 1], Some("state"))?;
    graph.set_node_value(state, Some(&Tensor::zeros(&[1, 1])))?;

    // 使用 state 但不建立循环连接
    let add = graph.new_add_node(&[input, state], Some("add"))?;
    let output = graph.new_tanh_node(add, Some("output"))?;

    // 这应该能正常前向传播（state 作为常量）
    graph.set_node_value(input, Some(&Tensor::new(&[1.0], &[1, 1])))?;
    graph.forward(output)?;

    let val = graph.get_node_value(output)?.unwrap().data_as_slice()[0];
    // tanh(1.0 + 0.0) = tanh(1.0) ≈ 0.7616
    assert!((val - 0.7616).abs() < 0.01, "输出应为 tanh(1) ≈ 0.7616");
    Ok(())
}

/// 测试重复建立循环连接
#[test]
fn test_duplicate_recurrent_connection_error() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let state = graph.new_state_node(&[1, 1], Some("state"))?;
    graph.set_node_value(state, Some(&Tensor::zeros(&[1, 1])))?;

    let add = graph.new_add_node(&[input, state], None)?;
    let hidden = graph.new_tanh_node(add, Some("hidden"))?;

    // 第一次连接应该成功
    graph.connect_recurrent(hidden, state)?;

    // 第二次连接到同一个 state 应该失败
    let result = graph.connect_recurrent(hidden, state);
    assert_err!(
        result,
        GraphError::InvalidOperation(msg) if msg.contains("已经有循环连接源")
    );
    Ok(())
}

/// 测试 State 节点的 grad 在 BPTT 场景下的行为
///
/// BPTT 需要 State 节点接收并传递梯度（跨时间步）
/// 使用 backward_through_time 专用方法
#[test]
fn test_state_grad_in_bptt() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    graph.set_train_mode();

    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let state = graph.new_state_node(&[1, 1], Some("state"))?;
    graph.set_node_value(state, Some(&Tensor::zeros(&[1, 1])))?;

    let w = graph.new_parameter_node(&[1, 1], Some("w"))?;
    graph.set_node_value(w, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // 简单网络：output = tanh(input + state) * w
    let add = graph.new_add_node(&[input, state], None)?;
    let hidden = graph.new_tanh_node(add, None)?;
    let output = graph.new_mat_mul_node(hidden, w, Some("output"))?;

    let target = graph.new_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    graph.connect_recurrent(hidden, state)?;

    // 前向传播多步
    graph.set_node_value(target, Some(&Tensor::new(&[0.5], &[1, 1])))?;
    for &x in &[1.0, 0.5] {
        graph.set_node_value(input, Some(&Tensor::new(&[x], &[1, 1])))?;
        graph.step(loss)?;
    }

    // 使用 BPTT（会自动包含 State 节点）
    graph.backward_through_time(&[w], loss)?;

    // w 应该有梯度（通过 get_node_grad 获取）
    let w_grad = graph.get_node_grad(w)?;
    assert!(w_grad.is_some(), "w 应在 BPTT 后有 grad");
    Ok(())
}

/// 测试 State 节点形状不匹配的循环连接
#[test]
fn test_state_shape_mismatch_recurrent() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 使用无法广播的形状：[2, 3] 和 [2, 4]
    // 第二维 3 != 4 且都不是 1，无法广播
    let input = graph.new_input_node(&[2, 3], Some("input"))?;
    let state = graph.new_state_node(&[2, 4], Some("state"))?;
    graph.set_node_value(state, Some(&Tensor::zeros(&[2, 4])))?;

    // 这应该在 Add 节点创建时就报错（形状无法广播）
    let result = graph.new_add_node(&[input, state], None);
    assert_err!(result, GraphError::ShapeMismatch { .. });
    Ok(())
}

/// 测试 zero_grad 对 State 节点的影响
#[test]
fn test_zero_grad_on_state() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    graph.set_train_mode();

    // 构建一个会产生 State grad 的网络
    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let state = graph.new_state_node(&[1, 1], Some("state"))?;
    graph.set_node_value(state, Some(&Tensor::zeros(&[1, 1])))?;

    let w = graph.new_parameter_node(&[1, 1], Some("w"))?;
    graph.set_node_value(w, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    let add = graph.new_add_node(&[input, state], None)?;
    let hidden = graph.new_tanh_node(add, None)?;
    let output = graph.new_mat_mul_node(hidden, w, Some("output"))?;

    let target = graph.new_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    graph.connect_recurrent(hidden, state)?;

    // 前向和反向传播，产生 grad
    graph.set_node_value(input, Some(&Tensor::new(&[1.0], &[1, 1])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[0.5], &[1, 1])))?;
    graph.forward(loss)?;
    graph.backward(loss)?;

    // 验证 State 有 grad
    let state_grad = graph.get_node_grad(state)?;
    assert!(state_grad.is_some(), "State 节点在 backward 后应有 grad");

    // zero_grad 应该清除 State 的 grad
    graph.zero_grad()?;

    let cleared_grad = graph.get_node_grad(state)?;
    assert!(
        cleared_grad.is_none(),
        "State grad 应在 zero_grad() 后被清除"
    );
    Ok(())
}
