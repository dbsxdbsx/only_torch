/*
 * State 节点单元测试
 *
 * State 节点用于 RNN 的时间状态（如隐藏状态 h、LSTM 的 c）。
 *
 * 与 Input 节点的关键区别：
 *   - State 可以接收并存储 jacobi（用于 BPTT 梯度传递）
 *   - Input 不能接收 jacobi
 *
 * 与 Parameter 节点的关键区别：
 *   - State 不被优化器更新（不在 get_trainable_nodes() 中）
 *   - Parameter 被优化器更新
 */

use crate::assert_err;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

/// 测试 State 节点的基本创建
#[test]
fn test_state_node_creation() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let state = graph.new_state_node(&[1, 64], Some("hidden"))?;

    // 验证节点存在且可获取
    assert!(graph.get_node_value(state)?.is_none()); // 初始值为 None
    Ok(())
}

/// 测试 State 节点的值设置
#[test]
fn test_state_node_set_value() -> Result<(), GraphError> {
    let mut graph = Graph::new();
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
    let mut graph = Graph::new();

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

/// 测试 State 节点可以接收 jacobi（与 Input 的关键区别）
#[test]
fn test_state_accepts_jacobi() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    graph.set_train_mode();

    // 创建网络：state -> add -> output
    let state = graph.new_state_node(&[1, 2], Some("state"))?;
    graph.set_node_value(state, Some(&Tensor::new(&[1.0, 2.0], &[1, 2])))?;

    let param = graph.new_parameter_node(&[1, 2], Some("param"))?;
    graph.set_node_value(param, Some(&Tensor::new(&[0.5, 0.5], &[1, 2])))?;

    // state + param -> add
    let add = graph.new_add_node(&[state, param], Some("add"))?;

    // 前向传播
    graph.forward_node(add)?;

    // 反向传播
    graph.backward_nodes(&[state, param], add)?;

    // State 应该能接收 jacobi
    let state_jacobi = graph.get_node_jacobi(state)?;
    assert!(
        state_jacobi.is_some(),
        "State 节点在 backward 后应有 jacobi"
    );

    // 验证 jacobi 值（Add 节点对两个输入的 jacobi 都是单位矩阵）
    let jacobi = state_jacobi.unwrap();
    assert_eq!(jacobi.shape(), &[2, 2]); // [输出维度, 输入维度]

    Ok(())
}

/// 测试 Input 节点不能接收 jacobi（对照测试）
#[test]
fn test_input_rejects_jacobi() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    graph.set_train_mode();

    // 创建网络：input -> add -> output
    let input = graph.new_input_node(&[1, 2], Some("input"))?;
    graph.set_node_value(input, Some(&Tensor::new(&[1.0, 2.0], &[1, 2])))?;

    let param = graph.new_parameter_node(&[1, 2], Some("param"))?;
    graph.set_node_value(param, Some(&Tensor::new(&[0.5, 0.5], &[1, 2])))?;

    let add = graph.new_add_node(&[input, param], Some("add"))?;

    graph.forward_node(add)?;

    // 只对 param 做 backward（不包含 input）
    graph.backward_nodes(&[param], add)?;

    // Input 节点查询 jacobi 应该返回错误（设计如此）
    let jacobi_result = graph.get_node_jacobi(input);
    assert_err!(
        jacobi_result,
        GraphError::InvalidOperation(msg) if msg.contains("不应该有雅可比矩阵")
    );

    // 尝试对 Input 节点做 backward 应该失败
    let backward_result = graph.backward_nodes(&[input], add);
    assert_err!(
        backward_result,
        GraphError::InvalidOperation(msg) if msg.contains("不应该有雅可比矩阵")
    );

    Ok(())
}

/// 测试 State 节点在 forward_node 中的行为
#[test]
fn test_state_forward_behavior() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let state = graph.new_state_node(&[1, 4], Some("state"))?;

    // 尝试对未设值的 State 进行前向传播应该失败
    let result = graph.forward_node(state);
    assert_err!(
        result,
        GraphError::InvalidOperation(msg) if msg.contains("是输入/参数/状态节点")
    );

    // 设置值后，State 不应该被 forward_node 直接调用（它的值由外部设置）
    graph.set_node_value(state, Some(&Tensor::zeros(&[1, 4])))?;
    let result = graph.forward_node(state);
    assert_err!(
        result,
        GraphError::InvalidOperation(msg) if msg.contains("是输入/参数/状态节点")
    );

    Ok(())
}

/// 测试 State 节点在简单 RNN 结构中的使用
#[test]
fn test_state_in_rnn_structure() -> Result<(), GraphError> {
    let mut graph = Graph::new();
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

    // 前向传播
    graph.forward_node(hidden)?;

    // 反向传播到所有参数和状态
    graph.backward_nodes(&[w, h_prev], hidden)?;

    // 验证 W 有梯度
    let w_jacobi = graph.get_node_jacobi(w)?;
    assert!(w_jacobi.is_some(), "W 应有 jacobi");

    // 验证 h_prev 也有梯度（这是 State 与 Input 的关键区别）
    let h_prev_jacobi = graph.get_node_jacobi(h_prev)?;
    assert!(h_prev_jacobi.is_some(), "h_prev (State) 应有 jacobi");

    Ok(())
}

/// 测试 State 节点与循环连接的配合
#[test]
fn test_state_with_recurrent_connection() -> Result<(), GraphError> {
    let mut graph = Graph::new();
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

/// 测试 State 节点的 clear_jacobi
#[test]
fn test_state_clear_jacobi() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    graph.set_train_mode();

    let state = graph.new_state_node(&[1, 2], Some("state"))?;
    graph.set_node_value(state, Some(&Tensor::new(&[1.0, 2.0], &[1, 2])))?;

    let param = graph.new_parameter_node(&[1, 2], Some("param"))?;
    graph.set_node_value(param, Some(&Tensor::new(&[0.5, 0.5], &[1, 2])))?;

    let add = graph.new_add_node(&[state, param], Some("add"))?;

    graph.forward_node(add)?;
    graph.backward_nodes(&[state, param], add)?;

    // 验证有 jacobi
    assert!(graph.get_node_jacobi(state)?.is_some());

    // 清除 jacobi
    graph.clear_jacobi()?;

    // State 的 jacobi 应该被清除（与 Parameter 不同）
    // 注意：clear_jacobi 保留 Parameter 的 jacobi，但清除其他节点的
    // 这里需要检查 State 是否被正确清除
    // 根据 reset_intermediate_jacobi 的实现，只有 Parameter 的 jacobi 被保留
    let state_jacobi_after = graph.get_node_jacobi(state)?;
    assert!(
        state_jacobi_after.is_none(),
        "State jacobi 应被 clear_jacobi() 清除"
    );

    Ok(())
}

/// 测试 State 节点的 reset 行为
#[test]
fn test_state_reset_behavior() -> Result<(), GraphError> {
    let mut graph = Graph::new();

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
    let mut graph = Graph::new();

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
    let mut graph = Graph::new();

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
    let mut graph = Graph::new();

    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let state = graph.new_state_node(&[1, 1], Some("state"))?;
    // 故意不设置 state 的值

    // 创建使用 state 的计算节点
    let add = graph.new_add_node(&[input, state], Some("add"))?;

    // 设置 input 值
    graph.set_node_value(input, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    // forward 时，state 没有值应该会导致错误
    let result = graph.forward_node(add);

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
    let mut graph = Graph::new();
    graph.set_train_mode();

    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let state = graph.new_state_node(&[1, 1], Some("state"))?;
    graph.set_node_value(state, Some(&Tensor::zeros(&[1, 1])))?;

    // 使用 state 但不建立循环连接
    let add = graph.new_add_node(&[input, state], Some("add"))?;
    let output = graph.new_tanh_node(add, Some("output"))?;

    // 这应该能正常前向传播（state 作为常量）
    graph.set_node_value(input, Some(&Tensor::new(&[1.0], &[1, 1])))?;
    graph.forward_node(output)?;

    let val = graph.get_node_value(output)?.unwrap().data_as_slice()[0];
    // tanh(1.0 + 0.0) = tanh(1.0) ≈ 0.7616
    assert!((val - 0.7616).abs() < 0.01, "输出应为 tanh(1) ≈ 0.7616");
    Ok(())
}

/// 测试重复建立循环连接
#[test]
fn test_duplicate_recurrent_connection_error() -> Result<(), GraphError> {
    let mut graph = Graph::new();

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

/// 测试 State 节点的 jacobi 在 BPTT 场景下的行为
///
/// 普通 backward 只计算指定 target_nodes 的梯度，
/// 如果 State 不在 target_nodes 中，它可能没有 jacobi。
/// 但在 BPTT 中，State 节点会被自动包含以支持跨时间梯度传递。
#[test]
fn test_state_jacobi_in_bptt() -> Result<(), GraphError> {
    let mut graph = Graph::new();
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

    // w 应该有梯度
    let w_jacobi = graph.get_node_jacobi(w)?;
    assert!(w_jacobi.is_some(), "w 应在 BPTT 后有 jacobi");
    Ok(())
}

/// 测试 State 节点形状不匹配的循环连接
#[test]
fn test_state_shape_mismatch_recurrent() -> Result<(), GraphError> {
    let mut graph = Graph::new();

    let input = graph.new_input_node(&[1, 1], Some("input"))?;
    let state = graph.new_state_node(&[1, 2], Some("state"))?; // 形状 [1, 2]
    graph.set_node_value(state, Some(&Tensor::zeros(&[1, 2])))?;

    // hidden 形状会是 [1, 1]（与 input 相同）
    // 但 state 形状是 [1, 2]
    // 这应该在 Add 节点创建时就报错
    let result = graph.new_add_node(&[input, state], None);
    assert_err!(result, GraphError::ShapeMismatch { .. });
    Ok(())
}

/// 测试 clear_jacobi 对 State 节点的影响
#[test]
fn test_clear_jacobi_on_state() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    graph.set_train_mode();

    // 构建一个会产生 State jacobi 的网络
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

    // 前向和反向传播，产生 jacobi
    graph.set_node_value(input, Some(&Tensor::new(&[1.0], &[1, 1])))?;
    graph.set_node_value(target, Some(&Tensor::new(&[0.5], &[1, 1])))?;
    graph.step(loss)?;
    graph.backward_nodes(&[w, state], loss)?;

    // 验证 State 有 jacobi
    let state_jacobi = graph.get_node_jacobi(state)?;
    assert!(
        state_jacobi.is_some(),
        "State 节点在 backward 后应有 jacobi"
    );

    // clear_jacobi 应该清除 State 的 jacobi
    graph.clear_jacobi()?;

    let cleared_jacobi = graph.get_node_jacobi(state)?;
    assert!(
        cleared_jacobi.is_none(),
        "State jacobi 应在 clear_jacobi() 后被清除"
    );
    Ok(())
}
