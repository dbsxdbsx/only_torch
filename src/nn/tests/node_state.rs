/*
 * @Author       : 老董
 * @Description  : State 节点单元测试
 *
 * State 节点用于 RNN 的时间状态（如隐藏状态 h、LSTM 的 c）。
 * 与 Input 的区别：State 可接收梯度（用于 BPTT）。
 * 与 Parameter 的区别：State 不被优化器更新。
 *
 * 注：循环连接相关测试（connect_recurrent/step/BPTT）已移至展开式 RNN 测试。
 */

use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use std::rc::Rc;

// ==================== 基础创建与值操作测试 ====================

/// 测试 State 节点的基本创建
#[test]
fn test_state_node_creation() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let state = inner
        .borrow_mut()
        .create_state_node(&[1, 64], Some("hidden"))
        .unwrap();

    // 初始值为 None
    assert!(state.value().is_none());
    assert_eq!(state.shape(), vec![1, 64]);
    assert_eq!(state.name(), Some("hidden"));
    assert!(state.is_leaf());
    assert!(state.parents().is_empty());
}

/// 测试 State 节点的值设置
#[test]
fn test_state_node_set_value() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let state = inner
        .borrow_mut()
        .create_state_node(&[1, 4], Some("hidden"))
        .unwrap();

    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    state.set_value(Some(&tensor)).unwrap();

    let value = state.value().unwrap();
    assert_eq!(value.data_as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(value.shape(), &[1, 4]);
}

/// 测试 State 节点不在可训练参数中
///
/// State 不被优化器更新，不应出现在 parameters 注册表中
#[test]
fn test_state_not_trainable() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let _input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("input"))
        .unwrap();
    let _state = inner
        .borrow_mut()
        .create_state_node(&[1, 4], Some("hidden"))
        .unwrap();
    let param = inner
        .borrow_mut()
        .create_parameter_node(&[4, 4], Some("weight"))
        .unwrap();

    // 注册 Parameter
    inner
        .borrow_mut()
        .register_parameter("weight".to_string(), Rc::downgrade(&param))
        .unwrap();

    // 只有 Parameter 在 parameters 注册表中
    let all_params = inner.borrow().get_all_parameters();
    assert_eq!(all_params.len(), 1, "只有 Parameter 应在注册表中");
    assert_eq!(all_params[0].0, "weight");
}

// ==================== 前向传播测试 ====================

/// 测试 State 节点在 forward 中的行为
///
/// State 是叶子节点，值由外部管理：
/// - 没有值时：forward 报错
/// - 有值时：forward 静默成功
#[test]
fn test_state_forward_behavior() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let state = inner
        .borrow_mut()
        .create_state_node(&[1, 4], Some("state"))?;

    // 未设值时 forward 应该失败
    let result = inner.borrow_mut().forward_via_node_inner(&state);
    assert!(result.is_err(), "未设值的 State 节点 forward 应失败");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("没有值"),
        "错误应提示没有值，实际: {}",
        err_msg
    );

    // 设置值后 forward 应成功
    state.set_value(Some(&Tensor::zeros(&[1, 4])))?;
    inner.borrow_mut().forward_via_node_inner(&state)?;

    Ok(())
}

/// 测试 State 节点作为计算输入（无循环连接）
///
/// State 可以作为普通输入参与前向计算
#[test]
fn test_state_without_recurrent_connection() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1], Some("input"))?;
    input.set_value(Some(&Tensor::new(&[1.0], &[1, 1])))?;

    let state = inner
        .borrow_mut()
        .create_state_node(&[1, 1], Some("state"))?;
    state.set_value(Some(&Tensor::zeros(&[1, 1])))?;

    let add = inner
        .borrow_mut()
        .create_add_node(vec![input.clone(), state.clone()], Some("add"))?;
    let output = inner
        .borrow_mut()
        .create_tanh_node(add, Some("output"))?;

    inner.borrow_mut().forward_via_node_inner(&output)?;

    let val = output.value().unwrap();
    let v = val.data_as_slice()[0];
    // tanh(1.0 + 0.0) = tanh(1.0) ≈ 0.7616
    assert!(
        (v - 0.7616).abs() < 0.01,
        "输出应为 tanh(1) ≈ 0.7616，实际: {}",
        v
    );

    Ok(())
}

// ==================== 反向传播 & 梯度测试 ====================

/// 测试 State 节点可以接收梯度（与 Input 的关键区别）
///
/// 在 backward 后，State 节点应有 grad
#[test]
fn test_state_accepts_grad() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 创建网络：state + param -> add -> MSE loss
    let state = inner
        .borrow_mut()
        .create_state_node(&[1, 2], Some("state"))?;
    state.set_value(Some(&Tensor::new(&[1.0, 2.0], &[1, 2])))?;

    let param = inner
        .borrow_mut()
        .create_parameter_node(&[1, 2], Some("param"))?;
    inner
        .borrow_mut()
        .register_parameter("param".to_string(), Rc::downgrade(&param))?;
    param.set_value(Some(&Tensor::new(&[0.5, 0.5], &[1, 2])))?;

    let add = inner
        .borrow_mut()
        .create_add_node(vec![state.clone(), param.clone()], Some("add"))?;

    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("target"))?;
    target.set_value(Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;

    let loss = inner
        .borrow_mut()
        .create_mse_mean_node(add, target, Some("loss"))?;

    // 设置训练模式、前向传播、反向传播
    inner.borrow_mut().set_train_mode();
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    inner.borrow_mut().backward_via_node_inner(&loss, false)?;

    // State 应该能接收 grad
    let state_grad = state.grad();
    assert!(state_grad.is_some(), "State 节点在 backward 后应有 grad");
    assert_eq!(state_grad.unwrap().shape(), &[1, 2]);

    Ok(())
}

/// 测试 Input 节点不能有梯度（对照测试）
///
/// Input 节点在 backward 后不应有 grad（与 State 对比）
#[test]
fn test_input_has_no_grad() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("input"))?;
    input.set_value(Some(&Tensor::new(&[1.0, 2.0], &[1, 2])))?;

    let param = inner
        .borrow_mut()
        .create_parameter_node(&[1, 2], Some("param"))?;
    inner
        .borrow_mut()
        .register_parameter("param".to_string(), Rc::downgrade(&param))?;
    param.set_value(Some(&Tensor::new(&[0.5, 0.5], &[1, 2])))?;

    let add = inner
        .borrow_mut()
        .create_add_node(vec![input.clone(), param.clone()], Some("add"))?;

    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("target"))?;
    target.set_value(Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;

    let loss = inner
        .borrow_mut()
        .create_mse_mean_node(add, target, Some("loss"))?;

    inner.borrow_mut().set_train_mode();
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    inner.borrow_mut().backward_via_node_inner(&loss, false)?;

    // Input 节点不应有 grad（accumulate_grad 静默跳过不支持梯度的节点）
    assert!(
        input.grad().is_none(),
        "Input 节点不应有 grad（梯度汇点）"
    );

    // 对比：param 应有 grad
    assert!(param.grad().is_some(), "Parameter 应有 grad");

    Ok(())
}

/// 测试 State 节点在简单 RNN 结构中的梯度传递
///
/// 验证 State 作为 h_prev 参与计算时能正确接收梯度
#[test]
fn test_state_in_rnn_structure() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 简单 RNN: hidden = tanh(h_prev + input * W)
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1], Some("input"))?;
    input.set_value(Some(&Tensor::new(&[1.0], &[1, 1])))?;

    let h_prev = inner
        .borrow_mut()
        .create_state_node(&[1, 1], Some("h_prev"))?;
    h_prev.set_value(Some(&Tensor::zeros(&[1, 1])))?;

    let w = inner
        .borrow_mut()
        .create_parameter_node(&[1, 1], Some("W"))?;
    inner
        .borrow_mut()
        .register_parameter("W".to_string(), Rc::downgrade(&w))?;
    w.set_value(Some(&Tensor::new(&[0.5], &[1, 1])))?;

    // input * W
    let scaled = inner
        .borrow_mut()
        .create_mat_mul_node(vec![input.clone(), w.clone()], Some("scaled"))?;

    // h_prev + scaled
    let pre_hidden = inner
        .borrow_mut()
        .create_add_node(vec![h_prev.clone(), scaled], Some("pre_hidden"))?;

    // tanh(pre_hidden)
    let hidden = inner
        .borrow_mut()
        .create_tanh_node(pre_hidden, Some("hidden"))?;

    // target 和 loss
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1], Some("target"))?;
    target.set_value(Some(&Tensor::new(&[0.5], &[1, 1])))?;

    let loss = inner
        .borrow_mut()
        .create_mse_mean_node(hidden, target, Some("loss"))?;

    inner.borrow_mut().set_train_mode();
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    inner.borrow_mut().backward_via_node_inner(&loss, false)?;

    // W 应有梯度
    assert!(w.grad().is_some(), "W 应有 grad");

    // h_prev (State) 应有梯度 —— 与 Input 的关键区别
    assert!(h_prev.grad().is_some(), "h_prev (State) 应有 grad");

    Ok(())
}

/// 测试 State 节点的 zero_grad 行为
///
/// zero_grad 只清除 Parameter 注册表中的参数梯度。
/// State 不在注册表中，其梯度不受 zero_grad 影响。
/// 需手动调用 clear_grad() 清除 State 梯度。
#[test]
fn test_state_zero_grad() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let state = inner
        .borrow_mut()
        .create_state_node(&[1, 2], Some("state"))?;
    state.set_value(Some(&Tensor::new(&[1.0, 2.0], &[1, 2])))?;

    let param = inner
        .borrow_mut()
        .create_parameter_node(&[1, 2], Some("param"))?;
    inner
        .borrow_mut()
        .register_parameter("param".to_string(), Rc::downgrade(&param))?;
    param.set_value(Some(&Tensor::new(&[0.5, 0.5], &[1, 2])))?;

    let add = inner
        .borrow_mut()
        .create_add_node(vec![state.clone(), param.clone()], Some("add"))?;

    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("target"))?;
    target.set_value(Some(&Tensor::new(&[1.0, 1.0], &[1, 2])))?;

    let loss = inner
        .borrow_mut()
        .create_mse_mean_node(add, target, Some("loss"))?;

    inner.borrow_mut().set_train_mode();
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    inner.borrow_mut().backward_via_node_inner(&loss, false)?;

    // backward 后 State 和 Parameter 都有 grad
    assert!(state.grad().is_some(), "State 应有 grad");
    assert!(param.grad().is_some(), "Parameter 应有 grad");

    // zero_grad 只清除 Parameter 的 grad
    inner.borrow_mut().zero_grad()?;
    assert!(
        param.grad().is_none(),
        "Parameter grad 应被 zero_grad() 清除"
    );

    // State 的 grad 需手动清除
    state.clear_grad()?;
    assert!(
        state.grad().is_none(),
        "State grad 应在 clear_grad() 后被清除"
    );

    Ok(())
}

// ==================== 多 State 节点测试 ====================

/// 测试多个 State 节点（如 LSTM 的 h 和 c）
#[test]
fn test_multiple_state_nodes() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let h = inner
        .borrow_mut()
        .create_state_node(&[1, 64], Some("hidden"))
        .unwrap();
    let c = inner
        .borrow_mut()
        .create_state_node(&[1, 64], Some("cell"))
        .unwrap();

    h.set_value(Some(&Tensor::zeros(&[1, 64]))).unwrap();
    c.set_value(Some(&Tensor::zeros(&[1, 64]))).unwrap();

    // 两个 State 独立存在
    assert!(h.value().is_some());
    assert!(c.value().is_some());
    assert_ne!(h.id(), c.id(), "两个 State 应有不同 ID");

    // 两个都不在 parameters 注册表中
    let all_params = inner.borrow().get_all_parameters();
    assert!(
        all_params.is_empty(),
        "State 不应出现在 parameters 注册表中"
    );
}

// ==================== 维度验证测试 ====================

/// 测试 State 节点的维度验证
///
/// 支持 2D-4D，1D 和 5D 应失败
#[test]
fn test_state_dimension_validation() {
    use crate::assert_err;

    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 2D: 基础 RNN [batch, hidden_size]
    assert!(inner.borrow_mut().create_state_node(&[1, 64], None).is_ok());

    // 3D: 序列状态 [batch, seq_len, hidden_size]
    assert!(
        inner
            .borrow_mut()
            .create_state_node(&[2, 10, 64], None)
            .is_ok()
    );

    // 4D: ConvLSTM [batch, C, H, W]
    assert!(
        inner
            .borrow_mut()
            .create_state_node(&[2, 32, 7, 7], None)
            .is_ok()
    );

    // 1D 应失败
    assert_err!(
        inner.borrow_mut().create_state_node(&[64], None),
        GraphError::DimensionMismatch { expected, got, .. } if *expected == 2 && *got == 1
    );

    // 5D 应失败
    assert_err!(
        inner
            .borrow_mut()
            .create_state_node(&[1, 2, 3, 4, 5], None),
        GraphError::DimensionMismatch { expected, got, .. } if *expected == 2 && *got == 5
    );
}

// ==================== 误用场景测试 ====================

/// 测试 State 节点未初始化值时的行为
///
/// State 作为计算节点的输入，若无值则 forward 应报错
#[test]
fn test_state_used_without_value() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1], Some("input"))?;
    input.set_value(Some(&Tensor::new(&[1.0], &[1, 1])))?;

    let state = inner
        .borrow_mut()
        .create_state_node(&[1, 1], Some("state"))?;
    // 故意不设置 state 的值

    let add = inner
        .borrow_mut()
        .create_add_node(vec![input, state], Some("add"))?;

    // forward 时 state 无值应报错
    let result = inner.borrow_mut().forward_via_node_inner(&add);
    assert!(result.is_err(), "State 无值时 forward 应失败");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("没有值"),
        "错误应提示没有值，实际: {}",
        err_msg
    );

    Ok(())
}

// ==================== 动态形状测试 ====================

/// 测试 State 节点的动态形状传播
///
/// State 的 dynamic_expected_shape 第一维应为 None（动态 batch）
#[test]
fn test_state_dynamic_shape_propagation() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let state = inner
        .borrow_mut()
        .create_state_node(&[4, 64], Some("hidden"))
        .unwrap();

    let dyn_shape = state.dynamic_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(64), "特征维度应该是 64");
}

/// 测试 State 节点在不同维度下的动态形状
#[test]
fn test_state_dynamic_shape_various_dims() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 2D: [batch, hidden_size]
    let state_2d = inner
        .borrow_mut()
        .create_state_node(&[4, 64], Some("hidden_2d"))
        .unwrap();
    let dyn_2d = state_2d.dynamic_shape();
    assert!(dyn_2d.is_dynamic(0), "2D: batch 应该是动态的");
    assert!(!dyn_2d.is_dynamic(1), "2D: hidden_size 应该是固定的");

    // 3D: [batch, seq_len, hidden_size]
    let state_3d = inner
        .borrow_mut()
        .create_state_node(&[4, 10, 64], Some("hidden_3d"))
        .unwrap();
    let dyn_3d = state_3d.dynamic_shape();
    assert!(dyn_3d.is_dynamic(0), "3D: batch 应该是动态的");
    assert!(!dyn_3d.is_dynamic(1), "3D: seq_len 应该是固定的");
    assert!(!dyn_3d.is_dynamic(2), "3D: hidden_size 应该是固定的");

    // 4D: [batch, C, H, W]（ConvLSTM）
    let state_4d = inner
        .borrow_mut()
        .create_state_node(&[4, 32, 7, 7], Some("hidden_4d"))
        .unwrap();
    let dyn_4d = state_4d.dynamic_shape();
    assert!(dyn_4d.is_dynamic(0), "4D: batch 应该是动态的");
    assert!(!dyn_4d.is_dynamic(1), "4D: channels 应该是固定的");
    assert!(!dyn_4d.is_dynamic(2), "4D: height 应该是固定的");
    assert!(!dyn_4d.is_dynamic(3), "4D: width 应该是固定的");
}

/// 测试 State 节点在不同 batch_size 下的前向计算
#[test]
fn test_state_dynamic_batch_forward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("input"))?;
    let state = inner
        .borrow_mut()
        .create_state_node(&[2, 4], Some("state"))?;

    input.set_value(Some(&Tensor::ones(&[2, 4])))?;
    state.set_value(Some(&Tensor::zeros(&[2, 4])))?;

    let add = inner
        .borrow_mut()
        .create_add_node(vec![input.clone(), state.clone()], Some("add"))?;
    let output = inner
        .borrow_mut()
        .create_tanh_node(add, Some("output"))?;

    // 第一次 forward: batch=2
    inner.borrow_mut().forward_via_node_inner(&output)?;
    let value1 = output.value().unwrap();
    assert_eq!(value1.shape(), &[2, 4], "第一次 forward: batch=2");

    // 更新为不同 batch 大小
    input.set_value(Some(&Tensor::ones(&[6, 4])))?;
    state.set_value(Some(&Tensor::zeros(&[6, 4])))?;

    // 第二次 forward: batch=6
    inner.borrow_mut().forward_via_node_inner(&output)?;
    let value2 = output.value().unwrap();
    assert_eq!(value2.shape(), &[6, 4], "第二次 forward: batch=6");

    Ok(())
}

/// 测试 State 节点在不同 batch_size 下的反向传播
#[test]
fn test_state_dynamic_batch_backward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 网络：input @ weight + state -> tanh -> output -> MSE loss
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("input"))?;
    let state = inner
        .borrow_mut()
        .create_state_node(&[2, 4], Some("state"))?;
    let weight = inner
        .borrow_mut()
        .create_parameter_node(&[4, 4], Some("weight"))?;
    inner
        .borrow_mut()
        .register_parameter("weight".to_string(), Rc::downgrade(&weight))?;

    input.set_value(Some(&Tensor::ones(&[2, 4])))?;
    state.set_value(Some(&Tensor::ones(&[2, 4])))?;
    weight.set_value(Some(&Tensor::normal_seeded(0.0, 0.1, &[4, 4], 42)))?;

    // input @ weight
    let proj = inner
        .borrow_mut()
        .create_mat_mul_node(vec![input.clone(), weight.clone()], Some("proj"))?;
    // proj + state
    let add = inner
        .borrow_mut()
        .create_add_node(vec![proj, state.clone()], Some("add"))?;
    // tanh
    let output = inner
        .borrow_mut()
        .create_tanh_node(add, Some("output"))?;

    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("target"))?;
    target.set_value(Some(&Tensor::zeros(&[2, 4])))?;

    let loss = inner
        .borrow_mut()
        .create_mse_mean_node(output, target.clone(), Some("loss"))?;

    // 第一次训练: batch=2
    inner.borrow_mut().set_train_mode();
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    inner.borrow_mut().zero_grad()?;
    inner.borrow_mut().backward_via_node_inner(&loss, false)?;

    // 验证 State 有梯度
    let state_grad1 = state.grad();
    assert!(state_grad1.is_some(), "State 应有梯度");
    assert_eq!(state_grad1.unwrap().shape(), &[2, 4]);

    // 验证 weight 梯度形状
    let weight_grad1 = weight.grad();
    assert!(weight_grad1.is_some(), "Weight 应有梯度");
    assert_eq!(weight_grad1.unwrap().shape(), &[4, 4]);

    // 更新为不同 batch 大小
    input.set_value(Some(&Tensor::ones(&[5, 4])))?;
    state.set_value(Some(&Tensor::ones(&[5, 4])))?;
    state.clear_grad()?;
    target.set_value(Some(&Tensor::zeros(&[5, 4])))?;

    // 第二次训练: batch=5
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    inner.borrow_mut().zero_grad()?;
    inner.borrow_mut().backward_via_node_inner(&loss, false)?;

    // 验证 State 梯度形状随 batch 变化
    let state_grad2 = state.grad();
    assert!(state_grad2.is_some(), "State 应有梯度");
    assert_eq!(
        state_grad2.unwrap().shape(),
        &[5, 4],
        "State 梯度应适应新 batch 大小"
    );

    // weight 梯度形状应保持不变
    let weight_grad2 = weight.grad();
    assert!(weight_grad2.is_some(), "Weight 应有梯度");
    assert_eq!(
        weight_grad2.unwrap().shape(),
        &[4, 4],
        "Weight 梯度形状应保持不变"
    );

    Ok(())
}

// ==================== 方案 C：新节点创建 API 测试 ====================

#[test]
fn test_create_state_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 创建 State 节点
    let state = inner
        .borrow_mut()
        .create_state_node(&[4, 64], Some("hidden"))
        .unwrap();

    // 验证节点属性
    assert_eq!(state.shape(), vec![4, 64]);
    assert_eq!(state.name(), Some("hidden"));
    assert!(state.is_leaf());
    assert!(state.parents().is_empty());

    // State 初始值为 None（由 reset() 设置）
    assert!(state.value().is_none());
}

#[test]
fn test_create_state_auto_name() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let state = inner
        .borrow_mut()
        .create_state_node(&[4, 64], None)
        .unwrap();

    let name = state.name().unwrap();
    assert!(name.contains("state"), "名称应包含 'state': {}", name);
}

#[test]
fn test_create_state_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak;
    {
        let state = inner
            .borrow_mut()
            .create_state_node(&[4, 64], None)
            .unwrap();
        weak = Rc::downgrade(&state);
        assert!(weak.upgrade().is_some());
    }
    // state 离开作用域，节点被释放
    assert!(weak.upgrade().is_none());
}

#[test]
fn test_create_state_various_shapes() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 2D: RNN 隐藏状态 [batch, hidden_size]
    let rnn_state = inner
        .borrow_mut()
        .create_state_node(&[32, 128], None)
        .unwrap();
    assert_eq!(rnn_state.shape(), vec![32, 128]);

    // 3D: 序列隐藏状态 [batch, seq_len, hidden_size]
    let seq_state = inner
        .borrow_mut()
        .create_state_node(&[32, 10, 128], None)
        .unwrap();
    assert_eq!(seq_state.shape(), vec![32, 10, 128]);

    // 4D: ConvLSTM 状态 [batch, C, H, W]
    let conv_state = inner
        .borrow_mut()
        .create_state_node(&[32, 64, 8, 8], None)
        .unwrap();
    assert_eq!(conv_state.shape(), vec![32, 64, 8, 8]);
}

#[test]
fn test_create_state_invalid_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 1D 形状应该失败
    let result = inner.borrow_mut().create_state_node(&[64], None);
    assert!(result.is_err());

    // 5D 形状也应该失败
    let result = inner
        .borrow_mut()
        .create_state_node(&[1, 2, 3, 4, 5], None);
    assert!(result.is_err());
}
