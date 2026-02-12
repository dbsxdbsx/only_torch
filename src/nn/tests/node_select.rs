/*
 * @Author       : 老董
 * @Description  : Select 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ basic 3D; 首/末时间步; axis=0; 错误处理; cannot_set_value
 * 2. VJP 单元测试（底层）→ scatter unit; scatter non-unit
 * 3. E2E 反向传播（高层）→ sigmoid chain; 多 select RNN; RNN 单步; 两步展开
 * 4. 动态形状（KEEP AS-IS）
 * 5. Create API（KEEP AS-IS）
 *
 * Select 从张量中选择指定轴的切片。VJP: 全零张量 scatter 上游梯度到选中索引位置
 */

use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarLossOps, VarMatrixOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试（高层 Graph + Var API）====================

/// 基本 select：从 [2, 3, 4] 中选择 axis=1, index=1 → [2, 4]
#[test]
fn test_select_forward_basic_3d() {
    let graph = Graph::new();

    // 填充：input[i, j, k] = i*100 + j*10 + k
    let mut input_data = Tensor::zeros(&[2, 3, 4]);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                input_data[[i, j, k]] = (i * 100 + j * 10 + k) as f32;
            }
        }
    }

    let x = graph.input(&input_data).unwrap();
    let selected = x.select(1, 1).unwrap();

    selected.forward().unwrap();

    let output = selected.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 4]);

    // output[i, k] = input[i, 1, k] = i*100 + 10 + k
    for i in 0..2 {
        for k in 0..4 {
            let expected = (i * 100 + 10 + k) as f32;
            assert_abs_diff_eq!(output[[i, k]], expected, epsilon = 1e-6);
        }
    }
}

/// 选择第一个时间步 (index=0)
#[test]
fn test_select_forward_first_timestep() {
    let graph = Graph::new();

    let input_data = Tensor::normal_seeded(0.0, 1.0, &[4, 5, 3], 42);
    let x = graph.input(&input_data).unwrap();
    let selected = x.select(1, 0).unwrap();

    selected.forward().unwrap();

    let output = selected.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[4, 3]);

    for i in 0..4 {
        for k in 0..3 {
            assert_abs_diff_eq!(output[[i, k]], input_data[[i, 0, k]], epsilon = 1e-6);
        }
    }
}

/// 选择最后一个时间步
#[test]
fn test_select_forward_last_timestep() {
    let graph = Graph::new();

    let seq_len = 5;
    let input_data = Tensor::normal_seeded(0.0, 1.0, &[2, seq_len, 3], 42);
    let x = graph.input(&input_data).unwrap();
    let selected = x.select(1, seq_len - 1).unwrap();

    selected.forward().unwrap();

    let output = selected.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 3]);

    for i in 0..2 {
        for k in 0..3 {
            assert_abs_diff_eq!(
                output[[i, k]],
                input_data[[i, seq_len - 1, k]],
                epsilon = 1e-6
            );
        }
    }
}

/// axis=0 选择（选择 batch 维度中的一行）
#[test]
fn test_select_forward_axis_0() {
    let graph = Graph::new();

    let input_data = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
            9.0, 10.0, 11.0, 12.0, // row 2
        ],
        &[3, 4],
    );
    let x = graph.input(&input_data).unwrap();
    let selected = x.select(0, 1).unwrap();

    selected.forward().unwrap();

    let output = selected.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[4]);

    let expected = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[4]);
    assert_eq!(output, &expected);
}

/// axis 越界错误
#[test]
fn test_select_error_axis_out_of_bounds() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 3, 4])).unwrap();

    // axis=3 超出 3 维张量的范围
    let result = x.select(3, 0);
    assert!(result.is_err());
}

/// index 越界错误
#[test]
fn test_select_error_index_out_of_bounds() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 3, 4])).unwrap();

    // index=5 超出 axis=1 的大小 3
    let result = x.select(1, 5);
    assert!(result.is_err());
}

/// Select 节点不能直接设置值
#[test]
fn test_select_cannot_set_value() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 3, 4])).unwrap();
    let selected = x.select(1, 0).unwrap();

    let err = selected.set_value(&Tensor::zeros(&[2, 4]));
    assert!(err.is_err(), "Select 节点不应支持直接设值");
}

// ==================== 2. VJP 单元测试（底层 calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证 scatter 梯度。
// Select VJP: 全零张量中只在 [:, index, :] 填入 upstream

/// VJP scatter：unit upstream（全 1.0 上游梯度）
///
/// input [2, 3, 4], select axis=1, index=1
/// upstream [2, 4] = 全 1.0
/// grad [2, 3, 4]: 只有 [:, 1, :] = 1.0，其余 = 0.0
#[test]
fn test_select_vjp_scatter_unit() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3, 4], Some("input"))
        .unwrap();
    let selected = inner
        .borrow_mut()
        .create_select_node(input.clone(), 1, 1, Some("sel"))
        .unwrap();

    input
        .set_value(Some(&Tensor::normal_seeded(0.0, 1.0, &[2, 3, 4], 42)))
        .unwrap();
    selected.forward_recursive(1, false).unwrap();

    let upstream = Tensor::ones(&[2, 4]);
    let grad = selected.calc_grad_to_parent_index(0, &upstream)?;

    assert_eq!(grad.shape(), &[2, 3, 4]);

    // 只有 [:, 1, :] 处为 1.0，其他为 0.0
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                let expected = if j == 1 { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(grad[[i, j, k]], expected, epsilon = 1e-6);
            }
        }
    }

    Ok(())
}

/// VJP scatter：非 unit upstream
///
/// input [2, 3, 4], select axis=1, index=2
/// upstream [2, 4]: upstream[i, k] = i*10 + k
/// grad [2, 3, 4]: 只有 [:, 2, :] = upstream 值，其余 = 0.0
#[test]
fn test_select_vjp_scatter_non_unit() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3, 4], Some("input"))
        .unwrap();
    let selected = inner
        .borrow_mut()
        .create_select_node(input.clone(), 1, 2, Some("sel"))
        .unwrap();

    input
        .set_value(Some(&Tensor::zeros(&[2, 3, 4])))
        .unwrap();
    selected.forward_recursive(1, false).unwrap();

    // 非 unit 上游梯度
    let mut upstream = Tensor::zeros(&[2, 4]);
    for i in 0..2 {
        for k in 0..4 {
            upstream[[i, k]] = (i * 10 + k) as f32;
        }
    }

    let grad = selected.calc_grad_to_parent_index(0, &upstream)?;

    assert_eq!(grad.shape(), &[2, 3, 4]);

    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                let expected = if j == 2 { (i * 10 + k) as f32 } else { 0.0 };
                assert_abs_diff_eq!(grad[[i, j, k]], expected, epsilon = 1e-6);
            }
        }
    }

    Ok(())
}

// ==================== 3. 端到端反向传播测试（高层 Graph + Var API）====================

/// sigmoid chain: output = sigmoid(matmul(select(input, 1, 0), W))
///
/// input: [2, 3, 4] -> select(axis=1, idx=0) -> [2, 4] -> matmul(W:[4,2]) -> [2, 2] -> sigmoid -> MSE loss
/// 验证 input 梯度只在 [:, 0, :] 处有非零值
#[test]
fn test_select_e2e_sigmoid_chain() {
    let graph = Graph::new();

    let input = graph
        .parameter(
            &[2, 3, 4],
            Init::Zeros,
            "input",
        )
        .unwrap();
    input
        .set_value(&Tensor::normal_seeded(0.0, 1.0, &[2, 3, 4], 42))
        .unwrap();

    let selected = input.select(1, 0).unwrap();

    let w = graph.parameter(&[4, 2], Init::Zeros, "w").unwrap();
    w.set_value(&Tensor::normal_seeded(0.0, 0.5, &[4, 2], 100))
        .unwrap();

    let mm = selected.matmul(&w).unwrap();
    let sig = mm.sigmoid();

    let target = graph.input(&Tensor::zeros(&[2, 2])).unwrap();
    let loss = sig.mse_loss(&target).unwrap();

    graph.zero_grad().unwrap();
    let loss_val = loss.backward().unwrap();
    assert!(loss_val > 0.0);

    let input_grad = input.grad().unwrap().unwrap();
    assert_eq!(input_grad.shape(), &[2, 3, 4]);

    // 只有 [:, 0, :] 处有非零梯度
    let mut has_nonzero_at_0 = false;
    let mut has_nonzero_at_other = false;
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                if input_grad[[i, j, k]].abs() > 1e-10 {
                    if j == 0 {
                        has_nonzero_at_0 = true;
                    } else {
                        has_nonzero_at_other = true;
                    }
                }
            }
        }
    }
    assert!(has_nonzero_at_0, "index=0 处应有非零梯度");
    assert!(!has_nonzero_at_other, "其他 index 处应全为零");
}

/// 多个 select 模拟 RNN 展开：选择 3 个时间步求和
///
/// x_seq [2, 3, 4] -> select(t=0) + select(t=1) + select(t=2) -> MSE loss
/// 所有时间步都应有梯度
#[test]
fn test_select_e2e_multi_select_rnn() {
    let graph = Graph::new();

    let batch = 2;
    let seq_len = 3;
    let input_size = 4;

    let x_seq = graph
        .parameter(&[batch, seq_len, input_size], Init::Zeros, "x_seq")
        .unwrap();
    x_seq
        .set_value(&Tensor::normal_seeded(
            0.0,
            1.0,
            &[batch, seq_len, input_size],
            42,
        ))
        .unwrap();

    // 选择 3 个时间步
    let x_0 = x_seq.select(1, 0).unwrap();
    let x_1 = x_seq.select(1, 1).unwrap();
    let x_2 = x_seq.select(1, 2).unwrap();

    // 求和
    let sum_01 = &x_0 + &x_1;
    let sum_all = &sum_01 + &x_2;

    let target = graph
        .input(&Tensor::zeros(&[batch, input_size]))
        .unwrap();
    let loss = sum_all.mse_loss(&target).unwrap();

    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    let x_grad = x_seq.grad().unwrap().unwrap();
    assert_eq!(x_grad.shape(), &[batch, seq_len, input_size]);

    // 所有时间步都应有梯度
    for j in 0..seq_len {
        let mut has_nonzero = false;
        for i in 0..batch {
            for k in 0..input_size {
                if x_grad[[i, j, k]].abs() > 1e-10 {
                    has_nonzero = true;
                }
            }
        }
        assert!(has_nonzero, "时间步 {} 应有非零梯度", j);
    }
}

/// RNN 单步：h = tanh(x_0 @ W_ih)
///
/// x_seq [2, 4, 3] -> select(t=0) -> [2, 3] -> matmul(W_ih:[3, 5]) -> [2, 5] -> tanh -> MSE loss
#[test]
fn test_select_e2e_rnn_single_step() {
    let graph = Graph::new();

    let batch = 2;
    let seq_len = 4;
    let input_size = 3;
    let hidden_size = 5;

    let x_seq = graph
        .input(&Tensor::normal_seeded(
            0.0,
            1.0,
            &[batch, seq_len, input_size],
            100,
        ))
        .unwrap();

    let w_ih = graph
        .parameter(&[input_size, hidden_size], Init::Zeros, "w_ih")
        .unwrap();
    w_ih.set_value(&Tensor::normal_seeded(
        0.0,
        0.5,
        &[input_size, hidden_size],
        42,
    ))
    .unwrap();

    let x_0 = x_seq.select(1, 0).unwrap();
    let mm = x_0.matmul(&w_ih).unwrap();
    let h = mm.tanh();

    let target = graph
        .input(&Tensor::zeros(&[batch, hidden_size]))
        .unwrap();
    let loss = h.mse_loss(&target).unwrap();

    graph.zero_grad().unwrap();
    let loss_val = loss.backward().unwrap();
    assert!(loss_val > 0.0);

    let w_grad = w_ih.grad().unwrap().unwrap();
    assert_eq!(w_grad.shape(), &[input_size, hidden_size]);
    assert!(
        w_grad.data_as_slice().iter().any(|&v| v.abs() > 1e-10),
        "w_ih 梯度应非零"
    );
}

/// RNN 两步展开（简化 BPTT）
///
/// h_1 = tanh(x_0 @ W_ih + h_0 @ W_hh)
/// h_2 = tanh(x_1 @ W_ih + h_1 @ W_hh)
/// loss = MSE(h_2, target)
///
/// W_ih 和 W_hh 在两个时间步中共享，验证两者梯度非零
#[test]
fn test_select_e2e_rnn_two_step_unroll() {
    let graph = Graph::new();

    let batch = 2;
    let input_size = 3;
    let hidden_size = 4;

    let x_seq = graph
        .input(&Tensor::normal_seeded(
            0.0,
            1.0,
            &[batch, 2, input_size],
            100,
        ))
        .unwrap();

    let w_ih = graph
        .parameter(&[input_size, hidden_size], Init::Zeros, "w_ih")
        .unwrap();
    w_ih.set_value(&Tensor::normal_seeded(
        0.0,
        0.5,
        &[input_size, hidden_size],
        42,
    ))
    .unwrap();

    let w_hh = graph
        .parameter(&[hidden_size, hidden_size], Init::Zeros, "w_hh")
        .unwrap();
    w_hh.set_value(&Tensor::normal_seeded(
        0.0,
        0.5,
        &[hidden_size, hidden_size],
        200,
    ))
    .unwrap();

    let h_0 = graph
        .input(&Tensor::zeros(&[batch, hidden_size]))
        .unwrap();

    // === 时间步 0 ===
    let x_0 = x_seq.select(1, 0).unwrap();
    let xw_0 = x_0.matmul(&w_ih).unwrap();
    let hw_0 = h_0.matmul(&w_hh).unwrap();
    let pre_h_1 = &xw_0 + &hw_0;
    let h_1 = pre_h_1.tanh();

    // === 时间步 1 ===
    let x_1 = x_seq.select(1, 1).unwrap();
    let xw_1 = x_1.matmul(&w_ih).unwrap();
    let hw_1 = h_1.matmul(&w_hh).unwrap();
    let pre_h_2 = &xw_1 + &hw_1;
    let h_2 = pre_h_2.tanh();

    let target = graph
        .input(&Tensor::zeros(&[batch, hidden_size]))
        .unwrap();
    let loss = h_2.mse_loss(&target).unwrap();

    graph.zero_grad().unwrap();
    let loss_val = loss.backward().unwrap();
    assert!(loss_val > 0.0);

    // 验证参数梯度
    let w_ih_grad = w_ih.grad().unwrap().unwrap();
    let w_hh_grad = w_hh.grad().unwrap().unwrap();

    assert_eq!(w_ih_grad.shape(), &[input_size, hidden_size]);
    assert_eq!(w_hh_grad.shape(), &[hidden_size, hidden_size]);

    // 两个时间步都用了这些参数，梯度应非零
    assert!(
        w_ih_grad.data_as_slice().iter().any(|&v| v.abs() > 1e-10),
        "w_ih 梯度应非零"
    );
    assert!(
        w_hh_grad.data_as_slice().iter().any(|&v| v.abs() > 1e-10),
        "w_hh 梯度应非零"
    );
}

// ==================== 4. 动态形状测试（KEEP AS-IS）====================

/// 测试 Select 节点的动态形状传播
/// 使用 3D 输入（符合 Select 典型业务场景：从序列中选择时间步）
/// 注：Input 节点默认支持动态 batch
#[test]
fn test_select_dynamic_shape_propagation() {
    let graph = Graph::new();

    // 创建 3D 输入：[batch, seq_len, features]
    // Input 节点默认支持动态 batch
    let x_seq = graph.input(&Tensor::zeros(&[4, 3, 16])).unwrap();

    // Select：沿 axis=1 选择 index=1，输出应为 [?, 16]
    let selected = x_seq.select(1, 1).unwrap();

    // 验证动态形状传播
    let dyn_shape = selected.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
    assert_eq!(dyn_shape.dims().len(), 2, "输出应该是 2 维");
}

/// 测试 Select 节点在不同 batch_size 下的前向计算
/// 使用 3D 输入（符合 Select 典型业务场景）
#[test]
fn test_select_dynamic_batch_forward() {
    let graph = Graph::new();

    // 创建 3D 输入：[batch, seq_len, features]
    // Input 节点默认支持动态 batch
    let x_seq = graph.input(&Tensor::zeros(&[2, 4, 16])).unwrap();

    // Select 沿 axis=1 选择 index=2
    let selected = x_seq.select(1, 2).unwrap();

    // 第一次 forward：batch=2
    selected.forward().unwrap();
    let value1 = selected.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 16], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    x_seq.set_value(&Tensor::zeros(&[5, 4, 16])).unwrap();

    // 第二次 forward：batch=5
    selected.forward().unwrap();
    let value2 = selected.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[5, 16], "第二次 forward: batch=5");
}

/// 测试 Select 节点在不同 batch_size 下的反向传播
/// 使用 3D 输入（模拟 RNN 输入序列），符合 Select 的典型业务场景
#[test]
fn test_select_dynamic_batch_backward() {
    let graph = Graph::new();

    // 创建 3D 输入序列：[batch, seq_len, features]
    // Input 节点默认支持动态 batch
    let x_seq = graph
        .input(&Tensor::normal_seeded(0.0, 1.0, &[2, 3, 16], 42))
        .unwrap();

    // Select 沿 axis=1 选择 index=0，模拟选择第一个时间步
    let selected = x_seq.select(1, 0).unwrap();

    // 创建 target
    let target = graph.input(&Tensor::zeros(&[2, 16])).unwrap();
    let loss = selected.mse_loss(&target).unwrap();

    // 第一次训练：batch=2
    loss.forward().unwrap();
    let loss_val1 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0);
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size（模拟不同 batch 的序列数据）
    x_seq
        .set_value(&Tensor::normal_seeded(0.0, 1.0, &[5, 3, 16], 100))
        .unwrap();
    target.set_value(&Tensor::zeros(&[5, 16])).unwrap();

    // 第二次训练：batch=5
    loss.forward().unwrap();
    let loss_val2 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);
    graph.zero_grad().unwrap();
    loss.backward().unwrap();
}

// ==================== 5. 节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_select_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // [2, 3, 4] select axis=1, index=0 -> [2, 4]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4], Some("input"))
        .unwrap();

    let selected = inner
        .borrow_mut()
        .create_select_node(input.clone(), 1, 0, Some("selected"))
        .unwrap();

    assert_eq!(selected.shape(), vec![2, 4]);
    assert_eq!(selected.name(), Some("selected"));
    assert!(!selected.is_leaf());
    assert_eq!(selected.parents().len(), 1);
}

#[test]
fn test_create_select_node_axis0() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // [3, 4, 5] select axis=0, index=2 -> [4, 5]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4, 5], None)
        .unwrap();

    let selected = inner
        .borrow_mut()
        .create_select_node(input, 0, 2, None)
        .unwrap();

    assert_eq!(selected.shape(), vec![4, 5]);
}

#[test]
fn test_create_select_node_last_axis() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // [2, 3, 4] select axis=2, index=1 -> [2, 3]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4], None)
        .unwrap();

    let selected = inner
        .borrow_mut()
        .create_select_node(input, 2, 1, None)
        .unwrap();

    assert_eq!(selected.shape(), vec![2, 3]);
}

#[test]
fn test_create_select_node_index_out_of_bounds() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 4], None)
        .unwrap();

    // index=5 超出 axis=1 的大小 3
    let result = inner
        .borrow_mut()
        .create_select_node(input, 1, 5, None);
    assert!(result.is_err());
}

#[test]
fn test_create_select_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_selected;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let selected = inner
            .borrow_mut()
            .create_select_node(input, 1, 0, None)
            .unwrap();
        weak_selected = Rc::downgrade(&selected);

        assert!(weak_selected.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_selected.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
