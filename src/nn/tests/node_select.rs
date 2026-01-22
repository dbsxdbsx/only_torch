/*
 * @Author       : 老董
 * @Date         : 2026-01-21
 * @Description  : Select 节点单元测试
 *
 * Select 节点用于从张量中选择指定轴和索引的切片，
 * 主要用于 RNN 展开式设计：从 [batch, seq_len, input_size] 提取单个时间步。
 */

use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试基本的 select 功能：从 [2, 3, 4] 中选择 axis=1, index=1
#[test]
fn test_select_basic_3d() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建输入节点 [2, 3, 4] (batch=2, seq_len=3, input_size=4)
    let input = graph.new_input_node(&[2, 3, 4], Some("input"))?;

    // 选择 axis=1, index=1 (第 2 个时间步)
    let selected = graph.new_select_node(input, 1, 1, Some("selected"))?;

    // 设置输入值
    let mut input_data = Tensor::zeros(&[2, 3, 4]);
    // 填充：第 i 个 batch 的第 j 个时间步的第 k 个特征 = i*100 + j*10 + k
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                input_data[[i, j, k]] = (i * 100 + j * 10 + k) as f32;
            }
        }
    }
    graph.set_node_value(input, Some(&input_data))?;

    // 前向传播
    graph.forward(selected)?;

    // 验证输出形状 [2, 4]
    let output = graph.get_node_value(selected)?.unwrap();
    assert_eq!(output.shape(), &[2, 4]);

    // 验证输出值（axis=1, index=1 意味着 j=1）
    // output[i, k] = input[i, 1, k] = i*100 + 10 + k
    for i in 0..2 {
        for k in 0..4 {
            let expected = (i * 100 + 10 + k) as f32;
            assert_abs_diff_eq!(output[[i, k]], expected, epsilon = 1e-6);
        }
    }

    Ok(())
}

/// 测试 select 选择第一个时间步 (index=0)
#[test]
fn test_select_first_timestep() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[4, 5, 3], Some("input"))?;
    let selected = graph.new_select_node(input, 1, 0, Some("t0"))?;

    let input_data = Tensor::normal_seeded(0.0, 1.0, &[4, 5, 3], 42);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward(selected)?;

    let output = graph.get_node_value(selected)?.unwrap();
    assert_eq!(output.shape(), &[4, 3]);

    // 验证 output[i, k] == input[i, 0, k]
    for i in 0..4 {
        for k in 0..3 {
            assert_abs_diff_eq!(output[[i, k]], input_data[[i, 0, k]], epsilon = 1e-6);
        }
    }

    Ok(())
}

/// 测试 select 选择最后一个时间步
#[test]
fn test_select_last_timestep() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let seq_len = 5;
    let input = graph.new_input_node(&[2, seq_len, 3], Some("input"))?;
    let selected = graph.new_select_node(input, 1, seq_len - 1, Some("t_last"))?;

    let input_data = Tensor::normal_seeded(0.0, 1.0, &[2, seq_len, 3], 42);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward(selected)?;

    let output = graph.get_node_value(selected)?.unwrap();
    assert_eq!(output.shape(), &[2, 3]);

    // 验证 output[i, k] == input[i, seq_len-1, k]
    for i in 0..2 {
        for k in 0..3 {
            assert_abs_diff_eq!(
                output[[i, k]],
                input_data[[i, seq_len - 1, k]],
                epsilon = 1e-6
            );
        }
    }

    Ok(())
}

/// 测试 select 在 axis=0 上（选择 batch）
#[test]
fn test_select_axis_0() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[3, 4], Some("input"))?;
    let selected = graph.new_select_node(input, 0, 1, Some("row1"))?;

    let input_data = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
            9.0, 10.0, 11.0, 12.0, // row 2
        ],
        &[3, 4],
    );
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward(selected)?;

    let output = graph.get_node_value(selected)?.unwrap();
    assert_eq!(output.shape(), &[4]); // 维度减少了

    let expected = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[4]);
    assert_eq!(output, &expected);

    Ok(())
}

// ==================== VJP 单元测试（直接调用 calc_grad_to_parent）====================

/// 测试 Select VJP（scatter 操作）
///
/// 对于 y = select(x, axis, index)，反向传播时：
/// grad_to_input 是一个全零张量，只在 [:, index, :] 处填入 upstream_grad
#[test]
fn test_select_backward_vjp() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3, 4], Some("input"))?;
    let select_id = graph.new_select_node(input_id, 1, 1, Some("selected"))?;

    let input_data = Tensor::normal_seeded(0.0, 1.0, &[2, 3, 4], 42);
    graph.set_node_value(input_id, Some(&input_data))?;
    graph.forward(select_id)?;

    // 直接测试 VJP
    let upstream_grad = Tensor::ones(&[2, 4]);
    let select_node = graph.get_node(select_id)?;
    let input_node = graph.get_node(input_id)?;
    let grad = select_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 验证梯度形状
    assert_eq!(grad.shape(), &[2, 3, 4]);

    // 验证梯度值：只有 [:, 1, :] 处为 1.0，其他为 0.0
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

/// 测试 Select VJP（非单位上游梯度）
#[test]
fn test_select_backward_vjp_non_unit_upstream() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3, 4], Some("input"))?;
    let select_id = graph.new_select_node(input_id, 1, 2, Some("selected"))?;

    let input_data = Tensor::zeros(&[2, 3, 4]);
    graph.set_node_value(input_id, Some(&input_data))?;
    graph.forward(select_id)?;

    // 非单位上游梯度
    let mut upstream_grad = Tensor::zeros(&[2, 4]);
    for i in 0..2 {
        for k in 0..4 {
            upstream_grad[[i, k]] = (i * 10 + k) as f32;
        }
    }

    let select_node = graph.get_node(select_id)?;
    let input_node = graph.get_node(input_id)?;
    let grad = select_node.calc_grad_to_parent(input_node, &upstream_grad, None)?;

    // 验证梯度：只有 [:, 2, :] 处有值
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

// ==================== 端到端反向传播测试（通过 graph.backward）====================

/// 测试 Select 通过 graph.backward() 的端到端反向传播
#[test]
fn test_select_backward_e2e() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 计算图：output = sigmoid(matmul(select(input, 1, 0), W))
    // input: [2, 3, 4] -> select -> [2, 4] -> matmul(W:[4,2]) -> [2, 2] -> sigmoid
    let input = graph.new_parameter_node(&[2, 3, 4], Some("input"))?;
    let selected = graph.new_select_node(input, 1, 0, Some("selected"))?;
    let w = graph.new_parameter_node(&[4, 2], Some("w"))?;
    let mm = graph.new_mat_mul_node(selected, w, Some("mm"))?;
    let sigmoid = graph.new_sigmoid_node(mm, Some("sigmoid"))?;

    // loss = MSE(sigmoid, target)
    let target = graph.new_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(sigmoid, target, Some("loss"))?;

    // 设置值
    let input_data = Tensor::normal_seeded(0.0, 1.0, &[2, 3, 4], 42);
    let w_data = Tensor::normal_seeded(0.0, 0.5, &[4, 2], 100);
    graph.set_node_value(input, Some(&input_data))?;
    graph.set_node_value(w, Some(&w_data))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    let loss_returned = graph.backward(loss)?;
    assert!(loss_returned > 0.0);

    // 验证梯度存在且形状正确
    let input_grad = graph.get_node(input)?.grad().expect("input 应有 grad");
    assert_eq!(input_grad.shape(), &[2, 3, 4]);

    // 验证只有 [:, 0, :] 处有非零梯度（因为 select 只选择了 index=0）
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

    Ok(())
}

/// 测试多个 Select（模拟 RNN 展开）
#[test]
fn test_select_multiple_timesteps() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 模拟 RNN 展开：从 [batch, seq_len, input] 选择多个时间步
    let batch = 2;
    let seq_len = 3;
    let input_size = 4;

    let x_seq = graph.new_parameter_node(&[batch, seq_len, input_size], Some("x_seq"))?;

    // 选择 3 个时间步
    let x_0 = graph.new_select_node(x_seq, 1, 0, Some("x_0"))?;
    let x_1 = graph.new_select_node(x_seq, 1, 1, Some("x_1"))?;
    let x_2 = graph.new_select_node(x_seq, 1, 2, Some("x_2"))?;

    // 简单求和作为输出
    let sum_01 = graph.new_add_node(&[x_0, x_1], Some("sum_01"))?;
    let sum_all = graph.new_add_node(&[sum_01, x_2], Some("sum_all"))?;

    // 使用 MSE loss
    let target = graph.new_input_node(&[batch, input_size], Some("target"))?;
    let loss = graph.new_mse_loss_node(sum_all, target, Some("loss"))?;

    // 设置值
    let x_data = Tensor::normal_seeded(0.0, 1.0, &[batch, seq_len, input_size], 42);
    graph.set_node_value(x_seq, Some(&x_data))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[batch, input_size])))?;

    // 前向传播
    graph.forward(loss)?;

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证梯度存在
    let x_grad = graph.get_node(x_seq)?.grad().expect("x_seq 应有 grad");
    assert_eq!(x_grad.shape(), &[batch, seq_len, input_size]);

    // 所有时间步都应有梯度（因为都被 select 了）
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

    Ok(())
}

// ==================== RNN 展开式场景测试 ====================

/// 测试类似 RNN 单步计算的场景
#[test]
fn test_select_rnn_like_single_step() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    let batch = 2;
    let seq_len = 4;
    let input_size = 3;
    let hidden_size = 5;

    // 输入序列
    let x_seq = graph.new_input_node(&[batch, seq_len, input_size], Some("x_seq"))?;

    // RNN 参数
    let w_ih = graph.new_parameter_node(&[input_size, hidden_size], Some("w_ih"))?;

    // 选择第一个时间步
    let x_0 = graph.new_select_node(x_seq, 1, 0, Some("x_0"))?;

    // 计算 h = tanh(x_0 @ W_ih)
    let mm = graph.new_mat_mul_node(x_0, w_ih, Some("mm"))?;
    let h = graph.new_tanh_node(mm, Some("h"))?;

    // 简单的 loss
    let target = graph.new_input_node(&[batch, hidden_size], Some("target"))?;
    let loss = graph.new_mse_loss_node(h, target, Some("loss"))?;

    // 设置值
    let x_data = Tensor::normal_seeded(0.0, 1.0, &[batch, seq_len, input_size], 100);
    graph.set_node_value(x_seq, Some(&x_data))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[batch, hidden_size])))?;

    // 训练一步
    graph.forward(loss)?;
    graph.zero_grad()?;
    let loss_val = graph.backward(loss)?;

    assert!(loss_val > 0.0);

    // 验证参数梯度存在
    let w_grad = graph.get_node(w_ih)?.grad().expect("w_ih 应有 grad");
    assert_eq!(w_grad.shape(), &[input_size, hidden_size]);

    Ok(())
}

/// 测试 RNN 两步展开（模拟简化的 BPTT）
#[test]
fn test_select_rnn_two_step_unroll() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    let batch = 2;
    let input_size = 3;
    let hidden_size = 4;

    // 输入序列 [batch, 2, input_size]
    let x_seq = graph.new_input_node(&[batch, 2, input_size], Some("x_seq"))?;

    // RNN 参数（共享）
    let w_ih = graph.new_parameter_node(&[input_size, hidden_size], Some("w_ih"))?;
    let w_hh = graph.new_parameter_node(&[hidden_size, hidden_size], Some("w_hh"))?;

    // 初始隐藏状态（零）
    let h_0 = graph.new_input_node(&[batch, hidden_size], Some("h_0"))?;

    // === 时间步 0 ===
    let x_0 = graph.new_select_node(x_seq, 1, 0, Some("x_0"))?;
    let xw_0 = graph.new_mat_mul_node(x_0, w_ih, Some("xw_0"))?;
    let hw_0 = graph.new_mat_mul_node(h_0, w_hh, Some("hw_0"))?;
    let pre_h_1 = graph.new_add_node(&[xw_0, hw_0], Some("pre_h_1"))?;
    let h_1 = graph.new_tanh_node(pre_h_1, Some("h_1"))?;

    // === 时间步 1 ===
    let x_1 = graph.new_select_node(x_seq, 1, 1, Some("x_1"))?;
    let xw_1 = graph.new_mat_mul_node(x_1, w_ih, Some("xw_1"))?;
    let hw_1 = graph.new_mat_mul_node(h_1, w_hh, Some("hw_1"))?;
    let pre_h_2 = graph.new_add_node(&[xw_1, hw_1], Some("pre_h_2"))?;
    let h_2 = graph.new_tanh_node(pre_h_2, Some("h_2"))?;

    // Loss
    let target = graph.new_input_node(&[batch, hidden_size], Some("target"))?;
    let loss = graph.new_mse_loss_node(h_2, target, Some("loss"))?;

    // 设置值
    let x_data = Tensor::normal_seeded(0.0, 1.0, &[batch, 2, input_size], 100);
    graph.set_node_value(x_seq, Some(&x_data))?;
    graph.set_node_value(h_0, Some(&Tensor::zeros(&[batch, hidden_size])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[batch, hidden_size])))?;

    // 训练
    graph.forward(loss)?;
    graph.zero_grad()?;
    let loss_val = graph.backward(loss)?;

    assert!(loss_val > 0.0);

    // 验证参数梯度
    let w_ih_grad = graph.get_node(w_ih)?.grad().expect("w_ih 应有 grad");
    let w_hh_grad = graph.get_node(w_hh)?.grad().expect("w_hh 应有 grad");

    assert_eq!(w_ih_grad.shape(), &[input_size, hidden_size]);
    assert_eq!(w_hh_grad.shape(), &[hidden_size, hidden_size]);

    // 梯度应非零（因为两个时间步都使用了这些参数）
    assert!(w_ih_grad.data_as_slice().iter().any(|&v| v.abs() > 1e-10));
    assert!(w_hh_grad.data_as_slice().iter().any(|&v| v.abs() > 1e-10));

    Ok(())
}

// ==================== 错误处理测试 ====================

/// 测试 axis 越界错误
#[test]
fn test_select_axis_out_of_bounds() {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[2, 3, 4], Some("input")).unwrap();

    // axis=3 超出 3 维张量的范围
    let result = graph.new_select_node(input, 3, 0, Some("bad_select"));

    assert!(result.is_err());
    if let Err(GraphError::InvalidOperation(msg)) = result {
        assert!(msg.contains("axis") || msg.contains("超出"));
    } else {
        panic!("应返回 InvalidOperation 错误");
    }
}

/// 测试 index 越界错误
#[test]
fn test_select_index_out_of_bounds() {
    let mut graph = GraphInner::new();

    let input = graph.new_input_node(&[2, 3, 4], Some("input")).unwrap();

    // index=5 超出 axis=1 的大小 3
    let result = graph.new_select_node(input, 1, 5, Some("bad_select"));

    assert!(result.is_err());
    if let Err(GraphError::InvalidOperation(msg)) = result {
        assert!(msg.contains("index") || msg.contains("超出"));
    } else {
        panic!("应返回 InvalidOperation 错误");
    }
}

// ==================== 动态形状测试 ====================

/// 测试 Select 节点的动态形状传播
/// 使用 3D 输入（符合 Select 典型业务场景：从序列中选择时间步）
/// 注：Input 节点默认支持动态 batch
#[test]
fn test_select_dynamic_shape_propagation() {
    use crate::nn::var_ops::VarShapeOps;
    use crate::nn::Graph;

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
    use crate::nn::var_ops::VarShapeOps;
    use crate::nn::Graph;

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
    use crate::nn::var_ops::{VarLossOps, VarShapeOps};
    use crate::nn::Graph;

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
    graph.zero_grad();
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
    graph.zero_grad();
    loss.backward().unwrap();
}
