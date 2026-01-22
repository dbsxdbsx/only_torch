/*
 * ZerosLike 节点单元测试
 *
 * ZerosLike 节点根据参考节点的 batch_size 动态生成零张量。
 * 用于 RNN/LSTM/GRU 的初始隐藏状态。
 */

use crate::nn::Graph;
use crate::nn::var_ops::VarMatrixOps;
use crate::tensor::Tensor;

/// 测试 ZerosLike 基本功能
#[test]
fn test_zeros_like_basic() {
    let graph = Graph::new();

    // 创建输入节点 [batch=4, features=3]
    let x = graph.input(&Tensor::zeros(&[4, 3])).unwrap();

    // 创建 ZerosLike 节点，输出形状 [?, hidden_size=5]
    let h0 = graph.zeros_like(&x, &[5], Some("h0")).unwrap();

    // 触发前向计算
    h0.forward().unwrap();

    // 验证 h0 的形状是 [4, 5]
    let h0_value = h0.value().unwrap().unwrap();
    assert_eq!(h0_value.shape(), &[4, 5]);

    // 验证值是全零
    for i in 0..4 {
        for j in 0..5 {
            assert_eq!(h0_value.get(&[i, j]).get_data_number().unwrap(), 0.0);
        }
    }
}

/// 测试 ZerosLike 动态 batch：更新参考节点后重新计算
#[test]
fn test_zeros_like_dynamic_batch() {
    let graph = Graph::new();

    // 创建输入节点 [batch=2, features=3]
    let x = graph.input(&Tensor::zeros(&[2, 3])).unwrap();

    // 创建 ZerosLike 节点
    let h0 = graph.zeros_like(&x, &[5], Some("h0")).unwrap();

    // 第一次前向计算
    h0.forward().unwrap();
    let h0_value = h0.value().unwrap().unwrap();
    assert_eq!(h0_value.shape(), &[2, 5], "首次 forward: batch=2");

    // 更新输入节点的值（不同 batch_size）
    x.set_value(&Tensor::zeros(&[8, 3])).unwrap();

    // 重新前向计算
    h0.forward().unwrap();
    let h0_value = h0.value().unwrap().unwrap();
    assert_eq!(h0_value.shape(), &[8, 5], "再次 forward: batch=8");
}

/// 测试 ZerosLike 与 MatMul 结合使用（模拟 RNN 场景）
#[test]
fn test_zeros_like_with_matmul() {
    let graph = Graph::new();

    // 创建输入 [batch=3, input_size=4]
    let x = graph.input(&Tensor::normal(0.0, 1.0, &[3, 4])).unwrap();

    // 创建权重 [input_size=4, hidden_size=5]
    let w = graph
        .parameter(&[4, 5], crate::nn::Init::Kaiming, "w")
        .unwrap();

    // 创建初始隐藏状态 h0 = ZerosLike(x) -> [batch=3, hidden_size=5]
    let h0 = graph.zeros_like(&x, &[5], Some("h0")).unwrap();

    // 计算 xw = x @ w -> [3, 5]
    let xw = x.matmul(&w).unwrap();

    // 计算 output = xw + h0 -> [3, 5]
    let output = &xw + &h0;

    // 前向计算
    output.forward().unwrap();

    // 验证输出形状
    let output_value = output.value().unwrap().unwrap();
    assert_eq!(output_value.shape(), &[3, 5]);

    // h0 是零，所以 output 应该等于 xw
    let xw_value = xw.value().unwrap().unwrap();
    for i in 0..3 {
        for j in 0..5 {
            let out_val = output_value.get(&[i, j]).get_data_number().unwrap();
            let xw_val = xw_value.get(&[i, j]).get_data_number().unwrap();
            let diff = (out_val - xw_val).abs();
            assert!(
                diff < 1e-6,
                "output[{},{}] = {} != xw[{},{}] = {}",
                i,
                j,
                out_val,
                i,
                j,
                xw_val
            );
        }
    }
}

/// 测试 ZerosLike 动态 batch 与 MatMul 结合
#[test]
fn test_zeros_like_dynamic_batch_with_matmul() {
    let graph = Graph::new();

    // 创建输入 [batch=2, input_size=4]
    let x = graph.input(&Tensor::normal(0.0, 1.0, &[2, 4])).unwrap();

    // 创建权重 [hidden_size=5, hidden_size=5]（h @ W_hh）
    let w_hh = graph
        .parameter(&[5, 5], crate::nn::Init::Kaiming, "w_hh")
        .unwrap();

    // 创建初始隐藏状态 h0 = ZerosLike(x) -> [batch=2, hidden_size=5]
    let h0 = graph.zeros_like(&x, &[5], Some("h0")).unwrap();

    // 计算 hw = h0 @ W_hh -> [2, 5]
    let hw = h0.matmul(&w_hh).unwrap();

    // 第一次前向计算
    hw.forward().unwrap();
    let hw_value = hw.value().unwrap().unwrap();
    assert_eq!(hw_value.shape(), &[2, 5], "首次 forward: batch=2");

    // 验证结果是全零（因为 h0 是零）
    for i in 0..2 {
        for j in 0..5 {
            assert_eq!(hw_value.get(&[i, j]).get_data_number().unwrap(), 0.0);
        }
    }

    // 更新输入节点的值（不同 batch_size）
    x.set_value(&Tensor::normal(0.0, 1.0, &[6, 4])).unwrap();

    // 重新前向计算
    hw.forward().unwrap();
    let hw_value = hw.value().unwrap().unwrap();
    assert_eq!(hw_value.shape(), &[6, 5], "再次 forward: batch=6");
}

/// 测试 ZerosLike 的 3D 输入（RNN 场景）
#[test]
fn test_zeros_like_3d_input() {
    let graph = Graph::new();

    // 创建 3D 输入 [batch=4, seq_len=8, features=3]
    let x = graph.input(&Tensor::zeros(&[4, 8, 3])).unwrap();

    // 创建 ZerosLike 节点，输出形状 [?, hidden_size=16]
    let h0 = graph.zeros_like(&x, &[16], Some("h0")).unwrap();

    // 触发前向计算
    h0.forward().unwrap();

    // 验证 h0 的形状是 [4, 16]（batch_size 从 x 的第一维获取）
    let h0_value = h0.value().unwrap().unwrap();
    assert_eq!(h0_value.shape(), &[4, 16]);
}

/// 测试 ZerosLike 与 MatMul 结合使用并进行反向传播
#[test]
fn test_zeros_like_with_matmul_backward() {
    use crate::nn::var_ops::VarLossOps;

    let graph = Graph::new();

    // 创建输入 [batch=3, input_size=4]
    let x = graph.input(&Tensor::normal(0.0, 1.0, &[3, 4])).unwrap();

    // 创建权重 [hidden_size=5, hidden_size=5]
    let w_hh = graph
        .parameter(&[5, 5], crate::nn::Init::Kaiming, "w_hh")
        .unwrap();

    // 创建初始隐藏状态 h0 = ZerosLike(x) -> [batch=3, hidden_size=5]
    let h0 = graph.zeros_like(&x, &[5], Some("h0")).unwrap();

    // 计算 hw = h0 @ W_hh -> [3, 5]
    let hw = h0.matmul(&w_hh).unwrap();

    // 创建目标张量并计算 MSE Loss（标量）
    let target = graph.input(&Tensor::zeros(&[3, 5])).unwrap();
    let loss = hw.mse_loss(&target).unwrap();

    // 前向计算
    loss.forward().unwrap();

    // 反向传播
    let loss_val = loss.backward().unwrap();
    assert!(loss_val >= 0.0, "backward 应该成功，loss >= 0");

    // 验证 w_hh 有梯度（虽然是零，因为 h0 是零）
    let w_grad = w_hh.grad().unwrap();
    assert!(w_grad.is_some(), "w_hh 应该有梯度");
}

/// 测试 ZerosLike 动态 batch 与反向传播
#[test]
fn test_zeros_like_dynamic_batch_with_backward() {
    use crate::nn::var_ops::VarLossOps;

    let graph = Graph::new();

    // 创建输入 [batch=2, input_size=4]
    let x = graph.input(&Tensor::normal(0.0, 1.0, &[2, 4])).unwrap();

    // 创建权重
    let w_hh = graph
        .parameter(&[5, 5], crate::nn::Init::Kaiming, "w_hh")
        .unwrap();

    // 创建初始隐藏状态
    let h0 = graph.zeros_like(&x, &[5], Some("h0")).unwrap();

    // 计算
    let hw = h0.matmul(&w_hh).unwrap();

    // 创建目标张量并计算 MSE Loss
    let target = graph.input(&Tensor::zeros(&[2, 5])).unwrap();
    let loss = hw.mse_loss(&target).unwrap();

    // 第一次前向和反向
    loss.forward().unwrap();
    assert_eq!(hw.value().unwrap().unwrap().shape(), &[2, 5]);
    loss.backward().unwrap();

    // 更新输入（不同 batch_size）
    x.set_value(&Tensor::normal(0.0, 1.0, &[6, 4])).unwrap();
    target.set_value(&Tensor::zeros(&[6, 5])).unwrap();

    // 第二次前向和反向
    loss.forward().unwrap();
    assert_eq!(
        hw.value().unwrap().unwrap().shape(),
        &[6, 5],
        "动态 batch: 形状应该是 [6, 5]"
    );
    loss.backward().unwrap();

    // 验证 w_hh 有梯度
    let w_grad = w_hh.grad().unwrap();
    assert!(w_grad.is_some(), "w_hh 应该有梯度");
}
