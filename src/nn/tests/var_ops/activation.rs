/*
 * @Description  : VarActivationOps trait 测试
 *
 * 测试激活函数扩展 trait 的独立功能：
 * - relu, sigmoid, tanh, leaky_relu, softmax, softplus, step, sign
 */

use crate::nn::graph::Graph;
use crate::nn::{VarActivationOps, VarLossOps};
use crate::tensor::Tensor;

#[test]
fn test_var_relu() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[-1.0, 0.0, 1.0, 2.0], &[4, 1]))
        .unwrap();
    let y = x.relu();
    y.forward().unwrap();
    assert_eq!(
        y.value().unwrap().unwrap().data_as_slice(),
        &[0.0, 0.0, 1.0, 2.0]
    );
}

#[test]
fn test_var_sigmoid() {
    let graph = Graph::new();
    let x = graph.input(&Tensor::new(&[0.0], &[1, 1])).unwrap();
    let y = x.sigmoid();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap().data_as_slice()[0];
    assert!((result - 0.5).abs() < 1e-5);
}

#[test]
fn test_var_tanh() {
    let graph = Graph::new();
    let x = graph.input(&Tensor::new(&[0.0], &[1, 1])).unwrap();
    let y = x.tanh();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap().data_as_slice()[0];
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn test_var_leaky_relu() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[-1.0, 0.0, 1.0], &[3, 1]))
        .unwrap();
    let y = x.leaky_relu(0.1);
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    assert!((result.data_as_slice()[0] - (-0.1)).abs() < 1e-5);
    assert!((result.data_as_slice()[1] - 0.0).abs() < 1e-5);
    assert!((result.data_as_slice()[2] - 1.0).abs() < 1e-5);
}

#[test]
fn test_var_step() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[-1.0, 0.0, 1.0], &[3, 1]))
        .unwrap();
    let y = x.step();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    // step(x) = 1 if x >= 0 else 0
    assert_eq!(result.data_as_slice(), &[0.0, 1.0, 1.0]);
}

#[test]
fn test_var_softplus() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[-1.0, 0.0, 1.0], &[3, 1]))
        .unwrap();
    let y = x.softplus();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    // softplus(x) = log(1 + exp(x))
    // softplus(-1) ≈ 0.3133, softplus(0) ≈ 0.6931, softplus(1) ≈ 1.3133
    assert!((result.data_as_slice()[0] - 0.3133).abs() < 0.01);
    assert!((result.data_as_slice()[1] - 0.6931).abs() < 0.01);
    assert!((result.data_as_slice()[2] - 1.3133).abs() < 0.01);
}

#[test]
fn test_var_sign() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[-2.0, 0.0, 3.0], &[3, 1]))
        .unwrap();
    let y = x.sign();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    // sign(x) = -1 if x < 0, 0 if x == 0, 1 if x > 0
    assert_eq!(result.data_as_slice(), &[-1.0, 0.0, 1.0]);
}

#[test]
fn test_var_softmax() {
    let graph = Graph::new();
    // 输入 [batch=2, classes=3]
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))
        .unwrap();
    let y = x.softmax();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();

    // 第一行：softmax([1,2,3]) ≈ [0.09, 0.24, 0.67]
    let row0_sum: f32 = result.data_as_slice()[0..3].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "第一行应该归一化为1");

    // 第二行：softmax([1,1,1]) = [1/3, 1/3, 1/3]
    let row1 = &result.data_as_slice()[3..6];
    assert!((row1[0] - 1.0 / 3.0).abs() < 1e-5);
    assert!((row1[1] - 1.0 / 3.0).abs() < 1e-5);
    assert!((row1[2] - 1.0 / 3.0).abs() < 1e-5);
}

#[test]
fn test_var_softmax_backward() {
    use crate::nn::var::Init;

    let graph = Graph::new();

    // 简单网络：parameter -> softmax -> mse_loss
    let x = graph.parameter(&[1, 3], Init::Ones, "x").unwrap();
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0], &[1, 3]))
        .unwrap();

    let probs = x.softmax();
    let loss = probs.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 验证 x 有梯度
    let x_grad = x.grad().unwrap();
    assert!(x_grad.is_some());
    // 梯度应该存在且非零
    let grad_tensor = x_grad.unwrap();
    let grad_data = grad_tensor.data_as_slice();
    assert!(grad_data.iter().any(|&g| g.abs() > 1e-6));
}

#[test]
fn test_var_chained_activations() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[-1.0, 0.5, 1.0], &[3, 1]))
        .unwrap();
    let y = x.relu().sigmoid();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    // relu: [0, 0.5, 1.0] -> sigmoid: [0.5, 0.622..., 0.731...]
    assert!((result.data_as_slice()[0] - 0.5).abs() < 1e-5);
}

#[test]
fn test_var_chained_softplus_sigmoid() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[-1.0, 0.0, 1.0], &[3, 1]))
        .unwrap();
    // softplus -> sigmoid 是平滑的复合非线性
    let y = x.softplus().sigmoid();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    // softplus(-1) ≈ 0.3133 -> sigmoid(0.3133) ≈ 0.5777
    assert!(result.data_as_slice()[0] > 0.5 && result.data_as_slice()[0] < 0.6);
}

// ==================== Softmax 动态形状测试 ====================

/// 测试 Softmax 节点的动态形状传播
#[test]
fn test_softmax_dynamic_shape_propagation() {
    let graph = Graph::new();

    // 创建 2D 输入（Softmax 要求 2D）：[batch, num_classes]
    // Input 节点默认支持动态 batch
    let x = graph.input(&Tensor::zeros(&[4, 10])).unwrap();

    // Softmax
    let probs = x.softmax();

    // 验证动态形状传播
    let dyn_shape = probs.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "num_classes 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(10), "num_classes 应该是 10");
}

/// 测试 Softmax 节点在不同 batch_size 下的前向计算
#[test]
fn test_softmax_dynamic_batch_forward() {
    let graph = Graph::new();

    // 创建 2D 输入
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))
        .unwrap();

    // Softmax
    let probs = x.softmax();

    // 第一次 forward：batch=2
    probs.forward().unwrap();
    let value1 = probs.value().unwrap().unwrap();
    assert_eq!(value1.shape(), &[2, 3], "第一次 forward: batch=2");
    // 验证每行归一化
    let row0_sum: f32 = value1.data_as_slice()[0..3].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5);

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        &[4, 3],
    ))
    .unwrap();

    // 第二次 forward：batch=4
    probs.forward().unwrap();
    let value2 = probs.value().unwrap().unwrap();
    assert_eq!(value2.shape(), &[4, 3], "第二次 forward: batch=4");
}

/// 测试 Softmax 节点在不同 batch_size 下的反向传播
#[test]
fn test_softmax_dynamic_batch_backward() {
    use crate::nn::var_ops::VarLossOps;

    let graph = Graph::new();

    // 创建参数和目标
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[2, 3]))
        .unwrap();

    // Softmax + MSE
    let probs = x.softmax();
    let loss = probs.mse_loss(&target).unwrap();

    // 第一次训练：batch=2
    loss.forward().unwrap();
    let loss_val1 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0);
    graph.zero_grad();
    loss.backward().unwrap();

    // 更新输入为不同的 batch_size
    x.set_value(&Tensor::normal_seeded(0.0, 1.0, &[5, 3], 42))
        .unwrap();
    target.set_value(&Tensor::zeros(&[5, 3])).unwrap();

    // 第二次训练：batch=5
    loss.forward().unwrap();
    let loss_val2 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);
    graph.zero_grad();
    loss.backward().unwrap();
}
