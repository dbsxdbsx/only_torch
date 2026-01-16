/*
 * @Description  : VarActivationOps trait 测试
 *
 * 测试激活函数扩展 trait 的独立功能：
 * - relu, sigmoid, tanh, leaky_relu, softmax, softplus, step, sign
 */

use crate::nn::graph::GraphHandle;
use crate::nn::{VarActivationOps, VarLossOps};
use crate::tensor::Tensor;

#[test]
fn test_var_relu() {
    let graph = GraphHandle::new();
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
    let graph = GraphHandle::new();
    let x = graph.input(&Tensor::new(&[0.0], &[1, 1])).unwrap();
    let y = x.sigmoid();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap().data_as_slice()[0];
    assert!((result - 0.5).abs() < 1e-5);
}

#[test]
fn test_var_tanh() {
    let graph = GraphHandle::new();
    let x = graph.input(&Tensor::new(&[0.0], &[1, 1])).unwrap();
    let y = x.tanh();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap().data_as_slice()[0];
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn test_var_leaky_relu() {
    let graph = GraphHandle::new();
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
    let graph = GraphHandle::new();
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
    let graph = GraphHandle::new();
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
    let graph = GraphHandle::new();
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
    let graph = GraphHandle::new();
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

    let graph = GraphHandle::new();

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
    let graph = GraphHandle::new();
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
    let graph = GraphHandle::new();
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
