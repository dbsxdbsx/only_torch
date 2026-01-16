/*
 * @Description  : Var-based Linear 层测试
 *
 * 测试新版 Linear 层（基于 Var API）
 */

use crate::nn::graph::Graph;
use crate::nn::layer::Linear;
use crate::nn::{Module, VarActivationOps, VarLossOps};
use crate::tensor::Tensor;

#[test]
fn test_linear_forward() {
    let graph = Graph::new();

    // 创建 Linear 层：3 -> 2
    let fc = Linear::new(&graph, 3, 2, true, "fc").unwrap();

    // 创建输入：[2, 3] (batch=2, in_features=3)
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    // 前向传播
    let y = fc.forward(&x);
    y.forward().unwrap();

    // 验证输出形状
    let result = y.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_linear_no_bias() {
    let graph = Graph::new();

    // 创建无 bias 的 Linear 层
    let fc = Linear::new(&graph, 3, 2, false, "fc_no_bias").unwrap();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();

    let y = fc.forward(&x);
    y.forward().unwrap();

    let result = y.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[1, 2]);
}

#[test]
fn test_linear_parameters() {
    let graph = Graph::new();

    // 有 bias 的 Linear
    let fc_with_bias = Linear::new(&graph, 4, 3, true, "fc1").unwrap();
    assert_eq!(fc_with_bias.parameters().len(), 2);
    assert_eq!(fc_with_bias.num_params(), 2);

    // 无 bias 的 Linear
    let fc_no_bias = Linear::new(&graph, 4, 3, false, "fc2").unwrap();
    assert_eq!(fc_no_bias.parameters().len(), 1);
    assert_eq!(fc_no_bias.num_params(), 1);
}

#[test]
fn test_linear_chained_with_activation() {
    let graph = Graph::new();

    let fc = Linear::new(&graph, 3, 2, true, "fc").unwrap();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();

    // 链式调用：fc.forward(x).relu()
    let y = fc.forward(&x).relu();
    y.forward().unwrap();

    let result = y.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[1, 2]);
    // ReLU 后所有值应该 >= 0
    assert!(result.data_as_slice().iter().all(|&v| v >= 0.0));
}

#[test]
fn test_linear_backward() {
    let graph = Graph::new();

    let fc = Linear::new(&graph, 3, 2, true, "fc").unwrap();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0, 0.0], &[1, 2])).unwrap();

    let y = fc.forward(&x);
    let loss = y.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 权重和偏置都应该有梯度
    let params = fc.parameters();
    for p in params {
        let grad = p.grad().unwrap();
        assert!(grad.is_some(), "参数应该有梯度");
    }
}

#[test]
fn test_mlp_two_layers() {
    let graph = Graph::new();

    // 简单 MLP：3 -> 4 -> 2
    let fc1 = Linear::new(&graph, 3, 4, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 4, 2, true, "fc2").unwrap();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0, 0.0], &[1, 2])).unwrap();

    // 前向：fc1 -> relu -> fc2
    let h = fc1.forward(&x).relu();
    let y = fc2.forward(&h);
    let loss = y.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 两层的参数都应该有梯度
    for p in fc1.parameters() {
        assert!(p.grad().unwrap().is_some());
    }
    for p in fc2.parameters() {
        assert!(p.grad().unwrap().is_some());
    }
}
