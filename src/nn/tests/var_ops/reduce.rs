/*
 * @Author       : 老董
 * @Description  : VarReduceOps 单元测试
 */

use crate::nn::var::Init;
use crate::nn::var_ops::{VarLossOps, VarReduceOps};
use crate::nn::Graph;
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== Sum 测试 ====================

#[test]
fn test_var_sum_global() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let y = x.sum();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();

    assert_eq!(result.shape(), &[1, 1]);
    assert_abs_diff_eq!(result[[0, 0]], 21.0, epsilon = 1e-5);
}

#[test]
fn test_var_sum_axis0() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let y = x.sum_axis(0);
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();

    // [[1,2,3], [4,5,6]] sum axis=0 -> [[5,7,9]]
    assert_eq!(result.shape(), &[1, 3]);
    assert_abs_diff_eq!(result[[0, 0]], 5.0, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[0, 1]], 7.0, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[0, 2]], 9.0, epsilon = 1e-5);
}

#[test]
fn test_var_sum_axis1() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let y = x.sum_axis(1);
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();

    // [[1,2,3], [4,5,6]] sum axis=1 -> [[6], [15]]
    assert_eq!(result.shape(), &[2, 1]);
    assert_abs_diff_eq!(result[[0, 0]], 6.0, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[1, 0]], 15.0, epsilon = 1e-5);
}

#[test]
fn test_var_sum_backward_global() {
    let graph = Graph::new();

    // parameter -> sum -> mse_loss
    let x = graph.parameter(&[2, 3], Init::Ones, "x").unwrap();
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[20.0], &[1, 1])).unwrap();

    let sum_x = x.sum();
    let loss = sum_x.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 验证 x 有梯度
    let x_grad = x.grad().unwrap();
    assert!(x_grad.is_some());
    let grad_tensor = x_grad.unwrap();
    assert_eq!(grad_tensor.shape(), &[2, 3]);

    // 全局求和梯度：所有元素梯度相等
    let first_grad = grad_tensor[[0, 0]];
    for val in grad_tensor.data_as_slice() {
        assert_abs_diff_eq!(*val, first_grad, epsilon = 1e-6);
    }
}

#[test]
fn test_var_sum_backward_axis() {
    let graph = Graph::new();

    // parameter -> sum_axis(1) -> mse_loss
    let x = graph.parameter(&[2, 3], Init::Ones, "x").unwrap();
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[5.0, 14.0], &[2, 1])).unwrap();

    let sum_x = x.sum_axis(1);
    let loss = sum_x.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 验证 x 有梯度
    let x_grad = x.grad().unwrap();
    assert!(x_grad.is_some());
    let grad_tensor = x_grad.unwrap();
    assert_eq!(grad_tensor.shape(), &[2, 3]);

    // 按轴求和梯度：同一行的梯度相等
    assert_abs_diff_eq!(grad_tensor[[0, 0]], grad_tensor[[0, 1]], epsilon = 1e-6);
    assert_abs_diff_eq!(grad_tensor[[0, 1]], grad_tensor[[0, 2]], epsilon = 1e-6);
    assert_abs_diff_eq!(grad_tensor[[1, 0]], grad_tensor[[1, 1]], epsilon = 1e-6);
    assert_abs_diff_eq!(grad_tensor[[1, 1]], grad_tensor[[1, 2]], epsilon = 1e-6);
}

#[test]
fn test_var_sum_chained() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[2, 2, 3],
        ))
        .unwrap();

    // 先沿 axis=2 求和，再沿 axis=1 求和
    let sum1 = x.sum_axis(2); // [2, 2, 3] -> [2, 2, 1]
    let sum2 = sum1.sum_axis(1); // [2, 2, 1] -> [2, 1, 1]

    sum2.forward().unwrap();
    let result = sum2.value().unwrap().unwrap();

    assert_eq!(result.shape(), &[2, 1, 1]);
    // 第一个 batch: 1+2+3+4+5+6 = 21
    // 第二个 batch: 7+8+9+10+11+12 = 57
    assert_abs_diff_eq!(result[[0, 0, 0]], 21.0, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[1, 0, 0]], 57.0, epsilon = 1e-5);
}

// ==================== Mean 测试 ====================

#[test]
fn test_var_mean_global() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let y = x.mean();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();

    assert_eq!(result.shape(), &[1, 1]);
    // mean = (1+2+3+4+5+6) / 6 = 3.5
    assert_abs_diff_eq!(result[[0, 0]], 3.5, epsilon = 1e-5);
}

#[test]
fn test_var_mean_axis0() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let y = x.mean_axis(0);
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();

    // [[1,2,3], [4,5,6]] mean axis=0 -> [[2.5, 3.5, 4.5]]
    assert_eq!(result.shape(), &[1, 3]);
    assert_abs_diff_eq!(result[[0, 0]], 2.5, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[0, 1]], 3.5, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[0, 2]], 4.5, epsilon = 1e-5);
}

#[test]
fn test_var_mean_axis1() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let y = x.mean_axis(1);
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();

    // [[1,2,3], [4,5,6]] mean axis=1 -> [[2], [5]]
    assert_eq!(result.shape(), &[2, 1]);
    assert_abs_diff_eq!(result[[0, 0]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[1, 0]], 5.0, epsilon = 1e-5);
}

#[test]
fn test_var_mean_backward_global() {
    let graph = Graph::new();

    // parameter -> mean -> mse_loss
    let x = graph.parameter(&[2, 3], Init::Ones, "x").unwrap();
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[3.0], &[1, 1])).unwrap();

    let mean_x = x.mean();
    let loss = mean_x.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 验证 x 有梯度
    let x_grad = x.grad().unwrap();
    assert!(x_grad.is_some());
    let grad_tensor = x_grad.unwrap();
    assert_eq!(grad_tensor.shape(), &[2, 3]);

    // 全局均值梯度：所有元素梯度相等
    let first_grad = grad_tensor[[0, 0]];
    for val in grad_tensor.data_as_slice() {
        assert_abs_diff_eq!(*val, first_grad, epsilon = 1e-6);
    }
}

#[test]
fn test_var_mean_backward_axis() {
    let graph = Graph::new();

    // parameter -> mean_axis(1) -> mse_loss
    let x = graph.parameter(&[2, 3], Init::Ones, "x").unwrap();
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[2.0, 5.0], &[2, 1])).unwrap();

    let mean_x = x.mean_axis(1);
    let loss = mean_x.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 验证 x 有梯度
    let x_grad = x.grad().unwrap();
    assert!(x_grad.is_some());
    let grad_tensor = x_grad.unwrap();
    assert_eq!(grad_tensor.shape(), &[2, 3]);

    // 按轴均值梯度：同一行的梯度相等
    assert_abs_diff_eq!(grad_tensor[[0, 0]], grad_tensor[[0, 1]], epsilon = 1e-6);
    assert_abs_diff_eq!(grad_tensor[[0, 1]], grad_tensor[[0, 2]], epsilon = 1e-6);
    assert_abs_diff_eq!(grad_tensor[[1, 0]], grad_tensor[[1, 1]], epsilon = 1e-6);
    assert_abs_diff_eq!(grad_tensor[[1, 1]], grad_tensor[[1, 2]], epsilon = 1e-6);
}

#[test]
fn test_var_mean_chained() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[2, 2, 3],
        ))
        .unwrap();

    // 先沿 axis=2 求均值，再沿 axis=1 求均值
    let mean1 = x.mean_axis(2); // [2, 2, 3] -> [2, 2, 1]
    let mean2 = mean1.mean_axis(1); // [2, 2, 1] -> [2, 1, 1]

    mean2.forward().unwrap();
    let result = mean2.value().unwrap().unwrap();

    assert_eq!(result.shape(), &[2, 1, 1]);
    // 第一个 batch: mean of [1,2,3,4,5,6] = 3.5
    // 第二个 batch: mean of [7,8,9,10,11,12] = 9.5
    assert_abs_diff_eq!(result[[0, 0, 0]], 3.5, epsilon = 1e-5);
    assert_abs_diff_eq!(result[[1, 0, 0]], 9.5, epsilon = 1e-5);
}

/// 验证 Var 层 mean 与 sum/n 等价
#[test]
fn test_var_mean_equals_sum_div_n() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    let mean_result = x.mean();
    let sum_result = x.sum();

    mean_result.forward().unwrap();
    sum_result.forward().unwrap();

    let mean_val = mean_result.value().unwrap().unwrap()[[0, 0]];
    let sum_val = sum_result.value().unwrap().unwrap()[[0, 0]];

    // mean = sum / n
    assert_abs_diff_eq!(mean_val, sum_val / 6.0, epsilon = 1e-6);
}
