/*
 * @Description  : VarShapeOps trait 测试 + Var::stack 测试
 *
 * 测试形状变换扩展 trait 的独立功能：
 * - reshape, flatten, select
 * - Var::stack（关联函数）
 */

use crate::nn::graph::Graph;
use crate::nn::var::Var;
use crate::nn::{Init, VarLossOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_var_reshape() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let y = x.reshape(&[3, 2]).unwrap();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.data_as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_var_reshape_to_vector() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    // Reshape 到 [4, 1]（列向量）
    let y = x.reshape(&[4, 1]).unwrap();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[4, 1]);
    assert_eq!(result.data_as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_var_reshape_invalid_size() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    // 4 个元素无法 reshape 为 [3, 2]（需要 6 个元素）
    let result = x.reshape(&[3, 2]);
    assert!(result.is_err());
}

#[test]
fn test_var_flatten() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let y = x.flatten().unwrap();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    // Flatten(keep_first_dim=true) 将 [2, 3] 展平为 [2, 3]（保持 batch 维度）
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.data_as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_var_flatten_already_flat() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))
        .unwrap();
    let y = x.flatten().unwrap();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    // [3, 1] flatten(keep_first_dim=true) -> [3, 1]（保持 batch 维度）
    assert_eq!(result.shape(), &[3, 1]);
}

#[test]
fn test_var_reshape_chain() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    // [2, 3] -> [6, 1] -> [3, 2]
    let y = x.reshape(&[6, 1]).unwrap().reshape(&[3, 2]).unwrap();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.data_as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// ==================== Var::stack 测试 ====================

/// 测试 Var::stack - Stack 模式（new_dim=true）
#[test]
fn test_var_stack_stack_mode() {
    let graph = Graph::new();
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();

    // Stack 模式：[2, 2] + [2, 2] -> [2, 2, 2]
    let stacked = Var::stack(&[&a, &b], 0, true).unwrap();
    stacked.forward().unwrap();

    let result = stacked.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2, 2]);
    assert_eq!(
        result.data_as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

/// 测试 Var::stack - Concat 模式（new_dim=false）
#[test]
fn test_var_stack_concat_mode() {
    let graph = Graph::new();
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();

    // Concat 模式：[2, 2] + [2, 2] -> [4, 2]
    let concat = Var::stack(&[&a, &b], 0, false).unwrap();
    concat.forward().unwrap();

    let result = concat.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[4, 2]);
    assert_eq!(
        result.data_as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

/// 测试 Var::stack - Concat 模式沿 axis=1
#[test]
fn test_var_stack_concat_axis1() {
    let graph = Graph::new();
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3]))
        .unwrap();

    // Concat 沿 axis=1：[2, 2] + [2, 3] -> [2, 5]
    let concat = Var::stack(&[&a, &b], 1, false).unwrap();
    concat.forward().unwrap();

    let result = concat.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 5]);
    // stack 自动保证连续内存，可以直接调用 data_as_slice
    assert_eq!(
        result.data_as_slice(),
        &[1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]
    );
}

/// 测试 Var::stack - 三个 Var
#[test]
fn test_var_stack_three_vars() {
    let graph = Graph::new();
    let a = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let b = graph.input(&Tensor::new(&[3.0, 4.0], &[1, 2])).unwrap();
    let c = graph.input(&Tensor::new(&[5.0, 6.0], &[1, 2])).unwrap();

    // Concat：[1,2] + [1,2] + [1,2] -> [3, 2]
    let concat = Var::stack(&[&a, &b, &c], 0, false).unwrap();
    concat.forward().unwrap();

    let result = concat.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.data_as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// 测试 Var::stack - 空列表应报错
#[test]
fn test_var_stack_empty_error() {
    let empty: Vec<&Var> = vec![];
    let result = Var::stack(&empty, 0, false);
    assert!(result.is_err());
}

/// 测试 Var::stack - 不同 Graph 应报错
#[test]
fn test_var_stack_different_graph_error() {
    let graph1 = Graph::new();
    let graph2 = Graph::new();
    let a = graph1.input(&Tensor::ones(&[2, 2])).unwrap();
    let b = graph2.input(&Tensor::ones(&[2, 2])).unwrap();

    let result = Var::stack(&[&a, &b], 0, false);
    assert!(result.is_err());
}

/// 测试 Var::stack 反向传播（端到端）
#[test]
fn test_var_stack_backward() {
    let graph = Graph::new();

    // 创建参数节点
    let p1 = graph.parameter(&[1, 2], Init::Zeros, "p1").unwrap();
    let p2 = graph.parameter(&[1, 2], Init::Zeros, "p2").unwrap();
    p1.set_value(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    p2.set_value(&Tensor::new(&[3.0, 4.0], &[1, 2])).unwrap();

    // Stack: [1, 2] + [1, 2] -> [2, 2]
    let stacked = Var::stack(&[&p1, &p2], 0, false).unwrap();

    // 创建 target 和 loss
    let target = graph.input(&Tensor::zeros(&[2, 2])).unwrap();
    let loss = stacked.mse_loss(&target).unwrap();

    // 反向传播
    let loss_val = loss.backward().unwrap();
    assert_abs_diff_eq!(loss_val, 7.5, epsilon = 1e-6); // mean([1,4,9,16]) = 7.5

    // 检查梯度
    let p1_grad = p1.grad().unwrap().unwrap();
    let p2_grad = p2.grad().unwrap().unwrap();

    // ∂loss/∂result = 2*(result - 0)/4 = result/2
    // p1 grad = [1, 2] / 2 = [0.5, 1.0]
    // p2 grad = [3, 4] / 2 = [1.5, 2.0]
    assert_eq!(p1_grad.shape(), &[1, 2]);
    assert_eq!(p2_grad.shape(), &[1, 2]);
    assert_abs_diff_eq!(&p1_grad, &Tensor::new(&[0.5, 1.0], &[1, 2]), epsilon = 1e-6);
    assert_abs_diff_eq!(&p2_grad, &Tensor::new(&[1.5, 2.0], &[1, 2]), epsilon = 1e-6);
}
