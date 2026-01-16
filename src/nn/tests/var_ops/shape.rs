/*
 * @Description  : VarShapeOps trait 测试
 *
 * 测试形状变换扩展 trait 的独立功能：
 * - reshape, flatten
 */

use crate::nn::graph::GraphHandle;
use crate::nn::VarShapeOps;
use crate::tensor::Tensor;

#[test]
fn test_var_reshape() {
    let graph = GraphHandle::new();
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
    let graph = GraphHandle::new();
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
    let graph = GraphHandle::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    // 4 个元素无法 reshape 为 [3, 2]（需要 6 个元素）
    let result = x.reshape(&[3, 2]);
    assert!(result.is_err());
}

#[test]
fn test_var_flatten() {
    let graph = GraphHandle::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let y = x.flatten().unwrap();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    // Flatten(keep_first_dim=false) 将 [2, 3] 展平为 [1, 6]（行向量）
    assert_eq!(result.shape(), &[1, 6]);
    assert_eq!(result.data_as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_var_flatten_already_flat() {
    let graph = GraphHandle::new();
    let x = graph.input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1])).unwrap();
    let y = x.flatten().unwrap();
    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();
    // [3, 1] flatten(keep_first_dim=false) -> [1, 3]（行向量）
    assert_eq!(result.shape(), &[1, 3]);
}

#[test]
fn test_var_reshape_chain() {
    let graph = GraphHandle::new();
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
