/*
 * @Description  : VarMatrixOps trait 测试
 *
 * 测试矩阵运算扩展 trait 的独立功能：
 * - matmul（及形状检查、跨图安全性）
 */

use crate::nn::VarMatrixOps;
use crate::nn::graph::GraphHandle;
use crate::tensor::Tensor;

#[test]
fn test_var_matmul() {
    let graph = GraphHandle::new();
    // [2, 3] @ [3, 2] = [2, 2]
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]))
        .unwrap();
    let c = a.matmul(&b).unwrap();
    c.forward().unwrap();
    let result = c.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    // [1,2,3] @ [1,2; 3,4; 5,6] = [22, 28]
    // [4,5,6] @ [1,2; 3,4; 5,6] = [49, 64]
    assert_eq!(result.data_as_slice(), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_var_matmul_vector() {
    let graph = GraphHandle::new();
    // [2, 3] @ [3, 1] = [2, 1]
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))
        .unwrap();
    let c = a.matmul(&b).unwrap();
    c.forward().unwrap();
    let result = c.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 1]);
    // [1,2,3] @ [1,2,3]^T = 1+4+9 = 14
    // [4,5,6] @ [1,2,3]^T = 4+10+18 = 32
    assert_eq!(result.data_as_slice(), &[14.0, 32.0]);
}

#[test]
#[should_panic(expected = "不能对来自不同 Graph 的 Var 进行操作")]
fn test_var_matmul_different_graph_panic() {
    let graph1 = GraphHandle::new();
    let graph2 = GraphHandle::new();
    let a = graph1.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let b = graph2.input(&Tensor::new(&[3.0, 4.0], &[2, 1])).unwrap();
    // 应该 panic（不同 Graph）
    let _ = a.matmul(&b);
}

#[test]
fn test_var_matmul_shape_mismatch() {
    let graph = GraphHandle::new();
    // [2, 3] @ [2, 3] = 形状不匹配（内维度 3 != 2）
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let result = a.matmul(&b);
    assert!(result.is_err());
}
