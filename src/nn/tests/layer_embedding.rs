/*
 * @Author       : 老董
 * @Description  : Embedding 层单元测试
 */

use crate::nn::{Embedding, Graph, GraphError, Module, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// Embedding 基本前向传播
#[test]
fn test_embedding_forward_1d() -> Result<(), GraphError> {
    let graph = Graph::new();

    let emb = Embedding::new(&graph, 5, 3, "emb")?;

    // 手动设置权重用于可预测测试
    // weight[i] = [i*10, i*10+1, i*10+2]
    let mut weight_data = vec![0.0f32; 15];
    for i in 0..5 {
        weight_data[i * 3] = (i * 10) as f32;
        weight_data[i * 3 + 1] = (i * 10 + 1) as f32;
        weight_data[i * 3 + 2] = (i * 10 + 2) as f32;
    }
    emb.weight()
        .set_value(&Tensor::new(&weight_data, &[5, 3]))?;

    // 索引 [[0, 2, 4]] → 选取第 0, 2, 4 行
    let indices = graph.input(&Tensor::new(&[0.0, 2.0, 4.0], &[1, 3]))?;
    let y = emb.forward(&indices);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[1, 3, 3]);

    // 行 0: [0, 1, 2]
    assert_abs_diff_eq!(output[[0, 0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 2]], 2.0, epsilon = 1e-6);

    // 行 2: [20, 21, 22]
    assert_abs_diff_eq!(output[[0, 1, 0]], 20.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 1]], 21.0, epsilon = 1e-6);

    // 行 4: [40, 41, 42]
    assert_abs_diff_eq!(output[[0, 2, 0]], 40.0, epsilon = 1e-6);

    Ok(())
}

/// Embedding 2D 索引 [N, T]
#[test]
fn test_embedding_forward_2d() -> Result<(), GraphError> {
    let graph = Graph::new();

    let emb = Embedding::new(&graph, 10, 4, "emb")?;

    // 设置简单权重: weight[i] = [i, i, i, i]
    let mut weight_data = vec![0.0f32; 40];
    for i in 0..10 {
        for j in 0..4 {
            weight_data[i * 4 + j] = i as f32;
        }
    }
    emb.weight()
        .set_value(&Tensor::new(&weight_data, &[10, 4]))?;

    // [N=2, T=3] 索引
    let indices = graph.input(&Tensor::new(&[0.0, 3.0, 7.0, 1.0, 5.0, 9.0], &[2, 3]))?;
    let y = emb.forward(&indices);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[2, 3, 4]);

    // [0, 0, :] = weight[0] = [0, 0, 0, 0]
    assert_abs_diff_eq!(output[[0, 0, 0]], 0.0, epsilon = 1e-6);
    // [0, 1, :] = weight[3] = [3, 3, 3, 3]
    assert_abs_diff_eq!(output[[0, 1, 0]], 3.0, epsilon = 1e-6);
    // [1, 2, :] = weight[9] = [9, 9, 9, 9]
    assert_abs_diff_eq!(output[[1, 2, 0]], 9.0, epsilon = 1e-6);

    Ok(())
}

/// Embedding 参数数量
#[test]
fn test_embedding_parameters() -> Result<(), GraphError> {
    let graph = Graph::new();
    let emb = Embedding::new(&graph, 1000, 256, "emb")?;

    let params = emb.parameters();
    assert_eq!(params.len(), 1); // 只有 weight
    assert_eq!(emb.vocab_size(), 1000);
    assert_eq!(emb.embed_dim(), 256);

    Ok(())
}

/// Embedding 反向传播 — 权重接收梯度
#[test]
fn test_embedding_backward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let emb = Embedding::new(&graph, 5, 3, "emb")?;

    let indices = graph.input(&Tensor::new(&[0.0, 2.0], &[1, 2]))?;
    let y = emb.forward(&indices);

    // mse loss
    let target = graph.input(&Tensor::zeros(&[1, 2, 3]))?;
    let loss = y.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    // 权重应有梯度
    let weight_grad = emb.weight().grad()?.expect("weight 应有 grad");
    assert_eq!(weight_grad.shape(), &[5, 3]);

    Ok(())
}
