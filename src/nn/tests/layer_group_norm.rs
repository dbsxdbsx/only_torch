/*
 * @Author       : 老董
 * @Description  : GroupNorm 层单元测试
 */

use crate::nn::{Graph, GraphError, GroupNorm, Module};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// GroupNorm 基本前向传播
#[test]
fn test_group_norm_forward() -> Result<(), GraphError> {
    let graph = Graph::new();

    // 4 通道，2 组（每组 2 通道）
    let gn = GroupNorm::new(&graph, 2, 4, 1e-5, "gn")?;

    // [N=1, C=4, H=1, W=2]
    let x = graph.input(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        &[1, 4, 1, 2],
    ))?;

    let y = gn.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[1, 4, 1, 2]);

    Ok(())
}

/// GroupNorm 参数
#[test]
fn test_group_norm_parameters() -> Result<(), GraphError> {
    let graph = Graph::new();
    let gn = GroupNorm::new(&graph, 4, 16, 1e-5, "gn")?;

    let params = gn.parameters();
    assert_eq!(params.len(), 2); // gamma + beta

    Ok(())
}

/// GroupNorm(1, C) ≈ LayerNorm 行为
#[test]
fn test_group_norm_single_group() -> Result<(), GraphError> {
    let graph = Graph::new();

    // num_groups=1 → 所有通道一起归一化
    let gn = GroupNorm::new(&graph, 1, 4, 1e-5, "gn")?;

    let x = graph.input(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 4],
    ))?;
    let y = gn.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[1, 4]);
    // 均值=2.5, 方差=1.25, std≈1.118
    // (1-2.5)/1.118 ≈ -1.3416
    assert_abs_diff_eq!(output[[0, 0]], -1.3416, epsilon = 1e-3);

    Ok(())
}
