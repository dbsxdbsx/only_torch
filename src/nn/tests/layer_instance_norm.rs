/*
 * @Author       : 老董
 * @Description  : InstanceNorm 层单元测试
 */

use crate::nn::{Graph, GraphError, InstanceNorm, Module};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// InstanceNorm 基本前向传播
#[test]
fn test_instance_norm_forward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let inst_norm = InstanceNorm::new(&graph, 2, 1e-5, "in")?;

    // [N=1, C=2, H=2, W=2]
    let x = graph.input(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        &[1, 2, 2, 2],
    ))?;
    let y = inst_norm.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[1, 2, 2, 2]);

    // 通道 0: [1,2,3,4], 均值=2.5, 方差=1.25
    // (1-2.5)/sqrt(1.25+1e-5) ≈ -1.3416
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], -1.3416, epsilon = 1e-3);
    // 通道 1: [10,20,30,40], 均值=25, 方差=125
    // (10-25)/sqrt(125+1e-5) ≈ -1.3416
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], -1.3416, epsilon = 1e-3);

    Ok(())
}

/// InstanceNorm 参数
#[test]
fn test_instance_norm_parameters() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inst_norm = InstanceNorm::new(&graph, 32, 1e-5, "in")?;

    let params = inst_norm.parameters();
    assert_eq!(params.len(), 2); // gamma + beta

    Ok(())
}
