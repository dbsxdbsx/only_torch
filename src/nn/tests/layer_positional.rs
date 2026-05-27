/*
 * @Author       : 老董
 * @Description  : Positional Encoding 单元测试
 *   覆盖：形状正确性、Sinusoidal 公式数值（与 PyTorch 参考对齐 1e-5）、
 *         不同 embed_dim 奇偶维度、可训练版本反向梯度。
 */

use crate::nn::{
    Graph, GraphError, LearnableAbsolutePositionalEncoding, Module, SinusoidalPositionalEncoding,
    VarLossOps,
};
use crate::tensor::Tensor;

/// pos=0 时所有偶数维 sin(0)=0、奇数维 cos(0)=1
#[test]
fn sinusoidal_pe_pos0_is_zero_one() -> Result<(), GraphError> {
    let g = Graph::new();
    let pe = SinusoidalPositionalEncoding::new(&g, 8, 4, "pe")?;
    let x = g.input(&Tensor::zeros(&[1, 1, 4]))?;
    let y = pe.forward(&x);
    let v = y.value()?.unwrap();
    let data = v.data_as_slice();
    assert!((data[0]).abs() < 1e-6, "i=0 sin(0)=0，实际 {}", data[0]);
    assert!(
        (data[1] - 1.0).abs() < 1e-6,
        "i=1 cos(0)=1，实际 {}",
        data[1]
    );
    assert!((data[2]).abs() < 1e-6, "i=2 sin(0)=0，实际 {}", data[2]);
    assert!(
        (data[3] - 1.0).abs() < 1e-6,
        "i=3 cos(0)=1，实际 {}",
        data[3]
    );
    Ok(())
}

/// 形状测试：[B, T, D] 输入 → [B, T, D] 输出，T < max_len
#[test]
fn sinusoidal_pe_output_shape_matches_input() -> Result<(), GraphError> {
    let g = Graph::new();
    let pe = SinusoidalPositionalEncoding::new(&g, 16, 8, "pe")?;
    let x = g.input(&Tensor::ones(&[2, 5, 8]))?;
    let y = pe.forward(&x);
    y.forward()?;
    assert_eq!(y.value()?.unwrap().shape(), &[2, 5, 8]);
    Ok(())
}

/// 验证 pos=1 处具体公式数值
#[test]
fn sinusoidal_pe_specific_values() -> Result<(), GraphError> {
    let g = Graph::new();
    let pe = SinusoidalPositionalEncoding::new(&g, 4, 4, "pe")?;
    let x = g.input(&Tensor::zeros(&[1, 4, 4]))?;
    let y = pe.forward(&x);
    let v = y.value()?.unwrap();
    let arr = v.data_as_slice();
    // pos=1, d=4：
    //   PE(1, 0) = sin(1 / 10000^(0/4)) = sin(1)
    //   PE(1, 1) = cos(1 / 10000^(0/4)) = cos(1)
    //   PE(1, 2) = sin(1 / 10000^(2/4)) = sin(1/100) = sin(0.01)
    //   PE(1, 3) = cos(1 / 10000^(2/4)) = cos(0.01)
    let pos = 1;
    let d = 4;
    let off = pos * d;
    assert!((arr[off] - 1.0_f32.sin()).abs() < 1e-5);
    assert!((arr[off + 1] - 1.0_f32.cos()).abs() < 1e-5);
    assert!((arr[off + 2] - 0.01_f32.sin()).abs() < 1e-5);
    assert!((arr[off + 3] - 0.01_f32.cos()).abs() < 1e-5);
    Ok(())
}

/// 不同 embed_dim（含奇数 6）下 pos=0 仍然是 [0,1,0,1,0,1]
#[test]
fn sinusoidal_pe_embed_dim_6() -> Result<(), GraphError> {
    let g = Graph::new();
    let pe = SinusoidalPositionalEncoding::new(&g, 4, 6, "pe")?;
    let x = g.input(&Tensor::zeros(&[1, 1, 6]))?;
    let y = pe.forward(&x);
    let v = y.value()?.unwrap();
    let arr = v.data_as_slice();
    assert!(arr[0].abs() < 1e-6);
    assert!((arr[1] - 1.0).abs() < 1e-6);
    assert!(arr[2].abs() < 1e-6);
    assert!((arr[3] - 1.0).abs() < 1e-6);
    assert!(arr[4].abs() < 1e-6);
    assert!((arr[5] - 1.0).abs() < 1e-6);
    Ok(())
}

/// Sinusoidal PE 是无参数的
#[test]
fn sinusoidal_pe_has_no_parameters() -> Result<(), GraphError> {
    let g = Graph::new();
    let pe = SinusoidalPositionalEncoding::new(&g, 8, 4, "pe")?;
    assert_eq!(pe.parameters().len(), 0);
    assert_eq!(pe.max_len(), 8);
    assert_eq!(pe.embed_dim(), 4);
    Ok(())
}

/// 学习式 PE 含一个 [max_len, D] 参数
#[test]
fn learnable_pe_has_weight_param() -> Result<(), GraphError> {
    let g = Graph::new();
    let pe = LearnableAbsolutePositionalEncoding::new(&g, 8, 4, "lpe")?;
    let params = pe.parameters();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].node().shape(), &[8, 4]);
    Ok(())
}

/// 学习式 PE forward 形状测试
#[test]
fn learnable_pe_forward_shape() -> Result<(), GraphError> {
    let g = Graph::new();
    let pe = LearnableAbsolutePositionalEncoding::new(&g, 16, 8, "lpe")?;
    let x = g.input(&Tensor::ones(&[3, 5, 8]))?;
    let y = pe.forward(&x);
    y.forward()?;
    assert_eq!(y.value()?.unwrap().shape(), &[3, 5, 8]);
    Ok(())
}

/// 学习式 PE 反向梯度通畅
#[test]
fn learnable_pe_backward_flows() -> Result<(), GraphError> {
    let g = Graph::new();
    let pe = LearnableAbsolutePositionalEncoding::new(&g, 8, 4, "lpe")?;
    let x = g.input(&Tensor::ones(&[2, 3, 4]))?;
    let y = pe.forward(&x);
    let target = g.input(&Tensor::zeros(&[2, 3, 4]))?;
    let loss = y.mse_loss(&target)?;
    g.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);
    let weight = pe.weight();
    let grad = weight.grad()?;
    assert!(grad.is_some(), "weight 应有梯度");
    Ok(())
}
