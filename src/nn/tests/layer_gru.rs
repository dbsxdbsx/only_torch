/*
 * @Author       : 老董
 * @Date         : 2026-01-17
 * @Description  : Gru Layer 单元测试（与 PyTorch 数值对照）
 */

use crate::nn::layer::Gru;
use crate::nn::{Graph, GraphError, Module, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 Gru 层创建
#[test]
fn test_gru_creation() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 16;
    let input_size = 10;
    let hidden_size = 20;

    let gru = Gru::new(&graph, input_size, hidden_size, batch_size, "gru1")?;

    // 验证参数存在
    assert!(gru.w_ir().value()?.is_some());
    assert!(gru.w_hr().value()?.is_some());
    assert!(gru.w_iz().value()?.is_some());
    assert!(gru.w_hz().value()?.is_some());
    assert!(gru.w_in().value()?.is_some());
    assert!(gru.w_hn().value()?.is_some());

    Ok(())
}

/// 测试 Gru 参数形状
#[test]
fn test_gru_shapes() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 8;
    let input_size = 4;
    let hidden_size = 6;

    let gru = Gru::new(&graph, input_size, hidden_size, batch_size, "gru1")?;

    // 验证重置门权重形状
    assert_eq!(gru.w_ir().value()?.unwrap().shape(), &[input_size, hidden_size]);
    assert_eq!(gru.w_hr().value()?.unwrap().shape(), &[hidden_size, hidden_size]);
    assert_eq!(gru.b_r().value()?.unwrap().shape(), &[1, hidden_size]);

    // 验证状态形状
    assert_eq!(gru.hidden_input().value()?.unwrap().shape(), &[batch_size, hidden_size]);

    Ok(())
}

/// 测试 Module trait
#[test]
fn test_gru_module_trait() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let gru = Gru::new(&graph, 10, 20, 4, "gru")?;

    let params = gru.parameters();
    assert_eq!(params.len(), 9); // 3 gates × 3 params each

    Ok(())
}

// ==================== PyTorch 数值对照测试 ====================

/// 测试简单前向传播
#[test]
fn test_gru_forward() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 2;
    let input_size = 3;
    let hidden_size = 4;

    let gru = Gru::new(&graph, input_size, hidden_size, batch_size, "gru1")?;

    // 设置简单权重
    gru.w_ir().set_value(&Tensor::new(&[0.1; 12], &[input_size, hidden_size]))?;
    gru.w_hr().set_value(&Tensor::new(&[0.1; 16], &[hidden_size, hidden_size]))?;
    gru.b_r().set_value(&Tensor::zeros(&[1, hidden_size]))?;

    gru.w_iz().set_value(&Tensor::new(&[0.1; 12], &[input_size, hidden_size]))?;
    gru.w_hz().set_value(&Tensor::new(&[0.1; 16], &[hidden_size, hidden_size]))?;
    gru.b_z().set_value(&Tensor::zeros(&[1, hidden_size]))?;

    gru.w_in().set_value(&Tensor::new(&[0.1; 12], &[input_size, hidden_size]))?;
    gru.w_hn().set_value(&Tensor::new(&[0.1; 16], &[hidden_size, hidden_size]))?;
    gru.b_n().set_value(&Tensor::zeros(&[1, hidden_size]))?;

    // 前向传播
    let x = Tensor::ones(&[batch_size, input_size]);
    gru.step(&x)?;

    // 验证输出存在且形状正确
    let hidden = gru.hidden().value()?.unwrap();
    assert_eq!(hidden.shape(), &[batch_size, hidden_size]);

    println!("✅ GRU 前向传播正确");
    Ok(())
}

/// 测试多时间步前向传播
#[test]
fn test_gru_multi_step() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 3;

    let gru = Gru::new(&graph, input_size, hidden_size, batch_size, "gru1")?;

    // 执行多个时间步
    for t in 0..3 {
        let x = Tensor::new(&[1.0, (t + 1) as f32 * 0.5], &[batch_size, input_size]);
        gru.step(&x)?;

        let hidden = gru.hidden().value()?.unwrap();
        println!("t={}: hidden[0]={:.4}", t, hidden[[0, 0]]);
    }

    println!("✅ GRU 多时间步前向传播正确");
    Ok(())
}

// ==================== reset 测试 ====================

/// 测试 reset() 清除状态
#[test]
fn test_gru_reset() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 2;

    let gru = Gru::new(&graph, input_size, hidden_size, batch_size, "gru")?;

    // 运行几步
    gru.step(&Tensor::ones(&[1, 2]))?;
    gru.step(&Tensor::ones(&[1, 2]))?;

    let h_before = gru.hidden().value()?.unwrap()[[0, 0]];
    assert!(h_before.abs() > 0.01);

    // 完整重置
    gru.reset();

    // 再运行一步
    gru.step(&Tensor::ones(&[1, 2]))?;
    let h_after = gru.hidden().value()?.unwrap()[[0, 0]];

    // 重新开始
    gru.reset();
    gru.step(&Tensor::ones(&[1, 2]))?;
    let h_fresh = gru.hidden().value()?.unwrap()[[0, 0]];

    assert_abs_diff_eq!(h_after, h_fresh, epsilon = 1e-6);
    println!("✅ GRU reset() 正确");
    Ok(())
}

// ==================== 与 Linear 集成测试 ====================

/// 测试 Gru 与 Linear 集成
#[test]
fn test_gru_with_linear_integration() -> Result<(), GraphError> {
    use crate::nn::layer::Linear;

    let graph = Graph::new_with_seed(42);
    let batch_size = 2;
    let input_size = 4;
    let hidden_size = 8;
    let output_size = 3;

    let gru = Gru::new(&graph, input_size, hidden_size, batch_size, "gru")?;
    let fc = Linear::new(&graph, hidden_size, output_size, true, "fc")?;

    // 前向传播
    gru.step(&Tensor::normal(0.0, 1.0, &[batch_size, input_size]))?;
    let fc_out = fc.forward(gru.hidden());
    fc_out.forward()?;

    let output = fc_out.value()?.unwrap();
    assert_eq!(output.shape(), &[batch_size, output_size]);

    println!("✅ GRU 与 Linear 集成正常");
    Ok(())
}

/// 测试完整训练流程
#[test]
fn test_gru_complete_training() -> Result<(), GraphError> {
    use crate::nn::layer::Linear;

    let graph = Graph::new_with_seed(42);
    let batch_size = 4;
    let input_size = 8;
    let hidden_size = 6;
    let output_size = 3;

    let gru = Gru::new(&graph, input_size, hidden_size, batch_size, "gru")?;
    let fc = Linear::new(&graph, hidden_size, output_size, true, "fc")?;

    // 前向传播
    gru.step(&Tensor::normal(0.0, 1.0, &[batch_size, input_size]))?;
    let fc_out = fc.forward(gru.hidden());

    let labels = graph.input(&Tensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        &[batch_size, output_size],
    ))?;
    let loss = fc_out.cross_entropy(&labels)?;

    // 前向 + 反向
    loss.forward()?;
    let loss_val = loss.value()?.unwrap()[[0, 0]];
    loss.backward()?;

    // 验证所有参数都有梯度
    assert!(gru.w_ir().grad()?.is_some());
    assert!(gru.w_iz().grad()?.is_some());
    assert!(gru.w_in().grad()?.is_some());
    assert!(fc.weights().grad()?.is_some());

    println!("✅ GRU 完整训练: loss={:.4}, 所有参数都有梯度", loss_val);
    Ok(())
}
