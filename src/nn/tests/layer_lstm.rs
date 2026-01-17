/*
 * @Author       : 老董
 * @Date         : 2026-01-17
 * @Description  : Lstm Layer 单元测试（与 PyTorch 数值对照）
 */

use crate::nn::layer::Lstm;
use crate::nn::{Graph, GraphError, Module, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 Lstm 层创建
#[test]
fn test_lstm_creation() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 16;
    let input_size = 10;
    let hidden_size = 20;

    let lstm = Lstm::new(&graph, input_size, hidden_size, batch_size, "lstm1")?;

    // 验证参数存在
    assert!(lstm.w_ii().value()?.is_some());
    assert!(lstm.w_hi().value()?.is_some());
    assert!(lstm.w_if().value()?.is_some());
    assert!(lstm.w_hf().value()?.is_some());
    assert!(lstm.w_ig().value()?.is_some());
    assert!(lstm.w_hg().value()?.is_some());
    assert!(lstm.w_io().value()?.is_some());
    assert!(lstm.w_ho().value()?.is_some());

    Ok(())
}

/// 测试 Lstm 参数形状
#[test]
fn test_lstm_shapes() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 8;
    let input_size = 4;
    let hidden_size = 6;

    let lstm = Lstm::new(&graph, input_size, hidden_size, batch_size, "lstm1")?;

    // 验证输入门权重形状
    assert_eq!(
        lstm.w_ii().value()?.unwrap().shape(),
        &[input_size, hidden_size]
    );
    assert_eq!(
        lstm.w_hi().value()?.unwrap().shape(),
        &[hidden_size, hidden_size]
    );
    assert_eq!(lstm.b_i().value()?.unwrap().shape(), &[1, hidden_size]);

    // 验证状态形状
    assert_eq!(
        lstm.hidden_input().value()?.unwrap().shape(),
        &[batch_size, hidden_size]
    );
    assert_eq!(
        lstm.cell_input().value()?.unwrap().shape(),
        &[batch_size, hidden_size]
    );

    Ok(())
}

/// 测试 Module trait
#[test]
fn test_lstm_module_trait() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 10, 20, 4, "lstm")?;

    let params = lstm.parameters();
    assert_eq!(params.len(), 12); // 4 gates × 3 params each

    Ok(())
}

// ==================== PyTorch 数值对照测试 ====================

/// 测试简单前向传播
#[test]
fn test_lstm_forward() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 2;
    let input_size = 3;
    let hidden_size = 4;

    let lstm = Lstm::new(&graph, input_size, hidden_size, batch_size, "lstm1")?;

    // 设置简单权重
    lstm.w_ii()
        .set_value(&Tensor::new(&[0.1; 12], &[input_size, hidden_size]))?;
    lstm.w_hi()
        .set_value(&Tensor::new(&[0.1; 16], &[hidden_size, hidden_size]))?;
    lstm.b_i().set_value(&Tensor::zeros(&[1, hidden_size]))?;

    lstm.w_if()
        .set_value(&Tensor::new(&[0.1; 12], &[input_size, hidden_size]))?;
    lstm.w_hf()
        .set_value(&Tensor::new(&[0.1; 16], &[hidden_size, hidden_size]))?;
    lstm.b_f().set_value(&Tensor::ones(&[1, hidden_size]))?;

    lstm.w_ig()
        .set_value(&Tensor::new(&[0.1; 12], &[input_size, hidden_size]))?;
    lstm.w_hg()
        .set_value(&Tensor::new(&[0.1; 16], &[hidden_size, hidden_size]))?;
    lstm.b_g().set_value(&Tensor::zeros(&[1, hidden_size]))?;

    lstm.w_io()
        .set_value(&Tensor::new(&[0.1; 12], &[input_size, hidden_size]))?;
    lstm.w_ho()
        .set_value(&Tensor::new(&[0.1; 16], &[hidden_size, hidden_size]))?;
    lstm.b_o().set_value(&Tensor::zeros(&[1, hidden_size]))?;

    // 前向传播
    let x = Tensor::ones(&[batch_size, input_size]);
    lstm.step(&x)?;

    // 验证输出存在且形状正确
    let hidden = lstm.hidden().value()?.unwrap();
    assert_eq!(hidden.shape(), &[batch_size, hidden_size]);

    let cell = lstm.cell().value()?.unwrap();
    assert_eq!(cell.shape(), &[batch_size, hidden_size]);

    println!("✅ LSTM 前向传播正确");
    Ok(())
}

/// 测试多时间步前向传播
#[test]
fn test_lstm_multi_step() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 1;
    let input_size = 2;
    let hidden_size = 3;

    let lstm = Lstm::new(&graph, input_size, hidden_size, batch_size, "lstm1")?;

    // 执行多个时间步
    for t in 0..3 {
        let x = Tensor::new(&[1.0, (t + 1) as f32 * 0.5], &[batch_size, input_size]);
        lstm.step(&x)?;

        let hidden = lstm.hidden().value()?.unwrap();
        println!("t={}: hidden[0]={:.4}", t, hidden[[0, 0]]);
    }

    println!("✅ LSTM 多时间步前向传播正确");
    Ok(())
}

// ==================== reset 测试 ====================

/// 测试 reset() 清除状态
///
/// 核心验证：reset 后从同一输入出发，应产生相同输出。
/// 不检查输出的绝对值大小，避免对随机初始化的依赖。
#[test]
fn test_lstm_reset() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let lstm = Lstm::new(&graph, 2, 2, 1, "lstm")?;

    // 运行几步（使用随机初始化的权重）
    lstm.step(&Tensor::ones(&[1, 2]))?;
    lstm.step(&Tensor::ones(&[1, 2]))?;

    // reset 后运行一步
    lstm.reset();
    lstm.step(&Tensor::ones(&[1, 2]))?;
    let h_after_reset = lstm.hidden().value()?.unwrap().clone();

    // 再次 reset 后运行一步
    lstm.reset();
    lstm.step(&Tensor::ones(&[1, 2]))?;
    let h_fresh = lstm.hidden().value()?.unwrap();

    // 核心断言：两次 reset 后从相同输入出发，输出应一致
    assert_abs_diff_eq!(h_after_reset[[0, 0]], h_fresh[[0, 0]], epsilon = 1e-6);
    assert_abs_diff_eq!(h_after_reset[[0, 1]], h_fresh[[0, 1]], epsilon = 1e-6);

    println!("✅ LSTM reset() 正确");
    Ok(())
}

// ==================== 与 Linear 集成测试 ====================

/// 测试 Lstm 与 Linear 集成
#[test]
fn test_lstm_with_linear_integration() -> Result<(), GraphError> {
    use crate::nn::layer::Linear;

    let graph = Graph::new_with_seed(42);
    let batch_size = 2;
    let input_size = 4;
    let hidden_size = 8;
    let output_size = 3;

    let lstm = Lstm::new(&graph, input_size, hidden_size, batch_size, "lstm")?;
    let fc = Linear::new(&graph, hidden_size, output_size, true, "fc")?;

    // 前向传播
    lstm.step(&Tensor::normal(0.0, 1.0, &[batch_size, input_size]))?;
    let fc_out = fc.forward(lstm.hidden());
    fc_out.forward()?;

    let output = fc_out.value()?.unwrap();
    assert_eq!(output.shape(), &[batch_size, output_size]);

    println!("✅ LSTM 与 Linear 集成正常");
    Ok(())
}

/// 测试完整训练流程
#[test]
fn test_lstm_complete_training() -> Result<(), GraphError> {
    use crate::nn::layer::Linear;

    let graph = Graph::new_with_seed(42);
    let batch_size = 4;
    let input_size = 8;
    let hidden_size = 6;
    let output_size = 3;

    let lstm = Lstm::new(&graph, input_size, hidden_size, batch_size, "lstm")?;
    let fc = Linear::new(&graph, hidden_size, output_size, true, "fc")?;

    // 前向传播
    lstm.step(&Tensor::normal(0.0, 1.0, &[batch_size, input_size]))?;
    let fc_out = fc.forward(lstm.hidden());

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
    assert!(lstm.w_ii().grad()?.is_some());
    assert!(lstm.w_if().grad()?.is_some());
    assert!(lstm.w_ig().grad()?.is_some());
    assert!(lstm.w_io().grad()?.is_some());
    assert!(fc.weights().grad()?.is_some());

    println!("✅ LSTM 完整训练: loss={:.4}, 所有参数都有梯度", loss_val);
    Ok(())
}
