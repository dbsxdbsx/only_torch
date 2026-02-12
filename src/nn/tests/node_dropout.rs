/*
 * Dropout 节点单元测试
 *
 * 测试策略（六段式）：
 * 1. 基础创建 → 默认 p=0.5; 自定义 p; 无效 p; cannot_set_value
 * 2. 训练/评估模式行为 → 训练丢弃; 评估直通; 训练→评估切换; 评估→训练切换; 多次 forward 不同 mask
 * 3. 反向传播 → 训练 mask 梯度; p=0 梯度完全通过; 评估 vs 训练对比
 * 4. 确定性 → 相同 seed 同结果; 不同 seed 不同结果（底层 API 指定 seed）
 * 5. 统计特性 → 丢弃率符合 p; 期望值保持（Inverted Dropout）; p=0 为 identity
 * 6. 动态 batch + Create API（KEEP AS-IS）
 *
 * Inverted Dropout: 训练时保留元素按 1/(1-p) 缩放，保证期望不变。
 * 高层 .dropout(p) 使用系统时间 seed（不确定），确定性测试用底层 API 指定 seed。
 */

use crate::nn::{Graph, GraphError, Init, Var, VarLossOps, VarRegularizationOps};
use crate::tensor::Tensor;

// ==================== 1. 基础创建测试 ====================

/// 高层 API 创建 Dropout（默认 p=0.5）
#[test]
fn test_dropout_create_default() {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::ones(&[2, 4])).unwrap();
    let dropped = x.dropout(0.5).unwrap();

    // 验证输出形状与输入一致
    assert_eq!(dropped.value_expected_shape(), vec![2, 4]);
}

/// 高层 API 创建不同 p 值的 Dropout
#[test]
fn test_dropout_create_custom_p() {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::ones(&[2, 4])).unwrap();

    // 各种合法 p 值
    let _d1 = x.dropout(0.1).unwrap();
    let _d2 = x.dropout(0.3).unwrap();
    let _d3 = x.dropout(0.7).unwrap();

    // p=0 等价 identity，应创建成功
    let _d0 = x.dropout(0.0).unwrap();
}

/// 无效 p 值应报错
#[test]
fn test_dropout_invalid_p() {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::ones(&[2, 4])).unwrap();

    // p >= 1.0
    assert!(x.dropout(1.0).is_err());
    assert!(x.dropout(1.5).is_err());

    // p < 0
    assert!(x.dropout(-0.1).is_err());
}

/// Dropout 节点不能直接设置值
#[test]
fn test_dropout_cannot_set_value() {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::ones(&[2, 4])).unwrap();
    let dropped = x.dropout(0.5).unwrap();

    let test_value = Tensor::ones(&[2, 4]);
    let err = dropped.set_value(&test_value);
    assert!(err.is_err(), "Dropout 节点不应支持直接设值");
}

// ==================== 2. 训练/评估模式行为 ====================

/// 训练模式下输出应有元素被置零（Inverted Dropout 缩放为 2.0）
#[test]
fn test_dropout_train_mode_drops_elements() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    graph.inner_mut().set_train_mode();

    let x = graph.input(&Tensor::ones(&[1, 100]))?;
    let dropped = x.dropout(0.5)?;

    dropped.forward()?;
    let output = dropped.value()?.unwrap();

    // p=0.5 → 保留元素缩放为 1/(1-0.5)=2.0, 丢弃为 0
    let mut zeros = 0;
    let mut twos = 0;
    for i in 0..100 {
        let v = output[[0, i]];
        if v == 0.0 {
            zeros += 1;
        } else if (v - 2.0).abs() < 1e-6 {
            twos += 1;
        }
    }

    assert!(zeros > 0, "训练模式应有元素被丢弃");
    assert!(twos > 0, "保留元素应被缩放为 2.0");
    assert_eq!(zeros + twos, 100, "所有元素应为 0 或 2.0");

    Ok(())
}

/// 评估模式下输出等于输入（直通）
#[test]
fn test_dropout_eval_mode_passthrough() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    graph.inner_mut().set_eval_mode();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let x = graph.input(&input_data)?;
    let dropped = x.dropout(0.5)?;

    dropped.forward()?;
    let output = dropped.value()?.unwrap();

    for i in 0..2 {
        for j in 0..4 {
            assert!(
                (output[[i, j]] - input_data[[i, j]]).abs() < 1e-6,
                "评估模式输出应等于输入"
            );
        }
    }

    Ok(())
}

/// 训练→评估模式切换后行为正确
#[test]
fn test_dropout_mode_switch_train_to_eval() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::ones(&[1, 100]))?;
    let dropped = x.dropout(0.5)?;

    // 1. 训练模式
    graph.inner_mut().set_train_mode();
    dropped.forward()?;
    let train_output = dropped.value()?.unwrap();

    let has_zeros = (0..100).any(|i| train_output[[0, i]] == 0.0);
    assert!(has_zeros, "训练模式应有元素被丢弃");

    // 2. 切换到评估模式，重新 forward
    graph.inner_mut().set_eval_mode();
    dropped.forward()?;
    let eval_output = dropped.value()?.unwrap();

    for i in 0..100 {
        assert!(
            (eval_output[[0, i]] - 1.0).abs() < 1e-6,
            "评估模式应直通，元素 {i} 值 {} != 1.0",
            eval_output[[0, i]]
        );
    }

    Ok(())
}

/// 评估→训练模式切换后行为正确
#[test]
fn test_dropout_mode_switch_eval_to_train() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::ones(&[1, 100]))?;
    let dropped = x.dropout(0.5)?;

    // 1. 评估模式
    graph.inner_mut().set_eval_mode();
    dropped.forward()?;
    let eval_output = dropped.value()?.unwrap();

    for i in 0..100 {
        assert!((eval_output[[0, i]] - 1.0).abs() < 1e-6, "评估模式应直通");
    }

    // 2. 切换到训练模式，重新 forward
    graph.inner_mut().set_train_mode();
    dropped.forward()?;
    let train_output = dropped.value()?.unwrap();

    let has_zeros = (0..100).any(|i| train_output[[0, i]] == 0.0);
    assert!(has_zeros, "切换到训练模式后应有元素被丢弃");

    Ok(())
}

/// 训练模式下多次 forward 产生不同 mask
#[test]
fn test_dropout_different_masks_each_forward() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    graph.inner_mut().set_train_mode();

    let x = graph.input(&Tensor::ones(&[1, 100]))?;
    let dropped = x.dropout(0.5)?;

    // 第一次 forward
    dropped.forward()?;
    let output1 = dropped.value()?.unwrap();

    // 第二次 forward（pass_id 递增，触发重新计算）
    dropped.forward()?;
    let output2 = dropped.value()?.unwrap();

    // 两次结果应不同（概率极高）
    let found_diff = (0..100).any(|i| (output1[[0, i]] - output2[[0, i]]).abs() > 1e-6);
    assert!(found_diff, "训练模式下多次 forward 应产生不同 mask");

    Ok(())
}

// ==================== 3. 反向传播测试 ====================

/// 训练模式下梯度在被丢弃位置为 0
#[test]
fn test_dropout_backward_train_mode() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    graph.inner_mut().set_train_mode();

    // w -> dropout -> mse_loss
    let w = graph.parameter(&[1, 10], Init::Ones, "w")?;
    let dropped = w.dropout(0.5)?;
    let target = graph.input(&Tensor::zeros(&[1, 10]))?;
    let loss = dropped.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    let grad = w.grad()?.expect("参数 w 应有梯度");
    let zeros_count = (0..10).filter(|&i| grad[[0, i]] == 0.0).count();
    assert!(zeros_count > 0, "训练模式下被丢弃位置的梯度应为 0");

    Ok(())
}

/// p=0 时梯度完全通过（等价 identity）
#[test]
fn test_dropout_backward_p_zero_passthrough() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    graph.inner_mut().set_train_mode();

    let w = graph.parameter(&[1, 4], Init::Ones, "w")?;
    w.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]))?;

    // p=0 → identity，梯度应直接通过
    let dropped = w.dropout(0.0)?;
    let target = graph.input(&Tensor::zeros(&[1, 4]))?;
    let loss = dropped.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    // MSE Mean VJP: grad = 2*(pred - target)/N = 2*[1,2,3,4]/4 = [0.5, 1.0, 1.5, 2.0]
    let grad = w.grad()?.expect("参数 w 应有梯度");
    let expected = [0.5, 1.0, 1.5, 2.0];
    for (i, &e) in expected.iter().enumerate() {
        let g = grad[[0, i]];
        assert!(
            (g - e).abs() < 1e-5,
            "位置 {i} 梯度不匹配: got {g}, expected {e}"
        );
    }

    Ok(())
}

/// 评估模式 vs 训练模式前向对比
#[test]
fn test_dropout_eval_vs_train_same_input() -> Result<(), GraphError> {
    // 训练模式图
    let graph_train = Graph::new_with_seed(42);
    graph_train.inner_mut().set_train_mode();

    let x_train = graph_train.input(&Tensor::ones(&[1, 100]))?;
    let d_train = x_train.dropout(0.5)?;

    d_train.forward()?;
    let output_train = d_train.value()?.unwrap();

    // 评估模式图
    let graph_eval = Graph::new_with_seed(42);
    graph_eval.inner_mut().set_eval_mode();

    let x_eval = graph_eval.input(&Tensor::ones(&[1, 100]))?;
    let d_eval = x_eval.dropout(0.5)?;

    d_eval.forward()?;
    let output_eval = d_eval.value()?.unwrap();

    // 训练模式应有 0 值
    let train_has_zeros = (0..100).any(|i| output_train[[0, i]] == 0.0);
    assert!(train_has_zeros, "训练模式应有元素被丢弃");

    // 评估模式应全为 1（无丢弃）
    for i in 0..100 {
        assert!(
            (output_eval[[0, i]] - 1.0).abs() < 1e-6,
            "评估模式输出应等于输入"
        );
    }

    Ok(())
}

// ==================== 4. 确定性测试（底层 API 指定 seed）====================

/// 辅助函数：用底层 API 创建带指定 seed 的 Dropout Var
fn create_dropout_with_seed(
    graph: &Graph,
    input: &Var,
    p: f32,
    seed: u64,
    name: Option<&str>,
) -> Result<Var, GraphError> {
    let inner = graph.inner_rc();
    let node =
        inner
            .borrow_mut()
            .create_dropout_node(std::rc::Rc::clone(input.node()), p, seed, name)?;
    Ok(Var::new_with_rc_graph(node, &inner))
}

/// 相同 seed → 相同结果
#[test]
fn test_dropout_deterministic_same_seed() -> Result<(), GraphError> {
    // 第一次
    let graph1 = Graph::new_with_seed(42);
    graph1.inner_mut().set_train_mode();
    let x1 = graph1.input(&Tensor::ones(&[1, 100]))?;
    let d1 = create_dropout_with_seed(&graph1, &x1, 0.5, 42, Some("d"))?;
    d1.forward()?;
    let output1 = d1.value()?.unwrap();

    // 第二次（相同 seed）
    let graph2 = Graph::new_with_seed(42);
    graph2.inner_mut().set_train_mode();
    let x2 = graph2.input(&Tensor::ones(&[1, 100]))?;
    let d2 = create_dropout_with_seed(&graph2, &x2, 0.5, 42, Some("d"))?;
    d2.forward()?;
    let output2 = d2.value()?.unwrap();

    // 应完全相同
    for i in 0..100 {
        assert!(
            (output1[[0, i]] - output2[[0, i]]).abs() < 1e-6,
            "相同 seed 应产生相同结果，位置 {i}: {} vs {}",
            output1[[0, i]],
            output2[[0, i]]
        );
    }

    Ok(())
}

/// 不同 seed → 不同结果
#[test]
fn test_dropout_deterministic_different_seed() -> Result<(), GraphError> {
    // seed = 42
    let graph1 = Graph::new_with_seed(42);
    graph1.inner_mut().set_train_mode();
    let x1 = graph1.input(&Tensor::ones(&[1, 100]))?;
    let d1 = create_dropout_with_seed(&graph1, &x1, 0.5, 42, Some("d"))?;
    d1.forward()?;
    let output1 = d1.value()?.unwrap();

    // seed = 123
    let graph2 = Graph::new_with_seed(123);
    graph2.inner_mut().set_train_mode();
    let x2 = graph2.input(&Tensor::ones(&[1, 100]))?;
    let d2 = create_dropout_with_seed(&graph2, &x2, 0.5, 123, Some("d"))?;
    d2.forward()?;
    let output2 = d2.value()?.unwrap();

    let found_diff = (0..100).any(|i| (output1[[0, i]] - output2[[0, i]]).abs() > 1e-6);
    assert!(found_diff, "不同 seed 应产生不同结果");

    Ok(())
}

// ==================== 5. 统计特性测试 ====================

/// 丢弃率大致符合 p
#[test]
fn test_dropout_drop_rate() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    graph.inner_mut().set_train_mode();

    let x = graph.input(&Tensor::ones(&[1, 10000]))?;
    let dropped = x.dropout(0.3)?;

    dropped.forward()?;
    let output = dropped.value()?.unwrap();

    let zeros_count = (0..10000).filter(|&i| output[[0, i]] == 0.0).count();
    let drop_rate = zeros_count as f64 / 10000.0;

    assert!(
        (drop_rate - 0.3).abs() < 0.05,
        "丢弃率 {drop_rate} 应接近 0.3"
    );

    Ok(())
}

/// Inverted Dropout 保持期望值不变
#[test]
fn test_dropout_expected_value_preserved() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    graph.inner_mut().set_train_mode();

    let x = graph.input(&Tensor::ones(&[1, 10000]))?;
    let dropped = x.dropout(0.5)?;

    dropped.forward()?;
    let output = dropped.value()?.unwrap();

    let sum: f64 = (0..10000).map(|i| output[[0, i]] as f64).sum();
    let mean = sum / 10000.0;

    // E[output] = (1-p) * (1/(1-p)) * input = input = 1.0
    assert!(
        (mean - 1.0).abs() < 0.1,
        "输出均值 {mean} 应接近 1.0（Inverted Dropout）"
    );

    Ok(())
}

/// p=0 时等价 identity
#[test]
fn test_dropout_p_zero_is_identity() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    graph.inner_mut().set_train_mode();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let x = graph.input(&input_data)?;
    let dropped = x.dropout(0.0)?;

    dropped.forward()?;
    let output = dropped.value()?.unwrap();

    for i in 0..2 {
        for j in 0..4 {
            assert!(
                (output[[i, j]] - input_data[[i, j]]).abs() < 1e-6,
                "p=0 时输出应等于输入"
            );
        }
    }

    Ok(())
}

// ==================== 6. 动态 batch + Create API（KEEP AS-IS）====================

use std::rc::Rc;

#[test]
fn test_create_dropout_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("input"))
        .unwrap();

    let dropout = inner
        .borrow_mut()
        .create_dropout_node(input.clone(), 0.5, 42, Some("dropout"))
        .unwrap();

    assert_eq!(dropout.shape(), vec![3, 4]);
    assert_eq!(dropout.name(), Some("dropout"));
}

#[test]
fn test_create_dropout_node_invalid_p() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], None)
        .unwrap();

    // p >= 1.0 无效
    let result = inner
        .borrow_mut()
        .create_dropout_node(input.clone(), 1.0, 42, None);
    assert!(result.is_err());

    // p < 0 无效
    let result2 = inner
        .borrow_mut()
        .create_dropout_node(input, -0.1, 42, None);
    assert!(result2.is_err());
}

#[test]
fn test_create_dropout_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_dropout;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let dropout = inner
            .borrow_mut()
            .create_dropout_node(input, 0.3, 42, None)
            .unwrap();
        weak_dropout = Rc::downgrade(&dropout);

        assert!(weak_dropout.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_dropout.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
