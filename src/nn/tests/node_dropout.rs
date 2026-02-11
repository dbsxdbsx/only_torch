/*
 * Dropout 节点单元测试
 *
 * 测试覆盖：
 * 1. 基本创建和参数验证
 * 2. 训练模式 vs 评估模式行为
 * 3. 前向传播（Inverted Dropout 缩放）
 * 4. 反向传播（梯度也需要 mask 和缩放）
 * 5. 确定性（相同 seed 产生相同结果）
 * 6. 动态形状支持
 */

use crate::nn::GraphError;
use crate::nn::graph::GraphInner;
use crate::tensor::Tensor;

// ==================== 基本创建测试 ====================

/// 测试默认 p=0.5 的 Dropout 创建
#[cfg(any())]
#[test]
fn test_dropout_create_default() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    let input = graph.new_basic_input_node(&[2, 4], Some("input"))?;
    let dropout = graph.new_dropout_node(input, 0.5, Some("dropout"))?;

    // 验证节点创建成功
    assert!(graph.get_node(dropout).is_ok());

    Ok(())
}

/// 测试自定义 p 的 Dropout 创建
#[cfg(any())]
#[test]
fn test_dropout_create_custom_p() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    let input = graph.new_basic_input_node(&[2, 4], Some("input"))?;

    // 测试不同的 p 值
    let _d1 = graph.new_dropout_node(input, 0.1, Some("d1"))?;
    let _d2 = graph.new_dropout_node(input, 0.3, Some("d2"))?;
    let _d3 = graph.new_dropout_node(input, 0.7, Some("d3"))?;

    // p=0.0 应该也可以创建（相当于 identity）
    let _d0 = graph.new_dropout_node(input, 0.0, Some("d0"))?;

    Ok(())
}

/// 测试无效 p 值应该报错
#[cfg(any())]
#[test]
fn test_dropout_invalid_p() {
    let mut graph = GraphInner::new_with_seed(42);
    let input = graph.new_basic_input_node(&[2, 4], Some("input")).unwrap();

    // p >= 1.0 应该失败
    assert!(graph.new_dropout_node(input, 1.0, None).is_err());
    assert!(graph.new_dropout_node(input, 1.5, None).is_err());

    // p < 0 应该失败
    assert!(graph.new_dropout_node(input, -0.1, None).is_err());
}

// ==================== 训练/评估模式测试 ====================

/// 测试：训练模式下输出应该有元素被置零
#[cfg(any())]
#[test]
fn test_dropout_train_mode_drops_elements() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 100], Some("input"))?;
    let dropout = graph.new_dropout_node(input, 0.5, Some("dropout"))?;

    // 设置输入为全 1
    let input_data = Tensor::ones(&[1, 100]);
    graph.set_node_value(input, Some(&input_data))?;

    // 前向传播
    graph.forward(dropout)?;

    let output = graph.get_node_value(dropout)?.unwrap();

    // 训练模式下，应该有一些元素被置零（或被缩放）
    // 由于 p=0.5，大约一半元素应该是 0，另一半是 2.0（1 / (1-0.5) = 2）
    let mut zeros_count = 0;
    let mut twos_count = 0;
    for i in 0..100 {
        let val = output[[0, i]];
        if val == 0.0 {
            zeros_count += 1;
        } else if (val - 2.0).abs() < 1e-6 {
            twos_count += 1;
        }
    }

    // 验证有元素被丢弃
    assert!(zeros_count > 0, "训练模式应该有元素被丢弃");
    // 验证保留的元素被缩放
    assert!(twos_count > 0, "保留的元素应该被缩放为 2.0");
    // 验证总数正确
    assert_eq!(zeros_count + twos_count, 100, "所有元素应该是 0 或 2.0");

    Ok(())
}

/// 测试：评估模式下输出应该等于输入
#[cfg(any())]
#[test]
fn test_dropout_eval_mode_passthrough() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_eval_mode(); // 评估模式

    let input = graph.new_basic_input_node(&[2, 4], Some("input"))?;
    let dropout = graph.new_dropout_node(input, 0.5, Some("dropout"))?;

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    graph.set_node_value(input, Some(&input_data))?;

    // 前向传播
    graph.forward(dropout)?;

    let output = graph.get_node_value(dropout)?.unwrap();

    // 评估模式下，输出应该完全等于输入
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

/// 测试：训练→评估模式切换后行为正确
#[cfg(any())]
#[test]
fn test_dropout_mode_switch_train_to_eval() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    let input = graph.new_basic_input_node(&[1, 100], Some("input"))?;
    let dropout = graph.new_dropout_node(input, 0.5, Some("dropout"))?;

    let input_data = Tensor::ones(&[1, 100]);
    graph.set_node_value(input, Some(&input_data))?;

    // 1. 训练模式
    graph.set_train_mode();
    graph.forward(dropout)?;
    let train_output = graph.get_node_value(dropout)?.unwrap().clone();

    // 训练模式应该有变化
    let mut has_zeros = false;
    for i in 0..100 {
        if train_output[[0, i]] == 0.0 {
            has_zeros = true;
            break;
        }
    }
    assert!(has_zeros, "训练模式应该有元素被丢弃");

    // 清除值以便重新计算
    graph.get_node_mut(dropout)?.clear_value()?;

    // 2. 评估模式
    graph.set_eval_mode();
    graph.forward(dropout)?;
    let eval_output = graph.get_node_value(dropout)?.unwrap();

    // 评估模式应该全是 1
    for i in 0..100 {
        assert!(
            (eval_output[[0, i]] - 1.0).abs() < 1e-6,
            "评估模式应该直接通过"
        );
    }

    Ok(())
}

/// 测试：评估→训练模式切换后行为正确
#[cfg(any())]
#[test]
fn test_dropout_mode_switch_eval_to_train() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    let input = graph.new_basic_input_node(&[1, 100], Some("input"))?;
    let dropout = graph.new_dropout_node(input, 0.5, Some("dropout"))?;

    let input_data = Tensor::ones(&[1, 100]);
    graph.set_node_value(input, Some(&input_data))?;

    // 1. 先评估模式
    graph.set_eval_mode();
    graph.forward(dropout)?;
    let eval_output = graph.get_node_value(dropout)?.unwrap().clone();

    // 评估模式应该全是 1（无丢弃）
    for i in 0..100 {
        assert!(
            (eval_output[[0, i]] - 1.0).abs() < 1e-6,
            "评估模式应该直接通过"
        );
    }

    // 清除值以便重新计算
    graph.get_node_mut(dropout)?.clear_value()?;

    // 2. 切换到训练模式
    graph.set_train_mode();
    graph.forward(dropout)?;
    let train_output = graph.get_node_value(dropout)?.unwrap();

    // 训练模式应该有丢弃
    let mut has_zeros = false;
    for i in 0..100 {
        if train_output[[0, i]] == 0.0 {
            has_zeros = true;
            break;
        }
    }
    assert!(has_zeros, "切换到训练模式后应该有元素被丢弃");

    Ok(())
}

/// 测试：训练模式下多次 forward 产生不同的 mask
#[cfg(any())]
#[test]
fn test_dropout_different_masks_each_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 100], Some("input"))?;
    let dropout = graph.new_dropout_node(input, 0.5, Some("dropout"))?;

    graph.set_node_value(input, Some(&Tensor::ones(&[1, 100])))?;

    // 第一次 forward
    graph.forward(dropout)?;
    let output1 = graph.get_node_value(dropout)?.unwrap().clone();

    // 清除值，重新 forward
    graph.get_node_mut(dropout)?.clear_value()?;
    graph.forward(dropout)?;
    let output2 = graph.get_node_value(dropout)?.unwrap();

    // 两次 forward 应该产生不同的 mask（概率极高）
    let mut found_diff = false;
    for i in 0..100 {
        if (output1[[0, i]] - output2[[0, i]]).abs() > 1e-6 {
            found_diff = true;
            break;
        }
    }
    assert!(found_diff, "训练模式下每次 forward 应该产生不同的随机 mask");

    Ok(())
}

// ==================== 反向传播测试 ====================

/// 测试：训练模式下反向传播正确（通过参数节点验证）
#[cfg(any())]
#[test]
fn test_dropout_backward_train_mode() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    // 使用 Parameter 节点作为输入，这样可以检查梯度
    // w -> dropout -> loss
    let w = graph.new_parameter_node(&[1, 10], Some("w"))?;
    graph.set_node_value(w, Some(&Tensor::ones(&[1, 10])))?;

    let dropout = graph.new_dropout_node(w, 0.5, Some("dropout"))?;
    let target = graph.new_basic_input_node(&[1, 10], Some("target"))?;
    let loss = graph.new_mse_loss_node(dropout, target, Some("loss"))?;

    // 设置 target
    graph.set_node_value(target, Some(&Tensor::zeros(&[1, 10])))?;

    // 前向 + 反向
    graph.forward(loss)?;
    graph.backward(loss)?;

    // 检查参数 w 的梯度
    let w_grad = graph.get_node(w)?.grad();
    assert!(w_grad.is_some(), "参数 w 应该有梯度");

    // 由于训练模式下 Dropout 会丢弃一些元素，
    // 梯度在被丢弃位置应该是 0
    let grad = w_grad.unwrap();
    let mut zeros_count = 0;
    for i in 0..10 {
        if grad[[0, i]] == 0.0 {
            zeros_count += 1;
        }
    }
    assert!(zeros_count > 0, "训练模式下梯度在被丢弃位置应该是 0");

    Ok(())
}

/// 测试：p=0 时梯度完全通过（无丢弃，相当于 identity）
#[cfg(any())]
#[test]
fn test_dropout_backward_p_zero_passthrough() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    // 使用 Parameter 节点
    let w = graph.new_parameter_node(&[1, 4], Some("w"))?;
    graph.set_node_value(w, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4])))?;

    // p=0 相当于 identity，梯度应该直接通过
    let dropout = graph.new_dropout_node(w, 0.0, Some("dropout"))?;
    let target = graph.new_basic_input_node(&[1, 4], Some("target"))?;
    let loss = graph.new_mse_loss_node(dropout, target, Some("loss"))?;

    // 设置 target
    graph.set_node_value(target, Some(&Tensor::zeros(&[1, 4])))?;

    // 前向 + 反向
    graph.forward(loss)?;
    graph.backward(loss)?;

    // 检查参数 w 的梯度
    let w_grad = graph.get_node(w)?.grad();
    assert!(w_grad.is_some(), "参数 w 应该有梯度");

    // p=0 时梯度应该直接通过
    // 梯度应该是 2*(pred - target)/N = 2*[1,2,3,4]/4 = [0.5, 1.0, 1.5, 2.0]
    let grad = w_grad.unwrap();
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

/// 测试：评估模式下前向传播无丢弃（与训练模式对比）
#[cfg(any())]
#[test]
fn test_dropout_eval_vs_train_same_input() -> Result<(), GraphError> {
    // 使用相同的 seed 创建两个 graph
    let mut graph_train = GraphInner::new_with_seed(42);
    let mut graph_eval = GraphInner::new_with_seed(42);

    graph_train.set_train_mode();
    graph_eval.set_eval_mode();

    // 创建相同结构
    let input_train = graph_train.new_basic_input_node(&[1, 100], Some("input"))?;
    let dropout_train = graph_train.new_dropout_node(input_train, 0.5, Some("dropout"))?;

    let input_eval = graph_eval.new_basic_input_node(&[1, 100], Some("input"))?;
    let dropout_eval = graph_eval.new_dropout_node(input_eval, 0.5, Some("dropout"))?;

    // 设置相同输入
    let input_data = Tensor::ones(&[1, 100]);
    graph_train.set_node_value(input_train, Some(&input_data))?;
    graph_eval.set_node_value(input_eval, Some(&input_data))?;

    // 前向传播
    graph_train.forward(dropout_train)?;
    graph_eval.forward(dropout_eval)?;

    let output_train = graph_train.get_node_value(dropout_train)?.unwrap();
    let output_eval = graph_eval.get_node_value(dropout_eval)?.unwrap();

    // 训练模式应该有 0 值
    let mut train_has_zeros = false;
    for i in 0..100 {
        if output_train[[0, i]] == 0.0 {
            train_has_zeros = true;
            break;
        }
    }
    assert!(train_has_zeros, "训练模式应该有元素被丢弃");

    // 评估模式应该全是 1（无 0 值）
    for i in 0..100 {
        assert!(
            (output_eval[[0, i]] - 1.0).abs() < 1e-6,
            "评估模式输出应等于输入，无丢弃"
        );
    }

    Ok(())
}

// ==================== 确定性测试 ====================

/// 测试：相同 seed 产生相同结果
#[cfg(any())]
#[test]
fn test_dropout_deterministic() -> Result<(), GraphError> {
    // 第一次运行
    let mut graph1 = GraphInner::new_with_seed(42);
    graph1.set_train_mode();
    let input1 = graph1.new_basic_input_node(&[1, 100], Some("input"))?;
    let dropout1 = graph1.new_dropout_node(input1, 0.5, Some("dropout"))?;
    graph1.set_node_value(input1, Some(&Tensor::ones(&[1, 100])))?;
    graph1.forward(dropout1)?;
    let output1 = graph1.get_node_value(dropout1)?.unwrap().clone();

    // 第二次运行（相同 seed）
    let mut graph2 = GraphInner::new_with_seed(42);
    graph2.set_train_mode();
    let input2 = graph2.new_basic_input_node(&[1, 100], Some("input"))?;
    let dropout2 = graph2.new_dropout_node(input2, 0.5, Some("dropout"))?;
    graph2.set_node_value(input2, Some(&Tensor::ones(&[1, 100])))?;
    graph2.forward(dropout2)?;
    let output2 = graph2.get_node_value(dropout2)?.unwrap();

    // 应该完全相同
    for i in 0..100 {
        assert!(
            (output1[[0, i]] - output2[[0, i]]).abs() < 1e-6,
            "相同 seed 应该产生相同结果"
        );
    }

    Ok(())
}

/// 测试：不同 seed 产生不同结果
#[cfg(any())]
#[test]
fn test_dropout_different_seeds() -> Result<(), GraphError> {
    // seed = 42
    let mut graph1 = GraphInner::new_with_seed(42);
    graph1.set_train_mode();
    let input1 = graph1.new_basic_input_node(&[1, 100], Some("input"))?;
    let dropout1 = graph1.new_dropout_node(input1, 0.5, Some("dropout"))?;
    graph1.set_node_value(input1, Some(&Tensor::ones(&[1, 100])))?;
    graph1.forward(dropout1)?;
    let output1 = graph1.get_node_value(dropout1)?.unwrap().clone();

    // seed = 123
    let mut graph2 = GraphInner::new_with_seed(123);
    graph2.set_train_mode();
    let input2 = graph2.new_basic_input_node(&[1, 100], Some("input"))?;
    let dropout2 = graph2.new_dropout_node(input2, 0.5, Some("dropout"))?;
    graph2.set_node_value(input2, Some(&Tensor::ones(&[1, 100])))?;
    graph2.forward(dropout2)?;
    let output2 = graph2.get_node_value(dropout2)?.unwrap();

    // 应该不同（找到至少一个不同的元素）
    let mut found_diff = false;
    for i in 0..100 {
        if (output1[[0, i]] - output2[[0, i]]).abs() > 1e-6 {
            found_diff = true;
            break;
        }
    }
    assert!(found_diff, "不同 seed 应该产生不同结果");

    Ok(())
}

// ==================== p=0 特殊情况测试 ====================

/// 测试：p=0 时相当于 identity
#[cfg(any())]
#[test]
fn test_dropout_p_zero() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[2, 4], Some("input"))?;
    let dropout = graph.new_dropout_node(input, 0.0, Some("dropout"))?;

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    graph.set_node_value(input, Some(&input_data))?;

    graph.forward(dropout)?;
    let output = graph.get_node_value(dropout)?.unwrap();

    // p=0 时输出应该等于输入
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

// ==================== 动态形状测试 ====================

/// 测试：支持不同 batch size
#[cfg(any())]
#[test]
fn test_dropout_dynamic_batch() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[2, 4], Some("input"))?;
    let dropout = graph.new_dropout_node(input, 0.5, Some("dropout"))?;

    // batch=2
    graph.set_node_value(input, Some(&Tensor::ones(&[2, 4])))?;
    graph.forward(dropout)?;
    let output1 = graph.get_node_value(dropout)?.unwrap();
    assert_eq!(output1.shape(), &[2, 4]);

    // 清除并用不同 batch
    graph.get_node_mut(dropout)?.clear_value()?;
    graph.set_node_value(input, Some(&Tensor::ones(&[5, 4])))?;
    graph.forward(dropout)?;
    let output2 = graph.get_node_value(dropout)?.unwrap();
    assert_eq!(output2.shape(), &[5, 4]);

    Ok(())
}

// ==================== 统计特性测试 ====================

/// 测试：丢弃率大致符合 p
#[cfg(any())]
#[test]
fn test_dropout_drop_rate() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 10000], Some("input"))?;
    let dropout = graph.new_dropout_node(input, 0.3, Some("dropout"))?;

    graph.set_node_value(input, Some(&Tensor::ones(&[1, 10000])))?;
    graph.forward(dropout)?;

    let output = graph.get_node_value(dropout)?.unwrap();
    let mut zeros_count = 0;
    for i in 0..10000 {
        if output[[0, i]] == 0.0 {
            zeros_count += 1;
        }
    }
    let drop_rate = zeros_count as f64 / 10000.0;

    // 丢弃率应该大致等于 0.3（允许一定误差）
    assert!(
        (drop_rate - 0.3).abs() < 0.05,
        "丢弃率 {drop_rate} 应该接近 0.3"
    );

    Ok(())
}

/// 测试：保留元素的期望值不变（Inverted Dropout 特性）
#[cfg(any())]
#[test]
fn test_dropout_expected_value_preserved() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    graph.set_train_mode();

    let input = graph.new_basic_input_node(&[1, 10000], Some("input"))?;
    let dropout = graph.new_dropout_node(input, 0.5, Some("dropout"))?;

    // 输入全为 1.0
    graph.set_node_value(input, Some(&Tensor::ones(&[1, 10000])))?;
    graph.forward(dropout)?;

    let output = graph.get_node_value(dropout)?.unwrap();
    let mut sum = 0.0;
    for i in 0..10000 {
        sum += output[[0, i]];
    }
    let mean = sum / 10000.0;

    // 由于 Inverted Dropout，期望值应该接近 1.0
    // E[output] = (1-p) * (1/(1-p)) * input = input
    assert!(
        (mean - 1.0).abs() < 0.1,
        "输出均值 {mean} 应该接近输入均值 1.0（Inverted Dropout 特性）"
    );

    Ok(())
}

// ==================== 方案 C：新节点创建 API 测试 ====================

use crate::nn::Graph;
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
