/*
 * @Author       : 老董
 * @Description  : BCE（Binary Cross Entropy with Logits）损失节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ basic 0.4205; 2D 0.2201; 数值稳定性; 多标签; cannot_set_value
 * 2. VJP 单元测试（底层）→ Mean VJP; Sum VJP
 * 3. 端到端反向传播测试（高层）→ mean/sum/2D/batch
 * 4. 梯度累积测试
 * 5. 动态形状测试
 * 6. 新节点创建 API 测试（KEEP AS-IS）
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试（高层 Graph + Var API）====================

/// PyTorch 验证:
/// ```python
/// import torch
/// import torch.nn as nn
///
/// logits = torch.tensor([[0.5, -0.5, 1.0]], requires_grad=True)
/// target = torch.tensor([[1.0, 0.0, 1.0]])
/// loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, target)
/// print(f"loss = {loss.item()}")  # loss = 0.4204719067
/// ```
#[test]
fn test_bce_loss_forward_mean_basic() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(&[0.5, -0.5, 1.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.0, 0.0, 1.0], &[1, 3]))
        .unwrap();
    let loss = logits.bce_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.item().unwrap();
    // PyTorch: 0.4204719067
    assert_abs_diff_eq!(loss_val, 0.420_471_9, epsilon = 1e-5);
}

/// PyTorch 验证:
/// ```python
/// logits = torch.tensor([[1.0, 2.0], [-1.0, -2.0]], requires_grad=True)
/// target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
/// loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, target)
/// print(f"loss = {loss.item()}")  # loss = 0.2200948894
/// ```
#[test]
fn test_bce_loss_forward_2d_matrix() {
    let graph = Graph::new();

    #[rustfmt::skip]
    let logits = graph
        .input(&Tensor::new(&[1.0, 2.0, -1.0, -2.0], &[2, 2]))
        .unwrap();
    #[rustfmt::skip]
    let target = graph
        .input(&Tensor::new(&[1.0, 1.0, 0.0, 0.0], &[2, 2]))
        .unwrap();
    let loss = logits.bce_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.item().unwrap();
    // PyTorch: 0.2200948894
    assert_abs_diff_eq!(loss_val, 0.220_094_9, epsilon = 1e-5);
}

/// 测试大正数 logits 的数值稳定性
/// 大正数 logits 时 sigmoid ≈ 1，BCE 应该趋近于 0（当 target=1）
#[test]
fn test_bce_loss_large_positive_logits() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(&[10.0, 20.0, 30.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.0, 1.0, 1.0], &[1, 3]))
        .unwrap();
    let loss = logits.bce_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.item().unwrap();
    assert!(
        loss_val < 1e-4,
        "大正数 logits 配合 target=1 时损失应接近 0，实际: {}",
        loss_val
    );
    assert!(loss_val >= 0.0, "损失不应为负");
}

/// 测试大负数 logits 的数值稳定性
/// 大负数 logits 时 sigmoid ≈ 0，BCE 应该趋近于 0（当 target=0）
#[test]
fn test_bce_loss_large_negative_logits() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(&[-10.0, -20.0, -30.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3]))
        .unwrap();
    let loss = logits.bce_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.item().unwrap();
    assert!(
        loss_val < 1e-4,
        "大负数 logits 配合 target=0 时损失应接近 0，实际: {}",
        loss_val
    );
    assert!(loss_val >= 0.0, "损失不应为负");
}

/// 测试错误预测时的高损失
#[test]
fn test_bce_loss_wrong_prediction_high_loss() {
    let graph = Graph::new();

    // 大正数预测 0，大负数预测 1（完全错误）
    let logits = graph.input(&Tensor::new(&[10.0, -10.0], &[1, 2])).unwrap();
    let target = graph.input(&Tensor::new(&[0.0, 1.0], &[1, 2])).unwrap();
    let loss = logits.bce_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.item().unwrap();
    assert!(loss_val > 5.0, "错误预测时损失应该很大，实际: {}", loss_val);
}

/// 测试多标签分类场景 — BCE 相对于 Softmax CE 的核心优势
#[test]
fn test_bce_loss_multi_label_classification() {
    let graph = Graph::new();

    // 预测: [高概率, 高概率, 低概率]
    // 标签: [1, 1, 0] — 同时属于前两个类别
    let logits = graph
        .input(&Tensor::new(&[2.0, 2.0, -2.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.0, 1.0, 0.0], &[1, 3]))
        .unwrap();
    let loss = logits.bce_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.item().unwrap();
    assert!(loss_val < 0.2, "正确预测时损失应该很小，实际: {}", loss_val);
}

/// 测试 BCE loss 节点不能直接设置值
#[test]
fn test_bce_loss_cannot_set_value() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(&[0.5, -0.5, 1.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.0, 0.0, 1.0], &[1, 3]))
        .unwrap();
    let loss = logits.bce_loss(&target).unwrap();

    let err = loss.set_value(&Tensor::new(&[0.0], &[1, 1]));
    assert!(err.is_err(), "BCE loss 节点不应支持直接设值");
}

// ==================== 2. VJP 单元测试（底层 NodeInner + calc_grad_to_parent_index）====================
//
// BCE with logits 梯度公式:
//   Mean: grad = (sigmoid(logits) - target) / N
//   Sum:  grad = sigmoid(logits) - target

/// BCE Mean VJP 测试
///
/// logits = [0.5, -0.5, 1.0], target = [1.0, 0.0, 1.0]
/// sigmoid([0.5, -0.5, 1.0]) ≈ [0.6225, 0.3775, 0.7311]
/// (sigmoid - target) / 3 = [-0.1258, 0.1258, -0.0896]
#[test]
fn test_bce_vjp_mean() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("logits"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("target"))
        .unwrap();
    let bce = inner
        .borrow_mut()
        .create_bce_mean_node(logits.clone(), target.clone(), Some("bce"))
        .unwrap();

    logits
        .set_value(Some(&Tensor::new(&[0.5, -0.5, 1.0], &[1, 3])))
        .unwrap();
    target
        .set_value(Some(&Tensor::new(&[1.0, 0.0, 1.0], &[1, 3])))
        .unwrap();
    bce.forward_recursive(1, false).unwrap();

    // 上游梯度为标量 1.0（loss 输出 [1,1]）
    let upstream = Tensor::ones(&[1, 1]);
    let grad = bce
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    // PyTorch 验证: [-0.125_846_9, 0.125_846_9, -0.089_647_1]
    assert_eq!(grad.shape(), &[1, 3]);
    assert_abs_diff_eq!(grad[[0, 0]], -0.125_846_9, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 0.125_846_9, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 2]], -0.089_647_1, epsilon = 1e-5);

    Ok(())
}

/// BCE Sum VJP 测试
///
/// PyTorch 验证:
/// ```python
/// logits = torch.tensor([[0.5, -0.5, 1.0]], requires_grad=True)
/// target = torch.tensor([[1.0, 0.0, 1.0]])
/// loss = nn.BCEWithLogitsLoss(reduction='sum')(logits, target)
/// print(f"loss = {loss.item()}")  # loss = 1.2614157200
/// loss.backward()
/// print(f"grad = {logits.grad}")  # [-0.3775, 0.3775, -0.2689]
/// ```
#[test]
fn test_bce_vjp_sum() -> Result<(), GraphError> {
    use crate::nn::nodes::raw_node::Reduction;

    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("logits"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("target"))
        .unwrap();
    let bce = inner
        .borrow_mut()
        .create_bce_node(logits.clone(), target.clone(), Reduction::Sum, Some("bce"))
        .unwrap();

    logits
        .set_value(Some(&Tensor::new(&[0.5, -0.5, 1.0], &[1, 3])))
        .unwrap();
    target
        .set_value(Some(&Tensor::new(&[1.0, 0.0, 1.0], &[1, 3])))
        .unwrap();
    bce.forward_recursive(1, false).unwrap();

    // 验证 Sum 前向值
    let bce_val = bce.value().unwrap();
    assert_abs_diff_eq!(bce_val[[0, 0]], 1.261_415_7, epsilon = 1e-5);

    // VJP
    let upstream = Tensor::ones(&[1, 1]);
    let grad = bce
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    // sigmoid - target（不除以 N）
    assert_eq!(grad.shape(), &[1, 3]);
    assert_abs_diff_eq!(grad[[0, 0]], -0.377_540_7, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 0.377_540_7, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 2]], -0.268_941_4, epsilon = 1e-5);

    Ok(())
}

// ==================== 3. 端到端反向传播测试（高层 Graph + Var API）====================

/// PyTorch 验证:
/// ```python
/// logits = torch.tensor([[0.5, -0.5, 1.0]], requires_grad=True)
/// target = torch.tensor([[1.0, 0.0, 1.0]])
/// loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, target)
/// loss.backward()
/// print(f"grad = {logits.grad}")
/// # grad = tensor([[-0.1258,  0.1258, -0.0896]])
/// ```
#[test]
fn test_bce_loss_backward_e2e_mean() -> Result<(), GraphError> {
    let graph = Graph::new();

    let logits = graph.parameter(&[1, 3], Init::Zeros, "logits")?;
    logits.set_value(&Tensor::new(&[0.5, -0.5, 1.0], &[1, 3]))?;

    let target = graph.input(&Tensor::new(&[1.0, 0.0, 1.0], &[1, 3]))?;
    let loss = logits.bce_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    let grad = logits.grad()?.expect("logits 应有 grad");
    let expected_grad = Tensor::new(&[-0.125_846_9, 0.125_846_9, -0.089_647_1], &[1, 3]);
    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);

    Ok(())
}

/// 2D 矩阵的端到端反向传播
///
/// PyTorch 验证:
/// ```python
/// logits = torch.tensor([[1.0, 2.0], [-1.0, -2.0]], requires_grad=True)
/// target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
/// loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, target)
/// loss.backward()
/// print(f"grad = {logits.grad}")
/// # grad = tensor([[-0.0672, -0.0298],
/// #                [ 0.0672,  0.0298]])
/// ```
#[test]
fn test_bce_loss_backward_e2e_2d() -> Result<(), GraphError> {
    let graph = Graph::new();

    let logits = graph.parameter(&[2, 2], Init::Zeros, "logits")?;
    #[rustfmt::skip]
    logits.set_value(&Tensor::new(&[1.0, 2.0, -1.0, -2.0], &[2, 2]))?;

    #[rustfmt::skip]
    let target = graph.input(&Tensor::new(&[1.0, 1.0, 0.0, 0.0], &[2, 2]))?;
    let loss = logits.bce_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    let grad = logits.grad()?.expect("logits 应有 grad");
    #[rustfmt::skip]
    let expected_grad = Tensor::new(&[
        -0.067_235_4, -0.029_800_7,
         0.067_235_4,  0.029_800_7,
    ], &[2, 2]);
    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);

    Ok(())
}

/// 批量输入的端到端反向传播
///
/// PyTorch 验证:
/// ```python
/// logits = torch.tensor([
///     [0.5, -0.5, 1.0, -1.0],
///     [1.5, -1.5, 2.0, -2.0],
///     [0.0,  0.0, 0.5, -0.5]
/// ], requires_grad=True)
/// target = torch.tensor([
///     [1.0, 0.0, 1.0, 0.0],
///     [1.0, 0.0, 1.0, 0.0],
///     [0.0, 1.0, 0.0, 1.0]
/// ])
/// loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, target)
/// print(f"loss = {loss.item()}")  # loss = 0.4638172686
/// loss.backward()
/// ```
#[test]
fn test_bce_loss_backward_e2e_batch() -> Result<(), GraphError> {
    let graph = Graph::new();

    let logits = graph.parameter(&[3, 4], Init::Zeros, "logits")?;
    #[rustfmt::skip]
    logits.set_value(&Tensor::new(&[
        0.5, -0.5, 1.0, -1.0,
        1.5, -1.5, 2.0, -2.0,
        0.0,  0.0, 0.5, -0.5,
    ], &[3, 4]))?;

    #[rustfmt::skip]
    let target = graph.input(&Tensor::new(&[
        1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
    ], &[3, 4]))?;

    let loss = logits.bce_loss(&target)?;

    // 验证前向传播
    loss.forward().unwrap();
    let loss_val = loss.item().unwrap();
    assert_abs_diff_eq!(loss_val, 0.463_817_3, epsilon = 1e-5);

    // 反向传播
    graph.zero_grad()?;
    loss.backward()?;

    let grad = logits.grad()?.expect("logits 应有 grad");
    assert_eq!(grad.shape(), &[3, 4]);
    // 梯度值应该在合理范围内
    for &val in grad.flatten_view().iter() {
        assert!(val.abs() < 1.0, "梯度值应该在合理范围内");
    }

    Ok(())
}

/// 简单的二分类训练测试（端到端）
/// 目标: 学习一个线性分类器 y = sigmoid(w * x)
#[test]
fn test_bce_loss_simple_binary_classification_training() -> Result<(), GraphError> {
    let graph = Graph::new();

    // 创建 w 参数
    let w = graph.parameter(&[1, 1], Init::Zeros, "w")?;
    w.set_value(&Tensor::new(&[0.0], &[1, 1]))?;

    // 训练数据: x > 0 -> y = 1, x < 0 -> y = 0
    let training_data = [
        (1.0_f32, 1.0_f32),
        (2.0, 1.0),
        (3.0, 1.0),
        (-1.0, 0.0),
        (-2.0, 0.0),
        (-3.0, 0.0),
    ];

    let lr = 0.5;

    for _ in 0..100 {
        for &(x_val, y_val) in &training_data {
            let x = graph.input(&Tensor::new(&[x_val], &[1, 1]))?;
            let logit = x.matmul(&w)?;
            let target = graph.input(&Tensor::new(&[y_val], &[1, 1]))?;
            let loss = logit.bce_loss(&target)?;

            graph.zero_grad()?;
            loss.backward()?;

            // 手动 SGD 更新
            let w_val = w.value()?.unwrap();
            let w_grad = w.grad()?.unwrap();
            w.set_value(&(w_val - lr * &w_grad))?;
        }
    }

    let learned_w = w.item().unwrap();
    assert!(
        learned_w > 0.5,
        "学习到的权重应为正数（实际: {}）",
        learned_w
    );

    Ok(())
}

// ==================== 4. 梯度累积测试（高层 Graph + Var API）====================

/// 测试 BCE 梯度累积
///
/// 验证语义：参数的 grad 在多次 backward 之间累积，直到调用 zero_grad()。
#[test]
fn test_bce_loss_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let logits = graph.parameter(&[1, 3], Init::Zeros, "logits")?;
    logits.set_value(&Tensor::new(&[0.5, -0.5, 1.0], &[1, 3]))?;

    let target = graph.input(&Tensor::new(&[1.0, 0.0, 1.0], &[1, 3]))?;
    let loss = logits.bce_loss(&target)?;

    // 第 1 次反向传播
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = logits.grad()?.unwrap().clone();

    // 第 2 次反向传播（梯度累积）
    loss.backward()?;
    let grad_second = logits.grad()?.unwrap();
    assert_abs_diff_eq!(grad_second, &(&grad_first * 2.0), epsilon = 1e-6);

    // zero_grad 后重新计算
    graph.zero_grad()?;
    loss.backward()?;
    let grad_after_clear = logits.grad()?.unwrap();
    assert_abs_diff_eq!(grad_after_clear, &grad_first, epsilon = 1e-6);

    Ok(())
}

// ==================== 5. 动态形状测试 ====================

/// 测试 BCE 节点的动态形状（输出固定为标量 [1, 1]）
#[test]
fn test_bce_loss_dynamic_shape_output_fixed() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], Some("logits"))?;
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], Some("target"))?;
    let bce = inner
        .borrow_mut()
        .create_bce_mean_node(logits, target, Some("bce"))?;

    // BCE 输出形状固定为 [1, 1]
    let dyn_shape = bce.dynamic_shape();
    assert!(!dyn_shape.is_dynamic(0), "BCE 输出维度 0 应固定");
    assert!(!dyn_shape.is_dynamic(1), "BCE 输出维度 1 应固定");
    assert_eq!(dyn_shape.dim(0), Some(1));
    assert_eq!(dyn_shape.dim(1), Some(1));

    Ok(())
}

/// 测试 BCE 接受动态 batch 输入
#[test]
fn test_bce_loss_dynamic_batch_forward() -> Result<(), GraphError> {
    let graph = Graph::new();

    // 第一次 forward：batch=2
    let logits = graph.input(&Tensor::ones(&[2, 4]))?;
    let target = graph.input(&Tensor::ones(&[2, 4]))?;
    let loss = logits.bce_loss(&target)?;

    loss.forward()?;
    let loss_val1 = loss.value()?.unwrap();
    assert_eq!(loss_val1.shape(), &[1, 1], "BCE 输出应为标量");

    // 第二次 forward：batch=6（不同 batch 大小）
    logits.set_value(&Tensor::ones(&[6, 4]))?;
    target.set_value(&Tensor::ones(&[6, 4]))?;

    loss.forward()?;
    let loss_val2 = loss.value()?.unwrap();
    assert_eq!(loss_val2.shape(), &[1, 1], "BCE 输出应始终为标量");

    Ok(())
}

// ==================== 6. 节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_bce_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("logits"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("target"))
        .unwrap();

    let bce = inner
        .borrow_mut()
        .create_bce_mean_node(logits.clone(), target.clone(), Some("bce"))
        .unwrap();

    // BCE 输出形状固定为 [1, 1]
    assert_eq!(bce.shape(), vec![1, 1]);
    assert_eq!(bce.name(), Some("bce"));
    assert!(!bce.is_leaf());
    assert_eq!(bce.parents().len(), 2);
}

#[test]
fn test_create_bce_node_shape_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], None) // 形状不匹配
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_bce_mean_node(logits, target, None);

    assert!(result.is_err());
}

#[test]
fn test_create_bce_node_output_always_scalar() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 10], None)
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 10], None)
        .unwrap();
    let bce = inner
        .borrow_mut()
        .create_bce_mean_node(logits, target, None)
        .unwrap();
    assert_eq!(bce.shape(), vec![1, 1]);
}

#[test]
fn test_create_bce_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_bce;
    let weak_logits;
    let weak_target;
    {
        let logits = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_logits = Rc::downgrade(&logits);

        let target = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_target = Rc::downgrade(&target);

        let bce = inner
            .borrow_mut()
            .create_bce_mean_node(logits, target, None)
            .unwrap();
        weak_bce = Rc::downgrade(&bce);

        assert!(weak_bce.upgrade().is_some());
        assert!(weak_logits.upgrade().is_some());
        assert!(weak_target.upgrade().is_some());
    }
    assert!(weak_bce.upgrade().is_none());
    assert!(weak_logits.upgrade().is_none());
    assert!(weak_target.upgrade().is_none());
}
