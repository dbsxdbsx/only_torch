/*
 * @Author       : 老董
 * @Description  : SoftmaxCrossEntropy 损失节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ simple 0.4076; uniform 1.3863; 10 classes; cannot_set_value
 * 2. VJP 单元测试（底层）→ simple grad; uniform grad; 10 classes grad
 * 3. 端到端反向传播测试（高层）→ 含线性层的链式网络
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
/// logits = [1.0, 2.0, 3.0], labels = [0, 0, 1]
/// softmax = [0.09003057, 0.24472848, 0.66524094]
/// loss = 0.40760597
/// ```
#[test]
fn test_sce_forward_simple() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let labels = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0], &[1, 3]))
        .unwrap();
    let loss = logits.cross_entropy(&labels).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.item().unwrap();
    assert_abs_diff_eq!(loss_val, 0.40760597, epsilon = 1e-5);
}

/// PyTorch 验证:
/// ```python
/// logits = [1.0, 1.0, 1.0, 1.0], labels = [0, 1, 0, 0]
/// softmax = [0.25, 0.25, 0.25, 0.25]
/// loss = 1.3862944 (= -ln(0.25))
/// ```
#[test]
fn test_sce_forward_uniform() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 4]))
        .unwrap();
    let labels = graph
        .input(&Tensor::new(&[0.0, 1.0, 0.0, 0.0], &[1, 4]))
        .unwrap();
    let loss = logits.cross_entropy(&labels).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.item().unwrap();
    assert_abs_diff_eq!(loss_val, 1.3862944, epsilon = 1e-5);
}

/// 10 类分类前向传播
///
/// PyTorch 验证:
/// ```python
/// logits = [0.5, 1.5, -0.5, 2.0, 0.0, -1.0, 1.0, 0.5, -0.5, 1.5]
/// labels = [0,0,0,1,0,0,0,0,0,0] (类别 3)
/// loss = 1.2168376
/// ```
#[test]
fn test_sce_forward_10_classes() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(
            &[0.5, 1.5, -0.5, 2.0, 0.0, -1.0, 1.0, 0.5, -0.5, 1.5],
            &[1, 10],
        ))
        .unwrap();
    let mut labels_data = vec![0.0; 10];
    labels_data[3] = 1.0;
    let labels = graph.input(&Tensor::new(&labels_data, &[1, 10])).unwrap();
    let loss = logits.cross_entropy(&labels).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.item().unwrap();
    assert_abs_diff_eq!(loss_val, 1.2168376, epsilon = 1e-5);
}

/// SoftmaxCrossEntropy 损失节点不能直接设置值
#[test]
fn test_sce_cannot_set_value() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let labels = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0], &[1, 3]))
        .unwrap();
    let loss = logits.cross_entropy(&labels).unwrap();

    let err = loss.set_value(&Tensor::new(&[0.0], &[1, 1]));
    assert!(err.is_err(), "SCE 损失节点不应支持直接设值");
}

// ==================== 2. VJP 单元测试（底层 calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证梯度计算公式。
// SCE 梯度 = softmax(logits) - labels

/// PyTorch 验证:
/// ```python
/// logits = [1.0, 2.0, 3.0], labels = [0, 0, 1]
/// grad = [0.09003057, 0.24472848, -0.33475903]
/// ```
#[test]
fn test_sce_vjp_simple() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("logits"))
        .unwrap();
    let labels = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("labels"))
        .unwrap();
    let sce = inner
        .borrow_mut()
        .create_softmax_cross_entropy_node(logits.clone(), labels.clone(), Some("sce"))
        .unwrap();

    logits
        .set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    labels
        .set_value(Some(&Tensor::new(&[0.0, 0.0, 1.0], &[1, 3])))
        .unwrap();
    sce.forward_recursive(1, false).unwrap();

    let upstream = Tensor::ones(&[1, 1]);
    let grad = sce.calc_grad_to_parent_index(0, &upstream)?.resolve(&upstream);

    assert_eq!(grad.shape(), &[1, 3]);
    let expected = Tensor::new(&[0.09003057, 0.24472848, -0.33475903], &[1, 3]);
    assert_abs_diff_eq!(&grad, &expected, epsilon = 1e-5);

    Ok(())
}

/// PyTorch 验证:
/// ```python
/// logits = [1.0, 1.0, 1.0, 1.0], labels = [0, 1, 0, 0]
/// grad = [0.25, -0.75, 0.25, 0.25]
/// ```
#[test]
fn test_sce_vjp_uniform() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("logits"))
        .unwrap();
    let labels = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("labels"))
        .unwrap();
    let sce = inner
        .borrow_mut()
        .create_softmax_cross_entropy_node(logits.clone(), labels.clone(), Some("sce"))
        .unwrap();

    logits
        .set_value(Some(&Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 4])))
        .unwrap();
    labels
        .set_value(Some(&Tensor::new(&[0.0, 1.0, 0.0, 0.0], &[1, 4])))
        .unwrap();
    sce.forward_recursive(1, false).unwrap();

    let upstream = Tensor::ones(&[1, 1]);
    let grad = sce.calc_grad_to_parent_index(0, &upstream)?.resolve(&upstream);

    let expected = Tensor::new(&[0.25, -0.75, 0.25, 0.25], &[1, 4]);
    assert_abs_diff_eq!(&grad, &expected, epsilon = 1e-5);

    Ok(())
}

/// 10 类分类 VJP
///
/// PyTorch 验证:
/// ```python
/// logits = [0.5, 1.5, -0.5, 2.0, 0.0, -1.0, 1.0, 0.5, -0.5, 1.5]
/// labels = [0,0,0,1,0,0,0,0,0,0]
/// grad = [0.0660834, 0.1796333, 0.02431072, -0.7038347, 0.04008161,
///          0.0147452, 0.10895311, 0.0660834, 0.02431072, 0.1796333]
/// ```
#[test]
fn test_sce_vjp_10_classes() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 10], Some("logits"))
        .unwrap();
    let labels = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 10], Some("labels"))
        .unwrap();
    let sce = inner
        .borrow_mut()
        .create_softmax_cross_entropy_node(logits.clone(), labels.clone(), Some("sce"))
        .unwrap();

    logits
        .set_value(Some(&Tensor::new(
            &[0.5, 1.5, -0.5, 2.0, 0.0, -1.0, 1.0, 0.5, -0.5, 1.5],
            &[1, 10],
        )))
        .unwrap();
    let mut labels_data = vec![0.0; 10];
    labels_data[3] = 1.0;
    labels
        .set_value(Some(&Tensor::new(&labels_data, &[1, 10])))
        .unwrap();
    sce.forward_recursive(1, false).unwrap();

    // 验证前向值
    let loss_val = sce.value().unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], 1.2168376, epsilon = 1e-5);

    // VJP
    let upstream = Tensor::ones(&[1, 1]);
    let grad = sce.calc_grad_to_parent_index(0, &upstream)?.resolve(&upstream);

    #[rustfmt::skip]
    let expected = Tensor::new(
        &[0.0660834, 0.1796333, 0.02431072, -0.7038347, 0.04008161,
          0.0147452, 0.10895311, 0.0660834, 0.02431072, 0.1796333],
        &[1, 10]
    );
    assert_abs_diff_eq!(&grad, &expected, epsilon = 1e-5);

    Ok(())
}

// ==================== 3. 端到端反向传播测试（高层 Graph + Var API）====================

/// 简单反向传播：验证高层 API 梯度与 VJP 一致
#[test]
fn test_sce_backward_e2e_simple() {
    let graph = Graph::new();

    let logits = graph.parameter(&[1, 3], Init::Zeros, "logits").unwrap();
    logits
        .set_value(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let labels = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0], &[1, 3]))
        .unwrap();
    let loss = logits.cross_entropy(&labels).unwrap();

    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    let grad = logits.grad().unwrap().unwrap();
    let expected = Tensor::new(&[0.09003057, 0.24472848, -0.33475903], &[1, 3]);
    assert_abs_diff_eq!(&grad, &expected, epsilon = 1e-5);
}

/// 含线性层的链式网络反向传播
///
/// input [1, 2] -> matmul(weights [2, 3]) -> + bias [1, 3] -> cross_entropy -> loss
#[test]
fn test_sce_backward_with_linear_layer() {
    let graph = Graph::new();

    // 输入
    let input = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();

    // 权重和偏置（可训练参数）
    let weights = graph.parameter(&[2, 3], Init::Zeros, "weights").unwrap();
    weights
        .set_value(&Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]))
        .unwrap();
    let bias = graph.parameter(&[1, 3], Init::Zeros, "bias").unwrap();
    // bias 保持零初始化

    // input @ weights + bias -> logits
    let logits = input.matmul(&weights).unwrap() + &bias;

    // labels
    let labels = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0], &[1, 3]))
        .unwrap();
    let loss = logits.cross_entropy(&labels).unwrap();

    // 反向传播
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // 验证权重有梯度且形状正确
    let weights_grad = weights.grad().unwrap().unwrap();
    assert_eq!(weights_grad.shape(), &[2, 3]);

    // 验证偏置有梯度且形状正确
    let bias_grad = bias.grad().unwrap().unwrap();
    assert_eq!(bias_grad.shape(), &[1, 3]);
}

// ==================== 4. 梯度累积测试 ====================

#[test]
fn test_sce_gradient_accumulation() {
    let graph = Graph::new();

    let logits = graph.parameter(&[1, 3], Init::Zeros, "logits").unwrap();
    logits
        .set_value(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let labels = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0], &[1, 3]))
        .unwrap();
    let loss = logits.cross_entropy(&labels).unwrap();

    // 第一次前向+反向
    graph.zero_grad().unwrap();
    loss.backward().unwrap();
    let grad_first = logits.grad().unwrap().unwrap().clone();

    // 第二次反向传播（梯度累积，不 zero_grad）
    loss.backward().unwrap();
    let grad_second = logits.grad().unwrap().unwrap();
    assert_abs_diff_eq!(&grad_second, &(&grad_first * 2.0), epsilon = 1e-6);

    // zero_grad 后重新计算
    graph.zero_grad().unwrap();
    loss.backward().unwrap();
    let grad_after_clear = logits.grad().unwrap().unwrap();
    assert_abs_diff_eq!(&grad_after_clear, &grad_first, epsilon = 1e-6);
}

// ==================== 5. 动态形状测试 ====================

/// 测试 SoftmaxCrossEntropy 节点的动态形状传播
/// Loss 输出固定为 [1, 1]
#[test]
fn test_sce_dynamic_shape_propagation() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("logits"))
        .unwrap();
    let labels = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("labels"))
        .unwrap();
    let sce = inner
        .borrow_mut()
        .create_softmax_cross_entropy_node(logits, labels, Some("sce"))
        .unwrap();

    // Loss 输出固定为 [1, 1]
    assert_eq!(sce.shape(), vec![1, 1]);

    Ok(())
}

/// 测试 SoftmaxCrossEntropy 在不同 batch_size 下的前向计算
#[test]
fn test_sce_dynamic_batch_forward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("logits"))
        .unwrap();
    let labels = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("labels"))
        .unwrap();
    let sce = inner
        .borrow_mut()
        .create_softmax_cross_entropy_node(logits.clone(), labels.clone(), Some("sce"))
        .unwrap();

    // batch=2
    let mut labels_data = Tensor::zeros(&[2, 3]);
    labels_data[[0, 0]] = 1.0;
    labels_data[[1, 2]] = 1.0;
    logits
        .set_value(Some(&Tensor::normal_seeded(0.0, 1.0, &[2, 3], 42)))
        .unwrap();
    labels.set_value(Some(&labels_data)).unwrap();
    sce.forward_recursive(1, false).unwrap();
    let val1 = sce.value().unwrap();
    assert_eq!(val1.shape(), &[1, 1], "Loss 输出固定为 [1, 1]");
    assert!(val1[[0, 0]] > 0.0);

    // batch=5（不同 batch 大小）
    let mut labels_data2 = Tensor::zeros(&[5, 3]);
    for i in 0..5 {
        labels_data2[[i, i % 3]] = 1.0;
    }
    logits
        .set_value(Some(&Tensor::normal_seeded(0.0, 1.0, &[5, 3], 100)))
        .unwrap();
    labels.set_value(Some(&labels_data2)).unwrap();
    sce.forward_recursive(2, false).unwrap();
    let val2 = sce.value().unwrap();
    assert_eq!(val2.shape(), &[1, 1], "Loss 输出固定为 [1, 1]");
    assert!(val2[[0, 0]] > 0.0);

    Ok(())
}

/// 测试 SoftmaxCrossEntropy 在不同 batch_size 下的反向传播
///
/// 使用 backward_via_node_inner 确保中间节点梯度被正确清除，
/// 避免动态 batch 形状切换时的梯度形状冲突。
#[test]
fn test_sce_dynamic_batch_backward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // input -> matmul(weight) -> logits -> cross_entropy -> loss
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 5], Some("input"))
        .unwrap();
    let weight = inner
        .borrow_mut()
        .create_parameter_node(&[5, 3], Some("weight"))
        .unwrap();
    // 注册参数使 zero_grad 能正常工作
    inner
        .borrow_mut()
        .register_parameter("weight".to_string(), std::rc::Rc::downgrade(&weight))?;
    let logits = inner
        .borrow_mut()
        .create_mat_mul_node(vec![input.clone(), weight.clone()], Some("logits"))
        .unwrap();
    let labels = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("labels"))
        .unwrap();
    let sce = inner
        .borrow_mut()
        .create_softmax_cross_entropy_node(logits, labels.clone(), Some("sce"))
        .unwrap();

    // 初始化权重
    weight
        .set_value(Some(&Tensor::normal_seeded(0.0, 0.1, &[5, 3], 42)))
        .unwrap();

    // batch=2
    let mut labels_data = Tensor::zeros(&[2, 3]);
    labels_data[[0, 0]] = 1.0;
    labels_data[[1, 2]] = 1.0;
    input.set_value(Some(&Tensor::ones(&[2, 5]))).unwrap();
    labels.set_value(Some(&labels_data)).unwrap();

    sce.forward_recursive(1, false).unwrap();
    inner.borrow_mut().zero_grad()?;
    // backward_via_node_inner 会清除中间节点梯度、设置 loss grad=1 并反向传播
    inner.borrow_mut().backward_via_node_inner(&sce)?;
    let grad1 = weight.grad().unwrap().clone();
    assert_eq!(grad1.shape(), &[5, 3], "权重梯度形状应保持不变");

    // batch=4（不同 batch 大小）
    let mut labels_data2 = Tensor::zeros(&[4, 3]);
    for i in 0..4 {
        labels_data2[[i, i % 3]] = 1.0;
    }
    input.set_value(Some(&Tensor::ones(&[4, 5]))).unwrap();
    labels.set_value(Some(&labels_data2)).unwrap();

    sce.forward_recursive(2, false).unwrap();
    inner.borrow_mut().zero_grad()?;
    inner.borrow_mut().backward_via_node_inner(&sce)?;
    let grad2 = weight.grad().unwrap();
    assert_eq!(
        grad2.shape(),
        &[5, 3],
        "权重梯度形状应保持不变（与 batch 大小无关）"
    );

    Ok(())
}

// ==================== 节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_softmax_cross_entropy_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("logits"))
        .unwrap();
    let labels = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("labels"))
        .unwrap();

    let sce = inner
        .borrow_mut()
        .create_softmax_cross_entropy_node(logits.clone(), labels.clone(), Some("sce"))
        .unwrap();

    // SoftmaxCrossEntropy 输出形状固定为 [1, 1]
    assert_eq!(sce.shape(), vec![1, 1]);
    assert_eq!(sce.name(), Some("sce"));
    assert!(!sce.is_leaf());
    assert_eq!(sce.parents().len(), 2);
}

#[test]
fn test_create_softmax_cross_entropy_node_shape_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    let labels = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], None) // 形状不匹配
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_softmax_cross_entropy_node(logits, labels, None);

    assert!(result.is_err());
}

#[test]
fn test_create_softmax_cross_entropy_node_output_always_scalar() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 10], None)
        .unwrap();
    let labels = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 10], None)
        .unwrap();
    let sce = inner
        .borrow_mut()
        .create_softmax_cross_entropy_node(logits, labels, None)
        .unwrap();
    assert_eq!(sce.shape(), vec![1, 1]);
}

#[test]
fn test_create_softmax_cross_entropy_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_sce;
    let weak_logits;
    let weak_labels;
    {
        let logits = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_logits = Rc::downgrade(&logits);

        let labels = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_labels = Rc::downgrade(&labels);

        let sce = inner
            .borrow_mut()
            .create_softmax_cross_entropy_node(logits, labels, None)
            .unwrap();
        weak_sce = Rc::downgrade(&sce);

        assert!(weak_sce.upgrade().is_some());
        assert!(weak_logits.upgrade().is_some());
        assert!(weak_labels.upgrade().is_some());
    }
    assert!(weak_sce.upgrade().is_none());
    assert!(weak_logits.upgrade().is_none());
    assert!(weak_labels.upgrade().is_none());
}
