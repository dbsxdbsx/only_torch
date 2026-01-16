use approx::assert_abs_diff_eq;

use crate::nn::GraphInner;
use crate::tensor::Tensor;

#[test]
fn test_softmax_cross_entropy_creation() {
    let mut graph = GraphInner::new();

    // 创建 logits 和 labels 输入节点（必须是 2D）
    let logits_id = graph.new_input_node(&[1, 3], Some("logits")).unwrap();
    let labels_id = graph.new_input_node(&[1, 3], Some("labels")).unwrap();

    // 创建 SoftmaxCrossEntropy 节点
    let loss_id = graph
        .new_softmax_cross_entropy_node(logits_id, labels_id, Some("loss"))
        .unwrap();

    // 验证节点存在且预期形状正确
    assert_eq!(
        graph.get_node_value_expected_shape(loss_id).unwrap(),
        &[1, 1]
    );
}

#[test]
fn test_softmax_cross_entropy_shape_mismatch() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_input_node(&[1, 3], Some("logits")).unwrap();
    let labels_id = graph.new_input_node(&[1, 4], Some("labels")).unwrap(); // 形状不匹配

    let result = graph.new_softmax_cross_entropy_node(logits_id, labels_id, None);
    assert!(result.is_err());
}

#[test]
fn test_softmax_cross_entropy_forward_simple() {
    // PyTorch 验证值：
    // logits = [1.0, 2.0, 3.0], labels = [0, 0, 1]
    // softmax = [0.09003057, 0.24472848, 0.66524094]
    // loss = 0.40760597

    let mut graph = GraphInner::new();

    let logits_id = graph.new_input_node(&[1, 3], Some("logits")).unwrap();
    let labels_id = graph.new_input_node(&[1, 3], Some("labels")).unwrap();
    let loss_id = graph
        .new_softmax_cross_entropy_node(logits_id, labels_id, Some("loss"))
        .unwrap();

    // 设置输入值
    graph
        .set_node_value(logits_id, Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(labels_id, Some(&Tensor::new(&[0.0, 0.0, 1.0], &[1, 3])))
        .unwrap();

    // 前向传播
    graph.forward(loss_id).unwrap();

    // 验证损失值
    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    let expected_loss = Tensor::new(&[0.40760597], &[1, 1]);
    assert_abs_diff_eq!(loss, &expected_loss, epsilon = 1e-5);
}

#[test]
fn test_softmax_cross_entropy_forward_uniform() {
    // PyTorch 验证值：
    // logits = [1.0, 1.0, 1.0, 1.0], labels = [0, 1, 0, 0]
    // softmax = [0.25, 0.25, 0.25, 0.25]
    // loss = 1.3862944 (= -ln(0.25))

    let mut graph = GraphInner::new();

    let logits_id = graph.new_input_node(&[1, 4], Some("logits")).unwrap();
    let labels_id = graph.new_input_node(&[1, 4], Some("labels")).unwrap();
    let loss_id = graph
        .new_softmax_cross_entropy_node(logits_id, labels_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(
            logits_id,
            Some(&Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 4])),
        )
        .unwrap();
    graph
        .set_node_value(
            labels_id,
            Some(&Tensor::new(&[0.0, 1.0, 0.0, 0.0], &[1, 4])),
        )
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    let expected_loss = Tensor::new(&[1.3862944], &[1, 1]);
    assert_abs_diff_eq!(loss, &expected_loss, epsilon = 1e-5);
}

#[test]
fn test_softmax_cross_entropy_backward_simple() {
    // PyTorch 验证值：
    // logits = [1.0, 2.0, 3.0], labels = [0, 0, 1]
    // grad = [0.09003057, 0.24472848, -0.33475903]

    let mut graph = GraphInner::new();

    let logits_id = graph.new_parameter_node(&[1, 3], Some("logits")).unwrap();
    let labels_id = graph.new_input_node(&[1, 3], Some("labels")).unwrap();
    let loss_id = graph
        .new_softmax_cross_entropy_node(logits_id, labels_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(logits_id, Some(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(labels_id, Some(&Tensor::new(&[0.0, 0.0, 1.0], &[1, 3])))
        .unwrap();

    // 前向传播
    graph.forward(loss_id).unwrap();

    // 反向传播
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    // 验证 logits 的梯度
    let grad = graph.get_node(logits_id).unwrap().grad().unwrap();
    assert_eq!(grad.shape(), &[1, 3]);

    let expected_grad = Tensor::new(&[0.09003057, 0.24472848, -0.33475903], &[1, 3]);
    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

#[test]
fn test_softmax_cross_entropy_backward_uniform() {
    // PyTorch 验证值：
    // logits = [1.0, 1.0, 1.0, 1.0], labels = [0, 1, 0, 0]
    // grad = [0.25, -0.75, 0.25, 0.25]

    let mut graph = GraphInner::new();

    let logits_id = graph.new_parameter_node(&[1, 4], Some("logits")).unwrap();
    let labels_id = graph.new_input_node(&[1, 4], Some("labels")).unwrap();
    let loss_id = graph
        .new_softmax_cross_entropy_node(logits_id, labels_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(
            logits_id,
            Some(&Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 4])),
        )
        .unwrap();
    graph
        .set_node_value(
            labels_id,
            Some(&Tensor::new(&[0.0, 1.0, 0.0, 0.0], &[1, 4])),
        )
        .unwrap();

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(logits_id).unwrap().grad().unwrap();
    let expected_grad = Tensor::new(&[0.25, -0.75, 0.25, 0.25], &[1, 4]);
    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

#[test]
fn test_softmax_cross_entropy_10_classes() {
    // PyTorch 验证值：
    // logits = [0.5, 1.5, -0.5, 2.0, 0.0, -1.0, 1.0, 0.5, -0.5, 1.5]
    // labels = [0,0,0,1,0,0,0,0,0,0] (类别 3)
    // loss = 1.2168376
    // grad = [0.0660834, 0.1796333, 0.02431072, -0.7038347, ...]

    let mut graph = GraphInner::new();

    let logits_id = graph.new_parameter_node(&[1, 10], Some("logits")).unwrap();
    let labels_id = graph.new_input_node(&[1, 10], Some("labels")).unwrap();
    let loss_id = graph
        .new_softmax_cross_entropy_node(logits_id, labels_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(
            logits_id,
            Some(&Tensor::new(
                &[0.5, 1.5, -0.5, 2.0, 0.0, -1.0, 1.0, 0.5, -0.5, 1.5],
                &[1, 10],
            )),
        )
        .unwrap();

    let mut labels = vec![0.0; 10];
    labels[3] = 1.0;
    graph
        .set_node_value(labels_id, Some(&Tensor::new(&labels, &[1, 10])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    // 验证损失
    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    let expected_loss = Tensor::new(&[1.2168376], &[1, 1]);
    assert_abs_diff_eq!(loss, &expected_loss, epsilon = 1e-5);

    // 验证梯度
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();
    let grad = graph.get_node(logits_id).unwrap().grad().unwrap();

    #[rustfmt::skip]
    let expected_grad = Tensor::new(
        &[0.0660834, 0.1796333, 0.02431072, -0.7038347, 0.04008161,
          0.0147452, 0.10895311, 0.0660834, 0.02431072, 0.1796333],
        &[1, 10]
    );
    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

#[test]
fn test_softmax_cross_entropy_with_linear_layer() {
    // 简单网络: input -> linear -> softmax_cross_entropy
    // 验证梯度能正确传播到线性层权重

    let mut graph = GraphInner::new();

    // 输入: [1, 2] -> 线性层 -> [1, 3] -> softmax_cross_entropy -> loss
    let input_id = graph.new_input_node(&[1, 2], Some("input")).unwrap();
    let weights_id = graph.new_parameter_node(&[2, 3], Some("weights")).unwrap();
    let bias_id = graph.new_parameter_node(&[1, 3], Some("bias")).unwrap();

    // input @ weights + bias
    let matmul_id = graph
        .new_mat_mul_node(input_id, weights_id, Some("matmul"))
        .unwrap();
    let logits_id = graph
        .new_add_node(&[matmul_id, bias_id], Some("logits"))
        .unwrap();

    // labels 使用 [1, 3] 形状与 logits 匹配
    let labels_id = graph.new_input_node(&[1, 3], Some("labels")).unwrap();
    let loss_id = graph
        .new_softmax_cross_entropy_node(logits_id, labels_id, Some("loss"))
        .unwrap();

    // 设置值
    graph
        .set_node_value(input_id, Some(&Tensor::new(&[1.0, 2.0], &[1, 2])))
        .unwrap();
    graph
        .set_node_value(
            weights_id,
            Some(&Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3])),
        )
        .unwrap();
    graph
        .set_node_value(bias_id, Some(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(labels_id, Some(&Tensor::new(&[0.0, 0.0, 1.0], &[1, 3])))
        .unwrap();

    // 前向传播
    graph.forward(loss_id).unwrap();

    // 反向传播到权重
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    // 验证权重有梯度
    let weights_grad = graph.get_node(weights_id).unwrap().grad().unwrap();
    assert_eq!(weights_grad.shape(), &[2, 3]); // 权重形状 [2, 3]
}
