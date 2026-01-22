use approx::assert_abs_diff_eq;

use crate::nn::{GraphError, GraphInner};
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

// ==================== 动态形状测试 ====================

/// 测试 SoftmaxCrossEntropy 节点的动态形状传播
/// 注：Loss 节点输出固定为 [1, 1]，但输入可以有动态 batch
#[test]
fn test_softmax_cross_entropy_dynamic_shape_propagation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建 2D 输入：[batch, num_classes]
    let logits = graph.new_input_node(&[2, 3], Some("logits"))?;
    let labels = graph.new_input_node(&[2, 3], Some("labels"))?;

    let loss = graph.new_softmax_cross_entropy_node(logits, labels, Some("loss"))?;

    // Loss 输出固定为 [1, 1]
    let loss_node = graph.get_node(loss)?;
    assert_eq!(loss_node.value_expected_shape(), &[1, 1]);

    Ok(())
}

/// 测试 SoftmaxCrossEntropy 在不同 batch_size 下的前向计算
#[test]
fn test_softmax_cross_entropy_dynamic_batch_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建输入
    let logits = graph.new_input_node(&[2, 3], Some("logits"))?;
    let labels = graph.new_input_node(&[2, 3], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(logits, labels, Some("loss"))?;

    // 设置初始值：batch=2
    let mut labels_data = Tensor::zeros(&[2, 3]);
    labels_data[[0, 0]] = 1.0;
    labels_data[[1, 2]] = 1.0;
    graph.set_node_value(logits, Some(&Tensor::normal_seeded(0.0, 1.0, &[2, 3], 42)))?;
    graph.set_node_value(labels, Some(&labels_data))?;

    // 第一次 forward：batch=2
    graph.forward(loss)?;
    let value1 = graph.get_node_value(loss)?.unwrap();
    assert_eq!(value1.shape(), &[1, 1], "Loss 输出固定为 [1, 1]");
    let loss_val1 = value1[[0, 0]];
    assert!(loss_val1 > 0.0);

    // 更新输入为不同的 batch_size
    let mut labels_data2 = Tensor::zeros(&[5, 3]);
    for i in 0..5 {
        labels_data2[[i, i % 3]] = 1.0;
    }
    graph.set_node_value(logits, Some(&Tensor::normal_seeded(0.0, 1.0, &[5, 3], 100)))?;
    graph.set_node_value(labels, Some(&labels_data2))?;

    // 第二次 forward：batch=5
    graph.forward(loss)?;
    let value2 = graph.get_node_value(loss)?.unwrap();
    assert_eq!(value2.shape(), &[1, 1], "Loss 输出固定为 [1, 1]");
    let loss_val2 = value2[[0, 0]];
    assert!(loss_val2 > 0.0);

    Ok(())
}

/// 测试 SoftmaxCrossEntropy 在不同 batch_size 下的反向传播
#[test]
fn test_softmax_cross_entropy_dynamic_batch_backward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 使用 Input 节点接收动态 batch 数据
    // 构建 y = x * w 形式，其中 w 是可训练的 Parameter
    let input = graph.new_input_node(&[2, 5], Some("input"))?;
    let weight = graph.new_parameter_node(&[5, 3], Some("weight"))?;  // [5, 3] 权重
    let logits = graph.new_mat_mul_node(input, weight, Some("logits"))?;
    let labels = graph.new_input_node(&[2, 3], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(logits, labels, Some("loss"))?;

    // 初始化权重
    graph.set_node_value(weight, Some(&Tensor::normal_seeded(0.0, 0.1, &[5, 3], 42)))?;

    // 设置初始值：batch=2
    let mut labels_data = Tensor::zeros(&[2, 3]);
    labels_data[[0, 0]] = 1.0;
    labels_data[[1, 2]] = 1.0;
    graph.set_node_value(input, Some(&Tensor::ones(&[2, 5])))?;
    graph.set_node_value(labels, Some(&labels_data))?;

    // 第一次训练：batch=2
    graph.forward(loss)?;
    graph.zero_grad()?;
    graph.backward(loss)?;
    let grad1 = graph.get_node(weight)?.grad().unwrap().clone();
    assert_eq!(grad1.shape(), &[5, 3], "权重梯度形状应保持不变");

    // 更新输入为不同的 batch_size
    let mut labels_data2 = Tensor::zeros(&[4, 3]);
    for i in 0..4 {
        labels_data2[[i, i % 3]] = 1.0;
    }
    graph.set_node_value(input, Some(&Tensor::ones(&[4, 5])))?;
    graph.set_node_value(labels, Some(&labels_data2))?;

    // 第二次训练：batch=4
    graph.zero_grad()?;
    graph.forward(loss)?;
    graph.backward(loss)?;
    let grad2 = graph.get_node(weight)?.grad().unwrap();
    assert_eq!(
        grad2.shape(),
        &[5, 3],
        "权重梯度形状应保持不变（与 batch 大小无关）"
    );

    Ok(())
}
