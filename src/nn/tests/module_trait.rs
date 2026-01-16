/*
 * @Description  : Module trait 单元测试
 *
 * 测试 Module trait 的核心功能：
 * - parameters() 返回可训练参数
 * - num_params() 参数数量统计
 * - 组合模型的参数收集
 */

use crate::nn::Module;
use crate::nn::graph::Graph;
use crate::nn::layer::Linear;

/// 测试 Linear 层实现 Module trait
#[test]
fn test_linear_implements_module() {
    let graph = Graph::new_with_seed(42);

    // 带 bias 的 Linear
    let fc = Linear::new(&graph, 10, 5, true, "fc").unwrap();

    let params = fc.parameters();
    assert_eq!(params.len(), 2); // weights + bias

    // 验证形状
    let w_shape = params[0].value_expected_shape();
    let b_shape = params[1].value_expected_shape();
    assert_eq!(w_shape, vec![10, 5]); // [in, out]
    assert_eq!(b_shape, vec![1, 5]); // [1, out]
}

/// 测试不带 bias 的 Linear
#[test]
fn test_linear_no_bias() {
    let graph = Graph::new_with_seed(42);

    let fc = Linear::new(&graph, 8, 4, false, "fc_no_bias").unwrap();

    let params = fc.parameters();
    assert_eq!(params.len(), 1); // 只有 weights
    assert_eq!(fc.num_params(), 1);
}

/// 测试 num_params() 默认实现
#[test]
fn test_module_num_params() {
    let graph = Graph::new_with_seed(42);

    let fc = Linear::new(&graph, 16, 8, true, "fc").unwrap();
    assert_eq!(fc.num_params(), 2);

    let fc_no_bias = Linear::new(&graph, 16, 8, false, "fc2").unwrap();
    assert_eq!(fc_no_bias.num_params(), 1);
}

/// 测试组合模型的参数收集
///
/// 模拟一个简单的 MLP：fc1 -> fc2
#[test]
fn test_composite_model_parameters() {
    let graph = Graph::new_with_seed(42);

    // 构建简单 MLP
    let fc1 = Linear::new(&graph, 784, 128, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 128, 10, true, "fc2").unwrap();

    // 收集所有参数
    let all_params: Vec<_> = [fc1.parameters(), fc2.parameters()].concat();

    // fc1: weights + bias = 2
    // fc2: weights + bias = 2
    // 总计 4 个参数 Var
    assert_eq!(all_params.len(), 4);

    // 验证参数形状
    assert_eq!(all_params[0].value_expected_shape(), vec![784, 128]); // fc1.W
    assert_eq!(all_params[1].value_expected_shape(), vec![1, 128]); // fc1.b
    assert_eq!(all_params[2].value_expected_shape(), vec![128, 10]); // fc2.W
    assert_eq!(all_params[3].value_expected_shape(), vec![1, 10]); // fc2.b
}

/// 测试参数可用于优化器
#[test]
fn test_parameters_usable_with_optimizer() {
    use crate::nn::{Adam, Optimizer, VarActivationOps, VarLossOps};
    use crate::tensor::Tensor;

    let graph = Graph::new_with_seed(42);

    // 简单模型
    let fc = Linear::new(&graph, 4, 2, true, "fc").unwrap();

    // 输入/目标
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0, 0.0], &[1, 2])).unwrap();

    // 前向传播
    let y = fc.forward(&x).sigmoid();
    let loss = y.mse_loss(&target).unwrap();

    // 使用 Module::parameters() 创建优化器
    let mut optimizer = Adam::new(&graph, &fc.parameters(), 0.01);

    // 训练一步
    optimizer.zero_grad().unwrap();
    let loss_before = loss.backward().unwrap();
    optimizer.step().unwrap();

    // 再算一次 loss
    optimizer.zero_grad().unwrap();
    let loss_after = loss.backward().unwrap();

    // loss 应该下降
    assert!(
        loss_after < loss_before,
        "loss 应该下降: before={}, after={}",
        loss_before,
        loss_after
    );
}

/// 测试参数的 Var 可以 clone 且共享底层节点
#[test]
fn test_parameters_var_clone_semantics() {
    let graph = Graph::new_with_seed(42);

    let fc = Linear::new(&graph, 4, 2, true, "fc").unwrap();

    let params1 = fc.parameters();
    let params2 = fc.parameters();

    // 两次调用返回的 Var 应该指向同一个节点
    assert_eq!(params1[0].node_id(), params2[0].node_id());
    assert_eq!(params1[1].node_id(), params2[1].node_id());
}
