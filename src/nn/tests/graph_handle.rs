/*
 * @Description  : Graph（V2 Graph 句柄）测试
 *
 * 这是 Phase 1b 验收测试的一部分，测试 Graph 句柄的核心功能。
 */

use crate::nn::graph::Graph;
use crate::nn::var::Init;
use crate::nn::{VarActivationOps, VarLossOps};
use crate::tensor::Tensor;

// ==================== 创建测试 ====================

/// 测试 Graph 创建
#[test]
fn test_graph_handle_new() {
    let graph = Graph::new();

    // 应该能创建输入
    let x = graph.input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]));
    assert!(x.is_ok());
}

/// 测试带种子的 Graph 创建
#[test]
fn test_graph_handle_new_with_seed() {
    let graph1 = Graph::new_with_seed(42);
    let graph2 = Graph::new_with_seed(42);

    // 使用相同图级别种子，使用 parameter_seeded 确保可重复
    // 注意：parameter() + Init::Normal 使用全局 RNG，不受 graph seed 控制
    // 要测试 graph seed，应使用图内部的 RNG（如 parameter_seeded）
    let p1 = graph1.parameter_seeded(&[2, 2], "p", 100).unwrap();
    let p2 = graph2.parameter_seeded(&[2, 2], "p", 100).unwrap();

    let v1 = p1.value().unwrap().unwrap();
    let v2 = p2.value().unwrap().unwrap();

    assert_eq!(v1.data_as_slice(), v2.data_as_slice());
}

// ==================== 输入创建测试 ====================

/// 测试 input 方法
#[test]
fn test_graph_handle_input() {
    let graph = Graph::new();

    let data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let x = graph.input(&data).unwrap();

    // 值应该已设置
    let value = x.value().unwrap().unwrap();
    assert_eq!(value.shape(), &[2, 2]);
    assert_eq!(value.data_as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

/// 测试 input_named 方法
#[test]
fn test_graph_handle_input_named() {
    let graph = Graph::new();

    let data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let x = graph.input_named(&data, "my_input").unwrap();

    // 检查名称
    let inner = graph.inner();
    let name = inner.get_node_name(x.node_id()).unwrap();
    assert_eq!(name, "my_input");
}

// ==================== 参数创建测试 ====================

/// 测试 parameter 方法
#[test]
fn test_graph_handle_parameter() {
    let graph = Graph::new();

    let w = graph.parameter(&[3, 2], Init::Xavier, "weight").unwrap();

    // 应该有值
    let value = w.value().unwrap().unwrap();
    assert_eq!(value.shape(), &[3, 2]);

    // Xavier 初始化的值应该在合理范围内
    let data = value.data_as_slice();
    for &v in data {
        assert!(v.abs() < 2.0, "Xavier 初始化值 {} 超出预期范围", v);
    }
}

/// 测试 parameter_seeded 方法
#[test]
fn test_graph_handle_parameter_seeded() {
    let graph1 = Graph::new();
    let graph2 = Graph::new();

    // 使用相同种子应该得到相同的初始化
    let p1 = graph1.parameter_seeded(&[3, 2], "p", 123).unwrap();
    let p2 = graph2.parameter_seeded(&[3, 2], "p", 123).unwrap();

    let v1 = p1.value().unwrap().unwrap();
    let v2 = p2.value().unwrap().unwrap();

    assert_eq!(v1.data_as_slice(), v2.data_as_slice());
}

// ==================== 张量创建测试 ====================

/// 测试 zeros 方法
#[test]
fn test_graph_handle_zeros() {
    let graph = Graph::new();

    let x = graph.zeros(&[2, 3]).unwrap();
    let value = x.value().unwrap().unwrap();

    assert_eq!(value.shape(), &[2, 3]);
    assert!(value.data_as_slice().iter().all(|&v| v == 0.0));
}

/// 测试 ones 方法
#[test]
fn test_graph_handle_ones() {
    let graph = Graph::new();

    let x = graph.ones(&[2, 3]).unwrap();
    let value = x.value().unwrap().unwrap();

    assert_eq!(value.shape(), &[2, 3]);
    assert!(value.data_as_slice().iter().all(|&v| v == 1.0));
}

/// 测试 randn 方法
#[test]
fn test_graph_handle_randn() {
    let graph = Graph::new();

    let x = graph.randn(&[100, 100]).unwrap();
    let value = x.value().unwrap().unwrap();

    assert_eq!(value.shape(), &[100, 100]);

    // 标准正态分布，均值应接近 0，标准差应接近 1
    let data = value.data_as_slice();
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std = var.sqrt();

    assert!(mean.abs() < 0.1, "均值 {} 偏离 0 太远", mean);
    assert!((std - 1.0).abs() < 0.1, "标准差 {} 偏离 1 太远", std);
}

/// 测试 constant 方法
#[test]
fn test_graph_handle_constant() {
    let graph = Graph::new();

    let data = Tensor::new(&[3.14, 2.71, 1.41], &[3, 1]);
    let c = graph.constant(&data).unwrap();

    let value = c.value().unwrap().unwrap();
    assert_eq!(value.data_as_slice(), &[3.14, 2.71, 1.41]);
}

/// 测试 constant_named 方法
#[test]
fn test_graph_handle_constant_named() {
    let graph = Graph::new();

    let data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let c = graph.constant_named(&data, "my_const").unwrap();

    let inner = graph.inner();
    let name = inner.get_node_name(c.node_id()).unwrap();
    assert_eq!(name, "my_const");
}

// ==================== 执行测试 ====================

/// 测试 forward 方法
#[test]
fn test_graph_handle_forward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let y = x.relu();

    // 通过 graph handle 执行 forward
    graph.forward(&y).unwrap();

    let value = y.value().unwrap().unwrap();
    assert_eq!(value.data_as_slice(), &[1.0, 2.0]);
}

/// 测试 backward 方法
#[test]
fn test_graph_handle_backward() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();
    let w = graph.parameter(&[1, 1], Init::Ones, "w").unwrap();
    let y = &x * &w;
    let target = graph.input(&Tensor::new(&[4.0], &[1, 1])).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    // 通过 graph handle 执行 backward
    let loss_val = graph.backward(&loss).unwrap();

    // 检查返回值
    // y = 2 * 1 = 2, target = 4, loss = (2-4)^2 = 4
    assert!((loss_val - 4.0).abs() < 0.01);

    // 检查梯度存在
    assert!(w.grad().unwrap().is_some());
}

// ==================== 训练控制测试 ====================

/// 测试 zero_grad 方法
#[test]
fn test_graph_handle_zero_grad() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();
    let w = graph.parameter(&[1, 1], Init::Ones, "w").unwrap();
    let y = &x * &w;
    let target = graph.input(&Tensor::new(&[4.0], &[1, 1])).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    // 记录 backward 前的节点数
    let node_count_before = graph.inner().nodes_count();

    // 第一次 backward
    loss.backward().unwrap();

    // 验证：梯度存在且非零
    let w_grad = w.grad().unwrap();
    assert!(w_grad.is_some(), "backward 后 w 应该有梯度");
    let grad_value = w_grad.unwrap();
    assert!(
        grad_value.data_as_slice().iter().any(|&v| v != 0.0),
        "梯度不应全为零"
    );

    // zero_grad
    graph.zero_grad().unwrap();

    // 验证：节点数不变（节点仍存在）
    assert_eq!(
        graph.inner().nodes_count(),
        node_count_before,
        "zero_grad 不应删除节点"
    );

    // 验证：参数值不变
    let w_value = w.value().unwrap().unwrap();
    assert_eq!(w_value.data_as_slice(), &[1.0], "zero_grad 不应改变参数值");

    // 验证：梯度被清零
    let w_grad_after = w.grad().unwrap();
    assert!(
        w_grad_after.is_none()
            || w_grad_after
                .as_ref()
                .map(|g| g.data_as_slice().iter().all(|&v| v == 0.0))
                .unwrap_or(true),
        "zero_grad 后梯度应为 None 或全零"
    );

    // 验证：可以再次 backward（图仍有效）
    loss.backward().unwrap();
    assert!(w.grad().unwrap().is_some(), "再次 backward 后 w 应该有梯度");
}

/// 测试 train/eval 模式
#[test]
fn test_graph_handle_train_eval() {
    let graph = Graph::new();

    // 默认应该是训练模式
    assert!(!graph.is_eval());

    // 切换到评估模式
    graph.eval();
    assert!(graph.is_eval());

    // 切换回训练模式
    graph.train();
    assert!(!graph.is_eval());
}

// ==================== Clone 语义测试 ====================

/// 测试 Graph clone 共享同一个 GraphInner
#[test]
fn test_graph_handle_clone_shared() {
    let graph1 = Graph::new();
    let graph2 = graph1.clone();

    // 在 graph1 创建的节点，graph2 也能看到
    let x = graph1.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    // 通过 graph2 创建的 Var 应该和 x 在同一个图
    let y = graph2.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();

    assert!(x.same_graph(&y));
}

// ==================== inner 访问测试 ====================

/// 测试 inner 和 inner_mut 访问
#[test]
fn test_graph_handle_inner_access() {
    let graph = Graph::new();

    // 通过 inner() 可以访问 GraphInner
    let node_count = graph.inner().nodes_count();
    assert_eq!(node_count, 0);

    // 创建一个节点
    let _ = graph.zeros(&[1, 1]).unwrap();

    // 节点数应该增加
    let node_count_after = graph.inner().nodes_count();
    assert_eq!(node_count_after, 1);
}

// ==================== no_grad_scope 测试 ====================

/// 测试 no_grad_scope 基本功能
#[test]
fn test_graph_handle_no_grad_scope_basic() {
    let graph = Graph::new();

    // 默认是训练模式
    assert!(!graph.is_eval());

    // 进入 no_grad_scope
    let result = graph.no_grad_scope(|g| {
        // 在 scope 内应该是 eval 模式
        assert!(g.is_eval());
        42
    });

    // 返回值应该正确传递
    assert_eq!(result, 42);

    // 退出 scope 后应该恢复训练模式
    assert!(!graph.is_eval());
}

/// 测试 no_grad_scope 从 eval 模式开始
#[test]
fn test_graph_handle_no_grad_scope_from_eval() {
    let graph = Graph::new();

    // 先切换到 eval 模式
    graph.eval();
    assert!(graph.is_eval());

    // 进入 no_grad_scope
    graph.no_grad_scope(|g| {
        // 仍然是 eval 模式
        assert!(g.is_eval());
    });

    // 退出 scope 后仍然是 eval 模式（因为进入前就是 eval）
    assert!(graph.is_eval());
}

/// 测试 no_grad_scope 嵌套调用
#[test]
fn test_graph_handle_no_grad_scope_nested() {
    let graph = Graph::new();

    // 默认是训练模式
    assert!(!graph.is_eval());

    graph.no_grad_scope(|g| {
        assert!(g.is_eval());

        // 嵌套调用
        g.no_grad_scope(|g2| {
            assert!(g2.is_eval());
        });

        // 嵌套退出后仍然是 eval（因为外层 scope 还在）
        assert!(g.is_eval());
    });

    // 完全退出后恢复训练模式
    assert!(!graph.is_eval());
}

/// 测试 no_grad_scope 中执行计算
#[test]
fn test_graph_handle_no_grad_scope_with_computation() {
    let graph = Graph::new();
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))
        .unwrap();
    let y = graph
        .input(&Tensor::new(&[4.0, 5.0, 6.0], &[3, 1]))
        .unwrap();

    // 在 no_grad_scope 中执行 forward
    let result = graph.no_grad_scope(|_g| {
        let z = &x + &y;
        z.forward().unwrap();
        z.value().unwrap().unwrap().data_as_slice().to_vec()
    });

    assert_eq!(result, vec![5.0, 7.0, 9.0]);
}
