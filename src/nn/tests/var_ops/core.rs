/*
 * @Description  : Var 核心功能测试
 *
 * 测试 Smart Var 的核心功能：
 * - 算子重载（Add, Sub, Mul, Neg）
 * - 值访问（value, set_value, item, grad）
 * - 跨图安全性
 * - 梯度流控制（detach）
 */

use crate::nn::graph::Graph;
use crate::nn::{VarActivationOps, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;

// ==================== 算子重载测试 ====================

/// 测试 Var 加法算子重载
#[test]
fn test_var_add() {
    let graph = Graph::new();

    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[4.0, 5.0, 6.0], &[3, 1]))
        .unwrap();

    // &Var + &Var
    let c = &a + &b;
    c.forward().unwrap();
    let result = c.value().unwrap().unwrap();
    assert_eq!(result.data_as_slice(), &[5.0, 7.0, 9.0]);
}

/// 测试 Var 减法算子重载
#[test]
fn test_var_sub() {
    let graph = Graph::new();

    let a = graph
        .input(&Tensor::new(&[5.0, 7.0, 9.0], &[3, 1]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))
        .unwrap();

    let c = &a - &b;
    c.forward().unwrap();
    let result = c.value().unwrap().unwrap();
    assert_eq!(result.data_as_slice(), &[4.0, 5.0, 6.0]);
}

/// 测试 Var 乘法算子重载（逐元素）
#[test]
fn test_var_mul() {
    let graph = Graph::new();

    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[2.0, 3.0, 4.0], &[3, 1]))
        .unwrap();

    let c = &a * &b;
    c.forward().unwrap();
    let result = c.value().unwrap().unwrap();
    assert_eq!(result.data_as_slice(), &[2.0, 6.0, 12.0]);
}

/// 测试 Var 除法算子重载
#[test]
fn test_var_div() {
    let graph = Graph::new();

    let a = graph
        .input(&Tensor::new(&[6.0, 12.0, 24.0], &[3, 1]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[2.0, 3.0, 4.0], &[3, 1]))
        .unwrap();

    let c = &a / &b;
    c.forward().unwrap();
    let result = c.value().unwrap().unwrap();
    assert_eq!(result.data_as_slice(), &[3.0, 4.0, 6.0]);
}

/// 测试 Var 除法梯度
#[test]
fn test_var_div_backward() {
    use crate::nn::var::Init;

    let graph = Graph::new();

    // 构建简单网络：a / b -> mse_loss
    // 设置 a=4, b=2，target=1，则 output=2，loss=(2-1)^2=1
    let a = graph.parameter(&[1, 1], Init::Ones, "a").unwrap();
    let b = graph.parameter(&[1, 1], Init::Ones, "b").unwrap();
    // 手动设置值
    a.set_value(&Tensor::new(&[4.0], &[1, 1])).unwrap();
    b.set_value(&Tensor::new(&[2.0], &[1, 1])).unwrap();
    let target = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    let output = &a / &b; // output = 4/2 = 2
    let loss = output.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 验证梯度存在
    let a_grad = a.grad().unwrap();
    let b_grad = b.grad().unwrap();
    assert!(a_grad.is_some());
    assert!(b_grad.is_some());

    // 数值验证：
    // L = (a/b - t)^2，对 a 求导：dL/da = 2*(a/b - t) * (1/b) = 2*(2-1)*(1/2) = 1
    // 对 b 求导：dL/db = 2*(a/b - t) * (-a/b^2) = 2*(2-1)*(-4/4) = -2
    let a_grad_val = a_grad.unwrap().data_as_slice()[0];
    let b_grad_val = b_grad.unwrap().data_as_slice()[0];
    assert!((a_grad_val - 1.0).abs() < 1e-5, "a_grad={}", a_grad_val);
    assert!((b_grad_val - (-2.0)).abs() < 1e-5, "b_grad={}", b_grad_val);
}

/// 测试 Var 取反算子重载
#[test]
fn test_var_neg() {
    let graph = Graph::new();

    let a = graph
        .input(&Tensor::new(&[1.0, -2.0, 3.0], &[3, 1]))
        .unwrap();

    let b = -&a;
    b.forward().unwrap();
    let result = b.value().unwrap().unwrap();
    assert_eq!(result.data_as_slice(), &[-1.0, 2.0, -3.0]);
}

/// 测试混合算子
#[test]
fn test_var_mixed_operators() {
    let graph = Graph::new();

    let a = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let b = graph.input(&Tensor::new(&[3.0, 4.0], &[2, 1])).unwrap();

    // (a + b) * (a - b) = a^2 - b^2 = [1-9, 4-16] = [-8, -12]
    let sum = &a + &b;
    let diff = &a - &b;
    let prod = &sum * &diff;

    prod.forward().unwrap();
    let result = prod.value().unwrap().unwrap();
    assert_eq!(result.data_as_slice(), &[-8.0, -12.0]);
}

/// 测试各种所有权组合
#[test]
fn test_var_ownership_combinations() {
    let graph = Graph::new();

    let a = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let b = graph.input(&Tensor::new(&[3.0, 4.0], &[2, 1])).unwrap();

    // 测试各种组合
    let r1 = &a + &b; // &Var + &Var
    let r2 = a.clone() + b.clone(); // Var + Var
    let r3 = &a + b.clone(); // &Var + Var

    r1.forward().unwrap();
    r2.forward().unwrap();
    r3.forward().unwrap();

    assert_eq!(r1.value().unwrap().unwrap().data_as_slice(), &[4.0, 6.0]);
    assert_eq!(r2.value().unwrap().unwrap().data_as_slice(), &[4.0, 6.0]);
    assert_eq!(r3.value().unwrap().unwrap().data_as_slice(), &[4.0, 6.0]);
}

// ==================== 链式调用测试 ====================

/// 测试链式激活函数
#[test]
fn test_var_chain_activations() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[-1.0, 0.0, 1.0], &[3, 1]))
        .unwrap();

    // ReLU
    let r = x.relu();
    r.forward().unwrap();
    let result = r.value().unwrap().unwrap();
    assert_eq!(result.data_as_slice(), &[0.0, 0.0, 1.0]);

    // Sigmoid
    let s = x.sigmoid();
    s.forward().unwrap();
    let sig_result = s.value().unwrap().unwrap();
    // sigmoid(-1) ≈ 0.2689, sigmoid(0) = 0.5, sigmoid(1) ≈ 0.7311
    assert!((sig_result.data_as_slice()[0] - 0.2689).abs() < 0.01);
    assert!((sig_result.data_as_slice()[1] - 0.5).abs() < 0.01);
    assert!((sig_result.data_as_slice()[2] - 0.7311).abs() < 0.01);

    // Tanh
    let t = x.tanh();
    t.forward().unwrap();
    let tanh_result = t.value().unwrap().unwrap();
    // tanh(-1) ≈ -0.7616, tanh(0) = 0, tanh(1) ≈ 0.7616
    assert!((tanh_result.data_as_slice()[0] - (-0.7616)).abs() < 0.01);
    assert!((tanh_result.data_as_slice()[1] - 0.0).abs() < 0.01);
    assert!((tanh_result.data_as_slice()[2] - 0.7616).abs() < 0.01);
}

/// 测试链式矩阵乘法
#[test]
fn test_var_chain_matmul() {
    let graph = Graph::new();

    // x: [2, 3], w: [3, 2] -> result: [2, 2]
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let w = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]))
        .unwrap();

    let y = x.matmul(&w).unwrap();
    y.forward().unwrap();

    let result = y.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    // [1,2,3] @ [[1,2],[3,4],[5,6]] = [22, 28]
    // [4,5,6] @ [[1,2],[3,4],[5,6]] = [49, 64]
    assert_eq!(result.data_as_slice(), &[22.0, 28.0, 49.0, 64.0]);
}

/// 测试复合链式调用
#[test]
fn test_var_compound_chain() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let w = graph
        .input(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2]))
        .unwrap();
    let b = graph.input(&Tensor::new(&[0.1, 0.1], &[2, 1])).unwrap();

    // y = relu(w @ x + b)
    let wx = w.matmul(&x).unwrap();
    let z = &wx + &b;
    let y = z.relu();

    y.forward().unwrap();
    let result = y.value().unwrap().unwrap();

    // w @ x = [[0.5*1+0.5*2], [0.5*1+0.5*2]] = [[1.5], [1.5]]
    // z = [[1.5+0.1], [1.5+0.1]] = [[1.6], [1.6]]
    // relu(z) = [[1.6], [1.6]]
    assert_eq!(result.shape(), &[2, 1]);
    assert!((result.data_as_slice()[0] - 1.6).abs() < 0.01);
    assert!((result.data_as_slice()[1] - 1.6).abs() < 0.01);
}

// ==================== 值访问测试 ====================

/// 测试 set_value 和 value
#[test]
fn test_var_set_value() {
    let graph = Graph::new();

    let x = graph.zeros(&[2, 2]).unwrap();
    assert_eq!(
        x.value().unwrap().unwrap().data_as_slice(),
        &[0.0, 0.0, 0.0, 0.0]
    );

    // 更新值
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    assert_eq!(
        x.value().unwrap().unwrap().data_as_slice(),
        &[1.0, 2.0, 3.0, 4.0]
    );
}

/// 测试 item（获取标量）
#[test]
fn test_var_item() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[3.14], &[1, 1])).unwrap();
    let val = x.item().unwrap();
    assert!((val - 3.14).abs() < 0.001);
}

/// 测试 grad
#[test]
fn test_var_grad() {
    let graph = Graph::new();

    // 简单网络：x * w -> loss
    let x = graph.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();
    let w = graph
        .parameter(&[1, 1], crate::nn::var::Init::Ones, "w")
        .unwrap();
    let y = &x * &w;
    let target = graph.input(&Tensor::new(&[4.0], &[1, 1])).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    // 反向传播
    loss.backward().unwrap();

    // w 应该有梯度
    let w_grad = w.grad().unwrap();
    assert!(w_grad.is_some());
}

// ==================== 跨图安全性测试 ====================

/// 测试同一图的 Var 可以操作
#[test]
fn test_var_same_graph() {
    let graph = Graph::new();

    let a = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let b = graph.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();

    assert!(a.same_graph(&b));

    // 应该能正常操作
    let c = &a + &b;
    c.forward().unwrap();
    assert_eq!(c.value().unwrap().unwrap().data_as_slice(), &[3.0]);
}

/// 测试不同图的 Var 操作会 panic
#[test]
#[should_panic(expected = "不能对来自不同 Graph 的 Var 进行加法")]
fn test_var_different_graph_panic() {
    let graph1 = Graph::new();
    let graph2 = Graph::new();

    let a = graph1.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let b = graph2.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();

    // 这里应该 panic
    let _ = &a + &b;
}

// ==================== get_graph 测试 ====================

/// 测试 get_graph 能恢复图句柄
#[test]
fn test_var_get_graph() {
    let x = {
        let graph = Graph::new();
        graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap()
        // graph 在这里 drop
    };

    // 即使原始 graph handle 已 drop，x 仍持有 GraphInner
    let recovered_graph = x.get_graph();

    // 可以通过恢复的 graph 创建新 Var
    let y = recovered_graph
        .input(&Tensor::new(&[3.0, 4.0], &[2, 1]))
        .unwrap();

    // 两者应该在同一个图中
    assert!(x.same_graph(&y));
}

// ==================== 防御性测试 ====================

/// 验证算子操作不会修改原始 Var（防御性测试）
///
/// 虽然 Rust 类型系统已保证 `&Var` 不可变，但此测试明确记录了这一语义。
#[test]
fn test_var_operators_do_not_mutate_operands() {
    let graph = Graph::new();

    let a = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    let b = graph.input(&Tensor::new(&[3.0, 4.0], &[2, 1])).unwrap();

    // 记录操作前的状态
    let a_id = a.node_id();
    let b_id = b.node_id();
    let a_value_before = a.value().unwrap().unwrap().clone();
    let b_value_before = b.value().unwrap().unwrap().clone();

    // 执行多种操作
    let _add = &a + &b;
    let _sub = &a - &b;
    let _mul = &a * &b;
    let _neg = -&a;

    // 验证原始 Var 的 node_id 未变
    assert_eq!(a.node_id(), a_id, "a 的 node_id 不应改变");
    assert_eq!(b.node_id(), b_id, "b 的 node_id 不应改变");

    // 验证原始 Var 的值未变
    assert_eq!(
        a.value().unwrap().unwrap().data_as_slice(),
        a_value_before.data_as_slice(),
        "a 的值不应改变"
    );
    assert_eq!(
        b.value().unwrap().unwrap().data_as_slice(),
        b_value_before.data_as_slice(),
        "b 的值不应改变"
    );
}

// ==================== detach/attach 测试 ====================

/// 测试 detach 截断梯度流
#[test]
fn test_var_detach() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let w = graph
        .parameter(&[1, 1], crate::nn::var::Init::Ones, "w")
        .unwrap();

    let h = &x * &w;
    let h_detached = h.detach().unwrap();

    // detach 后的 Var 仍可参与计算
    let y = &h_detached * &w;
    let target = graph.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 梯度应该存在
    let w_grad = w.grad().unwrap();
    assert!(w_grad.is_some());
}
