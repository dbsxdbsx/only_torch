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
use crate::nn::{Init, VarActivationOps, VarLossOps, VarMatrixOps};
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
    // 在动态图架构下，Var 持有 Weak<GraphInner>，
    // 因此 Graph handle 必须保持存活，Var 才能恢复 Graph。
    let graph = Graph::new();
    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();

    // 通过 Var 恢复 Graph handle
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
    let h_detached = h.detach();

    // detach 后的 Var 仍可参与计算
    let y = &h_detached * &w;
    let target = graph.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    loss.backward().unwrap();

    // 梯度应该存在
    let w_grad = w.grad().unwrap();
    assert!(w_grad.is_some());
}

// ==================== Var-Tensor 混合运算测试 ====================

/// 测试 Var + Tensor 混合加法
#[test]
fn test_var_tensor_add() {
    let graph = Graph::new();
    let var = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))
        .unwrap();
    let tensor = Tensor::new(&[10.0, 20.0, 30.0], &[3, 1]);

    // Var + &Tensor
    let result = &var + &tensor;
    result.forward().unwrap();
    assert_eq!(
        result.value().unwrap().unwrap().data_as_slice(),
        &[11.0, 22.0, 33.0]
    );

    // &Tensor + Var
    let result2 = &tensor + var.clone();
    result2.forward().unwrap();
    assert_eq!(
        result2.value().unwrap().unwrap().data_as_slice(),
        &[11.0, 22.0, 33.0]
    );
}

/// 测试 Var - Tensor 混合减法
#[test]
fn test_var_tensor_sub() {
    let graph = Graph::new();
    let var = graph
        .input(&Tensor::new(&[10.0, 20.0, 30.0], &[3, 1]))
        .unwrap();
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);

    // Var - Tensor
    let result = &var - &tensor;
    result.forward().unwrap();
    assert_eq!(
        result.value().unwrap().unwrap().data_as_slice(),
        &[9.0, 18.0, 27.0]
    );

    // Tensor - Var（注意顺序）
    let result2 = &tensor - &var;
    result2.forward().unwrap();
    assert_eq!(
        result2.value().unwrap().unwrap().data_as_slice(),
        &[-9.0, -18.0, -27.0]
    );
}

/// 测试 Var * Tensor 混合乘法
#[test]
fn test_var_tensor_mul() {
    let graph = Graph::new();
    let var = graph
        .input(&Tensor::new(&[2.0, 3.0, 4.0], &[3, 1]))
        .unwrap();
    let tensor = Tensor::new(&[10.0, 10.0, 10.0], &[3, 1]);

    // Var * Tensor
    let result = &var * &tensor;
    result.forward().unwrap();
    assert_eq!(
        result.value().unwrap().unwrap().data_as_slice(),
        &[20.0, 30.0, 40.0]
    );

    // Tensor * Var
    let result2 = tensor * var;
    result2.forward().unwrap();
    assert_eq!(
        result2.value().unwrap().unwrap().data_as_slice(),
        &[20.0, 30.0, 40.0]
    );
}

/// 测试 Var / Tensor 混合除法
#[test]
fn test_var_tensor_div() {
    let graph = Graph::new();
    let var = graph
        .input(&Tensor::new(&[10.0, 20.0, 30.0], &[3, 1]))
        .unwrap();
    let tensor = Tensor::new(&[2.0, 4.0, 5.0], &[3, 1]);

    // Var / Tensor
    let result = &var / &tensor;
    result.forward().unwrap();
    assert_eq!(
        result.value().unwrap().unwrap().data_as_slice(),
        &[5.0, 5.0, 6.0]
    );

    // Tensor / Var
    let result2 = &tensor / &var;
    result2.forward().unwrap();
    assert_eq!(
        result2.value().unwrap().unwrap().data_as_slice(),
        &[0.2, 0.2, 1.0 / 6.0]
    );
}

/// 测试 mse_loss 接受 Tensor（通过 LossTarget trait）
#[test]
fn test_mse_loss_tensor() {
    use crate::nn::Init;

    let graph = Graph::new();
    let pred = graph.parameter(&[2, 1], Init::Zeros, "pred").unwrap();
    pred.set_value(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();

    // target 直接是 Tensor
    let target_tensor = Tensor::new(&[1.5, 2.5], &[2, 1]);

    // 直接使用 mse_loss（自动识别 Tensor 类型）
    let loss = pred.mse_loss(&target_tensor).unwrap();
    loss.forward().unwrap();

    // mse = ((1-1.5)^2 + (2-2.5)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
    let loss_val = loss.value().unwrap().unwrap();
    assert!((loss_val.get_data_number().unwrap() - 0.25).abs() < 1e-6);

    // 反向传播也应正常工作
    loss.backward().unwrap();
    let grad = pred.grad().unwrap().unwrap();
    // d(mse)/d(pred) = 2*(pred - target) / n
    assert!((grad[[0, 0]] - (-0.5)).abs() < 1e-6); // 2*(1-1.5)/2
    assert!((grad[[1, 0]] - (-0.5)).abs() < 1e-6); // 2*(2-2.5)/2
}

/// 测试 Var-Tensor 混合运算的反向传播
#[test]
fn test_var_tensor_mixed_backward() {
    use crate::nn::Init;

    let graph = Graph::new();
    let w = graph.parameter(&[1, 1], Init::Ones, "w").unwrap();
    let x = Tensor::new(&[2.0], &[1, 1]);

    // y = w * x (Var * Tensor)
    let y = &w * &x;

    // loss = (y - 4)^2  (target 也用 Tensor)
    let target = Tensor::new(&[4.0], &[1, 1]);
    let loss = y.mse_loss(&target).unwrap();  // 直接传 Tensor

    loss.forward().unwrap();
    loss.backward().unwrap();

    // y = w * 2 = 1 * 2 = 2
    // loss = (2 - 4)^2 = 4
    // d(loss)/d(w) = 2 * (y - target) * x = 2 * (2 - 4) * 2 = -8
    let w_grad = w.grad().unwrap().unwrap();
    assert!((w_grad[[0, 0]] - (-8.0)).abs() < 1e-6);
}

// ==================== 复杂场景综合测试 ====================

/// 复杂 Var-Tensor 混合运算的梯度追踪测试
///
/// 场景：多参数、多 Tensor、detach、交换顺序、多路径汇合
/// 验证：无论多复杂的计算图，梯度都能正确追踪
#[test]
fn test_complex_mixed_gradient_tracking() {
    use crate::nn::Init;

    let graph = Graph::new();

    // ========== 创建可训练参数 ==========
    let x = graph.parameter(&[2, 1], Init::Zeros, "x").unwrap();
    let y = graph.parameter(&[2, 1], Init::Zeros, "y").unwrap();
    x.set_value(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    y.set_value(&Tensor::new(&[3.0, 4.0], &[2, 1])).unwrap();

    // ========== 创建常量 Tensor ==========
    let t1 = Tensor::new(&[2.0, 2.0], &[2, 1]); // 乘法因子
    let t2 = Tensor::new(&[0.5, 0.5], &[2, 1]); // 加法常量

    // ========== 复杂计算图 ==========
    // Step 1: a = x * t1  (Var * Tensor)
    let a = &x * &t1;

    // Step 2: b = t2 + y  (Tensor + Var，测试交换律)
    let b = &t2 + &y;

    // Step 3: c = a + b   (两条路径汇合)
    let c = &a + &b;

    // Step 4: d = c * scalar_tensor (标量乘法)
    let scalar = Tensor::new(&[0.5, 0.5], &[2, 1]);
    let d = &c * &scalar;

    // Step 5: e = x.detach() * y (detach 后与另一个 Var 运算)
    // detach 会切断 x 方向的梯度，但 y 方向仍然有梯度
    let x_detached = x.detach();
    let e = &x_detached * &y;

    // Step 6: f = d + e (最终汇合)
    let f = &d + &e;

    // ========== 目标和损失 ==========
    let target = Tensor::new(&[10.0, 20.0], &[2, 1]);
    let loss = f.mse_loss(&target).unwrap();

    // ========== 前向传播 ==========
    loss.forward().unwrap();

    // 手算中间值验证：
    // a = x * t1 = [1, 2] * [2, 2] = [2, 4]
    // b = t2 + y = [0.5, 0.5] + [3, 4] = [3.5, 4.5]
    // c = a + b = [2, 4] + [3.5, 4.5] = [5.5, 8.5]
    // d = c * 0.5 = [2.75, 4.25]
    // e = x_detach * y = [1, 2] * [3, 4] = [3, 8]
    // f = d + e = [2.75, 4.25] + [3, 8] = [5.75, 12.25]
    // loss = mse(f, target) = ((5.75-10)^2 + (12.25-20)^2) / 2
    //      = (18.0625 + 60.0625) / 2 = 39.0625

    let loss_val = loss.value().unwrap().unwrap().get_data_number().unwrap();
    assert!(
        (loss_val - 39.0625).abs() < 1e-4,
        "loss 前向值错误: {} vs 39.0625",
        loss_val
    );

    // ========== 反向传播 ==========
    loss.backward().unwrap();

    // 手算梯度：
    // d(loss)/d(f) = 2*(f - target) / n = [(5.75-10), (12.25-20)] = [-4.25, -7.75] (已除以 n=2)
    // 实际 upstream = 2/n * (f - target) = [-4.25, -7.75]
    //
    // 路径 1: f <- d <- c <- a <- x
    //   d(f)/d(d) = 1
    //   d(d)/d(c) = scalar = 0.5
    //   d(c)/d(a) = 1
    //   d(a)/d(x) = t1 = 2
    //   所以 d(loss)/d(x) via 路径1 = upstream * 1 * 0.5 * 1 * 2 = upstream * 1.0
    //
    // 路径 2: f <- e <- x_detach (被 detach 切断，无梯度)
    //
    // 因此 x 的梯度 = [-4.25, -7.75] * 1.0 = [-4.25, -7.75]

    let x_grad = x.grad().unwrap().unwrap();
    assert!(
        (x_grad[[0, 0]] - (-4.25)).abs() < 1e-4,
        "x[0] 梯度错误: {} vs -4.25",
        x_grad[[0, 0]]
    );
    assert!(
        (x_grad[[1, 0]] - (-7.75)).abs() < 1e-4,
        "x[1] 梯度错误: {} vs -7.75",
        x_grad[[1, 0]]
    );

    // y 的梯度来自两条路径：
    // 路径 1: f <- d <- c <- b <- y
    //   d(b)/d(y) = 1
    //   所以贡献 = upstream * 1 * 0.5 * 1 * 1 = upstream * 0.5 = [-2.125, -3.875]
    //
    // 路径 2: f <- e <- y
    //   d(e)/d(y) = x_detach.value = [1, 2]
    //   所以贡献 = upstream * 1 * x_detach = [-4.25, -7.75] * [1, 2] = [-4.25, -15.5]
    //
    // 总梯度 = [-2.125, -3.875] + [-4.25, -15.5] = [-6.375, -19.375]

    let y_grad = y.grad().unwrap().unwrap();
    assert!(
        (y_grad[[0, 0]] - (-6.375)).abs() < 1e-4,
        "y[0] 梯度错误: {} vs -6.375",
        y_grad[[0, 0]]
    );
    assert!(
        (y_grad[[1, 0]] - (-19.375)).abs() < 1e-4,
        "y[1] 梯度错误: {} vs -19.375",
        y_grad[[1, 0]]
    );
}

/// 测试 Tensor 作为"常数因子"时的梯度正确性
///
/// 场景：同一个 Tensor 在计算图中多次使用
#[test]
fn test_tensor_as_constant_multiple_uses() {
    use crate::nn::Init;

    let graph = Graph::new();
    let w = graph.parameter(&[2, 1], Init::Zeros, "w").unwrap();
    w.set_value(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();

    // 同一个 Tensor 多次使用
    let scale = Tensor::new(&[3.0, 3.0], &[2, 1]);

    // a = w * scale
    let a = &w * &scale;
    // b = a + scale  (同一个 Tensor 再次使用)
    let b = &a + &scale;
    // c = b * scale  (第三次使用)
    let c = &b * &scale;

    let target = Tensor::new(&[30.0, 60.0], &[2, 1]);
    let loss = c.mse_loss(&target).unwrap();

    loss.forward().unwrap();

    // 手算：
    // a = [1, 2] * [3, 3] = [3, 6]
    // b = [3, 6] + [3, 3] = [6, 9]
    // c = [6, 9] * [3, 3] = [18, 27]
    // loss = ((18-30)^2 + (27-60)^2) / 2 = (144 + 1089) / 2 = 616.5

    let loss_val = loss.value().unwrap().unwrap().get_data_number().unwrap();
    assert!(
        (loss_val - 616.5).abs() < 1e-4,
        "loss 错误: {} vs 616.5",
        loss_val
    );

    loss.backward().unwrap();

    // 梯度分析：
    // d(loss)/d(c) = 2*(c - target) / n = [(18-30), (27-60)] = [-12, -33]
    // d(c)/d(b) = scale = [3, 3]
    // d(b)/d(a) = 1
    // d(a)/d(w) = scale = [3, 3]
    // d(loss)/d(w) = [-12, -33] * 3 * 1 * 3 = [-108, -297]

    let w_grad = w.grad().unwrap().unwrap();
    assert!(
        (w_grad[[0, 0]] - (-108.0)).abs() < 1e-4,
        "w[0] 梯度错误: {} vs -108",
        w_grad[[0, 0]]
    );
    assert!(
        (w_grad[[1, 0]] - (-297.0)).abs() < 1e-4,
        "w[1] 梯度错误: {} vs -297",
        w_grad[[1, 0]]
    );
}

/// 测试多参数共享路径时的梯度累加
///
/// 场景：一个 Var 同时出现在多个分支，梯度应正确累加
#[test]
fn test_gradient_accumulation_multiple_paths() {
    use crate::nn::Init;

    let graph = Graph::new();
    let x = graph.parameter(&[1, 1], Init::Zeros, "x").unwrap();
    x.set_value(&Tensor::new(&[2.0], &[1, 1])).unwrap();

    // x 同时出现在两个分支
    let t1 = Tensor::new(&[3.0], &[1, 1]);
    let t2 = Tensor::new(&[4.0], &[1, 1]);

    let branch1 = &x * &t1; // 3x
    let branch2 = &x * &t2; // 4x

    // 两个分支汇合
    let y = &branch1 + &branch2; // 3x + 4x = 7x

    let target = Tensor::new(&[21.0], &[1, 1]); // 期望 x = 3
    let loss = y.mse_loss(&target).unwrap();

    loss.forward().unwrap();

    // y = 7 * 2 = 14
    // loss = (14 - 21)^2 = 49

    loss.backward().unwrap();

    // d(loss)/d(y) = 2*(14 - 21) / 1 = -14
    // d(y)/d(x) = d(3x)/d(x) + d(4x)/d(x) = 3 + 4 = 7
    // d(loss)/d(x) = -14 * 7 = -98

    let x_grad = x.grad().unwrap().unwrap();
    assert!(
        (x_grad[[0, 0]] - (-98.0)).abs() < 1e-4,
        "x 梯度错误: {} vs -98",
        x_grad[[0, 0]]
    );
}

/// 测试 detach 完全切断梯度流
#[test]
fn test_detach_completely_blocks_gradient() {
    use crate::nn::Init;

    let graph = Graph::new();
    let x = graph.parameter(&[1, 1], Init::Zeros, "x").unwrap();
    x.set_value(&Tensor::new(&[5.0], &[1, 1])).unwrap();

    // 完全通过 detach 路径
    let x_detached = x.detach();
    let t = Tensor::new(&[2.0], &[1, 1]);
    let y = &x_detached * &t;

    let target = Tensor::new(&[20.0], &[1, 1]);
    let loss = y.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // x 应该没有梯度（被 detach 切断）
    let x_grad = x.grad();
    assert!(
        x_grad.is_err() || x_grad.unwrap().is_none(),
        "detach 后 x 不应有梯度"
    );
}

/// 测试 Var 与 Var 运算后再与 Tensor 运算
#[test]
fn test_var_var_then_tensor() {
    use crate::nn::Init;

    let graph = Graph::new();
    let a = graph.parameter(&[2, 1], Init::Zeros, "a").unwrap();
    let b = graph.parameter(&[2, 1], Init::Zeros, "b").unwrap();
    a.set_value(&Tensor::new(&[1.0, 2.0], &[2, 1])).unwrap();
    b.set_value(&Tensor::new(&[3.0, 4.0], &[2, 1])).unwrap();

    // Var + Var
    let c = &a + &b; // [4, 6]

    // 结果再与 Tensor 运算
    let t = Tensor::new(&[2.0, 2.0], &[2, 1]);
    let d = &c * &t; // [8, 12]

    // 再来一层
    let e = &t + &d; // [10, 14]

    let target = Tensor::new(&[10.0, 10.0], &[2, 1]);
    let loss = e.mse_loss(&target).unwrap();

    loss.forward().unwrap();

    // e = [10, 14], target = [10, 10]
    // loss = ((10-10)^2 + (14-10)^2) / 2 = 16 / 2 = 8

    let loss_val = loss.value().unwrap().unwrap().get_data_number().unwrap();
    assert!((loss_val - 8.0).abs() < 1e-4, "loss 错误: {} vs 8", loss_val);

    loss.backward().unwrap();

    // d(loss)/d(e) = 2*(e - target) / n = [0, 4]
    // d(e)/d(d) = 1
    // d(d)/d(c) = t = [2, 2]
    // d(c)/d(a) = 1, d(c)/d(b) = 1
    // d(loss)/d(a) = [0, 4] * 1 * 2 * 1 = [0, 8]
    // d(loss)/d(b) = [0, 4] * 1 * 2 * 1 = [0, 8]

    let a_grad = a.grad().unwrap().unwrap();
    let b_grad = b.grad().unwrap().unwrap();

    assert!(
        (a_grad[[0, 0]] - 0.0).abs() < 1e-4,
        "a[0] 梯度错误: {}",
        a_grad[[0, 0]]
    );
    assert!(
        (a_grad[[1, 0]] - 8.0).abs() < 1e-4,
        "a[1] 梯度错误: {}",
        a_grad[[1, 0]]
    );
    assert!(
        (b_grad[[0, 0]] - 0.0).abs() < 1e-4,
        "b[0] 梯度错误: {}",
        b_grad[[0, 0]]
    );
    assert!(
        (b_grad[[1, 0]] - 8.0).abs() < 1e-4,
        "b[1] 梯度错误: {}",
        b_grad[[1, 0]]
    );
}

// ==================== backward_ex 高层 API 测试 ====================

/// 多 loss 共享参数，backward_ex(true) + backward_ex(false) 梯度累积正确
#[test]
fn test_var_backward_ex_multi_loss() {
    use crate::nn::var_ops::{VarLossOps, VarMatrixOps};

    let graph = Graph::new();
    // 共享权重 [2, 1]
    let w = graph.parameter(&[2, 1], Init::Ones, "w").unwrap();
    // 两个不同 loss（输入 [1, 2] @ 权重 [2, 1] = [1, 1]）
    let x1 = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let y1 = x1.matmul(&w).unwrap();
    let t1 = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss1 = y1.mse_loss(&t1).unwrap();

    let x2 = graph.input(&Tensor::new(&[3.0, 4.0], &[1, 2])).unwrap();
    let y2 = x2.matmul(&w).unwrap();
    let t2 = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss2 = y2.mse_loss(&t2).unwrap();

    // 多 loss backward
    graph.zero_grad().unwrap();
    let v1 = loss1.backward_ex(true).unwrap();
    let v2 = loss2.backward_ex(false).unwrap();
    assert!(v1 > 0.0);
    assert!(v2 > 0.0);

    // w 应该有梯度（来自两个 loss 的累积）
    let w_grad_multi = w.grad().unwrap().unwrap();

    // 单独 backward loss1 验证梯度累积效果
    graph.zero_grad().unwrap();
    loss1.backward().unwrap();
    let w_grad_single = w.grad().unwrap().unwrap();

    // 两个 loss 累积的梯度应该比单 loss 的绝对值大
    assert!(
        w_grad_multi[[0, 0]].abs() > w_grad_single[[0, 0]].abs(),
        "多 loss 梯度应大于单 loss 梯度"
    );
}

/// 单 loss 场景，backward() 和 backward_ex(false) 行为一致
#[test]
fn test_var_backward_ex_single_loss() {
    use crate::nn::var_ops::{VarLossOps, VarMatrixOps};

    let graph = Graph::new_with_seed(42);
    let w = graph
        .parameter(&[2, 1], Init::Normal { mean: 0.0, std: 1.0 }, "w")
        .unwrap();
    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let y = x.matmul(&w).unwrap();
    let t = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = y.mse_loss(&t).unwrap();

    // backward()
    graph.zero_grad().unwrap();
    let v1 = loss.backward().unwrap();
    let grad1 = w.grad().unwrap().unwrap();

    // backward_ex(false)
    graph.zero_grad().unwrap();
    let v2 = loss.backward_ex(false).unwrap();
    let grad2 = w.grad().unwrap().unwrap();

    // 两者应完全一致
    assert!((v1 - v2).abs() < 1e-6, "loss 值应一致: {} vs {}", v1, v2);
    assert!(
        (grad1[[0, 0]] - grad2[[0, 0]]).abs() < 1e-6,
        "梯度应一致"
    );
}

/// 方案 C：同一 loss 可多次 backward（值由 Rc 管理，天然支持）
#[test]
fn test_var_backward_multiple_times() {
    use crate::nn::var_ops::{VarLossOps, VarMatrixOps};

    let graph = Graph::new_with_seed(42);
    let w = graph
        .parameter(&[2, 1], Init::Normal { mean: 0.0, std: 1.0 }, "w")
        .unwrap();
    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let y = x.matmul(&w).unwrap();
    let t = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = y.mse_loss(&t).unwrap();

    // 第一次 backward
    graph.zero_grad().unwrap();
    let v1 = loss.backward().unwrap();
    let grad1 = w.grad().unwrap().unwrap().clone();

    // 第二次 backward（方案 C 下不需要 retain_graph=true）
    graph.zero_grad().unwrap();
    let v2 = loss.backward().unwrap();
    let grad2 = w.grad().unwrap().unwrap();

    // 两次应完全一致
    assert!((v1 - v2).abs() < 1e-6, "两次 backward 的 loss 值应一致");
    assert!(
        (grad1[[0, 0]] - grad2[[0, 0]]).abs() < 1e-6,
        "两次 backward 的梯度应一致"
    );
}
