//! 梯度流控制机制测试
//!
//! 测试 `no_grad`、`detach`、多次 backward 等梯度控制机制
//! 参考设计文档: `.doc/design/gradient_flow_control_design.md`
//!
//! 使用底层 `Graph::new()` + `inner_rc()` + `GraphInner` API

use approx::assert_abs_diff_eq;

use crate::nn::Graph;
use crate::tensor::Tensor;
use std::rc::Rc;

// ============================================================================
// 1. no_grad 机制测试
// ============================================================================

/// 测试: is_grad_enabled 与 is_train_mode 一致
#[test]
fn test_is_grad_enabled_equals_is_train_mode() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 默认训练模式
    assert!(gi.is_grad_enabled());
    assert!(gi.is_train_mode());

    // 切换到评估模式
    gi.set_eval_mode();
    assert!(!gi.is_grad_enabled());
    assert!(!gi.is_train_mode());

    // 切换回训练模式
    gi.set_train_mode();
    assert!(gi.is_grad_enabled());
    assert!(gi.is_train_mode());
}

/// 测试: no_grad_scope 基本功能 - 临时禁用梯度
#[test]
fn test_no_grad_scope_basic() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 默认是训练模式
    assert!(gi.is_grad_enabled());

    // 进入 no_grad 上下文
    gi.no_grad_scope(|g| {
        // 应该处于评估模式
        assert!(!g.is_grad_enabled());
        assert!(!g.is_train_mode());
    });

    // 退出后恢复训练模式
    assert!(gi.is_grad_enabled());
    assert!(gi.is_train_mode());
}

/// 测试: no_grad_scope 从评估模式开始
#[test]
fn test_no_grad_scope_from_eval_mode() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 先切换到评估模式
    gi.set_eval_mode();
    assert!(!gi.is_grad_enabled());

    // 进入 no_grad 上下文（已经是评估模式）
    gi.no_grad_scope(|g| {
        assert!(!g.is_grad_enabled());
    });

    // 退出后应该保持评估模式（因为之前就是评估模式）
    assert!(!gi.is_grad_enabled());
}

/// 测试: no_grad_scope 返回值传递
#[test]
fn test_no_grad_scope_return_value() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 返回简单值
    let result: i32 = gi.no_grad_scope(|_g| 42);
    assert_eq!(result, 42);

    // 返回 Result
    let result: Result<f64, crate::nn::GraphError> = gi.no_grad_scope(|_g| Ok(3.14));
    assert_eq!(result.unwrap(), 3.14);

    // 返回复杂类型
    let result: Vec<i32> = gi.no_grad_scope(|_g| vec![1, 2, 3]);
    assert_eq!(result, vec![1, 2, 3]);
}

/// 测试: no_grad_scope 错误传播
#[test]
fn test_no_grad_scope_error_propagation() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 闭包返回错误
    let result: Result<(), crate::nn::GraphError> = gi.no_grad_scope(|_g| {
        Err(crate::nn::GraphError::InvalidOperation(
            "测试错误".to_string(),
        ))
    });

    // 错误应该被传播出来
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        crate::nn::GraphError::InvalidOperation("测试错误".to_string())
    );

    // 即使出错，模式也应该恢复
    assert!(gi.is_grad_enabled());
}

/// 测试: no_grad_scope 嵌套调用
#[test]
fn test_no_grad_scope_nested() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    assert!(gi.is_grad_enabled());

    gi.no_grad_scope(|g| {
        assert!(!g.is_grad_enabled());

        // 嵌套调用
        g.no_grad_scope(|g2| {
            assert!(!g2.is_grad_enabled());

            // 再嵌套一层
            g2.no_grad_scope(|g3| {
                assert!(!g3.is_grad_enabled());
            });

            // 退出第三层后仍在评估模式（因为第二层也是 no_grad）
            assert!(!g2.is_grad_enabled());
        });

        // 退出第二层后仍在评估模式（因为第一层也是 no_grad）
        assert!(!g.is_grad_enabled());
    });

    // 完全退出后恢复训练模式
    assert!(gi.is_grad_enabled());
}

/// 测试: no_grad_scope 与前向传播集成
#[test]
fn test_no_grad_scope_with_forward() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建简单网络: x -> tanh -> output
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let output = gi
        .create_tanh_node(Rc::clone(&x), Some("output"))
        .unwrap();

    // 设置输入
    let input_data = Tensor::new(&[0.5, -0.3], &[2, 1]);
    x.set_value(Some(&input_data)).unwrap();

    // 在 no_grad 上下文中前向传播
    let output_val = gi.no_grad_scope(|g| {
        assert!(!g.is_grad_enabled());
        g.forward_via_node_inner(&output).unwrap();
        output.value()
    });

    // 验证前向传播成功
    assert!(output_val.is_some());
    let val = output_val.unwrap();
    assert_eq!(val.shape(), &[2, 1]);

    // 验证模式恢复
    assert!(gi.is_grad_enabled());
}

/// 测试: no_grad_scope 核心保证 - 相同输入产生相同 loss，区别仅在于梯度
#[test]
fn test_no_grad_scope_same_input_same_loss_no_gradient() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建简单回归网络
    let x = gi
        .create_basic_input_node(&[1, 2], Some("x"))
        .unwrap();
    let w = gi.create_parameter_node(&[2, 1], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w))
        .unwrap();
    let pred = gi
        .create_mat_mul_node(vec![Rc::clone(&x), Rc::clone(&w)], Some("pred"))
        .unwrap();
    let y = gi
        .create_basic_input_node(&[1, 1], Some("y"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(Rc::clone(&pred), Rc::clone(&y), Some("loss"))
        .unwrap();

    // 使用相同的输入数据
    let input_x = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let target_y = Tensor::new(&[3.0], &[1, 1]);

    // ===== 正常模式：前向 + 反向 =====
    x.set_value(Some(&input_x)).unwrap();
    y.set_value(Some(&target_y)).unwrap();
    gi.forward_via_node_inner(&loss).unwrap();
    let normal_loss_value = loss.value().unwrap().get_data_number().unwrap();

    // 正常模式可以计算梯度
    gi.backward_via_node_inner(&loss).unwrap();
    let normal_gradient = w.grad();
    assert!(normal_gradient.is_some(), "正常模式应产生梯度");

    // 清除状态，准备下一次计算
    gi.zero_grad().unwrap();

    // ===== no_grad 模式：相同输入 =====
    let no_grad_loss_value = gi.no_grad_scope(|g| {
        // 使用完全相同的输入数据
        x.set_value(Some(&input_x)).unwrap();
        y.set_value(Some(&target_y)).unwrap();
        g.forward_via_node_inner(&loss).unwrap();
        loss.value().unwrap().get_data_number().unwrap()
    });

    // 核心验证 1: 相同输入 → 相同 loss
    assert_abs_diff_eq!(normal_loss_value, no_grad_loss_value, epsilon = 1e-6);

    // 核心验证 2: no_grad 模式后不应有残留梯度（zero_grad 已清除）
    let no_grad_gradient = w.grad();
    assert!(no_grad_gradient.is_none(), "no_grad 模式后不应有残留梯度");

    // 验证模式恢复
    assert!(gi.is_grad_enabled());
}

/// 测试: no_grad_scope 多次调用
#[test]
fn test_no_grad_scope_multiple_calls() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    for i in 0..5 {
        assert!(gi.is_grad_enabled(), "第 {i} 次调用前应为训练模式");

        gi.no_grad_scope(|g| {
            assert!(!g.is_grad_enabled(), "第 {i} 次调用中应为评估模式");
        });

        assert!(gi.is_grad_enabled(), "第 {i} 次调用后应恢复训练模式");
    }
}

/// 测试: no_grad_scope 与 set_eval_mode 交互
#[test]
fn test_no_grad_scope_interaction_with_eval_mode() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 场景 1: 在 no_grad 中手动切换模式
    gi.no_grad_scope(|g| {
        assert!(!g.is_grad_enabled());

        // 手动切换到训练模式（不推荐但应该能工作）
        g.set_train_mode();
        assert!(g.is_grad_enabled());

        // 切换回评估模式
        g.set_eval_mode();
        assert!(!g.is_grad_enabled());
    });

    // 退出后恢复（因为进入前是训练模式）
    assert!(gi.is_grad_enabled());
}

/// 测试: no_grad_scope 闭包中的可变借用
#[test]
fn test_no_grad_scope_mutable_operations() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建节点
    let x = gi
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let y_node = gi
        .create_tanh_node(Rc::clone(&x), Some("y"))
        .unwrap();

    // 在 no_grad 中执行可变操作
    gi.no_grad_scope(|g| {
        // 设置值
        let data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        x.set_value(Some(&data)).unwrap();

        // 前向传播
        g.forward_via_node_inner(&y_node).unwrap();

        // 验证结果
        let result = y_node.value();
        assert!(result.is_some());
    });
}

/// 测试: no_grad_scope 与种子的交互
#[test]
fn test_no_grad_scope_with_seeded_graph() {
    let graph = Graph::new_with_seed(42);
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    assert!(gi.has_seed());
    assert!(gi.is_grad_enabled());

    gi.no_grad_scope(|g| {
        assert!(g.has_seed()); // 种子应该保持不变
        assert!(!g.is_grad_enabled());

        // 可以在 no_grad 中创建参数节点
        let _w = g.create_parameter_node(&[2, 2], Some("w")).unwrap();
    });

    // 种子和模式都应该正确恢复
    assert!(gi.has_seed());
    assert!(gi.is_grad_enabled());
}

/// 测试: no_grad_scope 不阻止 backward 调用（与 PyTorch 有差异）
///
/// 说明：
/// - PyTorch 动态图：no_grad 内的 forward 结果无 grad_fn，backward 会失败
/// - only_torch 静态图：图在节点创建时已构建，backward 技术上可行
///
/// 设计决策：
/// - 不阻止执行（允许调试等合法用例）
/// - 输出警告提醒用户（大多数情况是误用）
///
/// 预期行为：
/// - backward 成功执行
/// - stderr 输出警告信息："[only_torch 警告] 在 no_grad/eval 模式下调用 backward..."
///
/// 参考: `.doc/design/gradient_flow_control_design.md` 1.7 节
#[test]
fn test_no_grad_scope_backward_still_works() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建简单网络: x -> w -> y (标量)
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let w = gi.create_parameter_node(&[1, 2], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w))
        .unwrap();
    let y = gi
        .create_mat_mul_node(vec![Rc::clone(&w), Rc::clone(&x)], Some("y"))
        .unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    x.set_value(Some(&input_data)).unwrap();

    // 在 no_grad_scope 内执行 forward + backward
    let backward_result = gi.no_grad_scope(|g| {
        assert!(!g.is_grad_enabled(), "应处于 no_grad 模式");

        // forward 应该正常工作
        g.forward_via_node_inner(&y).unwrap();
        assert!(y.value().is_some(), "forward 应产生值");

        // backward 也应该正常工作（打印警告但不报错）
        // y 是 [1,1]（标量），可以作为 loss
        g.backward_via_node_inner(&y)
    });

    // backward 应该成功执行
    assert!(
        backward_result.is_ok(),
        "no_grad_scope 内 backward 应该成功: {:?}",
        backward_result
    );

    // 退出 no_grad_scope 后模式应恢复
    assert!(gi.is_grad_enabled());
}

/// 测试: no_grad_scope 内创建的节点在退出后的梯度行为
#[test]
fn test_no_grad_scope_nodes_created_inside() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 先创建输入节点
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();

    // 在 no_grad_scope 内创建参数和运算节点
    let (w, y) = gi.no_grad_scope(|g| {
        let w = g.create_parameter_node(&[1, 2], Some("w")).unwrap();
        g.register_parameter("w".to_string(), Rc::downgrade(&w))
            .unwrap();
        let y = g
            .create_mat_mul_node(vec![Rc::clone(&w), Rc::clone(&x)], Some("y"))
            .unwrap();
        (w, y)
    });

    // 退出 no_grad_scope 后，这些节点应该可以正常参与训练
    assert!(gi.is_grad_enabled());

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    x.set_value(Some(&input_data)).unwrap();

    // 正常前向传播
    gi.forward_via_node_inner(&y).unwrap();
    assert!(y.value().is_some());

    // 正常反向传播应该工作（y 是 [1,1]，标量）
    gi.backward_via_node_inner(&y).unwrap();
    assert!(
        w.grad().is_some(),
        "退出 no_grad_scope 后，在其中创建的节点应能正常计算梯度"
    );
}

// ============================================================================
// 2. detach 机制测试
// ============================================================================

/// 测试: set_detached / is_detached 基本功能
#[test]
fn test_detach_basic() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建节点
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let w = gi.create_parameter_node(&[1, 2], Some("w")).unwrap();
    let y = gi
        .create_mat_mul_node(vec![Rc::clone(&w), Rc::clone(&x)], Some("y"))
        .unwrap();

    // 默认不是 detached
    assert!(!w.is_detached());
    assert!(!y.is_detached());

    // detach 节点
    y.set_detached(true);
    assert!(y.is_detached());
    assert!(!w.is_detached()); // w 不受影响
    assert!(!x.is_detached()); // x 不受影响

    // attach 恢复
    y.set_detached(false);
    assert!(!y.is_detached());
}

/// 测试: detach 阻止梯度回流
#[test]
fn test_detach_blocks_gradient_flow() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建简单网络: x -> w1 -> h -> w2 -> output (标量)
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let w1 = gi.create_parameter_node(&[2, 2], Some("w1")).unwrap();
    gi.register_parameter("w1".to_string(), Rc::downgrade(&w1))
        .unwrap();
    let h = gi
        .create_mat_mul_node(vec![Rc::clone(&w1), Rc::clone(&x)], Some("h"))
        .unwrap();
    let w2 = gi.create_parameter_node(&[1, 2], Some("w2")).unwrap();
    gi.register_parameter("w2".to_string(), Rc::downgrade(&w2))
        .unwrap();
    let output = gi
        .create_mat_mul_node(vec![Rc::clone(&w2), Rc::clone(&h)], Some("output"))
        .unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    x.set_value(Some(&input_data)).unwrap();

    // 前向传播
    gi.forward_via_node_inner(&output).unwrap();

    // ===== 场景 1: 正常反向传播 =====
    gi.backward_via_node_inner(&output).unwrap();

    // w1 和 w2 都应该有梯度
    assert!(w1.grad().is_some());
    assert!(w2.grad().is_some());

    // 清除梯度
    gi.zero_grad().unwrap();

    // ===== 场景 2: detach h，w1 不应该有梯度 =====
    h.set_detached(true);
    gi.forward_via_node_inner(&output).unwrap();
    gi.backward_via_node_inner(&output).unwrap();

    // w1 不应该有梯度（被 h 的 detach 阻断）
    assert!(
        w1.grad().is_none(),
        "w1 不应有梯度，因为 h 被 detach 阻断了梯度流"
    );
    // w2 应该有梯度（在 detach 点之后）
    assert!(w2.grad().is_some());

    // 清除并恢复
    gi.zero_grad().unwrap();
    h.set_detached(false);

    // ===== 场景 3: 恢复后 w1 应该有梯度 =====
    gi.forward_via_node_inner(&output).unwrap();
    gi.backward_via_node_inner(&output).unwrap();
    assert!(w1.grad().is_some());
    assert!(w2.grad().is_some());
}

/// 测试: detach 不影响前向传播
#[test]
fn test_detach_does_not_affect_forward() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建网络
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let y = gi
        .create_tanh_node(Rc::clone(&x), Some("y"))
        .unwrap();

    // 设置输入
    let input_data = Tensor::new(&[0.5, -0.3], &[2, 1]);
    x.set_value(Some(&input_data)).unwrap();

    // 正常前向传播
    gi.forward_via_node_inner(&y).unwrap();
    let result1 = y.value().unwrap();

    // detach y
    y.set_detached(true);

    // 重新设置输入并前向传播
    x.set_value(Some(&input_data)).unwrap();
    gi.forward_via_node_inner(&y).unwrap();
    let result2 = y.value().unwrap();

    // 结果应该相同
    assert_eq!(result1, result2);
}

/// 测试: 多次 detach/attach 切换（包括幂等性和实际 backward 效果验证）
#[test]
fn test_detach_attach_multiple_times() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建简单网络: x -> w -> y -> loss
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let w = gi.create_parameter_node(&[1, 2], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w))
        .unwrap();
    let y = gi
        .create_mat_mul_node(vec![Rc::clone(&w), Rc::clone(&x)], Some("y"))
        .unwrap();
    let target = gi
        .create_basic_input_node(&[1, 1], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(Rc::clone(&y), Rc::clone(&target), Some("loss"))
        .unwrap();

    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let target_data = Tensor::new(&[0.0], &[1, 1]);
    x.set_value(Some(&input_data)).unwrap();
    target.set_value(Some(&target_data)).unwrap();

    // ===== 1. 测试交替切换 =====
    for i in 0..10 {
        if i % 2 == 0 {
            w.set_detached(true);
            assert!(w.is_detached());
        } else {
            w.set_detached(false);
            assert!(!w.is_detached());
        }
    }

    // ===== 2. 测试幂等性：连续 detach =====
    w.set_detached(true);
    assert!(w.is_detached());
    w.set_detached(true); // 再次 detach
    assert!(w.is_detached()); // 仍然是 detached

    // ===== 3. 测试幂等性：连续 attach =====
    w.set_detached(false);
    assert!(!w.is_detached());
    w.set_detached(false); // 再次 attach
    assert!(!w.is_detached()); // 仍然是 attached

    // ===== 4. 验证 detach 状态确实影响 backward =====
    // 注意：方案 C 中 set_detached 阻止节点 **向上游传播** 梯度，
    // 而非阻止节点 **接收** 梯度。因此需要 detach 中间节点 y 来阻断流向 w 的梯度。

    // attached 状态：w 应该有梯度
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();
    assert!(w.grad().is_some());
    gi.zero_grad().unwrap();

    // detach y（中间节点）：梯度被 y 阻断，w 不应有梯度
    y.set_detached(true);
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();
    assert!(
        w.grad().is_none(),
        "y 被 detach 后，w 不应有梯度"
    );
    gi.zero_grad().unwrap();

    // 恢复 attached：w 应该又有梯度
    y.set_detached(false);
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();
    assert!(w.grad().is_some());
}

/// 测试: detach 对 Input 节点（语义上没意义但不应崩溃）
#[test]
fn test_detach_input_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    let x = gi
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();

    // 技术上可以 detach，但语义上没有意义
    x.set_detached(true);
    assert!(x.is_detached());

    // attach 也应该工作
    x.set_detached(false);
    assert!(!x.is_detached());
}

/// 测试: GAN 风格训练模式
#[test]
fn test_detach_gan_style_training() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 简化的 GAN 结构:
    // z(输入) -> G_w(生成器参数) -> fake_data -> D_w(判别器参数) -> d_output -> loss

    // 生成器部分
    let z = gi
        .create_basic_input_node(&[2, 1], Some("z"))
        .unwrap();
    let g_w = gi.create_parameter_node(&[3, 2], Some("g_w")).unwrap();
    gi.register_parameter("g_w".to_string(), Rc::downgrade(&g_w))
        .unwrap();
    let fake_data = gi
        .create_mat_mul_node(vec![Rc::clone(&g_w), Rc::clone(&z)], Some("fake"))
        .unwrap();

    // 判别器部分
    let d_w = gi.create_parameter_node(&[1, 3], Some("d_w")).unwrap();
    gi.register_parameter("d_w".to_string(), Rc::downgrade(&d_w))
        .unwrap();
    let d_output = gi
        .create_mat_mul_node(vec![Rc::clone(&d_w), Rc::clone(&fake_data)], Some("d_out"))
        .unwrap();

    // 添加 loss 节点使输出为标量
    let d_target = gi
        .create_basic_input_node(&[1, 1], Some("d_target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(Rc::clone(&d_output), Rc::clone(&d_target), Some("loss"))
        .unwrap();

    // 设置输入
    let noise = Tensor::new(&[0.5, -0.3], &[2, 1]);
    let target = Tensor::new(&[1.0], &[1, 1]); // 判别器目标
    z.set_value(Some(&noise)).unwrap();
    d_target.set_value(Some(&target)).unwrap();
    gi.forward_via_node_inner(&loss).unwrap();

    // ===== 训练判别器：detach fake_data =====
    fake_data.set_detached(true);
    gi.backward_via_node_inner(&loss).unwrap();

    // d_w 应该有梯度
    assert!(d_w.grad().is_some());
    // g_w 不应该有梯度（fake_data 被 detach，梯度不会传到 g_w）
    assert!(
        g_w.grad().is_none(),
        "g_w 不应有梯度，因为 fake_data 被 detach"
    );

    gi.zero_grad().unwrap();

    // ===== 训练生成器：attach fake_data =====
    fake_data.set_detached(false);
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();

    // g_w 应该有梯度
    assert!(g_w.grad().is_some());
    // d_w 也会有梯度（因为它在 loss 到 g_w 的路径上）
    // 在 only_torch 中，如果想冻结 d_w，需要 detach 它
}

/// 测试: detach 与批量输入的兼容性
#[test]
fn test_detach_with_batch_input() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建网络: x -> w -> y -> mse_loss
    let x = gi
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let w = gi.create_parameter_node(&[2, 2], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w))
        .unwrap();
    let y = gi
        .create_mat_mul_node(vec![Rc::clone(&w), Rc::clone(&x)], Some("y"))
        .unwrap();
    let target = gi
        .create_basic_input_node(&[2, 2], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(Rc::clone(&y), Rc::clone(&target), Some("loss"))
        .unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let target_data = Tensor::new(&[1.1, 2.1, 3.1, 4.1], &[2, 2]);
    x.set_value(Some(&input_data)).unwrap();
    target.set_value(Some(&target_data)).unwrap();

    // Batch 前向传播
    gi.forward_via_node_inner(&loss).unwrap();

    // detach y 后 batch 反向传播
    y.set_detached(true);
    gi.backward_via_node_inner(&loss).unwrap();

    // w 不应该有 batch 梯度（因为 y 被 detach，梯度不会传到 w）
    assert!(
        w.grad().is_none(),
        "w 不应有梯度，因为 y 被 detach 阻断了梯度流"
    );
}

/// 测试: detach 后的节点仍然可以正常操作，且已有的 grad 不会被清除（Single 模式）
#[test]
fn test_detach_node_still_functional() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建简单网络: x -> w -> y (标量)
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let w = gi.create_parameter_node(&[1, 2], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w))
        .unwrap();
    let y = gi
        .create_mat_mul_node(vec![Rc::clone(&w), Rc::clone(&x)], Some("y"))
        .unwrap();

    // 设置输入并执行 forward + backward
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    x.set_value(Some(&input_data)).unwrap();
    gi.forward_via_node_inner(&y).unwrap();
    gi.backward_via_node_inner(&y).unwrap();

    // 验证 w 有 grad
    let grad_before = w.grad().unwrap();
    assert!(grad_before.size() > 0);

    // detach w
    w.set_detached(true);
    assert!(w.is_detached());

    // 验证 detach 后 grad 仍然保留（detach 不清除已有梯度）
    let grad_after = w.grad().unwrap();
    assert_eq!(grad_before, grad_after);

    // 仍然可以获取值
    assert!(w.value().is_some());

    // 仍然可以设置值
    let new_value = Tensor::ones(&[1, 2]);
    w.set_value(Some(&new_value)).unwrap();

    // 仍然可以获取名称
    assert_eq!(w.name(), Some("w"));
}

/// 测试: detach 梯度数值与 PyTorch 精确对照
/// PyTorch 对照: tests/python/calc_jacobi_by_pytorch/detach_gradient_values.py
///
/// 拓扑: x(input) -> w1 -> h (detached) -> w2 -> output
///
/// 验证:
/// 1. w1 应无梯度 (被 detach 阻断)
/// 2. w2 的梯度数值与 PyTorch 精确匹配
#[test]
fn test_detach_gradient_values_match_pytorch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建网络: x -> w1 -> h -> w2 -> output
    // x: [2, 1], w1: [2, 2] -> h: [2, 1], w2: [1, 2] -> output: [1, 1]
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let w1 = gi.create_parameter_node(&[2, 2], Some("w1")).unwrap();
    gi.register_parameter("w1".to_string(), Rc::downgrade(&w1))
        .unwrap();
    let h = gi
        .create_mat_mul_node(vec![Rc::clone(&w1), Rc::clone(&x)], Some("h"))
        .unwrap();
    let w2 = gi.create_parameter_node(&[1, 2], Some("w2")).unwrap();
    gi.register_parameter("w2".to_string(), Rc::downgrade(&w2))
        .unwrap();
    let output = gi
        .create_mat_mul_node(vec![Rc::clone(&w2), Rc::clone(&h)], Some("output"))
        .unwrap();

    // 设置值（与 PyTorch 脚本一致）
    x.set_value(Some(&Tensor::new(&[1.0, 2.0], &[2, 1])))
        .unwrap();
    w1.set_value(Some(&Tensor::ones(&[2, 2]))).unwrap();
    w2.set_value(Some(&Tensor::ones(&[1, 2]))).unwrap();

    // 前向传播
    gi.forward_via_node_inner(&output).unwrap();

    // 验证中间值（与 PyTorch 一致）
    let h_value = h.value().unwrap();
    assert_eq!(
        h_value,
        Tensor::new(&[3.0, 3.0], &[2, 1]),
        "h 值应为 [3, 3]"
    );

    let output_value = output.value().unwrap();
    assert_eq!(
        output_value,
        Tensor::new(&[6.0], &[1, 1]),
        "output 值应为 [6]"
    );

    // ===== detach h =====
    h.set_detached(true);

    // 反向传播
    gi.backward_via_node_inner(&output).unwrap();

    // ===== 验证梯度（PyTorch 对照值）=====
    // w1 应无梯度（被 detach 阻断）
    assert!(
        w1.grad().is_none(),
        "w1 应无梯度 (被 h 的 detach 阻断)"
    );

    // w2 梯度应与 PyTorch 精确匹配
    // PyTorch 输出: w2.grad = [[3., 3.]]
    let expected_w2_grad = Tensor::new(&[3.0, 3.0], &[1, 2]);
    let actual_w2_grad = w2.grad().unwrap();
    assert_eq!(
        actual_w2_grad, expected_w2_grad,
        "w2 梯度应与 PyTorch 匹配"
    );
}

/// 测试: detach 后的节点仍然可以正常操作，且已有的 grad 不会被清除（Batch 模式）
#[test]
fn test_detach_node_still_functional_batch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建网络: x -> w -> y -> loss
    let x = gi
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let w = gi.create_parameter_node(&[2, 2], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w))
        .unwrap();
    let y = gi
        .create_mat_mul_node(vec![Rc::clone(&w), Rc::clone(&x)], Some("y"))
        .unwrap();
    let target = gi
        .create_basic_input_node(&[2, 2], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(Rc::clone(&y), Rc::clone(&target), Some("loss"))
        .unwrap();

    // 设置输入并执行 forward + backward
    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let target_data = Tensor::new(&[1.1, 2.1, 3.1, 4.1], &[2, 2]);
    x.set_value(Some(&input_data)).unwrap();
    target.set_value(Some(&target_data)).unwrap();
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();

    // 验证 w 有 grad
    let grad_before = w.grad().unwrap();
    assert!(grad_before.size() > 0);

    // detach w
    w.set_detached(true);
    assert!(w.is_detached());

    // 验证 detach 后 grad 仍然保留（detach 不清除已有梯度）
    let grad_after = w.grad().unwrap();
    assert_eq!(grad_before, grad_after);

    // 仍然可以获取值
    assert!(w.value().is_some());

    // 仍然可以设置值
    let new_value = Tensor::ones(&[2, 2]);
    w.set_value(Some(&new_value)).unwrap();

    // 仍然可以获取名称
    assert_eq!(w.name(), Some("w"));
}

// ============================================================================
// 3. 多次 backward 测试（动态图架构天然支持）
// ============================================================================

/// 测试: backward_via_node_inner 基本功能
#[test]
fn test_retain_graph_basic() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建简单网络: x -> w -> y (标量)
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let w = gi.create_parameter_node(&[1, 2], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w))
        .unwrap();
    let y = gi
        .create_mat_mul_node(vec![Rc::clone(&w), Rc::clone(&x)], Some("y"))
        .unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    x.set_value(Some(&input_data)).unwrap();

    // 前向传播
    gi.forward_via_node_inner(&y).unwrap();

    // 反向传播
    gi.backward_via_node_inner(&y).unwrap();

    // w 应该有梯度
    assert!(w.grad().is_some());

    // y 的值应该仍然存在（动态图中值由 Rc 管理，不会被释放）
    assert!(y.value().is_some());
}

/// 测试: 中间结果生命周期由 Rc 管理
///
/// 动态图架构下中间结果由 Rc<NodeInner> 引用计数管理，
/// 不会在 backward 后被释放。此测试验证这一语义。
#[test]
fn test_intermediate_results_managed_by_rc() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建网络: x -> w -> y -> z -> loss (标量)
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let w = gi.create_parameter_node(&[2, 2], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w))
        .unwrap();
    let y = gi
        .create_mat_mul_node(vec![Rc::clone(&w), Rc::clone(&x)], Some("y"))
        .unwrap();
    let z = gi
        .create_tanh_node(Rc::clone(&y), Some("z"))
        .unwrap();
    let target = gi
        .create_basic_input_node(&[2, 1], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(Rc::clone(&z), Rc::clone(&target), Some("loss"))
        .unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let target_data = Tensor::new(&[0.5, 0.5], &[2, 1]);
    x.set_value(Some(&input_data)).unwrap();
    target.set_value(Some(&target_data)).unwrap();

    // 前向传播
    gi.forward_via_node_inner(&loss).unwrap();

    // 验证中间节点有值
    assert!(y.value().is_some());
    assert!(z.value().is_some());

    // 反向传播
    gi.backward_via_node_inner(&loss).unwrap();

    // w（参数节点）应该有梯度
    assert!(w.grad().is_some());

    // 动态图：中间节点的值由 Rc 管理，backward 不清除
    assert!(y.value().is_some(), "动态图中值由 Rc 管理，不会被释放");
    assert!(z.value().is_some(), "动态图中值由 Rc 管理，不会被释放");

    // Input 和 Parameter 的值应该保留
    assert!(x.value().is_some());
    assert!(w.value().is_some());
}

/// 测试: 允许多次 backward（动态图中值不被释放，天然支持）
#[test]
fn test_multiple_backward() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建网络: x -> w -> y (标量)
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let w = gi.create_parameter_node(&[1, 2], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w))
        .unwrap();
    let y = gi
        .create_mat_mul_node(vec![Rc::clone(&w), Rc::clone(&x)], Some("y"))
        .unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    x.set_value(Some(&input_data)).unwrap();

    // 前向传播
    gi.forward_via_node_inner(&y).unwrap();

    // 第一次 backward
    gi.backward_via_node_inner(&y).unwrap();
    let grad1 = w.grad().unwrap();

    // 清除梯度
    gi.zero_grad().unwrap();

    // 第二次 backward（动态图中值不被释放，天然支持多次 backward）
    gi.backward_via_node_inner(&y).unwrap();
    let grad2 = w.grad().unwrap();

    // 两次计算应该得到相同的梯度
    assert_eq!(grad1, grad2);
}

/// 测试: 多任务学习场景 - 两个 loss 共享 backbone
/// PyTorch 对照: tests/python/calc_jacobi_by_pytorch/multi_task_learning_retain_graph.py
#[test]
fn test_multi_task_learning() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建共享 backbone: x -> w_shared -> features
    let x = gi
        .create_basic_input_node(&[4, 1], Some("x"))
        .unwrap();
    let w_shared = gi
        .create_parameter_node(&[2, 4], Some("w_shared"))
        .unwrap();
    gi.register_parameter("w_shared".to_string(), Rc::downgrade(&w_shared))
        .unwrap();
    let features = gi
        .create_mat_mul_node(
            vec![Rc::clone(&w_shared), Rc::clone(&x)],
            Some("features"),
        )
        .unwrap();

    // 任务 1: features -> w1 -> out1 -> loss1
    let w1 = gi.create_parameter_node(&[1, 2], Some("w1")).unwrap();
    gi.register_parameter("w1".to_string(), Rc::downgrade(&w1))
        .unwrap();
    let out1 = gi
        .create_mat_mul_node(
            vec![Rc::clone(&w1), Rc::clone(&features)],
            Some("out1"),
        )
        .unwrap();
    let target1 = gi
        .create_basic_input_node(&[1, 1], Some("target1"))
        .unwrap();
    let loss1 = gi
        .create_mse_mean_node(Rc::clone(&out1), Rc::clone(&target1), Some("loss1"))
        .unwrap();

    // 任务 2: features -> w2 -> out2 -> loss2
    let w2 = gi.create_parameter_node(&[1, 2], Some("w2")).unwrap();
    gi.register_parameter("w2".to_string(), Rc::downgrade(&w2))
        .unwrap();
    let out2 = gi
        .create_mat_mul_node(
            vec![Rc::clone(&w2), Rc::clone(&features)],
            Some("out2"),
        )
        .unwrap();
    let target2 = gi
        .create_basic_input_node(&[1, 1], Some("target2"))
        .unwrap();
    let loss2 = gi
        .create_mse_mean_node(Rc::clone(&out2), Rc::clone(&target2), Some("loss2"))
        .unwrap();

    // 设置输入和参数值（与 PyTorch 对照测试一致，使用全 1 参数）
    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4, 1]);
    let target_data = Tensor::zeros(&[1, 1]); // 目标为 0
    x.set_value(Some(&input_data)).unwrap();
    target1.set_value(Some(&target_data)).unwrap();
    target2.set_value(Some(&target_data)).unwrap();
    w_shared
        .set_value(Some(&Tensor::ones(&[2, 4])))
        .unwrap();
    w1.set_value(Some(&Tensor::ones(&[1, 2]))).unwrap();
    w2.set_value(Some(&Tensor::ones(&[1, 2]))).unwrap();

    // 前向传播两个任务
    gi.forward_via_node_inner(&loss1).unwrap();
    gi.forward_via_node_inner(&loss2).unwrap();

    // ========== 验证前向传播结果（PyTorch 对照值）==========
    let features_value = features.value().unwrap();
    let expected_features = Tensor::new(&[10.0, 10.0], &[2, 1]);
    assert_eq!(features_value, expected_features, "features 前向值不匹配");

    let out1_value = out1.value().unwrap();
    let expected_out1 = Tensor::new(&[20.0], &[1, 1]);
    assert_eq!(out1_value, expected_out1, "out1 前向值不匹配");

    let out2_value = out2.value().unwrap();
    let expected_out2 = Tensor::new(&[20.0], &[1, 1]);
    assert_eq!(out2_value, expected_out2, "out2 前向值不匹配");

    // ========== 任务 1 backward ==========
    gi.backward_via_node_inner(&loss1).unwrap();

    // 验证 w_shared 和 w1 的梯度
    let w_shared_grad_1 = w_shared.grad().unwrap();
    let w1_grad = w1.grad().unwrap();

    // w1 的梯度: d(loss1)/d(w1) = d(loss1)/d(out1) * d(out1)/d(w1)
    //                          = 40 * features^T = 40 * [10, 10] = [400, 400]
    let expected_w1_grad = Tensor::new(&[400.0, 400.0], &[1, 2]);
    assert_eq!(w1_grad, expected_w1_grad, "w1 梯度不匹配");

    // w2 此时不应有梯度（不在 loss1 的计算图中）
    assert!(
        w2.grad().is_none(),
        "w2 在 task1 backward 后不应有梯度"
    );

    // ========== 任务 2 backward（梯度自动累积）==========
    gi.backward_via_node_inner(&loss2).unwrap();

    // 验证 w2 的梯度
    let w2_grad = w2.grad().unwrap();
    let expected_w2_grad = Tensor::new(&[400.0, 400.0], &[1, 2]);
    assert_eq!(w2_grad, expected_w2_grad, "w2 梯度不匹配");

    // 验证 w_shared 的累积梯度（task1 + task2）
    let w_shared_grad_accumulated = w_shared.grad().unwrap();
    assert!(
        w_shared_grad_accumulated.size() > 0,
        "w_shared 应有累积梯度"
    );

    // 验证累积效果：第二次的梯度应该比第一次大
    let sum_after = w_shared_grad_accumulated.flatten_view().iter().sum::<f32>();
    let sum_before = w_shared_grad_1.flatten_view().iter().sum::<f32>();
    assert!(
        sum_after > sum_before,
        "累积后的梯度和应该更大: {} > {}",
        sum_after,
        sum_before
    );
}

/// 测试: backward 默认行为（方案 C 中值由 Rc 管理）
#[test]
fn test_backward_default_releases_graph() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建网络: x -> w -> y -> loss (标量)
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let w = gi.create_parameter_node(&[2, 2], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w))
        .unwrap();
    let y = gi
        .create_mat_mul_node(vec![Rc::clone(&w), Rc::clone(&x)], Some("y"))
        .unwrap();
    let target = gi
        .create_basic_input_node(&[2, 1], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(Rc::clone(&y), Rc::clone(&target), Some("loss"))
        .unwrap();

    // 设置输入
    let input_data = Tensor::new(&[0.5, -0.3], &[2, 1]);
    let target_data = Tensor::new(&[0.0, 0.0], &[2, 1]);
    x.set_value(Some(&input_data)).unwrap();
    w.set_value(Some(&Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2])))
        .unwrap();
    target.set_value(Some(&target_data)).unwrap();

    // 前向传播
    gi.forward_via_node_inner(&loss).unwrap();

    // 验证 forward 后中间节点有值
    assert!(y.value().is_some());

    // backward
    gi.backward_via_node_inner(&loss).unwrap();

    // w 应该有梯度
    assert!(w.grad().is_some());

    // 动态图：y 的值由 Rc 管理，不会在 backward 后被释放
    assert!(y.value().is_some(), "动态图中中间值不会被释放");

    // x 和 w（Input/Parameter）的值应该保留
    assert!(x.value().is_some());
    assert!(w.value().is_some());
}

/// 测试: 多次 backward 无需重新 forward（方案 C 中值不被释放）
#[test]
fn test_multiple_backward_without_new_forward() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建网络
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let w = gi.create_parameter_node(&[1, 2], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w))
        .unwrap();
    let y = gi
        .create_mat_mul_node(vec![Rc::clone(&w), Rc::clone(&x)], Some("y"))
        .unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    x.set_value(Some(&input_data)).unwrap();

    // 前向传播
    gi.forward_via_node_inner(&y).unwrap();

    // 第一次 backward
    gi.backward_via_node_inner(&y).unwrap();
    let grad1 = w.grad().unwrap();

    // 清除梯度
    gi.zero_grad().unwrap();

    // 动态图：值不被释放，无需重新 forward 即可再次 backward
    gi.backward_via_node_inner(&y).unwrap();
    let grad2 = w.grad().unwrap();

    // 两次应该得到相同结果
    assert_eq!(grad1, grad2, "多次 backward 应产生相同梯度");
}

/// 测试: 多次 backward 与 detach 混合使用
#[test]
fn test_multiple_backward_with_detach() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建网络: x -> w1 -> h -> w2 -> y -> loss
    let x = gi
        .create_basic_input_node(&[2, 1], Some("x"))
        .unwrap();
    let w1 = gi.create_parameter_node(&[2, 2], Some("w1")).unwrap();
    gi.register_parameter("w1".to_string(), Rc::downgrade(&w1))
        .unwrap();
    let h = gi
        .create_mat_mul_node(vec![Rc::clone(&w1), Rc::clone(&x)], Some("h"))
        .unwrap();
    let w2 = gi.create_parameter_node(&[1, 2], Some("w2")).unwrap();
    gi.register_parameter("w2".to_string(), Rc::downgrade(&w2))
        .unwrap();
    let y = gi
        .create_mat_mul_node(vec![Rc::clone(&w2), Rc::clone(&h)], Some("y"))
        .unwrap();
    let target = gi
        .create_basic_input_node(&[1, 1], Some("target"))
        .unwrap();
    let loss = gi
        .create_mse_mean_node(Rc::clone(&y), Rc::clone(&target), Some("loss"))
        .unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let target_data = Tensor::zeros(&[1, 1]);
    x.set_value(Some(&input_data)).unwrap();
    target.set_value(Some(&target_data)).unwrap();

    // 前向传播
    gi.forward_via_node_inner(&loss).unwrap();

    // detach h
    h.set_detached(true);

    // 反向传播
    gi.backward_via_node_inner(&loss).unwrap();

    // w2 应该有梯度
    assert!(w2.grad().is_some());

    // h 和 y 的值应该存在（方案 C 中由 Rc 管理）
    assert!(h.value().is_some());
    assert!(y.value().is_some());

    // 验证 detach 效果：方案 C 中 detach 节点仍可接收梯度（只阻止向上游传播）
    // h 从 y 收到梯度，但不向 w1 传播
    assert!(
        h.grad().is_some(),
        "方案 C: detach 节点仍可接收来自下游的梯度（只阻止向上传播）"
    );

    // w1 不应有梯度（因为 h 被 detach 阻断了梯度流）
    assert!(
        w1.grad().is_none(),
        "w1 不应有梯度，因为 h 被 detach 阻断了梯度流（PyTorch 标准行为）"
    );
}

/// 测试: 复杂拓扑下多参数节点的梯度累积（链式 + 分叉结构）
///
/// 验证关键机制：
/// 1. 传播信号必须使用"本次新算的值"，而非累积后的值（规则 4）
/// 2. 共享参数节点正确累积来自多个分支的贡献
/// 3. 中间特征节点不影响参数节点的累积正确性
///
/// 设计文档: `.doc/design/gradient_flow_control_design.md` 7.2 节
///
/// 拓扑:
/// ```
///   x → w_shared1 → shared_feat1 → w_shared2 → w_shared3 → shared_feat2 → w_task1 → out1 → loss1
///                                                                     └──→ w_task2 → out2 → loss2
/// ```
///
/// 参数节点: w_shared1, w_shared2, w_shared3 (共享链，两次 backward 都累积)
///          w_task1, w_task2 (分叉，各自只在对应任务累积)
/// 中间特征节点: shared_feat1, shared_feat2 (共享但不累积，每次重新计算)
///
/// 关键测试点: w_shared2 与 w_shared3 是**相邻参数节点**，验证链式累积是否正确
#[test]
fn test_backward_accumulation_for_complex_topology() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // ========== 构建拓扑 ==========
    // 输入
    let x = gi
        .create_basic_input_node(&[4, 1], Some("x"))
        .unwrap();

    // 共享链: x → w_shared1 → shared_feat1 → w_shared2 → w_shared3 → shared_feat2
    let w_shared1 = gi
        .create_parameter_node(&[2, 4], Some("w_shared1"))
        .unwrap();
    gi.register_parameter("w_shared1".to_string(), Rc::downgrade(&w_shared1))
        .unwrap();
    let shared_feat1 = gi
        .create_mat_mul_node(
            vec![Rc::clone(&w_shared1), Rc::clone(&x)],
            Some("shared_feat1"),
        )
        .unwrap();

    // w_shared2 和 w_shared3 是相邻参数节点（测试链式累积）
    let w_shared2 = gi
        .create_parameter_node(&[2, 2], Some("w_shared2"))
        .unwrap();
    gi.register_parameter("w_shared2".to_string(), Rc::downgrade(&w_shared2))
        .unwrap();
    let w_shared2_out = gi
        .create_mat_mul_node(
            vec![Rc::clone(&w_shared2), Rc::clone(&shared_feat1)],
            Some("w_shared2_out"),
        )
        .unwrap();

    let w_shared3 = gi
        .create_parameter_node(&[2, 2], Some("w_shared3"))
        .unwrap();
    gi.register_parameter("w_shared3".to_string(), Rc::downgrade(&w_shared3))
        .unwrap();
    let shared_feat2 = gi
        .create_mat_mul_node(
            vec![Rc::clone(&w_shared3), Rc::clone(&w_shared2_out)],
            Some("shared_feat2"),
        )
        .unwrap();

    // 分叉: shared_feat2 → w_task1 → out1 → loss1, shared_feat2 → w_task2 → out2 → loss2
    let w_task1 = gi
        .create_parameter_node(&[1, 2], Some("w_task1"))
        .unwrap();
    gi.register_parameter("w_task1".to_string(), Rc::downgrade(&w_task1))
        .unwrap();
    let out1 = gi
        .create_mat_mul_node(
            vec![Rc::clone(&w_task1), Rc::clone(&shared_feat2)],
            Some("out1"),
        )
        .unwrap();
    let target1 = gi
        .create_basic_input_node(&[1, 1], Some("target1"))
        .unwrap();
    let loss1 = gi
        .create_mse_mean_node(Rc::clone(&out1), Rc::clone(&target1), Some("loss1"))
        .unwrap();

    let w_task2 = gi
        .create_parameter_node(&[1, 2], Some("w_task2"))
        .unwrap();
    gi.register_parameter("w_task2".to_string(), Rc::downgrade(&w_task2))
        .unwrap();
    let out2 = gi
        .create_mat_mul_node(
            vec![Rc::clone(&w_task2), Rc::clone(&shared_feat2)],
            Some("out2"),
        )
        .unwrap();
    let target2 = gi
        .create_basic_input_node(&[1, 1], Some("target2"))
        .unwrap();
    let loss2 = gi
        .create_mse_mean_node(Rc::clone(&out2), Rc::clone(&target2), Some("loss2"))
        .unwrap();

    // ========== 设置固定值 ==========
    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4, 1])))
        .unwrap();
    target1
        .set_value(Some(&Tensor::zeros(&[1, 1])))
        .unwrap();
    target2
        .set_value(Some(&Tensor::zeros(&[1, 1])))
        .unwrap();
    w_shared1
        .set_value(Some(&Tensor::ones(&[2, 4])))
        .unwrap();
    w_shared2
        .set_value(Some(&Tensor::ones(&[2, 2])))
        .unwrap();
    w_shared3
        .set_value(Some(&Tensor::ones(&[2, 2])))
        .unwrap();
    w_task1
        .set_value(Some(&Tensor::ones(&[1, 2])))
        .unwrap();
    w_task2
        .set_value(Some(&Tensor::ones(&[1, 2])))
        .unwrap();

    // ========== 前向传播 ==========
    gi.forward_via_node_inner(&loss1).unwrap();
    gi.forward_via_node_inner(&loss2).unwrap();

    // ========== 第 1 次 backward (loss1) ==========
    gi.backward_via_node_inner(&loss1).unwrap();

    let w_shared1_after_task1 = w_shared1.grad().unwrap();
    let w_shared2_after_task1 = w_shared2.grad().unwrap();
    let w_shared3_after_task1 = w_shared3.grad().unwrap();
    let w_task1_after_task1 = w_task1.grad().unwrap();

    // w_task2 此时不应有梯度（不在 loss1 的计算图中）
    assert!(
        w_task2.grad().is_none(),
        "w_task2 在 task1 backward 后不应有梯度"
    );

    // 中间节点应该有值
    assert!(
        shared_feat1.value().is_some(),
        "shared_feat1 应有本次 forward 的值"
    );
    assert!(
        w_shared2_out.value().is_some(),
        "w_shared2_out 应有本次 forward 的值"
    );
    assert!(
        shared_feat2.value().is_some(),
        "shared_feat2 应有本次 forward 的值"
    );

    // ========== 第 2 次 backward (loss2) ==========
    gi.backward_via_node_inner(&loss2).unwrap();

    let w_shared1_accumulated = w_shared1.grad().unwrap();
    let w_shared2_accumulated = w_shared2.grad().unwrap();
    let w_shared3_accumulated = w_shared3.grad().unwrap();
    let w_task2_after_task2 = w_task2.grad().unwrap();

    // ========== 验证累积效果 ==========
    let sum_w_shared1_after = w_shared1_accumulated.flatten_view().iter().sum::<f32>();
    let sum_w_shared1_before = w_shared1_after_task1.flatten_view().iter().sum::<f32>();
    assert!(
        sum_w_shared1_after > sum_w_shared1_before,
        "w_shared1 累积梯度应该更大: {} > {}",
        sum_w_shared1_after,
        sum_w_shared1_before
    );

    let sum_w_shared2_after = w_shared2_accumulated.flatten_view().iter().sum::<f32>();
    let sum_w_shared2_before = w_shared2_after_task1.flatten_view().iter().sum::<f32>();
    assert!(
        sum_w_shared2_after > sum_w_shared2_before,
        "w_shared2 累积梯度应该更大"
    );

    let sum_w_shared3_after = w_shared3_accumulated.flatten_view().iter().sum::<f32>();
    let sum_w_shared3_before = w_shared3_after_task1.flatten_view().iter().sum::<f32>();
    assert!(
        sum_w_shared3_after > sum_w_shared3_before,
        "w_shared3 累积梯度应该更大（相邻参数链式累积）"
    );

    // w_task1 的梯度应该只有 task1 的贡献
    let w_task1_final = w_task1.grad().unwrap();
    assert_eq!(
        w_task1_after_task1, w_task1_final,
        "w_task1 只有 task1 的贡献，不应变化"
    );

    // w_task2 应该有 task2 的梯度
    assert!(w_task2_after_task2.size() > 0, "w_task2 应有 task2 的梯度");
}
