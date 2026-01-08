//! 梯度流控制机制测试
//!
//! 测试 `no_grad`、`detach`、`retain_graph` 三种梯度控制机制
//! 参考设计文档: `.doc/design/gradient_flow_control_design.md`

use approx::assert_abs_diff_eq;

use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

// ============================================================================
// 1. no_grad 机制测试
// ============================================================================

/// 测试: is_grad_enabled 与 is_train_mode 一致
#[test]
fn test_is_grad_enabled_equals_is_train_mode() {
    let mut graph = Graph::new();

    // 默认训练模式
    assert!(graph.is_grad_enabled());
    assert!(graph.is_train_mode());

    // 切换到评估模式
    graph.set_eval_mode();
    assert!(!graph.is_grad_enabled());
    assert!(!graph.is_train_mode());

    // 切换回训练模式
    graph.set_train_mode();
    assert!(graph.is_grad_enabled());
    assert!(graph.is_train_mode());
}

/// 测试: no_grad_scope 基本功能 - 临时禁用梯度
#[test]
fn test_no_grad_scope_basic() {
    let mut graph = Graph::new();

    // 默认是训练模式
    assert!(graph.is_grad_enabled());

    // 进入 no_grad 上下文
    graph.no_grad_scope(|g| {
        // 应该处于评估模式
        assert!(!g.is_grad_enabled());
        assert!(!g.is_train_mode());
    });

    // 退出后恢复训练模式
    assert!(graph.is_grad_enabled());
    assert!(graph.is_train_mode());
}

/// 测试: no_grad_scope 从评估模式开始
#[test]
fn test_no_grad_scope_from_eval_mode() {
    let mut graph = Graph::new();

    // 先切换到评估模式
    graph.set_eval_mode();
    assert!(!graph.is_grad_enabled());

    // 进入 no_grad 上下文（已经是评估模式）
    graph.no_grad_scope(|g| {
        assert!(!g.is_grad_enabled());
    });

    // 退出后应该保持评估模式（因为之前就是评估模式）
    assert!(!graph.is_grad_enabled());
}

/// 测试: no_grad_scope 返回值传递
#[test]
fn test_no_grad_scope_return_value() {
    let mut graph = Graph::new();

    // 返回简单值
    let result: i32 = graph.no_grad_scope(|_g| 42);
    assert_eq!(result, 42);

    // 返回 Result
    let result: Result<f64, GraphError> = graph.no_grad_scope(|_g| Ok(3.14));
    assert_eq!(result.unwrap(), 3.14);

    // 返回复杂类型
    let result: Vec<i32> = graph.no_grad_scope(|_g| vec![1, 2, 3]);
    assert_eq!(result, vec![1, 2, 3]);
}

/// 测试: no_grad_scope 错误传播
#[test]
fn test_no_grad_scope_error_propagation() {
    let mut graph = Graph::new();

    // 闭包返回错误
    let result: Result<(), GraphError> =
        graph.no_grad_scope(|_g| Err(GraphError::InvalidOperation("测试错误".to_string())));

    // 错误应该被传播出来
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        GraphError::InvalidOperation("测试错误".to_string())
    );

    // 即使出错，模式也应该恢复
    assert!(graph.is_grad_enabled());
}

/// 测试: no_grad_scope 嵌套调用
#[test]
fn test_no_grad_scope_nested() {
    let mut graph = Graph::new();

    assert!(graph.is_grad_enabled());

    graph.no_grad_scope(|g| {
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
    assert!(graph.is_grad_enabled());
}

/// 测试: no_grad_scope 与前向传播集成
#[test]
fn test_no_grad_scope_with_forward() {
    let mut graph = Graph::new();

    // 创建简单网络: x -> tanh -> output
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let output = graph.new_tanh_node(x, Some("output")).unwrap();

    // 设置输入
    let input_data = Tensor::new(&[0.5, -0.3], &[2, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();

    // 在 no_grad 上下文中前向传播
    let output_val: Result<Option<Tensor>, GraphError> = graph.no_grad_scope(|g| {
        assert!(!g.is_grad_enabled());
        g.forward(output)?;
        Ok(g.get_node_value(output)?.cloned())
    });

    // 验证前向传播成功
    let output_val = output_val.unwrap();
    assert!(output_val.is_some());
    let val = output_val.unwrap();
    assert_eq!(val.shape(), &[2, 1]);

    // 验证模式恢复
    assert!(graph.is_grad_enabled());
}

/// 测试: no_grad_scope 核心保证 - 相同输入产生相同 loss，区别仅在于梯度
#[test]
fn test_no_grad_scope_same_input_same_loss_no_gradient() {
    let mut graph = Graph::new();

    // 创建简单回归网络
    let x = graph.new_input_node(&[1, 2], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[2, 1], Some("w")).unwrap();
    let pred = graph.new_mat_mul_node(x, w, Some("pred")).unwrap();
    let y = graph.new_input_node(&[1, 1], Some("y")).unwrap();
    let loss = graph.new_mse_loss_node(pred, y, Some("loss")).unwrap();

    // 使用相同的输入数据
    let input_x = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let target_y = Tensor::new(&[3.0], &[1, 1]);

    // ===== 正常模式：前向 + 反向 =====
    graph.set_node_value(x, Some(&input_x)).unwrap();
    graph.set_node_value(y, Some(&target_y)).unwrap();
    graph.forward(loss).unwrap();
    let normal_loss_value = graph
        .get_node_value(loss)
        .unwrap()
        .unwrap()
        .get_data_number()
        .unwrap();

    // 正常模式可以计算梯度
    let backward_result = graph.backward(loss);
    assert!(backward_result.is_ok());
    let normal_gradient = graph.get_node_grad(w).unwrap();
    assert!(normal_gradient.is_some(), "正常模式应产生梯度");

    // 清除状态，准备下一次计算
    graph.zero_grad().unwrap();

    // ===== no_grad 模式：相同输入 =====
    let no_grad_loss_value: Result<f32, GraphError> = graph.no_grad_scope(|g| {
        // 使用完全相同的输入数据
        g.set_node_value(x, Some(&input_x))?;
        g.set_node_value(y, Some(&target_y))?;
        g.forward(loss)?;
        Ok(g.get_node_value(loss)?.unwrap().get_data_number().unwrap())
    });
    let no_grad_loss_value = no_grad_loss_value.unwrap();

    // 核心验证 1: 相同输入 → 相同 loss
    assert_abs_diff_eq!(normal_loss_value, no_grad_loss_value, epsilon = 1e-6);

    // 核心验证 2: no_grad 模式下无法计算梯度（已在其他测试中验证 backward 会失败）
    // 这里验证退出 no_grad_scope 后梯度状态
    let no_grad_gradient = graph.get_node_grad(w).unwrap();
    assert!(no_grad_gradient.is_none(), "no_grad 模式后不应有残留梯度");

    // 验证模式恢复
    assert!(graph.is_grad_enabled());
}

/// 测试: no_grad_scope 多次调用
#[test]
fn test_no_grad_scope_multiple_calls() {
    let mut graph = Graph::new();

    for i in 0..5 {
        assert!(graph.is_grad_enabled(), "第 {i} 次调用前应为训练模式");

        graph.no_grad_scope(|g| {
            assert!(!g.is_grad_enabled(), "第 {i} 次调用中应为评估模式");
        });

        assert!(graph.is_grad_enabled(), "第 {i} 次调用后应恢复训练模式");
    }
}

/// 测试: no_grad_scope 与 set_eval_mode 交互
#[test]
fn test_no_grad_scope_interaction_with_eval_mode() {
    let mut graph = Graph::new();

    // 场景 1: 在 no_grad 中手动切换模式
    graph.no_grad_scope(|g| {
        assert!(!g.is_grad_enabled());

        // 手动切换到训练模式（不推荐但应该能工作）
        g.set_train_mode();
        assert!(g.is_grad_enabled());

        // 切换回评估模式
        g.set_eval_mode();
        assert!(!g.is_grad_enabled());
    });

    // 退出后恢复（因为进入前是训练模式）
    assert!(graph.is_grad_enabled());
}

/// 测试: no_grad_scope 闭包中的可变借用
#[test]
fn test_no_grad_scope_mutable_operations() {
    let mut graph = Graph::new();

    // 创建节点
    let x = graph.new_input_node(&[2, 2], Some("x")).unwrap();
    let y = graph.new_tanh_node(x, Some("y")).unwrap();

    // 在 no_grad 中执行可变操作
    graph.no_grad_scope(|g| {
        // 设置值
        let data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        g.set_node_value(x, Some(&data)).unwrap();

        // 前向传播
        g.forward(y).unwrap();

        // 验证结果
        let result = g.get_node_value(y).unwrap();
        assert!(result.is_some());
    });
}

/// 测试: no_grad_scope 与种子的交互
#[test]
fn test_no_grad_scope_with_seeded_graph() {
    let mut graph = Graph::new_with_seed(42);

    assert!(graph.has_seed());
    assert!(graph.is_grad_enabled());

    graph.no_grad_scope(|g| {
        assert!(g.has_seed()); // 种子应该保持不变
        assert!(!g.is_grad_enabled());

        // 可以在 no_grad 中创建参数节点
        let _w = g.new_parameter_node(&[2, 2], Some("w")).unwrap();
    });

    // 种子和模式都应该正确恢复
    assert!(graph.has_seed());
    assert!(graph.is_grad_enabled());
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
    let mut graph = Graph::new();

    // 创建简单网络
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[1, 2], Some("w")).unwrap();
    let y = graph.new_mat_mul_node(w, x, Some("y")).unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();

    // 在 no_grad_scope 内执行 forward + backward
    let backward_result: Result<(), GraphError> = graph.no_grad_scope(|g| {
        assert!(!g.is_grad_enabled(), "应处于 no_grad 模式");

        // forward 应该正常工作
        g.forward(y)?;
        assert!(g.get_node_value(y)?.is_some(), "forward 应产生值");

        // backward 也应该正常工作（与 PyTorch 一致）
        // no_grad 不阻止 backward，只是影响前向时的缓存策略
        g.backward(y)?;

        Ok(())
    });

    // backward 应该成功执行
    assert!(
        backward_result.is_ok(),
        "no_grad_scope 内 backward 应该成功: {:?}",
        backward_result
    );

    // 退出 no_grad_scope 后模式应恢复
    assert!(graph.is_grad_enabled());

    // 注意：参数的梯度可能存在（取决于实现细节）
    // 这里验证的是 backward 不会报错，而非梯度语义
}

/// 测试: no_grad_scope 内创建的节点在退出后的梯度行为
#[test]
fn test_no_grad_scope_nodes_created_inside() {
    let mut graph = Graph::new();

    // 先创建输入节点
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();

    // 在 no_grad_scope 内创建参数和运算节点
    let (w, y) = graph.no_grad_scope(|g| {
        let w = g.new_parameter_node(&[1, 2], Some("w")).unwrap();
        let y = g.new_mat_mul_node(w, x, Some("y")).unwrap();
        (w, y)
    });

    // 退出 no_grad_scope 后，这些节点应该可以正常参与训练
    assert!(graph.is_grad_enabled());

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();

    // 正常前向传播
    graph.forward(y).unwrap();
    assert!(graph.get_node_value(y).unwrap().is_some());

    // 正常反向传播应该工作
    graph.backward(y).unwrap();
    assert!(
        graph.get_node_grad(w).unwrap().is_some(),
        "退出 no_grad_scope 后，在其中创建的节点应能正常计算梯度"
    );
}

// ============================================================================
// 2. detach 机制测试
// ============================================================================

/// 测试: detach_node / attach_node / is_node_detached 基本功能
#[test]
fn test_detach_basic() {
    let mut graph = Graph::new();

    // 创建节点
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[1, 2], Some("w")).unwrap();
    let y = graph.new_mat_mul_node(w, x, Some("y")).unwrap();

    // 默认不是 detached
    assert!(!graph.is_node_detached(w).unwrap());
    assert!(!graph.is_node_detached(y).unwrap());

    // detach 节点
    graph.detach_node(y).unwrap();
    assert!(graph.is_node_detached(y).unwrap());
    assert!(!graph.is_node_detached(w).unwrap()); // w 不受影响
    assert!(!graph.is_node_detached(x).unwrap()); // x 不受影响

    // attach 恢复
    graph.attach_node(y).unwrap();
    assert!(!graph.is_node_detached(y).unwrap());
}

/// 测试: detach 阻止梯度回流
#[test]
fn test_detach_blocks_gradient_flow() {
    let mut graph = Graph::new();

    // 创建简单网络: x -> w1 -> h -> w2 -> output
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w1 = graph.new_parameter_node(&[2, 2], Some("w1")).unwrap();
    let h = graph.new_mat_mul_node(w1, x, Some("h")).unwrap();
    let w2 = graph.new_parameter_node(&[1, 2], Some("w2")).unwrap();
    let output = graph.new_mat_mul_node(w2, h, Some("output")).unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();

    // 前向传播
    graph.forward(output).unwrap();

    // ===== 场景 1: 正常反向传播 =====
    graph.backward(output).unwrap();

    // w1 和 w2 都应该有梯度
    assert!(graph.get_node_grad(w1).unwrap().is_some());
    assert!(graph.get_node_grad(w2).unwrap().is_some());

    // 清除梯度
    graph.zero_grad().unwrap();

    // ===== 场景 2: detach h，w1 不应该有梯度 =====
    graph.detach_node(h).unwrap();
    graph.forward(output).unwrap();
    graph.backward(output).unwrap();

    // w1 不应该有梯度（被 h 的 detach 阻断）
    // PyTorch 标准行为：detach 后上游节点的梯度为 None
    assert!(
        graph.get_node_grad(w1).unwrap().is_none(),
        "w1 不应有梯度，因为 h 被 detach 阻断了梯度流"
    );
    // w2 应该有梯度（在 detach 点之后）
    assert!(graph.get_node_grad(w2).unwrap().is_some());

    // 清除并恢复
    graph.zero_grad().unwrap();
    graph.attach_node(h).unwrap();

    // ===== 场景 3: 恢复后 w1 应该有梯度 =====
    graph.forward(output).unwrap();
    graph.backward(output).unwrap();
    assert!(graph.get_node_grad(w1).unwrap().is_some());
    assert!(graph.get_node_grad(w2).unwrap().is_some());
}

/// 测试: detach 不影响前向传播
#[test]
fn test_detach_does_not_affect_forward() {
    let mut graph = Graph::new();

    // 创建网络
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let y = graph.new_tanh_node(x, Some("y")).unwrap();

    // 设置输入
    let input_data = Tensor::new(&[0.5, -0.3], &[2, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();

    // 正常前向传播
    graph.forward(y).unwrap();
    let result1 = graph.get_node_value(y).unwrap().unwrap().clone();

    // detach y
    graph.detach_node(y).unwrap();

    // 重新设置输入并前向传播
    graph.set_node_value(x, Some(&input_data)).unwrap();
    graph.forward(y).unwrap();
    let result2 = graph.get_node_value(y).unwrap().unwrap();

    // 结果应该相同
    assert_eq!(&result1, result2);
}

/// 测试: 多次 detach/attach 切换（包括幂等性和实际 backward 效果验证）
#[test]
fn test_detach_attach_multiple_times() {
    let mut graph = Graph::new();

    // 创建简单网络用于验证 backward 效果: x -> w -> y -> loss
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[1, 2], Some("w")).unwrap();
    let y = graph.new_mat_mul_node(w, x, Some("y")).unwrap();
    let target = graph.new_input_node(&[1, 1], Some("target")).unwrap();
    let loss = graph.new_mse_loss_node(y, target, Some("loss")).unwrap();

    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let target_data = Tensor::new(&[0.0], &[1, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();
    graph.set_node_value(target, Some(&target_data)).unwrap();

    // ===== 1. 测试交替切换 =====
    for i in 0..10 {
        if i % 2 == 0 {
            graph.detach_node(w).unwrap();
            assert!(graph.is_node_detached(w).unwrap());
        } else {
            graph.attach_node(w).unwrap();
            assert!(!graph.is_node_detached(w).unwrap());
        }
    }

    // ===== 2. 测试幂等性：连续 detach =====
    graph.detach_node(w).unwrap();
    assert!(graph.is_node_detached(w).unwrap());
    graph.detach_node(w).unwrap(); // 再次 detach
    assert!(graph.is_node_detached(w).unwrap()); // 仍然是 detached

    // ===== 3. 测试幂等性：连续 attach =====
    graph.attach_node(w).unwrap();
    assert!(!graph.is_node_detached(w).unwrap());
    graph.attach_node(w).unwrap(); // 再次 attach
    assert!(!graph.is_node_detached(w).unwrap()); // 仍然是 attached

    // ===== 4. 验证 detach 状态确实影响 backward =====
    // attached 状态：应该有梯度
    graph.forward(loss).unwrap();
    graph.backward_ex(loss, true).unwrap();
    assert!(graph.get_node_grad(w).unwrap().is_some());
    graph.zero_grad().unwrap();

    // detached 状态：不应该有梯度
    graph.detach_node(w).unwrap();
    graph.forward(loss).unwrap();
    graph.backward_ex(loss, true).unwrap();
    assert!(
        graph.get_node_grad(w).unwrap().is_none(),
        "detached 的参数节点不应有梯度"
    );
    graph.zero_grad().unwrap();

    // 恢复 attached：应该又有梯度
    graph.attach_node(w).unwrap();
    graph.forward(loss).unwrap();
    graph.backward(loss).unwrap();
    assert!(graph.get_node_grad(w).unwrap().is_some());
}

/// 测试: detach 节点不存在时返回错误
#[test]
fn test_detach_nonexistent_node() {
    let mut graph = Graph::new();
    let invalid_id = crate::nn::NodeId(999);

    assert!(graph.detach_node(invalid_id).is_err());
    assert!(graph.attach_node(invalid_id).is_err());
    assert!(graph.is_node_detached(invalid_id).is_err());
}

/// 测试: GAN 风格训练模式
#[test]
fn test_detach_gan_style_training() {
    let mut graph = Graph::new();

    // 简化的 GAN 结构:
    // z(输入) -> G_w(生成器参数) -> fake_data -> D_w(判别器参数) -> d_output -> loss

    // 生成器部分
    let z = graph.new_input_node(&[2, 1], Some("z")).unwrap();
    let g_w = graph.new_parameter_node(&[3, 2], Some("g_w")).unwrap();
    let fake_data = graph.new_mat_mul_node(g_w, z, Some("fake")).unwrap();

    // 判别器部分
    let d_w = graph.new_parameter_node(&[1, 3], Some("d_w")).unwrap();
    let d_output = graph
        .new_mat_mul_node(d_w, fake_data, Some("d_out"))
        .unwrap();

    // 添加 loss 节点使输出为标量
    let d_target = graph.new_input_node(&[1, 1], Some("d_target")).unwrap();
    let loss = graph
        .new_mse_loss_node(d_output, d_target, Some("loss"))
        .unwrap();

    // 设置输入
    let noise = Tensor::new(&[0.5, -0.3], &[2, 1]);
    let target = Tensor::new(&[1.0], &[1, 1]); // 判别器目标
    graph.set_node_value(z, Some(&noise)).unwrap();
    graph.set_node_value(d_target, Some(&target)).unwrap();
    graph.forward(loss).unwrap();

    // ===== 训练判别器：detach fake_data =====
    graph.detach_node(fake_data).unwrap();
    graph.backward_ex(loss, true).unwrap();

    // d_w 应该有梯度
    assert!(graph.get_node_grad(d_w).unwrap().is_some());
    // g_w 不应该有梯度（fake_data 被 detach，梯度不会传到 g_w）
    assert!(
        graph.get_node_grad(g_w).unwrap().is_none(),
        "g_w 不应有梯度，因为 fake_data 被 detach"
    );
    // fake_data 被 detach，不应有 grad（PyTorch 语义）
    assert!(
        graph.get_node_grad(fake_data).unwrap().is_none(),
        "detach 的节点不应有 grad"
    );

    graph.zero_grad().unwrap();

    // ===== 训练生成器：attach fake_data =====
    graph.attach_node(fake_data).unwrap();
    graph.forward(loss).unwrap();
    graph.backward(loss).unwrap();

    // g_w 应该有梯度
    assert!(graph.get_node_grad(g_w).unwrap().is_some());
    // d_w 也会有梯度（因为它在 loss 到 g_w 的路径上）
    // 这与 PyTorch 的 GAN 训练略有不同，PyTorch 通常会冻结 D 的参数
    // 在 only_torch 中，如果想冻结 d_w，需要 detach 它
}

/// 测试: detach 与 Batch 模式的兼容性
#[test]
fn test_detach_with_batch_mode() {
    let mut graph = Graph::new();

    // 创建网络: x -> w -> y -> mse_loss
    let x = graph.new_input_node(&[2, 2], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[2, 2], Some("w")).unwrap();
    let y = graph.new_mat_mul_node(w, x, Some("y")).unwrap();
    let target = graph.new_input_node(&[2, 2], Some("target")).unwrap();
    let loss = graph.new_mse_loss_node(y, target, Some("loss")).unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let target_data = Tensor::new(&[1.1, 2.1, 3.1, 4.1], &[2, 2]);
    graph.set_node_value(x, Some(&input_data)).unwrap();
    graph.set_node_value(target, Some(&target_data)).unwrap();

    // Batch 前向传播
    graph.forward(loss).unwrap();

    // detach y 后 batch 反向传播
    graph.detach_node(y).unwrap();
    graph.backward(loss).unwrap();

    // w 不应该有 batch 梯度（因为 y 被 detach，梯度不会传到 w）
    assert!(
        graph.get_node_grad(w).unwrap().is_none(),
        "w 不应有梯度，因为 y 被 detach 阻断了梯度流"
    );

    // 注意：y 作为 detach 节点，它可能仍有来自 loss 的梯度（具体取决于实现）
    // 关键验证点是 w 没有梯度
}

/// 测试: Input 节点不能被 detach（语义上没意义但不应报错）
#[test]
fn test_detach_input_node() {
    let mut graph = Graph::new();
    let x = graph.new_input_node(&[2, 2], Some("x")).unwrap();

    // 技术上可以 detach，但语义上没有意义
    // 因为 Input 节点本来就不参与反向传播
    graph.detach_node(x).unwrap();
    assert!(graph.is_node_detached(x).unwrap());

    // attach 也应该工作
    graph.attach_node(x).unwrap();
    assert!(!graph.is_node_detached(x).unwrap());
}

/// 测试: detach 后的节点仍然可以正常操作，且已有的 grad 不会被清除（Single 模式）
#[test]
fn test_detach_node_still_functional() {
    let mut graph = Graph::new();

    // 创建简单网络: x -> w -> y
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[1, 2], Some("w")).unwrap();
    let y = graph.new_mat_mul_node(w, x, Some("y")).unwrap();

    // 设置输入并执行 forward + backward
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();
    graph.forward(y).unwrap();
    graph.backward(y).unwrap();

    // 验证 w 有 grad（使用 single 模式的 get_node_grad）
    let grad_before = graph.get_node_grad(w).unwrap().unwrap().clone();
    assert!(grad_before.size() > 0);

    // detach w
    graph.detach_node(w).unwrap();
    assert!(graph.is_node_detached(w).unwrap());

    // 验证 detach 后 grad 仍然保留（detach 不清除已有梯度）
    let grad_after = graph.get_node_grad(w).unwrap().unwrap();
    assert_eq!(&grad_before, grad_after);

    // 仍然可以获取值
    assert!(graph.get_node_value(w).unwrap().is_some());

    // 仍然可以设置值
    let new_value = Tensor::ones(&[1, 2]);
    graph.set_node_value(w, Some(&new_value)).unwrap();

    // 仍然可以获取名称
    assert_eq!(graph.get_node_name(w).unwrap(), "w");
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
    let mut graph = Graph::new();

    // 创建网络: x -> w1 -> h -> w2 -> output
    // x: [2, 1], w1: [2, 2] -> h: [2, 1], w2: [1, 2] -> output: [1, 1]
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w1 = graph.new_parameter_node(&[2, 2], Some("w1")).unwrap();
    let h = graph.new_mat_mul_node(w1, x, Some("h")).unwrap();
    let w2 = graph.new_parameter_node(&[1, 2], Some("w2")).unwrap();
    let output = graph.new_mat_mul_node(w2, h, Some("output")).unwrap();

    // 设置值（与 PyTorch 脚本一致）
    graph
        .set_node_value(x, Some(&Tensor::new(&[1.0, 2.0], &[2, 1])))
        .unwrap();
    graph
        .set_node_value(w1, Some(&Tensor::ones(&[2, 2])))
        .unwrap();
    graph
        .set_node_value(w2, Some(&Tensor::ones(&[1, 2])))
        .unwrap();

    // 前向传播
    graph.forward(output).unwrap();

    // 验证中间值（与 PyTorch 一致）
    let h_value = graph.get_node_value(h).unwrap().unwrap();
    assert_eq!(
        h_value,
        &Tensor::new(&[3.0, 3.0], &[2, 1]),
        "h 值应为 [3, 3]"
    );

    let output_value = graph.get_node_value(output).unwrap().unwrap();
    assert_eq!(
        output_value,
        &Tensor::new(&[6.0], &[1, 1]),
        "output 值应为 [6]"
    );

    // ===== detach h =====
    graph.detach_node(h).unwrap();

    // 反向传播
    graph.backward(output).unwrap();

    // ===== 验证梯度（PyTorch 对照值）=====
    // w1 应无梯度（被 detach 阻断）
    assert!(
        graph.get_node_grad(w1).unwrap().is_none(),
        "w1 应无梯度 (被 h 的 detach 阻断)"
    );

    // w2 梯度应与 PyTorch 精确匹配
    // PyTorch 输出: w2.grad = [[3., 3.]]
    // Jacobi 格式: [1, 2]
    let expected_w2_grad = Tensor::new(&[3.0, 3.0], &[1, 2]);
    let actual_w2_grad = graph.get_node_grad(w2).unwrap().unwrap();
    assert_eq!(
        actual_w2_grad, &expected_w2_grad,
        "w2 梯度应与 PyTorch 匹配"
    );
}

/// 测试: detach 后的节点仍然可以正常操作，且已有的 grad 不会被清除（Batch 模式）
#[test]
fn test_detach_node_still_functional_batch() {
    let mut graph = Graph::new();

    // 创建网络: x -> w -> y -> loss
    let x = graph.new_input_node(&[2, 2], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[2, 2], Some("w")).unwrap();
    let y = graph.new_mat_mul_node(w, x, Some("y")).unwrap();
    let target = graph.new_input_node(&[2, 2], Some("target")).unwrap();
    let loss = graph.new_mse_loss_node(y, target, Some("loss")).unwrap();

    // 设置输入并执行 forward + backward
    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let target_data = Tensor::new(&[1.1, 2.1, 3.1, 4.1], &[2, 2]);
    graph.set_node_value(x, Some(&input_data)).unwrap();
    graph.set_node_value(target, Some(&target_data)).unwrap();
    graph.forward(loss).unwrap();
    graph.backward(loss).unwrap();

    // 验证 w 有 grad（使用 batch 模式的 get_node_grad_ref）
    let grad_before = graph.get_node_grad_ref(w).unwrap().unwrap().clone();
    assert!(grad_before.size() > 0);

    // detach w
    graph.detach_node(w).unwrap();
    assert!(graph.is_node_detached(w).unwrap());

    // 验证 detach 后 grad 仍然保留（detach 不清除已有梯度）
    let grad_after = graph.get_node_grad_ref(w).unwrap().unwrap();
    assert_eq!(&grad_before, grad_after);

    // 仍然可以获取值
    assert!(graph.get_node_value(w).unwrap().is_some());

    // 仍然可以设置值
    let new_value = Tensor::ones(&[2, 2]);
    graph.set_node_value(w, Some(&new_value)).unwrap();

    // 仍然可以获取名称
    assert_eq!(graph.get_node_name(w).unwrap(), "w");
}

// ============================================================================
// 3. retain_graph 机制测试
// ============================================================================

/// 测试: backward_ex 基本功能
#[test]
fn test_retain_graph_basic() {
    let mut graph = Graph::new();

    // 创建简单网络: x -> w -> y
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[1, 2], Some("w")).unwrap();
    let y = graph.new_mat_mul_node(w, x, Some("y")).unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();

    // 前向传播
    graph.forward(y).unwrap();

    // 使用 retain_graph=true 反向传播
    graph.backward_ex(y, true).unwrap();

    // w 应该有梯度
    assert!(graph.get_node_grad(w).unwrap().is_some());

    // y 的值应该仍然存在（因为 retain_graph=true）
    assert!(graph.get_node_value(y).unwrap().is_some());
}

/// 测试: retain_graph=false 释放中间节点的值和梯度
#[test]
fn test_retain_graph_false_releases_intermediate_results() {
    let mut graph = Graph::new();

    // 创建网络: x -> w -> y -> z -> loss (标量)
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[2, 2], Some("w")).unwrap();
    let y = graph.new_mat_mul_node(w, x, Some("y")).unwrap();
    let z = graph.new_tanh_node(y, Some("z")).unwrap();
    let target = graph.new_input_node(&[2, 1], Some("target")).unwrap();
    let loss = graph.new_mse_loss_node(z, target, Some("loss")).unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let target_data = Tensor::new(&[0.5, 0.5], &[2, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();
    graph.set_node_value(target, Some(&target_data)).unwrap();

    // 前向传播
    graph.forward(loss).unwrap();

    // 验证中间节点有值
    assert!(graph.get_node_value(y).unwrap().is_some());
    assert!(graph.get_node_value(z).unwrap().is_some());

    // 使用 retain_graph=false 反向传播
    graph.backward_ex(loss, false).unwrap();

    // w（参数节点）应该有梯度
    assert!(graph.get_node_grad(w).unwrap().is_some());

    // 中间节点的值应该被释放
    assert!(graph.get_node_value(y).unwrap().is_none());
    assert!(graph.get_node_value(z).unwrap().is_none());

    // 中间节点的梯度也应该被释放（与值保持一致，更接近 PyTorch 语义）
    assert!(graph.get_node_grad(y).unwrap().is_none());
    assert!(graph.get_node_grad(z).unwrap().is_none());

    // Input 和 Parameter 的值应该保留
    assert!(graph.get_node_value(x).unwrap().is_some());
    assert!(graph.get_node_value(w).unwrap().is_some());
}

/// 测试: retain_graph=true 允许多次 backward
#[test]
fn test_retain_graph_allows_multiple_backward() {
    let mut graph = Graph::new();

    // 创建网络
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[1, 2], Some("w")).unwrap();
    let y = graph.new_mat_mul_node(w, x, Some("y")).unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();

    // 前向传播
    graph.forward(y).unwrap();

    // 第一次 backward（保留图）
    graph.backward_ex(y, true).unwrap();
    let grad1 = graph.get_node_grad(w).unwrap().unwrap().clone();

    // 清除梯度
    graph.zero_grad().unwrap();

    // 第二次 backward（仍然可以因为保留了图）
    graph.backward_ex(y, true).unwrap();
    let grad2 = graph.get_node_grad(w).unwrap().unwrap();

    // 两次计算应该得到相同的梯度
    assert_eq!(&grad1, grad2);
}

/// 测试: 多任务学习场景 - 两个 loss 共享 backbone
/// PyTorch 对照: tests/python/calc_jacobi_by_pytorch/multi_task_learning_retain_graph.py
#[test]
fn test_retain_graph_multi_task_learning() {
    let mut graph = Graph::new();

    // 创建共享 backbone: x -> w_shared -> features
    let x = graph.new_input_node(&[4, 1], Some("x")).unwrap();
    let w_shared = graph.new_parameter_node(&[2, 4], Some("w_shared")).unwrap();
    let features = graph
        .new_mat_mul_node(w_shared, x, Some("features"))
        .unwrap();

    // 任务 1: features -> w1 -> out1 -> loss1
    let w1 = graph.new_parameter_node(&[1, 2], Some("w1")).unwrap();
    let out1 = graph.new_mat_mul_node(w1, features, Some("out1")).unwrap();
    let target1 = graph.new_input_node(&[1, 1], Some("target1")).unwrap();
    let loss1 = graph
        .new_mse_loss_node(out1, target1, Some("loss1"))
        .unwrap();

    // 任务 2: features -> w2 -> out2 -> loss2
    let w2 = graph.new_parameter_node(&[1, 2], Some("w2")).unwrap();
    let out2 = graph.new_mat_mul_node(w2, features, Some("out2")).unwrap();
    let target2 = graph.new_input_node(&[1, 1], Some("target2")).unwrap();
    let loss2 = graph
        .new_mse_loss_node(out2, target2, Some("loss2"))
        .unwrap();

    // 设置输入和参数值（与 PyTorch 对照测试一致，使用全 1 参数）
    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4, 1]);
    let target_data = Tensor::zeros(&[1, 1]); // 目标为 0，使 loss = out^2
    graph.set_node_value(x, Some(&input_data)).unwrap();
    graph.set_node_value(target1, Some(&target_data)).unwrap();
    graph.set_node_value(target2, Some(&target_data)).unwrap();
    graph
        .set_node_value(w_shared, Some(&Tensor::ones(&[2, 4])))
        .unwrap();
    graph
        .set_node_value(w1, Some(&Tensor::ones(&[1, 2])))
        .unwrap();
    graph
        .set_node_value(w2, Some(&Tensor::ones(&[1, 2])))
        .unwrap();

    // 前向传播两个任务
    graph.forward(loss1).unwrap();
    graph.forward(loss2).unwrap();

    // ========== 验证前向传播结果（PyTorch 对照值）==========
    let features_value = graph.get_node_value(features).unwrap().unwrap();
    let expected_features = Tensor::new(&[10.0, 10.0], &[2, 1]);
    assert_eq!(features_value, &expected_features, "features 前向值不匹配");

    let out1_value = graph.get_node_value(out1).unwrap().unwrap();
    let expected_out1 = Tensor::new(&[20.0], &[1, 1]);
    assert_eq!(out1_value, &expected_out1, "out1 前向值不匹配");

    let out2_value = graph.get_node_value(out2).unwrap().unwrap();
    let expected_out2 = Tensor::new(&[20.0], &[1, 1]);
    assert_eq!(out2_value, &expected_out2, "out2 前向值不匹配");

    // ========== 任务 1 backward（保留图，因为任务 2 也需要）==========
    // 使用新的 VJP API
    graph.backward_ex(loss1, true).unwrap();

    // 验证 w_shared 和 w1 的梯度
    // MSE loss 对 out1 的梯度是 2*(out1 - target1) = 2*20 = 40
    // 然后通过链式法则传播
    let w_shared_grad_1 = graph.get_node_grad(w_shared).unwrap().unwrap().clone();
    let w1_grad = graph.get_node_grad(w1).unwrap().unwrap();

    // w1 的梯度: d(loss1)/d(w1) = d(loss1)/d(out1) * d(out1)/d(w1)
    //                          = 40 * features^T = 40 * [10, 10] = [400, 400]
    let expected_w1_grad = Tensor::new(&[400.0, 400.0], &[1, 2]);
    assert_eq!(w1_grad, &expected_w1_grad, "w1 梯度不匹配");

    // w2 此时不应有梯度（不在 loss1 的计算图中）
    assert!(
        graph.get_node_grad(w2).unwrap().is_none(),
        "w2 在 task1 backward 后不应有梯度"
    );

    // ========== 任务 2 backward（不保留图，梯度累积）==========
    graph.backward_ex(loss2, false).unwrap();

    // 验证 w2 的梯度
    let w2_grad = graph.get_node_grad(w2).unwrap().unwrap();
    let expected_w2_grad = Tensor::new(&[400.0, 400.0], &[1, 2]);
    assert_eq!(w2_grad, &expected_w2_grad, "w2 梯度不匹配");

    // 验证 w_shared 的累积梯度（task1 + task2）
    let w_shared_grad_accumulated = graph.get_node_grad(w_shared).unwrap().unwrap();
    // 两次 backward 的梯度应该累积
    // 每次 backward 对 w_shared 的梯度贡献相同
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

    // 中间节点的值应该被释放（因为最后一次 retain_graph=false）
    assert!(
        graph.get_node_value(features).unwrap().is_none(),
        "features 值应被释放"
    );
}

/// 测试: backward 默认行为（等价于 retain_graph=false，与 PyTorch 一致）
#[test]
fn test_backward_default_releases_graph() {
    let mut graph = Graph::new();

    // 创建网络: x -> w -> y -> loss (标量)
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[2, 2], Some("w")).unwrap();
    let y = graph.new_mat_mul_node(w, x, Some("y")).unwrap();
    let target = graph.new_input_node(&[2, 1], Some("target")).unwrap();
    let loss = graph.new_mse_loss_node(y, target, Some("loss")).unwrap();

    // 设置输入
    let input_data = Tensor::new(&[0.5, -0.3], &[2, 1]);
    let target_data = Tensor::new(&[0.0, 0.0], &[2, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();
    graph
        .set_node_value(w, Some(&Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2])))
        .unwrap();
    graph.set_node_value(target, Some(&target_data)).unwrap();

    // 前向传播
    graph.forward(loss).unwrap();

    // 验证 forward 后中间节点有值
    assert!(graph.get_node_value(y).unwrap().is_some());

    // 使用默认的 backward（应该释放中间值，与 PyTorch 一致）
    graph.backward(loss).unwrap();

    // w 应该有梯度
    assert!(graph.get_node_grad(w).unwrap().is_some());

    // y 的值应该被释放（因为 retain_graph=false）
    assert!(graph.get_node_value(y).unwrap().is_none());

    // 但 x 和 w（Input/Parameter）的值应该保留
    assert!(graph.get_node_value(x).unwrap().is_some());
    assert!(graph.get_node_value(w).unwrap().is_some());
}

/// 测试: retain_graph=false 后再次 backward 需要重新 forward
#[test]
fn test_retain_graph_false_requires_new_forward() {
    let mut graph = Graph::new();

    // 创建网络
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[1, 2], Some("w")).unwrap();
    let y = graph.new_mat_mul_node(w, x, Some("y")).unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();

    // 前向传播
    graph.forward(y).unwrap();

    // 使用 retain_graph=false 反向传播
    graph.backward_ex(y, false).unwrap();

    // 清除梯度
    graph.zero_grad().unwrap();

    // 重新前向传播（必须，因为中间值被释放了）
    graph.forward(y).unwrap();

    // 再次 backward 应该成功
    graph.backward_ex(y, true).unwrap();
    assert!(graph.get_node_grad(w).unwrap().is_some());
}

/// 测试: 混合使用 retain_graph 和 detach
#[test]
fn test_retain_graph_with_detach() {
    let mut graph = Graph::new();

    // 创建网络: x -> w1 -> h -> w2 -> y -> loss
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w1 = graph.new_parameter_node(&[2, 2], Some("w1")).unwrap();
    let h = graph.new_mat_mul_node(w1, x, Some("h")).unwrap();
    let w2 = graph.new_parameter_node(&[1, 2], Some("w2")).unwrap();
    let y = graph.new_mat_mul_node(w2, h, Some("y")).unwrap();
    let target = graph.new_input_node(&[1, 1], Some("target")).unwrap();
    let loss = graph.new_mse_loss_node(y, target, Some("loss")).unwrap();

    // 设置输入
    let input_data = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let target_data = Tensor::zeros(&[1, 1]);
    graph.set_node_value(x, Some(&input_data)).unwrap();
    graph.set_node_value(target, Some(&target_data)).unwrap();

    // 前向传播
    graph.forward(loss).unwrap();

    // detach h
    graph.detach_node(h).unwrap();

    // 使用 retain_graph=true 反向传播
    graph.backward_ex(loss, true).unwrap();

    // w2 应该有梯度
    assert!(graph.get_node_grad(w2).unwrap().is_some());

    // h 和 y 的值应该仍然存在（retain_graph=true）
    assert!(graph.get_node_value(h).unwrap().is_some());
    assert!(graph.get_node_value(y).unwrap().is_some());

    // 验证 detach 效果：h 被 detach 后，h 不应有 grad
    // 梯度在 detach 点被阻断
    assert!(
        graph.get_node_grad(h).unwrap().is_none(),
        "detach 的节点不应有 grad"
    );

    // w1 不应有梯度（因为 h 被 detach 阻断了梯度流）
    assert!(
        graph.get_node_grad(w1).unwrap().is_none(),
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
    let mut graph = Graph::new();

    // ========== 构建拓扑 ==========
    // 输入
    let x = graph.new_input_node(&[4, 1], Some("x")).unwrap();

    // 共享链: x → w_shared1 → shared_feat1 → w_shared2 → w_shared3 → shared_feat2
    let w_shared1 = graph
        .new_parameter_node(&[2, 4], Some("w_shared1"))
        .unwrap();
    let shared_feat1 = graph
        .new_mat_mul_node(w_shared1, x, Some("shared_feat1"))
        .unwrap();

    // w_shared2 和 w_shared3 是相邻参数节点（测试链式累积）
    let w_shared2 = graph
        .new_parameter_node(&[2, 2], Some("w_shared2"))
        .unwrap();
    let w_shared2_out = graph
        .new_mat_mul_node(w_shared2, shared_feat1, Some("w_shared2_out"))
        .unwrap();

    let w_shared3 = graph
        .new_parameter_node(&[2, 2], Some("w_shared3"))
        .unwrap();
    let shared_feat2 = graph
        .new_mat_mul_node(w_shared3, w_shared2_out, Some("shared_feat2"))
        .unwrap();

    // 分叉: shared_feat2 → w_task1 → out1 → loss1, shared_feat2 → w_task2 → out2 → loss2
    let w_task1 = graph.new_parameter_node(&[1, 2], Some("w_task1")).unwrap();
    let out1 = graph
        .new_mat_mul_node(w_task1, shared_feat2, Some("out1"))
        .unwrap();
    let target1 = graph.new_input_node(&[1, 1], Some("target1")).unwrap();
    let loss1 = graph
        .new_mse_loss_node(out1, target1, Some("loss1"))
        .unwrap();

    let w_task2 = graph.new_parameter_node(&[1, 2], Some("w_task2")).unwrap();
    let out2 = graph
        .new_mat_mul_node(w_task2, shared_feat2, Some("out2"))
        .unwrap();
    let target2 = graph.new_input_node(&[1, 1], Some("target2")).unwrap();
    let loss2 = graph
        .new_mse_loss_node(out2, target2, Some("loss2"))
        .unwrap();

    // ========== 设置固定值 ==========
    graph
        .set_node_value(x, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4, 1])))
        .unwrap();
    graph
        .set_node_value(target1, Some(&Tensor::zeros(&[1, 1])))
        .unwrap();
    graph
        .set_node_value(target2, Some(&Tensor::zeros(&[1, 1])))
        .unwrap();
    graph
        .set_node_value(w_shared1, Some(&Tensor::ones(&[2, 4])))
        .unwrap();
    graph
        .set_node_value(w_shared2, Some(&Tensor::ones(&[2, 2])))
        .unwrap();
    graph
        .set_node_value(w_shared3, Some(&Tensor::ones(&[2, 2])))
        .unwrap();
    graph
        .set_node_value(w_task1, Some(&Tensor::ones(&[1, 2])))
        .unwrap();
    graph
        .set_node_value(w_task2, Some(&Tensor::ones(&[1, 2])))
        .unwrap();

    // ========== 前向传播 ==========
    graph.forward(loss1).unwrap();
    graph.forward(loss2).unwrap();

    // ========== 第 1 次 backward (loss1, retain_graph=true) ==========
    graph.backward_ex(loss1, true).unwrap();

    let w_shared1_after_task1 = graph.get_node_grad(w_shared1).unwrap().unwrap().clone();
    let w_shared2_after_task1 = graph.get_node_grad(w_shared2).unwrap().unwrap().clone();
    let w_shared3_after_task1 = graph.get_node_grad(w_shared3).unwrap().unwrap().clone();
    let w_task1_after_task1 = graph.get_node_grad(w_task1).unwrap().unwrap().clone();

    // w_task2 此时不应有梯度（不在 loss1 的计算图中）
    assert!(
        graph.get_node_grad(w_task2).unwrap().is_none(),
        "w_task2 在 task1 backward 后不应有梯度"
    );

    // 中间节点应该有本次的值（可访问，但 retain_graph=true 时保留）
    assert!(
        graph.get_node_value(shared_feat1).unwrap().is_some(),
        "shared_feat1 应有本次 forward 的值"
    );
    assert!(
        graph.get_node_value(w_shared2_out).unwrap().is_some(),
        "w_shared2_out 应有本次 forward 的值"
    );
    assert!(
        graph.get_node_value(shared_feat2).unwrap().is_some(),
        "shared_feat2 应有本次 forward 的值"
    );

    // ========== 第 2 次 backward (loss2, retain_graph=false) ==========
    // 这次累积所有共享参数
    graph.backward_ex(loss2, false).unwrap();

    let w_shared1_accumulated = graph.get_node_grad(w_shared1).unwrap().unwrap();
    let w_shared2_accumulated = graph.get_node_grad(w_shared2).unwrap().unwrap();
    let w_shared3_accumulated = graph.get_node_grad(w_shared3).unwrap().unwrap();
    let w_task2_after_task2 = graph.get_node_grad(w_task2).unwrap().unwrap();

    // ========== 验证累积效果 ==========
    // 验证累积后的梯度比第一次的梯度大（因为累积了两次）
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

    // w_task1 的梯度应该只有 task1 的贡献（task2 不会对 w_task1 产生梯度）
    let w_task1_final = graph.get_node_grad(w_task1).unwrap().unwrap();
    assert_eq!(
        &w_task1_after_task1, w_task1_final,
        "w_task1 只有 task1 的贡献，不应变化"
    );

    // w_task2 应该有 task2 的梯度
    assert!(w_task2_after_task2.size() > 0, "w_task2 应有 task2 的梯度");

    // 中间节点的值应该被释放（retain_graph=false 时释放）
    assert!(
        graph.get_node_value(shared_feat1).unwrap().is_none(),
        "shared_feat1 值应被释放"
    );
    assert!(
        graph.get_node_value(w_shared2_out).unwrap().is_none(),
        "w_shared2_out 值应被释放"
    );
    assert!(
        graph.get_node_value(shared_feat2).unwrap().is_none(),
        "shared_feat2 值应被释放"
    );
}
