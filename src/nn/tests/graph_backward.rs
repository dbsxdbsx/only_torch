/*!
 * Graph 反向传播测试（底层 GraphInner API）
 *
 * 使用 Graph::new() + inner_rc() + borrow_mut() 访问 GraphInner，
 * 验证：反向传播 pass ID、梯度获取、梯度累积、错误处理。
 */

use crate::assert_err;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

/// 测试节点梯度获取（VJP 模式）
/// 验证：输入节点不应该有梯度，参数节点反向传播后有梯度
#[test]
fn test_node_grad_retrieval() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建计算图：loss = MSE(y, target)，其中 y = wx + b
    let x = gi.create_basic_input_node(&[3, 1], Some("x")).unwrap();
    let w = gi.create_parameter_node(&[1, 3], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w)).unwrap();
    let b = gi.create_parameter_node(&[1, 1], Some("b")).unwrap();
    gi.register_parameter("b".to_string(), Rc::downgrade(&b)).unwrap();
    let wx = gi.create_mat_mul_node(vec![w.clone(), x.clone()], None).unwrap();
    let y = gi.create_add_node(vec![wx.clone(), b.clone()], None).unwrap();
    let target = gi.create_basic_input_node(&[1, 1], Some("target")).unwrap();
    let loss = gi.create_mse_mean_node(y.clone(), target.clone(), Some("loss")).unwrap();

    // 2. 测试未计算时的梯度获取
    // 2.1 输入节点不应该有梯度（新 API 下 grad() 返回 None）
    assert!(x.grad().is_none(), "输入节点不应有梯度");

    // 2.2 参数节点在反向传播前应该返回 None
    assert_eq!(w.grad(), None);
    assert_eq!(b.grad(), None);

    // 2.3 计算节点在反向传播前应该返回 None
    assert_eq!(wx.grad(), None);
    assert_eq!(y.grad(), None);

    // 3. 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    let w_value = Tensor::new(&[0.1, 0.2, 0.3], &[1, 3]);
    let b_value = Tensor::new(&[0.4], &[1, 1]);
    let target_value = Tensor::new(&[1.0], &[1, 1]);
    x.set_value(Some(&x_value)).unwrap();
    w.set_value(Some(&w_value)).unwrap();
    b.set_value(Some(&b_value)).unwrap();
    target.set_value(Some(&target_value)).unwrap();

    // 4. 前向传播
    gi.forward_via_node_inner(&loss).unwrap();

    // 5. 反向传播
    gi.backward_via_node_inner(&loss).unwrap();

    // 6. 验证梯度
    // 6.1 输入节点仍然不应该有梯度
    assert!(x.grad().is_none(), "输入节点不应有梯度");

    // 6.2 验证 w 有梯度（参与反向传播的参数节点）
    let w_grad = w.grad().unwrap();
    assert_eq!(w_grad.shape(), &[1, 3]); // 梯度形状应该与参数形状一致

    // 6.3 验证 b 有梯度（参与反向传播的参数节点）
    let b_grad = b.grad().unwrap();
    assert_eq!(b_grad.shape(), &[1, 1]); // 梯度形状应该与参数形状一致
}

/// 测试梯度计算正确性
/// 验证：y = wx 情况下，dL/dw 的正确性
#[test]
fn test_node_grad_computation() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建计算图：loss = MSE(y, target)，其中 y = wx
    let x = gi.create_basic_input_node(&[3, 1], Some("x")).unwrap();
    let w = gi.create_parameter_node(&[1, 3], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w)).unwrap();
    let y = gi.create_mat_mul_node(vec![w.clone(), x.clone()], None).unwrap();
    let target = gi.create_basic_input_node(&[1, 1], Some("target")).unwrap();
    let loss = gi.create_mse_mean_node(y.clone(), target.clone(), Some("loss")).unwrap();

    // 2. 设置输入值并进行前向和反向传播
    let x_value = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    let w_value = Tensor::new(&[0.1, 0.2, 0.3], &[1, 3]);
    let target_value = Tensor::new(&[2.0], &[1, 1]); // y_pred = 0.1*1 + 0.2*2 + 0.3*3 = 1.4
    x.set_value(Some(&x_value)).unwrap();
    w.set_value(Some(&w_value)).unwrap();
    target.set_value(Some(&target_value)).unwrap();

    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();

    // 3. 验证梯度形状与参数形状一致
    let w_grad = w.grad().unwrap();
    assert_eq!(w_grad.shape(), w_value.shape());

    // 4. 验证梯度数值正确性
    // y_pred = 1.4, target = 2.0, diff = -0.6
    // MSE = (y - target)^2 / n = 0.36 / 1 = 0.36
    // dL/dy = 2 * (y - target) / n = 2 * (-0.6) / 1 = -1.2
    // dy/dw = x^T = [1, 2, 3]
    // dL/dw = dL/dy * dy/dw = -1.2 * [1, 2, 3] = [-1.2, -2.4, -3.6]
    assert_abs_diff_eq!(w_grad[[0, 0]], -1.2, epsilon = 1e-5);
    assert_abs_diff_eq!(w_grad[[0, 1]], -2.4, epsilon = 1e-5);
    assert_abs_diff_eq!(w_grad[[0, 2]], -3.6, epsilon = 1e-5);
}

/// 测试连续反向传播时的梯度累积
/// 验证：多次 backward 会累积梯度，zero_grad 会清零
#[test]
fn test_continuous_backward_grad_accumulation() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建计算图：loss = MSE(y, target)，其中 y = x + b
    let x = gi.create_basic_input_node(&[2, 1], Some("x")).unwrap();
    let b = gi.create_parameter_node(&[2, 1], Some("b")).unwrap();
    gi.register_parameter("b".to_string(), Rc::downgrade(&b)).unwrap();
    let y = gi.create_add_node(vec![x.clone(), b.clone()], Some("y")).unwrap();
    let target = gi.create_basic_input_node(&[2, 1], Some("target")).unwrap();
    let loss = gi.create_mse_mean_node(y.clone(), target.clone(), Some("loss")).unwrap();

    // 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let b_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    let target_value = Tensor::new(&[1.0, 2.0], &[2, 1]); // y = x + b = [1.1, 2.2]，target = [1, 2]
    x.set_value(Some(&x_value)).unwrap();
    b.set_value(Some(&b_value)).unwrap();
    target.set_value(Some(&target_value)).unwrap();

    // 前向传播
    gi.forward_via_node_inner(&loss).unwrap();

    // 验证初始状态：没有梯度
    assert_eq!(b.grad(), None);

    // 第1次反向传播
    gi.backward_via_node_inner(&loss).unwrap();
    let first_grad = b.grad().unwrap().clone();

    // 对于 y = x + b，dy/db = I
    // MSE loss: L = mean((y - target)^2) = mean([0.1^2, 0.2^2]) = 0.025
    // dL/dy = 2 * (y - target) / n = 2 * [0.1, 0.2] / 2 = [0.1, 0.2]
    // dL/db = dL/dy * dy/db = [0.1, 0.2]
    assert_abs_diff_eq!(first_grad[[0, 0]], 0.1, epsilon = 1e-5);
    assert_abs_diff_eq!(first_grad[[1, 0]], 0.2, epsilon = 1e-5);

    // 第2次反向传播（连续）- 应该累积梯度
    gi.backward_via_node_inner(&loss).unwrap();
    let second_grad = b.grad().unwrap().clone();

    // 累积后应该是2倍
    assert_abs_diff_eq!(second_grad[[0, 0]], 0.2, epsilon = 1e-5);
    assert_abs_diff_eq!(second_grad[[1, 0]], 0.4, epsilon = 1e-5);

    // 第3次反向传播 - 继续累积
    gi.backward_via_node_inner(&loss).unwrap();
    let third_grad = b.grad().unwrap().clone();

    // 累积后应该是3倍
    assert_abs_diff_eq!(third_grad[[0, 0]], 0.3, epsilon = 1e-5);
    assert_abs_diff_eq!(third_grad[[1, 0]], 0.6, epsilon = 1e-5);

    // 测试 zero_grad 功能
    gi.zero_grad().unwrap();
    assert_eq!(b.grad(), None);

    // 清除后再次反向传播，应该重新开始
    gi.backward_via_node_inner(&loss).unwrap();
    let after_clear_grad = b.grad().unwrap().clone();
    assert_abs_diff_eq!(after_clear_grad[[0, 0]], 0.1, epsilon = 1e-5);
    assert_abs_diff_eq!(after_clear_grad[[1, 0]], 0.2, epsilon = 1e-5);

    // 测试 set_value 不会自动清除梯度的行为（需要手动 zero_grad）
    // 先进行反向传播，确保有梯度
    gi.backward_via_node_inner(&loss).unwrap();
    let grad_before_set = b.grad().unwrap().clone();

    // 设置新的参数值，梯度应该仍然存在（不自动清除）
    let new_b_value = Tensor::new(&[0.3, 0.4], &[2, 1]);
    b.set_value(Some(&new_b_value)).unwrap();
    let grad_after_set = b.grad().unwrap().clone();
    assert_eq!(grad_after_set, grad_before_set); // 梯度应该保持不变
}

/// 测试没有前向传播时的反向传播错误处理
/// 验证：如果 loss 节点没有值，backward 应该报错
#[test]
fn test_backward_without_any_forward() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建计算图：loss = MSE(y, target)，其中 y = wx + b
    let x = gi.create_basic_input_node(&[3, 1], Some("x")).unwrap();
    let w = gi.create_parameter_node(&[1, 3], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w)).unwrap();
    let b = gi.create_parameter_node(&[1, 1], Some("b")).unwrap();
    gi.register_parameter("b".to_string(), Rc::downgrade(&b)).unwrap();
    let wx = gi.create_mat_mul_node(vec![w.clone(), x.clone()], Some("wx")).unwrap();
    let y = gi.create_add_node(vec![wx.clone(), b.clone()], Some("y")).unwrap();
    let target = gi.create_basic_input_node(&[1, 1], Some("target")).unwrap();
    let loss = gi.create_mse_mean_node(y.clone(), target.clone(), Some("loss")).unwrap();

    // 设置输入值，但不进行任何前向传播
    let x_value = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    x.set_value(Some(&x_value)).unwrap();

    // 验证：所有节点的前向传播ID都是0（没有前向传播）
    assert_eq!(x.last_forward_pass_id(), 0);
    assert_eq!(w.last_forward_pass_id(), 0);
    assert_eq!(b.last_forward_pass_id(), 0);
    assert_eq!(wx.last_forward_pass_id(), 0);
    assert_eq!(y.last_forward_pass_id(), 0);
    assert_eq!(loss.last_forward_pass_id(), 0);

    // 验证：loss 节点没有值（因为没有前向传播）
    assert!(loss.value().is_none());

    // 关键测试：在没有任何前向传播的情况下尝试反向传播
    // 这应该失败，因为 loss 节点没有值
    assert_err!(
        gi.backward_via_node_inner(&loss),
        GraphError::ComputationError(msg) if msg.contains("没有值") && msg.contains("请先执行 forward")
    );

    // 验证：反向传播失败后，参数节点仍然没有梯度
    assert!(w.grad().is_none());
    assert!(b.grad().is_none());

    // 验证：图的反向传播ID没有增加（因为反向传播失败了）
    assert_eq!(gi.last_backward_pass_id(), 0);
}

/// 测试部分前向传播时的反向传播行为
/// 验证：即使参数节点有未前向传播的子节点分支，反向传播仍能正常工作
#[test]
fn test_backward_with_partial_forward_propagation() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 创建一个计算图：loss = MSE(z, target)
    // 其中 z = (x + a) + (y + b)，参数 a 有多个子节点，但只有部分子节点参与了前向传播
    // 结构：
    //   a -> left_add -> z -> loss (参与前向传播)
    //   a -> new_add (不参与前向传播)
    let x = gi.create_basic_input_node(&[2, 1], Some("x")).unwrap();
    let y = gi.create_basic_input_node(&[2, 1], Some("y")).unwrap();
    let a = gi.create_parameter_node(&[2, 1], Some("a")).unwrap();
    gi.register_parameter("a".to_string(), Rc::downgrade(&a)).unwrap();
    let b = gi.create_parameter_node(&[2, 1], Some("b")).unwrap();
    gi.register_parameter("b".to_string(), Rc::downgrade(&b)).unwrap();
    let c = gi.create_parameter_node(&[2, 1], Some("c")).unwrap();
    gi.register_parameter("c".to_string(), Rc::downgrade(&c)).unwrap();

    let left_add = gi.create_add_node(vec![x.clone(), a.clone()], Some("left_add")).unwrap();
    let right_add = gi.create_add_node(vec![y.clone(), b.clone()], Some("right_add")).unwrap();
    let z = gi.create_add_node(vec![left_add.clone(), right_add.clone()], Some("z")).unwrap();

    // 添加 loss 节点
    let target = gi.create_basic_input_node(&[2, 1], Some("target")).unwrap();
    let loss = gi.create_mse_mean_node(z.clone(), target.clone(), Some("loss")).unwrap();

    // 创建一个不参与主计算路径的分支
    let new_add = gi.create_add_node(vec![a.clone(), c.clone()], Some("new_add")).unwrap();

    // 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let y_value = Tensor::new(&[3.0, 4.0], &[2, 1]);
    let a_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    let b_value = Tensor::new(&[0.3, 0.4], &[2, 1]);
    let c_value = Tensor::new(&[0.5, 0.6], &[2, 1]);
    let target_value = Tensor::new(&[4.0, 6.0], &[2, 1]); // 近似于 z 的预期输出

    x.set_value(Some(&x_value)).unwrap();
    y.set_value(Some(&y_value)).unwrap();
    a.set_value(Some(&a_value)).unwrap();
    b.set_value(Some(&b_value)).unwrap();
    c.set_value(Some(&c_value)).unwrap();
    target.set_value(Some(&target_value)).unwrap();

    // 只对主路径进行前向传播，不对new_add分支进行前向传播
    gi.forward_via_node_inner(&loss).unwrap();

    // 验证：主路径已前向传播，new_add分支没有
    assert_eq!(left_add.last_forward_pass_id(), 1);
    assert_eq!(right_add.last_forward_pass_id(), 1);
    assert_eq!(z.last_forward_pass_id(), 1);
    assert_eq!(loss.last_forward_pass_id(), 1);
    assert_eq!(new_add.last_forward_pass_id(), 0);

    // 关键测试：反向传播
    // 即使 a 有一个子节点(new_add)没有前向传播，反向传播也应该成功
    // 并且只考虑已前向传播的子节点(left_add)
    gi.backward_via_node_inner(&loss).unwrap();

    // 验证反向传播成功 - a 有梯度
    let a_grad = a.grad().unwrap();
    // 对于 z = (x + a) + (y + b)，dz/da = I
    // 加上 MSE loss 的链式法则
    assert_eq!(a_grad.shape(), &[2, 1]);

    // 验证 new_add 分支确实没有参与反向传播
    assert_eq!(new_add.last_forward_pass_id(), 0);
    assert!(new_add.grad().is_none());

    // 这个测试证明了：即使参数节点有未前向传播的子节点分支，
    // 反向传播仍然能够正常工作，只考虑已前向传播的路径
}

/// 测试反向传播 pass_id 递增行为
/// 验证：每次 backward 后 pass_id 增加
#[test]
fn test_backward_pass_id_increment() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建计算图：loss = MSE(y, target)，其中 y = x + b
    let x = gi.create_basic_input_node(&[2, 1], Some("x")).unwrap();
    let b = gi.create_parameter_node(&[2, 1], Some("b")).unwrap();
    gi.register_parameter("b".to_string(), Rc::downgrade(&b)).unwrap();
    let y = gi.create_add_node(vec![x.clone(), b.clone()], Some("y")).unwrap();
    let target = gi.create_basic_input_node(&[2, 1], Some("target")).unwrap();
    let loss = gi.create_mse_mean_node(y.clone(), target.clone(), Some("loss")).unwrap();

    // 2. 初始状态：pass_id应该为0
    assert_eq!(gi.last_backward_pass_id(), 0);

    // 3. 设置输入值并前向传播
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let b_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    let target_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    x.set_value(Some(&x_value)).unwrap();
    b.set_value(Some(&b_value)).unwrap();
    target.set_value(Some(&target_value)).unwrap();
    gi.forward_via_node_inner(&loss).unwrap();

    // 4. 第1次反向传播
    gi.backward_via_node_inner(&loss).unwrap();
    assert_eq!(gi.last_backward_pass_id(), 1);

    // 5. 第2次反向传播
    gi.backward_via_node_inner(&loss).unwrap();
    assert_eq!(gi.last_backward_pass_id(), 2);

    // 6. 第3次反向传播（最后一次可以不保留图）
    gi.backward_via_node_inner(&loss).unwrap();
    assert_eq!(gi.last_backward_pass_id(), 3);
}

/// 测试节点 pass_id 同步行为
/// 验证：前向/反向传播后，参与的节点的 pass_id 与图一致
#[test]
fn test_node_pass_id_synchronization() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建计算图：loss = MSE(z, target)，其中 z = (x + y) * w
    let x = gi.create_basic_input_node(&[2, 1], Some("x")).unwrap();
    let y = gi.create_basic_input_node(&[2, 1], Some("y")).unwrap();
    let w = gi.create_parameter_node(&[1, 2], Some("w")).unwrap();
    gi.register_parameter("w".to_string(), Rc::downgrade(&w)).unwrap();
    let add = gi.create_add_node(vec![x.clone(), y.clone()], Some("add")).unwrap();
    let z = gi.create_mat_mul_node(vec![w.clone(), add.clone()], Some("z")).unwrap();
    let target = gi.create_basic_input_node(&[1, 1], Some("target")).unwrap();
    let loss = gi.create_mse_mean_node(z.clone(), target.clone(), Some("loss")).unwrap();

    // 2. 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let y_value = Tensor::new(&[0.5, 1.5], &[2, 1]);
    let w_value = Tensor::new(&[0.1, 0.2], &[1, 2]);
    let target_value = Tensor::new(&[1.0], &[1, 1]);
    x.set_value(Some(&x_value)).unwrap();
    y.set_value(Some(&y_value)).unwrap();
    w.set_value(Some(&w_value)).unwrap();
    target.set_value(Some(&target_value)).unwrap();

    // 3. 前向传播并验证节点pass_id同步
    gi.forward_via_node_inner(&loss).unwrap();
    let graph_forward_pass_id = gi.last_forward_pass_id();

    // 验证所有参与计算的节点的前向pass_id都与图的pass_id一致
    assert_eq!(x.last_forward_pass_id(), graph_forward_pass_id);
    assert_eq!(y.last_forward_pass_id(), graph_forward_pass_id);
    assert_eq!(w.last_forward_pass_id(), graph_forward_pass_id);
    assert_eq!(add.last_forward_pass_id(), graph_forward_pass_id);
    assert_eq!(z.last_forward_pass_id(), graph_forward_pass_id);
    assert_eq!(loss.last_forward_pass_id(), graph_forward_pass_id);

    // 4. 反向传播并验证节点pass_id同步
    gi.backward_via_node_inner(&loss).unwrap();
    let graph_backward_pass_id = gi.last_backward_pass_id();

    // 验证参与反向传播的节点的反向pass_id都与图的pass_id一致
    assert_eq!(w.last_backward_pass_id(), graph_backward_pass_id);
    assert_eq!(add.last_backward_pass_id(), graph_backward_pass_id);
    assert_eq!(z.last_backward_pass_id(), graph_backward_pass_id);
    assert_eq!(loss.last_backward_pass_id(), graph_backward_pass_id);

    // 输入节点不参与反向传播（梯度汇点），所以其反向pass_id应该仍为0
    assert_eq!(x.last_backward_pass_id(), 0);
    assert_eq!(y.last_backward_pass_id(), 0);
    assert_eq!(target.last_backward_pass_id(), 0);
}

/// 测试反向传播错误时的 pass_id 回滚
/// 验证：当 backward 失败时，pass_id 应该不变
#[test]
fn test_pass_id_rollback_on_backward_error() {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let mut gi = inner.borrow_mut();

    // 1. 创建计算图：loss = MSE(y, target)，其中 y = x + b
    let x = gi.create_basic_input_node(&[2, 1], Some("x")).unwrap();
    let b = gi.create_parameter_node(&[2, 1], Some("b")).unwrap();
    gi.register_parameter("b".to_string(), Rc::downgrade(&b)).unwrap();
    let y = gi.create_add_node(vec![x.clone(), b.clone()], Some("y")).unwrap();
    let target = gi.create_basic_input_node(&[2, 1], Some("target")).unwrap();
    let loss = gi.create_mse_mean_node(y.clone(), target.clone(), Some("loss")).unwrap();

    // 2. 设置输入值但不对 loss 进行前向传播（只对 y 进行前向传播）
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let b_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    let target_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    x.set_value(Some(&x_value)).unwrap();
    b.set_value(Some(&b_value)).unwrap();
    target.set_value(Some(&target_value)).unwrap();

    // 只对 y 进行前向传播，不对 loss 进行前向传播
    gi.forward_via_node_inner(&y).unwrap();

    // 3. 记录初始反向传播pass_id
    let initial_backward_pass_id = gi.last_backward_pass_id();
    assert_eq!(initial_backward_pass_id, 0);

    // 4. 尝试反向传播，应该失败（因为 loss 节点没有值）
    let backward_result = gi.backward_via_node_inner(&loss);
    assert!(backward_result.is_err());

    // 验证反向传播失败后pass_id被正确回滚
    assert_eq!(gi.last_backward_pass_id(), initial_backward_pass_id);

    // 5. 正确地进行前向传播后，反向传播应该成功
    gi.forward_via_node_inner(&loss).unwrap();
    gi.backward_via_node_inner(&loss).unwrap();
    assert_eq!(gi.last_backward_pass_id(), 1);
}
