/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Maximum 节点单元测试
 *
 * 测试覆盖：
 * - 前向传播（相同形状、广播）
 * - 反向传播（梯度计算、广播梯度收缩）
 * - 相等值时梯度各半
 * - 强化学习场景（PPO/TD3 风格）
 */

use crate::nn::graph::Graph;
use crate::nn::{Init, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试 ====================

/// 测试 Maximum 前向传播 - 相同形状
#[test]
fn test_maximum_forward_same_shape() {
    let graph = Graph::new();

    let a = graph.input(&Tensor::new(&[1.0, 5.0, 3.0], &[3, 1])).unwrap();
    let b = graph.input(&Tensor::new(&[2.0, 4.0, 6.0], &[3, 1])).unwrap();

    // 创建 Maximum 节点
    let max_id = graph
        .inner_mut()
        .new_maximum_node(a.node_id(), b.node_id(), Some("max"))
        .unwrap();
    let max_var = graph.wrap_node_id(max_id);

    // 前向传播
    max_var.forward().unwrap();

    // 验证结果
    let result = max_var.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[3, 1]);
    assert_eq!(result.data_as_slice(), &[2.0, 5.0, 6.0]); // max(1,2)=2, max(5,4)=5, max(3,6)=6
}

/// 测试 Maximum 前向传播 - 广播
#[test]
fn test_maximum_forward_broadcast() {
    let graph = Graph::new();

    // a: [3, 1], b: [1, 4] -> 广播后 [3, 4]
    let a = graph.input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1])).unwrap();
    let b = graph
        .input(&Tensor::new(&[0.0, 1.5, 2.5, 4.0], &[1, 4]))
        .unwrap();

    let max_id = graph
        .inner_mut()
        .new_maximum_node(a.node_id(), b.node_id(), None)
        .unwrap();
    let max_var = graph.wrap_node_id(max_id);

    max_var.forward().unwrap();

    let result = max_var.value().unwrap().unwrap();
    assert_eq!(result.shape(), &[3, 4]);

    // 验证部分值
    // max(1, [0, 1.5, 2.5, 4]) = [1, 1.5, 2.5, 4]
    // max(2, [0, 1.5, 2.5, 4]) = [2, 2, 2.5, 4]
    // max(3, [0, 1.5, 2.5, 4]) = [3, 3, 3, 4]
    let expected = [1.0, 1.5, 2.5, 4.0, 2.0, 2.0, 2.5, 4.0, 3.0, 3.0, 3.0, 4.0];
    assert_eq!(result.data_as_slice(), &expected);
}

// ==================== 反向传播测试 ====================

/// 测试 Maximum 反向传播 - 基本场景
#[test]
fn test_maximum_backward_basic() {
    let graph = Graph::new();

    let a = graph.parameter(&[3, 1], Init::Zeros, "a").unwrap();
    let b = graph.parameter(&[3, 1], Init::Zeros, "b").unwrap();
    a.set_value(&Tensor::new(&[1.0, 5.0, 3.0], &[3, 1])).unwrap();
    b.set_value(&Tensor::new(&[2.0, 4.0, 3.0], &[3, 1])).unwrap(); // 最后一个相等

    // max(a, b) = [2, 5, 3]
    let max_id = graph
        .inner_mut()
        .new_maximum_node(a.node_id(), b.node_id(), None)
        .unwrap();
    let max_var = graph.wrap_node_id(max_id);

    // 用 MSE loss 来触发反向传播
    let target = Tensor::new(&[3.0, 6.0, 4.0], &[3, 1]);
    let loss = max_var.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // 验证梯度
    // max = [2, 5, 3], target = [3, 6, 4]
    // d(mse)/d(max) = 2*(max - target) / n = 2*[-1, -1, -1] / 3 = [-2/3, -2/3, -2/3]
    //
    // 对于 a:
    //   位置 0: a=1 < b=2, mask=0, grad=0
    //   位置 1: a=5 > b=4, mask=1, grad=-2/3
    //   位置 2: a=3 = b=3, mask=0.5, grad=-1/3
    //
    // 对于 b:
    //   位置 0: a=1 < b=2, mask=1, grad=-2/3
    //   位置 1: a=5 > b=4, mask=0, grad=0
    //   位置 2: a=3 = b=3, mask=0.5, grad=-1/3

    let a_grad = a.grad().unwrap().unwrap();
    let b_grad = b.grad().unwrap().unwrap();

    assert_abs_diff_eq!(a_grad[[0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(a_grad[[1, 0]], -2.0 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(a_grad[[2, 0]], -1.0 / 3.0, epsilon = 1e-5);

    assert_abs_diff_eq!(b_grad[[0, 0]], -2.0 / 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(b_grad[[1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(b_grad[[2, 0]], -1.0 / 3.0, epsilon = 1e-5);
}

/// 测试 Maximum 反向传播 - 广播场景
#[test]
fn test_maximum_backward_broadcast() {
    let graph = Graph::new();

    // a: [3, 1], b: [1, 3] -> 广播后 [3, 3]
    let a = graph.parameter(&[3, 1], Init::Zeros, "a").unwrap();
    let b = graph.parameter(&[1, 3], Init::Zeros, "b").unwrap();
    a.set_value(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1])).unwrap();
    b.set_value(&Tensor::new(&[0.0, 2.0, 4.0], &[1, 3])).unwrap();

    let max_id = graph
        .inner_mut()
        .new_maximum_node(a.node_id(), b.node_id(), None)
        .unwrap();
    let max_var = graph.wrap_node_id(max_id);

    // max 结果 [3, 3]:
    // a_broadcast = [[1,1,1], [2,2,2], [3,3,3]]
    // b_broadcast = [[0,2,4], [0,2,4], [0,2,4]]
    // max = [[1, 2, 4],
    //        [2, 2, 4],
    //        [3, 3, 4]]

    // 用 MSE loss，target 全零，这样 d(mse)/d(max) = 2*max/n = 2*max/9
    let target = Tensor::zeros(&[3, 3]);
    let loss = max_var.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // upstream_grad = 2*max/9
    //
    // 对于 a (形状 [3, 1])：需要沿 axis=1 求和
    //   a[0]=1: [0,0] a>b, mask=1, grad=2*1/9; [0,1] a<b, mask=0; [0,2] a<b, mask=0
    //   sum = 2/9
    //
    //   a[1]=2: [1,0] a>b, mask=1, grad=2*2/9; [1,1] a=b, mask=0.5, grad=2*2/9*0.5; [1,2] a<b, mask=0
    //   sum = 4/9 + 2/9 = 6/9
    //
    //   a[2]=3: [2,0] a>b, mask=1, grad=2*3/9; [2,1] a>b, mask=1, grad=2*3/9; [2,2] a<b, mask=0
    //   sum = 6/9 + 6/9 = 12/9

    let a_grad = a.grad().unwrap().unwrap();
    assert_abs_diff_eq!(a_grad[[0, 0]], 2.0 / 9.0, epsilon = 1e-5);
    assert_abs_diff_eq!(a_grad[[1, 0]], 6.0 / 9.0, epsilon = 1e-5);
    assert_abs_diff_eq!(a_grad[[2, 0]], 12.0 / 9.0, epsilon = 1e-5);

    // 对于 b (形状 [1, 3])：需要沿 axis=0 求和
    //   b[0]=0: 列 0 全是 a 赢 -> 贡献 0
    //
    //   b[1]=2: [0,1] a<b, mask=1, grad=2*2/9; [1,1] a=b, mask=0.5, grad=2*2/9*0.5; [2,1] a>b, mask=0
    //   sum = 4/9 + 2/9 = 6/9
    //
    //   b[2]=4: [0,2] a<b, mask=1, grad=2*4/9; [1,2] a<b, mask=1, grad=2*4/9; [2,2] a<b, mask=1, grad=2*4/9
    //   sum = 8/9 * 3 = 24/9

    let b_grad = b.grad().unwrap().unwrap();
    assert_abs_diff_eq!(b_grad[[0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(b_grad[[0, 1]], 6.0 / 9.0, epsilon = 1e-5);
    assert_abs_diff_eq!(b_grad[[0, 2]], 24.0 / 9.0, epsilon = 1e-5);
}

// ==================== 强化学习场景测试 ====================

/// 测试 TD3 风格：min(Q1, Q2) 用于 actor loss
///
/// 在 TD3 中，actor 的目标是最大化 min(Q1(s, π(s)), Q2(s, π(s)))
/// 梯度只流向产生较小 Q 值的那个 critic
#[test]
fn test_maximum_td3_style() {
    let graph = Graph::new();

    // 模拟两个 Q 网络的输出
    let q1 = graph.parameter(&[4, 1], Init::Zeros, "Q1").unwrap();
    let q2 = graph.parameter(&[4, 1], Init::Zeros, "Q2").unwrap();
    q1.set_value(&Tensor::new(&[1.0, 3.0, 2.0, 4.0], &[4, 1]))
        .unwrap();
    q2.set_value(&Tensor::new(&[2.0, 2.0, 3.0, 3.0], &[4, 1]))
        .unwrap();

    // TD3 用 min(Q1, Q2)，这里用 maximum 测试梯度流向
    let max_id = graph
        .inner_mut()
        .new_maximum_node(q1.node_id(), q2.node_id(), None)
        .unwrap();
    let max_var = graph.wrap_node_id(max_id);

    // max(Q1, Q2) = [2, 3, 3, 4]
    // 用 MSE loss，target 全零
    let target = Tensor::zeros(&[4, 1]);
    let loss = max_var.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // d(mse)/d(max) = 2*(max - 0) / n = 2*max/4 = max/2
    // max = [2, 3, 3, 4], upstream = [1, 1.5, 1.5, 2]
    //
    // 位置 0: Q1=1 < Q2=2, max 来自 Q2 -> Q1 grad=0, Q2 grad=1
    // 位置 1: Q1=3 > Q2=2, max 来自 Q1 -> Q1 grad=1.5, Q2 grad=0
    // 位置 2: Q1=2 < Q2=3, max 来自 Q2 -> Q1 grad=0, Q2 grad=1.5
    // 位置 3: Q1=4 > Q2=3, max 来自 Q1 -> Q1 grad=2, Q2 grad=0

    let q1_grad = q1.grad().unwrap().unwrap();
    let q2_grad = q2.grad().unwrap().unwrap();

    assert_abs_diff_eq!(q1_grad[[0, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(q1_grad[[1, 0]], 1.5, epsilon = 1e-5);
    assert_abs_diff_eq!(q1_grad[[2, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(q1_grad[[3, 0]], 2.0, epsilon = 1e-5);

    assert_abs_diff_eq!(q2_grad[[0, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(q2_grad[[1, 0]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(q2_grad[[2, 0]], 1.5, epsilon = 1e-5);
    assert_abs_diff_eq!(q2_grad[[3, 0]], 0.0, epsilon = 1e-5);
}

/// 测试相等值时梯度各半
#[test]
fn test_maximum_equal_values_gradient_split() {
    let graph = Graph::new();

    let a = graph.parameter(&[2, 1], Init::Zeros, "a").unwrap();
    let b = graph.parameter(&[2, 1], Init::Zeros, "b").unwrap();
    a.set_value(&Tensor::new(&[3.0, 3.0], &[2, 1])).unwrap();
    b.set_value(&Tensor::new(&[3.0, 3.0], &[2, 1])).unwrap(); // 完全相等

    let max_id = graph
        .inner_mut()
        .new_maximum_node(a.node_id(), b.node_id(), None)
        .unwrap();
    let max_var = graph.wrap_node_id(max_id);

    let target = Tensor::new(&[5.0, 5.0], &[2, 1]);
    let loss = max_var.mse_loss(&target).unwrap();

    loss.forward().unwrap();
    loss.backward().unwrap();

    // max = [3, 3], target = [5, 5]
    // d(mse)/d(max) = 2*(3-5) / 2 = -2
    // 由于 a == b，梯度各半：-2 * 0.5 = -1

    let a_grad = a.grad().unwrap().unwrap();
    let b_grad = b.grad().unwrap().unwrap();

    assert_abs_diff_eq!(a_grad[[0, 0]], -1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(a_grad[[1, 0]], -1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(b_grad[[0, 0]], -1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(b_grad[[1, 0]], -1.0, epsilon = 1e-5);
}

// ==================== 错误处理测试 ====================

/// 测试形状不兼容应报错
#[test]
fn test_maximum_incompatible_shapes() {
    let graph = Graph::new();

    // 使用 2D 张量（框架要求 2-4 维）
    let a = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))
        .unwrap();
    let b = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4, 1]))
        .unwrap();

    let result = graph
        .inner_mut()
        .new_maximum_node(a.node_id(), b.node_id(), None);

    assert!(result.is_err());
}

// ==================== 方案 C：新节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_maximum_node_same_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("a"))
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], Some("b"))
        .unwrap();

    let max = inner
        .borrow_mut()
        .create_maximum_node(a.clone(), b.clone(), Some("max"))
        .unwrap();

    assert_eq!(max.shape(), vec![3, 4]);
    assert_eq!(max.name(), Some("max"));
    assert_eq!(max.parents().len(), 2);
}

#[test]
fn test_create_maximum_node_broadcast() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], None)
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], None)
        .unwrap();

    let max = inner
        .borrow_mut()
        .create_maximum_node(a.clone(), b.clone(), None)
        .unwrap();

    // 广播后形状 [3, 4]
    assert_eq!(max.shape(), vec![3, 4]);
}

#[test]
fn test_create_maximum_node_shape_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let a = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4], None)
        .unwrap();
    let b = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 6], None)
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_maximum_node(a, b, None);

    assert!(result.is_err());
}

#[test]
fn test_create_maximum_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_max;
    let weak_a;
    let weak_b;
    {
        let a = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_a = Rc::downgrade(&a);

        let b = inner
            .borrow_mut()
            .create_basic_input_node(&[3, 4], None)
            .unwrap();
        weak_b = Rc::downgrade(&b);

        let max = inner
            .borrow_mut()
            .create_maximum_node(a, b, None)
            .unwrap();
        weak_max = Rc::downgrade(&max);

        assert!(weak_max.upgrade().is_some());
        assert!(weak_a.upgrade().is_some());
        assert!(weak_b.upgrade().is_some());
    }
    assert!(weak_max.upgrade().is_none());
    assert!(weak_a.upgrade().is_none());
    assert!(weak_b.upgrade().is_none());
}
