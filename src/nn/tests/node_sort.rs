/*
 * @Author       : 老董
 * @Description  : SortNode 单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ 1D 升序; 1D 降序; 2D 沿 axis=1; 错误处理
 * 2. VJP 单元测试（底层）→ 逆置换 scatter; 非 unit upstream
 * 3. E2E 反向传播（高层）→ sort + MSE 链; 梯度累积
 * 4. 节点创建 API
 *
 * SortNode 沿轴排序。VJP: 利用缓存的 indices 将上游梯度逆置换回原位置
 *
 * Python 对照脚本: tests/python/calc_jacobi_by_pytorch/node_sort.py
 */

use crate::nn::{
    Graph, GraphError, Init, VarLossOps, VarReduceOps, VarSelectionOps, VarShapeOps,
};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试 ====================

/// 1D 升序排序：[3, 1, 2] → [1, 2, 3]
#[test]
fn test_sort_forward_ascending_1d() {
    let graph = Graph::new();

    let input = Tensor::new(&[3.0, 1.0, 2.0], &[1, 3]);
    let x = graph.input(&input).unwrap();
    let sorted = x.sort_values(1, false).unwrap();

    sorted.forward().unwrap();

    let output = sorted.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 3]);
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 3.0, epsilon = 1e-6);
}

/// 1D 降序排序：[3, 1, 2] → [3, 2, 1]
#[test]
fn test_sort_forward_descending_1d() {
    let graph = Graph::new();

    let input = Tensor::new(&[3.0, 1.0, 2.0], &[1, 3]);
    let x = graph.input(&input).unwrap();
    let sorted = x.sort_values(1, true).unwrap();

    sorted.forward().unwrap();

    let output = sorted.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 3]);
    assert_abs_diff_eq!(output[[0, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 1.0, epsilon = 1e-6);
}

/// 2D 沿 axis=1 排序
///
/// input = [[5, 3, 4],
///          [2, 6, 1]]
/// 升序结果：
/// sorted = [[3, 4, 5],
///           [1, 2, 6]]
#[test]
fn test_sort_forward_2d_axis1() {
    let graph = Graph::new();

    let input = Tensor::new(&[5.0, 3.0, 4.0, 2.0, 6.0, 1.0], &[2, 3]);
    let x = graph.input(&input).unwrap();
    let sorted = x.sort_values(1, false).unwrap();

    sorted.forward().unwrap();

    let output = sorted.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 3]);

    // 第一行：[5, 3, 4] → [3, 4, 5]
    assert_abs_diff_eq!(output[[0, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 5.0, epsilon = 1e-6);

    // 第二行：[2, 6, 1] → [1, 2, 6]
    assert_abs_diff_eq!(output[[1, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 2]], 6.0, epsilon = 1e-6);
}

/// 沿 axis=0 排序
///
/// input = [[5, 3],
///          [2, 6],
///          [4, 1]]
/// 升序结果（按列排）：
/// sorted = [[2, 1],
///           [4, 3],
///           [5, 6]]
#[test]
fn test_sort_forward_2d_axis0() {
    let graph = Graph::new();

    let input = Tensor::new(&[5.0, 3.0, 2.0, 6.0, 4.0, 1.0], &[3, 2]);
    let x = graph.input(&input).unwrap();
    let sorted = x.sort_values(0, false).unwrap();

    sorted.forward().unwrap();

    let output = sorted.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[3, 2]);

    // 第一列 [5, 2, 4] → [2, 4, 5]
    assert_abs_diff_eq!(output[[0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2, 0]], 5.0, epsilon = 1e-6);

    // 第二列 [3, 6, 1] → [1, 3, 6]
    assert_abs_diff_eq!(output[[0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2, 1]], 6.0, epsilon = 1e-6);
}

/// axis 越界错误
#[test]
fn test_sort_error_axis_out_of_bounds() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::zeros(&[2, 3])).unwrap();

    // axis=2 超出 2 维张量的范围
    let result = x.sort_values(2, false);
    assert!(result.is_err());
}

/// SortNode 不能直接设置值
#[test]
fn test_sort_cannot_set_value() {
    let graph = Graph::new();

    let x = graph.input(&Tensor::new(&[3.0, 1.0, 2.0], &[1, 3])).unwrap();
    let sorted = x.sort_values(1, false).unwrap();

    let err = sorted.set_value(&Tensor::zeros(&[1, 3]));
    assert!(err.is_err(), "SortNode 不应支持直接设值");
}

// ==================== 2. VJP 单元测试 ====================

/// VJP 逆置换：升序 [3, 1, 2] 的 indices = [1, 2, 0]
///
/// upstream = [10, 20, 30]（对应排序后的 [1, 2, 3]）
/// 逆置换：grad[indices[0]]=grad[1]=10, grad[indices[1]]=grad[2]=20, grad[indices[2]]=grad[0]=30
/// 即 grad = [30, 10, 20]
#[test]
fn test_sort_vjp_inverse_permutation() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[1, 3], Some("input"))
        .unwrap();
    let sorted = inner
        .borrow_mut()
        .create_sort_node(input.clone(), 1, false, Some("sort"))
        .unwrap();

    // 设置输入 [3, 1, 2]
    input
        .set_value(Some(&Tensor::new(&[3.0, 1.0, 2.0], &[1, 3])))
        .unwrap();
    sorted.forward_recursive(1, false).unwrap();

    // upstream = [10, 20, 30]
    let upstream = Tensor::new(&[10.0, 20.0, 30.0], &[1, 3]);
    let grad = sorted.calc_grad_to_parent_index(0, &upstream)?.resolve(&upstream);

    assert_eq!(grad.shape(), &[1, 3]);

    // 排序后 [1, 2, 3]，indices = [1, 2, 0]
    // grad[0] ← upstream[2] = 30（因为 indices[2]=0，即原位置 0 排到了位置 2）
    // grad[1] ← upstream[0] = 10（因为 indices[0]=1，即原位置 1 排到了位置 0）
    // grad[2] ← upstream[1] = 20（因为 indices[1]=2，即原位置 2 排到了位置 1）
    assert_abs_diff_eq!(grad[[0, 0]], 30.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 10.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 2]], 20.0, epsilon = 1e-6);

    Ok(())
}

/// VJP：2D 张量沿 axis=1 排序，非 unit upstream
///
/// input = [[5, 3, 4],
///          [2, 6, 1]]
/// 升序 sorted = [[3, 4, 5],  indices = [[1, 2, 0],
///                [1, 2, 6]]              [2, 0, 1]]
///
/// upstream = [[1, 2, 3],
///             [4, 5, 6]]
///
/// row 0: grad[indices[0]]=grad[1]=1, grad[indices[1]]=grad[2]=2, grad[indices[2]]=grad[0]=3
///        → [3, 1, 2]
/// row 1: grad[indices[0]]=grad[2]=4, grad[indices[1]]=grad[0]=5, grad[indices[2]]=grad[1]=6
///        → [5, 6, 4]
#[test]
fn test_sort_vjp_2d_non_unit() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_parameter_node(&[2, 3], Some("input"))
        .unwrap();
    let sorted = inner
        .borrow_mut()
        .create_sort_node(input.clone(), 1, false, Some("sort"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(
            &[5.0, 3.0, 4.0, 2.0, 6.0, 1.0],
            &[2, 3],
        )))
        .unwrap();
    sorted.forward_recursive(1, false).unwrap();

    let upstream = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let grad = sorted.calc_grad_to_parent_index(0, &upstream)?.resolve(&upstream);

    assert_eq!(grad.shape(), &[2, 3]);

    // row 0: input [5, 3, 4] → sorted [3, 4, 5], indices [1, 2, 0]
    //   grad[0] ← upstream[2]=3, grad[1] ← upstream[0]=1, grad[2] ← upstream[1]=2
    assert_abs_diff_eq!(grad[[0, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 1]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 2]], 2.0, epsilon = 1e-6);

    // row 1: input [2, 6, 1] → sorted [1, 2, 6], indices [2, 0, 1]
    //   grad[0] ← upstream[1]=5, grad[1] ← upstream[2]=6, grad[2] ← upstream[0]=4
    assert_abs_diff_eq!(grad[[1, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 1]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[1, 2]], 4.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 3. E2E 反向传播测试 ====================

/// sort + sum + MSE 端到端
///
/// input -> sort(axis=1, ascending) -> sum(axis=1) -> MSE(target)
/// 验证梯度回传到 input
#[test]
fn test_sort_e2e_backward() {
    let graph = Graph::new();

    let input = graph.parameter(&[2, 3], Init::Zeros, "input").unwrap();
    input
        .set_value(&Tensor::new(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3]))
        .unwrap();

    let sorted = input.sort_values(1, false).unwrap();
    let summed = sorted.sum();

    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = summed.mse_loss(&target).unwrap();

    graph.zero_grad().unwrap();
    let loss_val = loss.backward().unwrap();
    assert!(loss_val > 0.0);

    // input 梯度应非零
    let input_grad = input.grad().unwrap().unwrap();
    assert_eq!(input_grad.shape(), &[2, 3]);
    assert!(
        input_grad.data_as_slice().iter().any(|&v| v.abs() > 1e-10),
        "input 梯度应非零"
    );
}

/// 梯度累积：两条路径共享 input，梯度应叠加
///
/// input → sort(ascending) → sum → loss1
/// input → sort(descending) → sum → loss2
/// total_loss = loss1 + loss2
///
/// input 梯度 = sort_asc 的梯度 + sort_desc 的梯度
#[test]
fn test_sort_gradient_accumulation() {
    let graph = Graph::new();

    let input = graph.parameter(&[1, 3], Init::Zeros, "input").unwrap();
    input
        .set_value(&Tensor::new(&[3.0, 1.0, 2.0], &[1, 3]))
        .unwrap();

    // 路径 1：升序排序后求和
    let sorted_asc = input.sort_values(1, false).unwrap();
    let sum_asc = sorted_asc.sum();

    // 路径 2：降序排序后求和
    let sorted_desc = input.sort_values(1, true).unwrap();
    let sum_desc = sorted_desc.sum();

    // total = sum_asc + sum_desc
    let total = &sum_asc + &sum_desc;

    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = total.mse_loss(&target).unwrap();

    graph.zero_grad().unwrap();
    let loss_val = loss.backward().unwrap();
    assert!(loss_val > 0.0);

    // 验证梯度累积：input 梯度应非零
    let input_grad = input.grad().unwrap().unwrap();
    assert_eq!(input_grad.shape(), &[1, 3]);

    // 由于升序和降序排序后的 sum 结果相同（元素总和不变），
    // 但梯度逆置换不同，最终的累积梯度每个位置应都有值
    for i in 0..3 {
        assert!(
            input_grad[[0, i]].abs() > 1e-10,
            "位置 {} 的梯度应非零",
            i
        );
    }
}

/// sort + narrow 链式测试
///
/// input -> sort(axis=1, ascending) -> narrow(axis=1, 0, 2) -> MSE
/// 验证只取排序后前 2 个元素时梯度正确回传
#[test]
fn test_sort_e2e_sort_then_narrow() {
    let graph = Graph::new();

    let input = graph.parameter(&[1, 4], Init::Zeros, "input").unwrap();
    input
        .set_value(&Tensor::new(&[4.0, 1.0, 3.0, 2.0], &[1, 4]))
        .unwrap();

    // 升序排序 → [1, 2, 3, 4]，取前 2 个 → [1, 2]
    let sorted = input.sort_values(1, false).unwrap();
    let top2 = sorted.narrow(1, 0, 2).unwrap();

    let target = graph.input(&Tensor::zeros(&[1, 2])).unwrap();
    let loss = top2.mse_loss(&target).unwrap();

    graph.zero_grad().unwrap();
    let loss_val = loss.backward().unwrap();
    assert!(loss_val > 0.0);

    let input_grad = input.grad().unwrap().unwrap();
    assert_eq!(input_grad.shape(), &[1, 4]);

    // 升序排序后 [1, 2, 3, 4]，indices = [1, 3, 2, 0]
    // narrow 取前 2 个元素 → upstream_grad 只有位置 0, 1 有梯度
    // 逆置换后：原位置 1（indices[0]=1）和原位置 3（indices[1]=3）应有梯度
    // 原位置 0 和 2 应无梯度
    assert!(
        input_grad[[0, 0]].abs() < 1e-10,
        "原位置 0 (值=4) 不在前 2 个中，梯度应为 0"
    );
    assert!(
        input_grad[[0, 1]].abs() > 1e-10,
        "原位置 1 (值=1) 在前 2 个中，梯度应非零"
    );
    assert!(
        input_grad[[0, 2]].abs() < 1e-10,
        "原位置 2 (值=3) 不在前 2 个中，梯度应为 0"
    );
    assert!(
        input_grad[[0, 3]].abs() > 1e-10,
        "原位置 3 (值=2) 在前 2 个中，梯度应非零"
    );
}

// ==================== 4. 节点创建 API 测试 ====================

use std::rc::Rc;

#[test]
fn test_create_sort_node_basic() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();

    let sorted = inner
        .borrow_mut()
        .create_sort_node(input.clone(), 1, false, Some("sorted"))
        .unwrap();

    // 输出形状与输入相同
    assert_eq!(sorted.shape(), vec![2, 3]);
    assert_eq!(sorted.name(), Some("sorted"));
    assert!(!sorted.is_leaf());
    assert_eq!(sorted.parents().len(), 1);
}

#[test]
fn test_create_sort_node_descending() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 4, 5], None)
        .unwrap();

    let sorted = inner
        .borrow_mut()
        .create_sort_node(input, 2, true, None)
        .unwrap();

    assert_eq!(sorted.shape(), vec![3, 4, 5]);
}

#[test]
fn test_create_sort_node_axis_out_of_bounds() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();

    // axis=2 超出 2 维张量范围
    let result = inner.borrow_mut().create_sort_node(input, 2, false, None);
    assert!(result.is_err());
}

#[test]
fn test_create_sort_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_sorted;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let sorted = inner
            .borrow_mut()
            .create_sort_node(input, 1, false, None)
            .unwrap();
        weak_sorted = Rc::downgrade(&sorted);

        assert!(weak_sorted.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_sorted.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
