/*
 * @Author       : 老董
 * @Description  : WhereCond 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试 → all-true / all-false / mixed condition
 * 2. VJP 单元测试 → 梯度分流验证
 * 3. 端到端反向传播测试 → backward 链路
 * 4. 梯度累积测试 → where_cond 接入多路径
 * 5. 节点创建 API + 错误处理（形状不匹配）
 *
 * 梯度公式：
 *   output = where(condition, x, y)
 *   grad_x = condition * upstream_grad
 *   grad_y = (1 - condition) * upstream_grad
 *
 * Python 对照脚本: tests/python/calc_jacobi_by_pytorch/node_where_cond.py
 */

use crate::nn::{Graph, GraphError, Init, VarFilterOps, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试 ====================

/// all-true condition: output = x
#[test]
fn test_where_cond_forward_all_true() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let y = graph
        .input(&Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]))
        .unwrap();
    let cond = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

    let result = crate::nn::Var::where_cond(&cond, &x, &y).unwrap();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(output, expected);
}

/// all-false condition: output = y
#[test]
fn test_where_cond_forward_all_false() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let y = graph
        .input(&Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]))
        .unwrap();
    let cond = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);

    let result = crate::nn::Var::where_cond(&cond, &x, &y).unwrap();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    assert_eq!(output, expected);
}

/// mixed condition: 逐元素选择
#[test]
fn test_where_cond_forward_mixed() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let y = graph
        .input(&Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]))
        .unwrap();
    // condition: [true, false, false, true]
    let cond = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let result = crate::nn::Var::where_cond(&cond, &x, &y).unwrap();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    // [x[0], y[1], y[2], x[3]] = [1, 20, 30, 4]
    let expected = Tensor::new(&[1.0, 20.0, 30.0, 4.0], &[2, 2]);
    assert_eq!(output, expected);
}

/// condition 含非 0/1 值（如 5.0），应归一化为 1
#[test]
fn test_where_cond_forward_nonbinary_condition() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let y = graph
        .input(&Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]))
        .unwrap();
    // condition 含非 0/1 值：5.0 和 -3.0 均视为 true
    let cond = Tensor::new(&[5.0, 0.0, -3.0, 0.0], &[2, 2]);

    let result = crate::nn::Var::where_cond(&cond, &x, &y).unwrap();
    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    let expected = Tensor::new(&[1.0, 20.0, 3.0, 40.0], &[2, 2]);
    assert_eq!(output, expected);
}

// ==================== 2. VJP 单元测试 ====================

/// all-true condition: grad_x = upstream, grad_y = 0
#[test]
fn test_where_cond_vjp_all_true() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let y = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("y"))
        .unwrap();
    let cond = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

    let wc = inner
        .borrow_mut()
        .create_where_cond_node(x.clone(), y.clone(), cond, Some("wc"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    y.set_value(Some(&Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2])))
        .unwrap();
    wc.forward_recursive(1, false).unwrap();

    let upstream = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);

    // grad_x = cond * upstream = [2, 3, 4, 5]
    let grad_x = wc.calc_grad_to_parent_index(0, &upstream)?.resolve(&upstream);
    assert_eq!(&grad_x, &Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]));

    // grad_y = (1-cond) * upstream = [0, 0, 0, 0]
    let grad_y = wc.calc_grad_to_parent_index(1, &upstream)?.resolve(&upstream);
    assert_eq!(&grad_y, &Tensor::zeros(&[2, 2]));

    Ok(())
}

/// all-false condition: grad_x = 0, grad_y = upstream
#[test]
fn test_where_cond_vjp_all_false() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let y = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("y"))
        .unwrap();
    let cond = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);

    let wc = inner
        .borrow_mut()
        .create_where_cond_node(x.clone(), y.clone(), cond, Some("wc"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    y.set_value(Some(&Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2])))
        .unwrap();
    wc.forward_recursive(1, false).unwrap();

    let upstream = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);

    // grad_x = 0 * upstream = [0, 0, 0, 0]
    let grad_x = wc.calc_grad_to_parent_index(0, &upstream)?.resolve(&upstream);
    assert_eq!(&grad_x, &Tensor::zeros(&[2, 2]));

    // grad_y = 1 * upstream = [2, 3, 4, 5]
    let grad_y = wc.calc_grad_to_parent_index(1, &upstream)?.resolve(&upstream);
    assert_eq!(&grad_y, &Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]));

    Ok(())
}

/// mixed condition: 梯度按 mask 分流
#[test]
fn test_where_cond_vjp_mixed() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("x"))
        .unwrap();
    let y = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], Some("y"))
        .unwrap();
    // cond = [1, 0, 0, 1]
    let cond = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let wc = inner
        .borrow_mut()
        .create_where_cond_node(x.clone(), y.clone(), cond, Some("wc"))
        .unwrap();

    x.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    y.set_value(Some(&Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2])))
        .unwrap();
    wc.forward_recursive(1, false).unwrap();

    let upstream = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);

    // grad_x = cond * upstream = [2, 0, 0, 5]
    let grad_x = wc.calc_grad_to_parent_index(0, &upstream)?.resolve(&upstream);
    assert_eq!(&grad_x, &Tensor::new(&[2.0, 0.0, 0.0, 5.0], &[2, 2]));

    // grad_y = (1-cond) * upstream = [0, 3, 4, 0]
    let grad_y = wc.calc_grad_to_parent_index(1, &upstream)?.resolve(&upstream);
    assert_eq!(&grad_y, &Tensor::new(&[0.0, 3.0, 4.0, 0.0], &[2, 2]));

    Ok(())
}

// ==================== 3. 端到端反向传播测试 ====================

/// E2E backward: where_cond → mse_loss → backward
#[test]
fn test_where_cond_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;

    let y = graph.parameter(&[2, 2], Init::Zeros, "y")?;
    y.set_value(&Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]))?;

    // cond = [1, 0, 1, 0]
    let cond = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
    let result = crate::nn::Var::where_cond(&cond, &x, &y)?;

    // result = [1, 20, 3, 40]
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = result.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    // loss = mean([1^2, 20^2, 3^2, 40^2]) = mean([1, 400, 9, 1600]) = 502.5
    assert_abs_diff_eq!(loss_val, 502.5, epsilon = 1e-4);

    // MSE grad to result: 2 * (result - target) / N = 2 * [1, 20, 3, 40] / 4 = [0.5, 10, 1.5, 20]
    // grad_x = cond * mse_grad = [0.5, 0, 1.5, 0]
    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_abs_diff_eq!(
        &x_grad,
        &Tensor::new(&[0.5, 0.0, 1.5, 0.0], &[2, 2]),
        epsilon = 1e-5
    );

    // grad_y = (1-cond) * mse_grad = [0, 10, 0, 20]
    let y_grad = y.grad()?.expect("y 应有 grad");
    assert_abs_diff_eq!(
        &y_grad,
        &Tensor::new(&[0.0, 10.0, 0.0, 20.0], &[2, 2]),
        epsilon = 1e-5
    );

    Ok(())
}

// ==================== 4. 梯度累积测试 ====================

/// x 参与两条路径，梯度应累积
#[test]
fn test_where_cond_grad_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 2], Init::Zeros, "x")?;
    x.set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))?;

    let y = graph.parameter(&[2, 2], Init::Zeros, "y")?;
    y.set_value(&Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]))?;

    let cond = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let wc = crate::nn::Var::where_cond(&cond, &x, &y)?;

    // 路径 1: wc → loss（x 通过 where_cond）
    // 路径 2: x → loss（直接）
    // result = wc + x = 2x（因为 cond 全 true，wc = x）
    let sum = &wc + &x;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = sum.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;

    // sum = [2, 4, 6, 8]
    // MSE grad to sum = 2 * [2, 4, 6, 8] / 4 = [1, 2, 3, 4]
    // x 梯度来自两条路径:
    //   路径 wc → x: cond * [1,2,3,4] = [1,2,3,4]
    //   路径 sum → x: [1,2,3,4]
    //   合计: [2, 4, 6, 8]
    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_abs_diff_eq!(
        &x_grad,
        &Tensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2]),
        epsilon = 1e-5
    );

    Ok(())
}

// ==================== 5. 节点创建 + 错误处理 ====================

#[test]
fn test_create_where_cond_node_basic() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let y = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("y"))
        .unwrap();
    let cond = Tensor::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 3]);

    let wc = inner
        .borrow_mut()
        .create_where_cond_node(x.clone(), y.clone(), cond, Some("wc"))
        .unwrap();

    assert_eq!(wc.shape(), vec![2, 3]);
    assert_eq!(wc.name(), Some("wc"));
    assert!(!wc.is_leaf());
    assert_eq!(wc.parents().len(), 2);
}

/// condition 形状与 x 不匹配 → 报错
#[test]
fn test_where_cond_shape_mismatch_condition() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let y = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("y"))
        .unwrap();
    // condition 形状 [3, 2]，与 x [2, 3] 不匹配
    let cond = Tensor::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[3, 2]);

    let err = inner
        .borrow_mut()
        .create_where_cond_node(x, y, cond, Some("wc"));
    assert!(err.is_err(), "condition 形状不匹配应报错");
}

/// x 和 y 形状不匹配 → 报错
#[test]
fn test_where_cond_shape_mismatch_xy() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let y = inner
        .borrow_mut()
        .create_basic_input_node(&[3, 2], Some("y"))
        .unwrap();
    let cond = Tensor::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 3]);

    let err = inner
        .borrow_mut()
        .create_where_cond_node(x, y, cond, Some("wc"));
    assert!(err.is_err(), "x 和 y 形状不匹配应报错");
}

/// Var::where_cond 高层 API 错误：不同 Graph
#[test]
fn test_where_cond_different_graph() {
    let graph1 = Graph::new();
    let graph2 = Graph::new();

    let x = graph1
        .input(&Tensor::new(&[1.0, 2.0], &[1, 2]))
        .unwrap();
    let y = graph2
        .input(&Tensor::new(&[3.0, 4.0], &[1, 2]))
        .unwrap();
    let cond = Tensor::new(&[1.0, 0.0], &[1, 2]);

    let err = crate::nn::Var::where_cond(&cond, &x, &y);
    assert!(err.is_err(), "不同 Graph 应报错");
}
