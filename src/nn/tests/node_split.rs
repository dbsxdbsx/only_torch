/*
 * @Author       : 老董
 * @Description  : Split 便捷方法单元测试
 *
 * Split 是 Mode B（Var 便捷方法），内部创建 N 个 Narrow 节点实现沿轴拆分。
 * API: var.split(axis, sizes) -> Result<Vec<Var>, GraphError>
 *
 * 测试策略：
 * 1. 前向传播测试 → 等分 / 不等分 / axis=0 / 值验证
 * 2. split-concat 互逆测试 → split 后 concat 应恢复原始值
 * 3. 端到端反向传播 → split 取单段 + MSE loss，验证梯度稀疏性
 * 4. 多段梯度累积 → 多段同时参与 loss，验证梯度正确汇聚
 * 5. 错误处理 → sizes 之和不匹配 / axis 越界
 */

use crate::nn::{Graph, GraphError, Init, Var, VarLossOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试 ====================

/// 等分: [2, 6] split(axis=1, sizes=[2,2,2]) → 3 段各 [2, 2]
///
/// input:
///   [[1, 2, 3, 4, 5, 6],
///    [7, 8, 9, 10, 11, 12]]
/// split(1, [2, 2, 2]):
///   part0 = [[1, 2], [7, 8]]       shape [2, 2]
///   part1 = [[3, 4], [9, 10]]      shape [2, 2]
///   part2 = [[5, 6], [11, 12]]     shape [2, 2]
#[test]
fn test_split_equal() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[2, 6])).unwrap();
    let parts = x.split(1, &[2, 2, 2]).unwrap();

    assert_eq!(parts.len(), 3);

    for p in &parts {
        p.forward().unwrap();
    }

    for p in &parts {
        assert_eq!(p.value().unwrap().unwrap().shape(), &[2, 2]);
    }
}

/// 不等分: [2, 5] split(axis=1, sizes=[1,3,1]) → [2,1], [2,3], [2,1]
///
/// input:
///   [[1, 2, 3, 4, 5],
///    [6, 7, 8, 9, 10]]
/// split(1, [1, 3, 1]):
///   part0 = [[1], [6]]             shape [2, 1]
///   part1 = [[2, 3, 4], [7, 8, 9]] shape [2, 3]
///   part2 = [[5], [10]]            shape [2, 1]
#[test]
fn test_split_unequal() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=10).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[2, 5])).unwrap();
    let parts = x.split(1, &[1, 3, 1]).unwrap();

    assert_eq!(parts.len(), 3);

    for p in &parts {
        p.forward().unwrap();
    }

    assert_eq!(parts[0].value().unwrap().unwrap().shape(), &[2, 1]);
    assert_eq!(parts[1].value().unwrap().unwrap().shape(), &[2, 3]);
    assert_eq!(parts[2].value().unwrap().unwrap().shape(), &[2, 1]);
}

/// axis=0 拆分: [4, 3] split(axis=0, sizes=[1,3]) → [1, 3] 和 [3, 3]
///
/// input:
///   [[1, 2, 3],
///    [4, 5, 6],
///    [7, 8, 9],
///    [10, 11, 12]]
/// split(0, [1, 3]):
///   part0 = [[1, 2, 3]]                          shape [1, 3]
///   part1 = [[4, 5, 6], [7, 8, 9], [10, 11, 12]] shape [3, 3]
#[test]
fn test_split_axis0() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[4, 3])).unwrap();
    let parts = x.split(0, &[1, 3]).unwrap();

    assert_eq!(parts.len(), 2);

    for p in &parts {
        p.forward().unwrap();
    }

    assert_eq!(parts[0].value().unwrap().unwrap().shape(), &[1, 3]);
    assert_eq!(parts[1].value().unwrap().unwrap().shape(), &[3, 3]);

    // part0: [[1, 2, 3]]
    let out0 = parts[0].value().unwrap().unwrap();
    assert_abs_diff_eq!(out0[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(out0[[0, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(out0[[0, 2]], 3.0, epsilon = 1e-6);

    // part1 首行和末行
    let out1 = parts[1].value().unwrap().unwrap();
    assert_abs_diff_eq!(out1[[0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(out1[[2, 2]], 12.0, epsilon = 1e-6);
}

/// 值验证: [2, 3] data=[1,2,3,4,5,6] split(axis=1, sizes=[1,2])
///
/// input:
///   [[1, 2, 3],
///    [4, 5, 6]]
/// split(1, [1, 2]):
///   part0 = [[1], [4]]       逐元素验证
///   part1 = [[2, 3], [5, 6]] 逐元素验证
#[test]
fn test_split_values() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[2, 3])).unwrap();
    let parts = x.split(1, &[1, 2]).unwrap();

    for p in &parts {
        p.forward().unwrap();
    }

    // part0: [[1], [4]]
    let out0 = parts[0].value().unwrap().unwrap();
    assert_eq!(out0.shape(), &[2, 1]);
    assert_abs_diff_eq!(out0[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(out0[[1, 0]], 4.0, epsilon = 1e-6);

    // part1: [[2, 3], [5, 6]]
    let out1 = parts[1].value().unwrap().unwrap();
    assert_eq!(out1.shape(), &[2, 2]);
    assert_abs_diff_eq!(out1[[0, 0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(out1[[0, 1]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(out1[[1, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(out1[[1, 1]], 6.0, epsilon = 1e-6);
}

// ==================== 2. split-concat 互逆测试 ====================

/// split 后 concat 应恢复原始数据
///
/// [2, 6] split(1, [2, 4]) → 两段 → Var::concat → [2, 6]
/// 逐元素对比应完全一致
#[test]
fn test_split_concat_inverse() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let input_tensor = Tensor::new(&data, &[2, 6]);
    let x = graph.input(&input_tensor).unwrap();

    let parts = x.split(1, &[2, 4]).unwrap();

    // concat 回来
    let refs: Vec<&Var> = parts.iter().collect();
    let restored = Var::concat(&refs, 1).unwrap();

    restored.forward().unwrap();

    let output = restored.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 6]);

    // 数据应完全一致
    for i in 0..2 {
        for j in 0..6 {
            assert_abs_diff_eq!(output[[i, j]], input_tensor[[i, j]], epsilon = 1e-6);
        }
    }
}

// ==================== 3. 端到端反向传播测试 ====================

/// split → 仅取其中一段 → MSE loss → backward
/// 梯度应仅流向被选中的段，其余位置梯度为零（稀疏性）
///
/// x [2, 6] → split(1, [2, 4]) → 仅用 part0 [2, 2] → MSE(target=0)
/// part0 对应列 0..2 → 梯度非零
/// part1 对应列 2..6 → 梯度为零（未参与 loss）
#[test]
fn test_split_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 6], Init::Zeros, "x")?;
    let data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.1).collect();
    x.set_value(&Tensor::new(&data, &[2, 6]))?;

    let parts = x.split(1, &[2, 4])?;

    // 仅用 part0 计算 loss
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = parts[0].mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    assert!(loss_val > 0.0, "loss 应为正");
    assert!(loss_val.is_finite());

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 6]);

    // 列 0..2（part0 范围）应有非零梯度
    let mut has_nonzero = false;
    for i in 0..2 {
        for j in 0..2 {
            if x_grad[[i, j]].abs() > 1e-10 {
                has_nonzero = true;
            }
        }
    }
    assert!(has_nonzero, "part0 范围内应有非零梯度");

    // 列 2..6（part1 范围，未参与 loss）梯度应为零
    for i in 0..2 {
        for j in 2..6 {
            assert_abs_diff_eq!(x_grad[[i, j]], 0.0, epsilon = 1e-6);
        }
    }

    Ok(())
}

// ==================== 4. 多段梯度累积测试 ====================

/// 多段同时参与 loss，梯度应正确汇聚到所有位置
///
/// x [2, 6] → split(1, [2, 4]) → part0 MSE + part1 MSE → backward
/// 两段 loss 各覆盖一部分列，因此所有位置应有非零梯度
#[test]
fn test_split_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[2, 6], Init::Zeros, "x")?;
    let data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.1).collect();
    x.set_value(&Tensor::new(&data, &[2, 6]))?;

    let parts = x.split(1, &[2, 4])?;

    let target0 = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss0 = parts[0].mse_loss(&target0)?;

    let target1 = graph.input(&Tensor::zeros(&[2, 4]))?;
    let loss1 = parts[1].mse_loss(&target1)?;

    let total_loss = &loss0 + &loss1;

    graph.zero_grad()?;
    let loss_val = total_loss.backward()?;

    assert!(loss_val > 0.0, "loss 应为正");

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[2, 6]);

    // 所有位置应有非零梯度（两段 loss 各覆盖一部分列）
    assert!(
        x_grad.data_as_slice().iter().all(|&v| v.abs() > 1e-10),
        "所有位置应有非零梯度"
    );

    Ok(())
}

// ==================== 5. 错误处理测试 ====================

/// sizes 之和不等于轴大小应报错
///
/// [2, 6] split(1, [2, 3]) → 2+3=5 ≠ 6
#[test]
fn test_split_sizes_mismatch() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[2, 6])).unwrap();

    let result = x.split(1, &[2, 3]);
    assert!(result.is_err(), "sizes 之和不等于轴大小应报错");
}

/// axis 越界应报错
///
/// [2, 2] split(axis=2, ...) → axis=2 超出 2 维张量的范围
#[test]
fn test_split_axis_out_of_bounds() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();

    let result = x.split(2, &[1, 1]);
    assert!(result.is_err(), "axis 越界应报错");
}
