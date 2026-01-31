/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Tensor softmax 单元测试
 */

use crate::tensor::Tensor;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓softmax 基本功能↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

#[test]
fn test_softmax_basic_1d() {
    // 1D 向量
    let x = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let probs = x.softmax(0);

    // 检查形状
    assert_eq!(probs.shape(), &[3]);

    // 检查和为 1
    let sum = probs[[0]] + probs[[1]] + probs[[2]];
    assert!((sum - 1.0).abs() < 1e-6, "softmax 和应为 1，实际为 {}", sum);

    // softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
    assert!((probs[[0]] - 0.0900).abs() < 0.001);
    assert!((probs[[1]] - 0.2447).abs() < 0.001);
    assert!((probs[[2]] - 0.6652).abs() < 0.001);
}

#[test]
fn test_softmax_2d_axis1() {
    // 2D 张量，沿 axis=1（最后一维）计算
    let x = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    let probs = x.softmax(1);

    // 检查形状
    assert_eq!(probs.shape(), &[2, 3]);

    // 每行和为 1
    let row0_sum = probs[[0, 0]] + probs[[0, 1]] + probs[[0, 2]];
    let row1_sum = probs[[1, 0]] + probs[[1, 1]] + probs[[1, 2]];
    assert!((row0_sum - 1.0).abs() < 1e-6, "第 0 行和应为 1");
    assert!((row1_sum - 1.0).abs() < 1e-6, "第 1 行和应为 1");

    // 两行应该相同（输入相同）
    assert!((probs[[0, 0]] - probs[[1, 0]]).abs() < 1e-6);
    assert!((probs[[0, 1]] - probs[[1, 1]]).abs() < 1e-6);
    assert!((probs[[0, 2]] - probs[[1, 2]]).abs() < 1e-6);
}

#[test]
fn test_softmax_2d_axis0() {
    // 2D 张量，沿 axis=0 计算
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let probs = x.softmax(0);

    // 检查形状
    assert_eq!(probs.shape(), &[2, 3]);

    // 每列和为 1
    let col0_sum = probs[[0, 0]] + probs[[1, 0]];
    let col1_sum = probs[[0, 1]] + probs[[1, 1]];
    let col2_sum = probs[[0, 2]] + probs[[1, 2]];
    assert!((col0_sum - 1.0).abs() < 1e-6, "第 0 列和应为 1");
    assert!((col1_sum - 1.0).abs() < 1e-6, "第 1 列和应为 1");
    assert!((col2_sum - 1.0).abs() < 1e-6, "第 2 列和应为 1");
}

#[test]
fn test_softmax_last_dim() {
    // softmax_last_dim 等价于 softmax(dim-1)
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let probs1 = x.softmax(1);
    let probs2 = x.softmax_last_dim();

    // 应该完全相同
    for i in 0..2 {
        for j in 0..3 {
            assert!((probs1[[i, j]] - probs2[[i, j]]).abs() < 1e-6);
        }
    }
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑softmax 基本功能↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓数值稳定性测试↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

#[test]
fn test_softmax_numerical_stability_large_values() {
    // 大数值不应该溢出（因为先减去 max）
    let x = Tensor::new(&[1000.0, 1001.0, 1002.0], &[3]);
    let probs = x.softmax(0);

    // 检查和为 1（如果溢出会得到 NaN 或 Inf）
    let sum = probs[[0]] + probs[[1]] + probs[[2]];
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "大数值 softmax 和应为 1，实际为 {}",
        sum
    );
    assert!(!probs[[0]].is_nan(), "结果不应为 NaN");
    assert!(!probs[[0]].is_infinite(), "结果不应为 Inf");

    // 相对关系应该保持：softmax([1000,1001,1002]) ≈ softmax([0,1,2])
    let x2 = Tensor::new(&[0.0, 1.0, 2.0], &[3]);
    let probs2 = x2.softmax(0);
    assert!((probs[[0]] - probs2[[0]]).abs() < 1e-5);
    assert!((probs[[1]] - probs2[[1]]).abs() < 1e-5);
    assert!((probs[[2]] - probs2[[2]]).abs() < 1e-5);
}

#[test]
fn test_softmax_numerical_stability_negative_large() {
    // 大负数也不应该有问题
    let x = Tensor::new(&[-1000.0, -999.0, -998.0], &[3]);
    let probs = x.softmax(0);

    let sum = probs[[0]] + probs[[1]] + probs[[2]];
    assert!((sum - 1.0).abs() < 1e-6, "大负数 softmax 和应为 1");
    assert!(!probs[[0]].is_nan());
}

#[test]
fn test_softmax_uniform_input() {
    // 所有元素相同时，输出应该均匀分布
    let x = Tensor::new(&[5.0, 5.0, 5.0, 5.0], &[4]);
    let probs = x.softmax(0);

    // 每个元素应该是 0.25
    for i in 0..4 {
        assert!((probs[[i]] - 0.25).abs() < 1e-6);
    }
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑数值稳定性测试↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓边界情况↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

#[test]
fn test_softmax_single_element() {
    // 单元素，softmax 应该是 1.0
    let x = Tensor::new(&[42.0], &[1]);
    let probs = x.softmax(0);
    assert!((probs[[0]] - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_3d_tensor() {
    // 3D 张量
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let probs = x.softmax(2); // 沿最后一维

    // 检查形状
    assert_eq!(probs.shape(), &[2, 2, 2]);

    // 每个 [i, j, :] 切片和为 1
    for i in 0..2 {
        for j in 0..2 {
            let sum = probs[[i, j, 0]] + probs[[i, j, 1]];
            assert!((sum - 1.0).abs() < 1e-6, "切片 [{}, {}, :] 和应为 1", i, j);
        }
    }
}

#[test]
#[should_panic(expected = "softmax: axis")]
fn test_softmax_invalid_axis() {
    let x = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let _ = x.softmax(1); // axis=1 超出范围
}

#[test]
#[should_panic(expected = "softmax_last_dim: 张量维度必须大于 0")]
fn test_softmax_last_dim_scalar() {
    let x = Tensor::new(&[1.0], &[]);
    let _ = x.softmax_last_dim(); // 标量没有 last_dim
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑边界情况↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓RL 场景测试↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

#[test]
fn test_softmax_rl_action_probs() {
    // RL 场景：将 Q 值转换为动作概率
    // Q 值: [batch=2, actions=3]
    let q_values = Tensor::new(&[1.0, 3.0, 2.0, 0.5, 0.5, 1.0], &[2, 3]);

    // 温度参数 alpha
    let alpha = 1.0;
    let q_scaled = &q_values / alpha;
    let action_probs = q_scaled.softmax(1);

    // 检查形状
    assert_eq!(action_probs.shape(), &[2, 3]);

    // 每行和为 1
    let row0_sum = action_probs[[0, 0]] + action_probs[[0, 1]] + action_probs[[0, 2]];
    let row1_sum = action_probs[[1, 0]] + action_probs[[1, 1]] + action_probs[[1, 2]];
    assert!((row0_sum - 1.0).abs() < 1e-6);
    assert!((row1_sum - 1.0).abs() < 1e-6);

    // 第一行：Q=[1,3,2]，最大的是 Q[1]=3，所以 action_probs[0,1] 应该最大
    assert!(action_probs[[0, 1]] > action_probs[[0, 0]]);
    assert!(action_probs[[0, 1]] > action_probs[[0, 2]]);
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑RL 场景测试↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
