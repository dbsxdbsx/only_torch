/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Tensor log_softmax 单元测试
 */

use crate::tensor::Tensor;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓log_softmax 基本功能↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

#[test]
fn test_log_softmax_basic_1d() {
    // 1D 向量
    let x = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let log_probs = x.log_softmax(0);

    // 检查形状
    assert_eq!(log_probs.shape(), &[3]);

    // log_softmax 输出应该都是负数（因为 softmax < 1）
    assert!(log_probs[[0]] < 0.0);
    assert!(log_probs[[1]] < 0.0);
    assert!(log_probs[[2]] < 0.0);

    // exp(log_softmax) 应该等于 softmax
    let probs = log_probs.exp();
    let sum = probs[[0]] + probs[[1]] + probs[[2]];
    assert!((sum - 1.0).abs() < 1e-6, "exp(log_softmax) 和应为 1");

    // 对比直接计算的 softmax
    let direct_probs = x.softmax(0);
    assert!((probs[[0]] - direct_probs[[0]]).abs() < 1e-6);
    assert!((probs[[1]] - direct_probs[[1]]).abs() < 1e-6);
    assert!((probs[[2]] - direct_probs[[2]]).abs() < 1e-6);
}

#[test]
fn test_log_softmax_2d_axis1() {
    // 2D 张量，沿 axis=1 计算
    let x = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    let log_probs = x.log_softmax(1);

    // 检查形状
    assert_eq!(log_probs.shape(), &[2, 3]);

    // 每行 exp(log_probs) 和为 1
    for i in 0..2 {
        let row_exp_sum =
            log_probs[[i, 0]].exp() + log_probs[[i, 1]].exp() + log_probs[[i, 2]].exp();
        assert!((row_exp_sum - 1.0).abs() < 1e-6, "第 {} 行 exp 和应为 1", i);
    }

    // 两行应该相同（输入相同）
    assert!((log_probs[[0, 0]] - log_probs[[1, 0]]).abs() < 1e-6);
    assert!((log_probs[[0, 1]] - log_probs[[1, 1]]).abs() < 1e-6);
    assert!((log_probs[[0, 2]] - log_probs[[1, 2]]).abs() < 1e-6);
}

#[test]
fn test_log_softmax_last_dim() {
    // log_softmax_last_dim 等价于 log_softmax(dim-1)
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let log_probs1 = x.log_softmax(1);
    let log_probs2 = x.log_softmax_last_dim();

    // 应该完全相同
    for i in 0..2 {
        for j in 0..3 {
            assert!((log_probs1[[i, j]] - log_probs2[[i, j]]).abs() < 1e-6);
        }
    }
}

#[test]
fn test_log_softmax_vs_softmax_ln() {
    // 验证 log_softmax 等价于 softmax().ln()
    let x = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);

    let log_probs_direct = x.log_softmax(1);
    let log_probs_indirect = x.softmax(1).ln();

    // 应该几乎相同（数值误差在可接受范围内）
    for j in 0..3 {
        assert!(
            (log_probs_direct[[0, j]] - log_probs_indirect[[0, j]]).abs() < 1e-5,
            "log_softmax 与 softmax().ln() 应相等"
        );
    }
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑log_softmax 基本功能↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓数值稳定性测试↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

#[test]
fn test_log_softmax_numerical_stability_large_values() {
    // 大数值不应该溢出
    let x = Tensor::new(&[1000.0, 1001.0, 1002.0], &[3]);
    let log_probs = x.log_softmax(0);

    // 检查输出是有限的（不是 NaN 或 Inf）
    assert!(log_probs[[0]].is_finite(), "log_softmax 输出应为有限值");
    assert!(log_probs[[1]].is_finite(), "log_softmax 输出应为有限值");
    assert!(log_probs[[2]].is_finite(), "log_softmax 输出应为有限值");

    // exp 后和为 1
    let sum = log_probs[[0]].exp() + log_probs[[1]].exp() + log_probs[[2]].exp();
    assert!((sum - 1.0).abs() < 1e-5, "大数值 exp(log_softmax) 和应为 1");
}

#[test]
fn test_log_softmax_numerical_stability_small_probs() {
    // 当某些概率非常小时，log_softmax 应该能正确处理
    // 而 softmax().ln() 可能会因为小概率值产生精度问题
    let x = Tensor::new(&[0.0, 0.0, 100.0], &[3]); // 第三个元素概率接近 1

    let log_probs = x.log_softmax(0);

    // log_softmax 输出应为有限值
    assert!(log_probs[[0]].is_finite());
    assert!(log_probs[[1]].is_finite());
    assert!(log_probs[[2]].is_finite());

    // 第三个概率接近 1，log 接近 0
    assert!(log_probs[[2]] > -1.0); // log(~1) ≈ 0

    // 前两个概率很小，log 是大负数
    assert!(log_probs[[0]] < -90.0);
    assert!(log_probs[[1]] < -90.0);
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑数值稳定性测试↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓边界情况↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

#[test]
fn test_log_softmax_3d() {
    // 3D 张量
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let log_probs = x.log_softmax(2);

    // 检查形状
    assert_eq!(log_probs.shape(), &[2, 2, 2]);

    // 每个 [i, j, :] 切片 exp 和为 1
    for i in 0..2 {
        for j in 0..2 {
            let sum = log_probs[[i, j, 0]].exp() + log_probs[[i, j, 1]].exp();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "切片 [{}, {}, :] exp 和应为 1",
                i,
                j
            );
        }
    }
}

#[test]
#[should_panic(expected = "log_softmax: axis")]
fn test_log_softmax_invalid_axis() {
    let x = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let _ = x.log_softmax(1); // axis=1 超出范围
}

#[test]
#[should_panic(expected = "log_softmax_last_dim: 张量维度必须大于 0")]
fn test_log_softmax_last_dim_scalar() {
    let x = Tensor::new(&[1.0], &[]);
    let _ = x.log_softmax_last_dim(); // 标量没有 last_dim
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑边界情况↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓RL 场景测试↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

#[test]
fn test_log_softmax_rl_log_probs() {
    // RL 场景：计算动作的 log 概率（用于 SAC Actor Loss）
    // logits: [batch=2, actions=3]
    let logits = Tensor::new(&[1.0, 3.0, 2.0, 0.5, 0.5, 1.0], &[2, 3]);

    let log_probs = logits.log_softmax(1);

    // 检查形状
    assert_eq!(log_probs.shape(), &[2, 3]);

    // 所有 log_probs 应为负数
    for i in 0..2 {
        for j in 0..3 {
            assert!(log_probs[[i, j]] < 0.0, "log_probs 应为负数");
        }
    }

    // 第一行：logits=[1,3,2]，最大的是 logits[1]=3
    // 所以 log_probs[0,1] 应该最大（最接近 0）
    assert!(log_probs[[0, 1]] > log_probs[[0, 0]]);
    assert!(log_probs[[0, 1]] > log_probs[[0, 2]]);
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑RL 场景测试↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
