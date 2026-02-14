//! Compose 和 RandomApply 测试

use crate::data::transforms::{Compose, RandomApply, Transform};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// 简单的乘法变换（测试用）
struct MultiplyTransform {
    factor: f32,
}

impl Transform for MultiplyTransform {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        tensor * self.factor
    }
}

/// 简单的加法变换（测试用）
struct AddTransform {
    offset: f32,
}

impl Transform for AddTransform {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        tensor + self.offset
    }
}

#[test]
fn test_compose_empty() {
    let compose = Compose::new(vec![]);
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let output = compose.apply(&input);

    assert_eq!(output[[0]], 1.0);
    assert_eq!(output[[1]], 2.0);
    assert_eq!(output[[2]], 3.0);
}

#[test]
fn test_compose_single() {
    let compose = Compose::new(vec![Box::new(MultiplyTransform { factor: 2.0 })]);
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let output = compose.apply(&input);

    assert_abs_diff_eq!(output[[0]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2]], 6.0, epsilon = 1e-6);
}

#[test]
fn test_compose_chain_order() {
    // 先乘 2 再加 1: (x * 2) + 1
    let compose = Compose::new(vec![
        Box::new(MultiplyTransform { factor: 2.0 }),
        Box::new(AddTransform { offset: 1.0 }),
    ]);
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let output = compose.apply(&input);

    // 1*2+1=3, 2*2+1=5, 3*2+1=7
    assert_abs_diff_eq!(output[[0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2]], 7.0, epsilon = 1e-6);
}

#[test]
fn test_compose_chain_reverse_order() {
    // 先加 1 再乘 2: (x + 1) * 2 — 顺序不同结果不同
    let compose = Compose::new(vec![
        Box::new(AddTransform { offset: 1.0 }),
        Box::new(MultiplyTransform { factor: 2.0 }),
    ]);
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let output = compose.apply(&input);

    // (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
    assert_abs_diff_eq!(output[[0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2]], 8.0, epsilon = 1e-6);
}

#[test]
fn test_random_apply_always() {
    // p=1.0 → 总是应用
    let ra = RandomApply::new(Box::new(MultiplyTransform { factor: 0.0 }), 1.0);
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[3]);

    // 运行多次确认总是应用
    for _ in 0..20 {
        let output = ra.apply(&input);
        assert_abs_diff_eq!(output[[0]], 0.0, epsilon = 1e-6);
    }
}

#[test]
fn test_random_apply_never() {
    // p=0.0 → 永不应用
    let ra = RandomApply::new(Box::new(MultiplyTransform { factor: 0.0 }), 0.0);
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[3]);

    // 运行多次确认永不应用
    for _ in 0..20 {
        let output = ra.apply(&input);
        assert_abs_diff_eq!(output[[0]], 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_random_apply_probability() {
    // p=0.5 → 统计检验
    let ra = RandomApply::new(Box::new(MultiplyTransform { factor: 0.0 }), 0.5);
    let input = Tensor::new(&[1.0], &[1]);

    let trials = 1000;
    let mut applied_count = 0;
    for _ in 0..trials {
        let output = ra.apply(&input);
        if output[[0]] == 0.0 {
            applied_count += 1;
        }
    }

    // 期望约 500 次应用，允许宽松的统计偏差
    let ratio = applied_count as f64 / trials as f64;
    assert!(
        (0.35..=0.65).contains(&ratio),
        "RandomApply(p=0.5) 实际应用比例 {ratio:.2} 偏离期望过大"
    );
}

#[test]
#[should_panic(expected = "概率 p 必须在 [0, 1] 范围内")]
fn test_random_apply_invalid_p() {
    RandomApply::new(Box::new(MultiplyTransform { factor: 1.0 }), 1.5);
}
