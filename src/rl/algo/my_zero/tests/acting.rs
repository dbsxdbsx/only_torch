//! 交互一步 π：采样与默认预算。

use super::super::runner::sample_from_probs;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn greedy_picks_argmax() {
    let mut rng = StdRng::seed_from_u64(1);
    assert_eq!(sample_from_probs(&[0.1, 0.7, 0.2], 0.0, &mut rng), 1);
}

#[test]
fn unit_temperature_samples_inside_support() {
    let mut rng = StdRng::seed_from_u64(2);
    let mut seen = [0u32; 2];
    for _ in 0..200 {
        let i = sample_from_probs(&[0.9, 0.1], 1.0, &mut rng);
        seen[i] += 1;
    }
    assert!(seen[0] > seen[1], "高概率动作应更常被抽到: {seen:?}");
    assert!(seen[1] > 0, "低概率动作在 200 次里应至少出现一次");
}
