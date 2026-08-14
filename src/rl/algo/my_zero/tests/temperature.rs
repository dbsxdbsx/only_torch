//! 温度契约：非评测恒 1，评测恒 0。无 hold/decay 课程表。

use crate::rl::algo::my_zero::runner::{
    EVAL_TEMPERATURE, SELF_PLAY_TEMPERATURE, sample_from_probs,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn modes_are_binary() {
    assert_eq!(SELF_PLAY_TEMPERATURE, 1.0);
    assert_eq!(EVAL_TEMPERATURE, 0.0);
}

#[test]
fn eval_temperature_is_argmax() {
    let mut rng = StdRng::seed_from_u64(1);
    assert_eq!(
        sample_from_probs(&[0.1, 0.7, 0.2], EVAL_TEMPERATURE, &mut rng),
        1
    );
}

#[test]
fn self_play_temperature_follows_distribution() {
    let mut rng = StdRng::seed_from_u64(2);
    let mut seen = [0u32; 2];
    for _ in 0..200 {
        let i = sample_from_probs(&[0.9, 0.1], SELF_PLAY_TEMPERATURE, &mut rng);
        seen[i] += 1;
    }
    assert!(seen[0] > seen[1], "高概率动作应更常被抽到: {seen:?}");
    assert!(seen[1] > 0, "τ=1 应仍能抽到低概率动作");
}
