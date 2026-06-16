//! ActionSampler 接缝契约测试
//!
//! 验证离散默认采样器的「行为不变」契约：枚举固定动作集、不产 proposal prior。
//! 连续 / 混合采样器的性质测试随 GumbelPolicy 一起补。

use crate::rl::mcts::{
    ActionCandidates, ActionPayload, ActionSampleContext, ActionSampler, DiscreteActionSampler,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn test_discrete_action_sampler_enumerates_fixed_set() {
    let sampler = DiscreteActionSampler::from_count(3);
    let state: Vec<f32> = vec![0.0; 4];
    let ctx = ActionSampleContext {
        state: &state,
        depth: 0,
        to_play: 0,
        num_candidates: 3,
    };
    let mut rng = StdRng::seed_from_u64(0);
    let ActionCandidates { actions, priors } = sampler.sample(ctx, &mut rng);

    assert_eq!(actions.len(), 3);
    assert_eq!(actions[0], ActionPayload::Discrete(0));
    assert_eq!(actions[1], ActionPayload::Discrete(1));
    assert_eq!(actions[2], ActionPayload::Discrete(2));
    assert!(
        priors.is_none(),
        "离散默认采样器不产 proposal prior（上层均匀处理）"
    );
}

#[test]
fn test_discrete_action_sampler_from_explicit_actions() {
    let actions_in = vec![ActionPayload::Discrete(5), ActionPayload::Discrete(9)];
    let sampler = DiscreteActionSampler::new(actions_in.clone());
    // 泛型 State：采样器对任意 S 可用，这里用单元类型验证不依赖具体 state。
    let state = ();
    let ctx = ActionSampleContext {
        state: &state,
        depth: 2,
        to_play: 1,
        num_candidates: 2,
    };
    let mut rng = StdRng::seed_from_u64(1);
    let out = sampler.sample(ctx, &mut rng);
    assert_eq!(out.actions, actions_in);
}
