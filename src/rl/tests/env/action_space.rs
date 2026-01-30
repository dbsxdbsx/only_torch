//! 动作空间解析测试

use crate::rl::{ActionType, GymEnv};
use pyo3::Python;

/// 测试离散动作空间解析
#[test]
fn test_discrete_action_space() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");

        assert_eq!(env.get_action_type(), ActionType::SingleDiscrete);

        let action_ranges = env.get_all_action_valid_range();
        assert_eq!(action_ranges.len(), 1);
        assert!(action_ranges[0].is_discrete_action());

        // 离散动作数量
        let n = action_ranges[0].get_discrete_action_selectable_num();
        assert_eq!(n, 2);

        env.close();
    });
}

/// 测试单维连续动作空间解析
#[test]
fn test_continuous_action_space_1d() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "Pendulum-v1");

        assert_eq!(env.get_action_type(), ActionType::Continuous);

        let action_ranges = env.get_all_action_valid_range();
        assert_eq!(action_ranges.len(), 1);
        assert!(!action_ranges[0].is_discrete_action());

        let (low, high) = action_ranges[0].get_continuous_action_low_high();
        assert!((low - (-2.0)).abs() < 0.01);
        assert!((high - 2.0).abs() < 0.01);

        env.close();
    });
}

/// 测试多维连续动作空间解析
#[test]
fn test_continuous_action_space_multi_dim() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "BipedalWalker-v3");

        assert_eq!(env.get_action_type(), ActionType::Continuous);
        assert_eq!(env.get_action_num_for_each_step(), 4);

        let action_ranges = env.get_all_action_valid_range();
        assert_eq!(action_ranges.len(), 4);

        for range in &action_ranges {
            assert!(!range.is_discrete_action());
            let (low, high) = range.get_continuous_action_low_high();
            assert!((low - (-1.0)).abs() < 0.01);
            assert!((high - 1.0).abs() < 0.01);
        }

        env.close();
    });
}

/// 测试动作采样
#[test]
fn test_action_sampling() {
    Python::attach(|py| {
        // 离散动作采样
        let env = GymEnv::new(py, "CartPole-v1");
        for _ in 0..10 {
            let action = env.sample_action();
            assert_eq!(action.len(), 1);
            assert!(action[0] == 0.0 || action[0] == 1.0);
        }
        env.close();

        // 连续动作采样
        let env = GymEnv::new(py, "Pendulum-v1");
        for _ in 0..10 {
            let action = env.sample_action();
            assert_eq!(action.len(), 1);
            assert!(action[0] >= -2.0 && action[0] <= 2.0);
        }
        env.close();
    });
}
