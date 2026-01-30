//! GymEnv 环境覆盖测试
//!
//! 通过代表性环境验证 `GymEnv` 封装的核心功能：
//! - 离散动作环境（CartPole）
//! - 单维连续动作环境（Pendulum）
//! - 多维连续动作环境（BipedalWalker）
//! - 高维连续动作环境（MuJoCo Ant）
//! - 图像观察环境（Atari Breakout）
//!
//! 对应 Python 测试：
//! - `tests/python/gym/test_01_basic_discrete.py`
//! - `tests/python/gym/test_02_basic_continuous.py`
//! - `tests/python/gym/test_03_box2d.py`
//! - `tests/python/gym/test_04_mujoco.py`
//! - `tests/python/gym/test_05_atari.py`

use crate::rl::{ActionType, GymEnv, ObsType};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

// ============================================================================
// 离散动作环境
// ============================================================================

/// 测试离散动作环境（代表：CartPole-v1）
///
/// 验证点：
/// - 动作类型：SingleDiscrete
/// - 观察空间：4 维向量
/// - 动作空间：2 个离散动作
#[test]
fn test_discrete_env() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");

        // 验证动作类型
        assert_eq!(env.get_action_type(), ActionType::SingleDiscrete);
        assert_eq!(env.get_obs_type(), ObsType::Vector);

        // 验证观察空间
        let obs_prop = env.get_obs_prop();
        assert_eq!(obs_prop.len(), 1);
        assert_eq!(obs_prop[0].shape_vec, vec![4]);

        // 验证动作空间
        let action_ranges = env.get_all_action_valid_range();
        assert_eq!(action_ranges.len(), 1);
        assert!(action_ranges[0].is_discrete_action());
        assert_eq!(action_ranges[0].get_discrete_action_selectable_num(), 2);

        // reset + step 验证
        let obs = env.reset(Some(42));
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].len(), 4);

        let action = vec![0.0];
        let (next_obs, reward, _done) = env.step(&action);
        assert_eq!(next_obs[0].len(), 4);
        assert!(reward > 0.0); // CartPole 每步奖励 1.0

        // 采样验证
        let sampled = env.sample_action();
        assert_eq!(sampled.len(), 1);
        assert!(sampled[0] == 0.0 || sampled[0] == 1.0);

        env.close();
    });
}

// ============================================================================
// 连续动作环境
// ============================================================================

/// 测试单维连续动作环境（代表：Pendulum-v1）
///
/// 验证点：
/// - 动作类型：Continuous
/// - 观察空间：3 维向量
/// - 动作空间：1 维连续 [-2.0, 2.0]
#[test]
fn test_continuous_1d_env() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "Pendulum-v1");

        // 验证动作类型
        assert_eq!(env.get_action_type(), ActionType::Continuous);
        assert_eq!(env.get_obs_type(), ObsType::Vector);

        // 验证观察空间
        let obs_prop = env.get_obs_prop();
        assert_eq!(obs_prop[0].shape_vec, vec![3]);

        // 验证动作空间
        let action_ranges = env.get_all_action_valid_range();
        assert_eq!(action_ranges.len(), 1);
        assert!(!action_ranges[0].is_discrete_action());
        let (low, high) = action_ranges[0].get_continuous_action_low_high();
        assert!((low - (-2.0)).abs() < 0.01);
        assert!((high - 2.0).abs() < 0.01);

        // reset + step 验证
        let obs = env.reset(Some(42));
        assert_eq!(obs[0].len(), 3);

        let action = vec![0.5];
        let (next_obs, reward, _done) = env.step(&action);
        assert_eq!(next_obs[0].len(), 3);
        assert!(reward <= 0.0); // Pendulum 奖励通常为负

        // 采样验证
        let sampled = env.sample_action();
        assert_eq!(sampled.len(), 1);
        assert!(sampled[0] >= -2.0 && sampled[0] <= 2.0);

        env.close();
    });
}

/// 测试多维连续动作环境（代表：BipedalWalker-v3）
///
/// 验证点：
/// - 动作类型：Continuous
/// - 观察空间：24 维向量
/// - 动作空间：4 维连续 [-1.0, 1.0]
#[test]
fn test_continuous_multi_dim_env() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "BipedalWalker-v3");

        // 验证动作类型
        assert_eq!(env.get_action_type(), ActionType::Continuous);

        // 验证观察空间
        let obs_prop = env.get_obs_prop();
        assert_eq!(obs_prop[0].shape_vec, vec![24]);

        // 验证动作空间：4 维连续
        let action_ranges = env.get_all_action_valid_range();
        assert_eq!(action_ranges.len(), 4);
        assert_eq!(env.get_action_num_for_each_step(), 4);

        for range in &action_ranges {
            assert!(!range.is_discrete_action());
            let (low, high) = range.get_continuous_action_low_high();
            assert!((low - (-1.0)).abs() < 0.01);
            assert!((high - 1.0).abs() < 0.01);
        }

        // reset + step 验证
        let obs = env.reset(Some(42));
        assert_eq!(obs[0].len(), 24);

        let action = vec![0.0, 0.0, 0.0, 0.0];
        let (next_obs, _reward, _done) = env.step(&action);
        assert_eq!(next_obs[0].len(), 24);

        env.close();
    });
}

/// 测试高维连续动作环境（代表：Ant-v5，MuJoCo）
///
/// 验证点：
/// - 动作类型：Continuous
/// - 观察空间：高维向量（> 20 维）
/// - 动作空间：8 维连续
#[test]
fn test_high_dim_continuous_env() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "Ant-v5");

        // 验证动作类型
        assert_eq!(env.get_action_type(), ActionType::Continuous);

        // 验证观察空间：高维
        let obs_prop = env.get_obs_prop();
        assert!(obs_prop[0].shape_vec[0] > 20);

        // 验证动作空间：8 维连续
        let action_ranges = env.get_all_action_valid_range();
        assert_eq!(action_ranges.len(), 8);

        // reset + step 验证
        let obs = env.reset(Some(42));
        assert!(!obs[0].is_empty());

        let action = env.sample_action();
        assert_eq!(action.len(), 8);

        let (next_obs, _reward, _done) = env.step(&action);
        assert!(!next_obs[0].is_empty());

        env.close();
    });
}

// ============================================================================
// 图像观察环境
// ============================================================================

/// 注册 Atari 环境（需要先注册 ale_py）
fn register_atari_envs(py: Python<'_>) {
    let ale_py = py.import("ale_py").expect("import ale_py 失败");
    let gymnasium = py.import("gymnasium").expect("import gymnasium 失败");
    gymnasium
        .call_method1("register_envs", (ale_py,))
        .expect("注册 Atari 环境失败");
}

/// 测试图像观察环境（代表：ALE/Breakout-v5，Atari）
///
/// 验证点：
/// - 动作类型：SingleDiscrete
/// - 观察类型：ChannelLast（HWC 格式）
/// - 观察空间：(210, 160, 3) 图像
/// - 动作空间：4 个离散动作
#[test]
fn test_image_obs_env() {
    Python::attach(|py| {
        register_atari_envs(py);

        let env = GymEnv::new(py, "ALE/Breakout-v5");

        // 验证动作类型
        assert_eq!(env.get_action_type(), ActionType::SingleDiscrete);

        // 验证观察类型：图像，通道在后（HWC）
        assert_eq!(env.get_obs_type(), ObsType::ChannelLast);

        // 验证观察空间：(210, 160, 3)
        let obs_prop = env.get_obs_prop();
        assert_eq!(obs_prop.len(), 1);
        assert_eq!(obs_prop[0].shape_vec, vec![210, 160, 3]);

        // 验证动作空间：4 个离散动作
        let action_ranges = env.get_all_action_valid_range();
        assert_eq!(action_ranges.len(), 1);
        assert!(action_ranges[0].is_discrete_action());
        assert_eq!(action_ranges[0].get_discrete_action_selectable_num(), 4);

        // 验证扁平化长度
        assert_eq!(env.get_flatten_observation_len(), 210 * 160 * 3);

        // reset + step 验证
        let obs = env.reset(Some(42));
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].len(), 210 * 160 * 3);

        let action = vec![0.0]; // NOOP
        let (next_obs, _reward, _done) = env.step(&action);
        assert_eq!(next_obs[0].len(), 210 * 160 * 3);

        // 采样验证
        let sampled = env.sample_action();
        assert_eq!(sampled.len(), 1);
        assert!(sampled[0] >= 0.0 && sampled[0] < 4.0);

        env.close();
    });
}

// ============================================================================
// 多步执行测试
// ============================================================================

/// 测试多步执行循环
///
/// 验证 GymEnv 在多步交互中的稳定性
#[test]
fn test_multiple_steps() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");
        let _obs = env.reset(Some(42));
        let mut total_reward = 0.0;

        for _ in 0..10 {
            let action = env.sample_action();
            let (_next_obs, reward, done) = env.step(&action);
            total_reward += reward;

            if done {
                break;
            }
        }

        assert!(total_reward > 0.0);
        env.close();
    });
}
