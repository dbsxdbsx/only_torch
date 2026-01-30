//! 观察空间解析测试

use crate::rl::{GymEnv, ObsType};
use pyo3::Python;

/// 测试向量观察空间
#[test]
fn test_vector_obs_space() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");

        assert_eq!(env.get_obs_type(), ObsType::Vector);

        let obs_prop = env.get_obs_prop();
        assert_eq!(obs_prop.len(), 1);
        assert_eq!(obs_prop[0].shape_vec, vec![4]);

        assert_eq!(env.get_flatten_observation_len(), 4);

        env.close();
    });
}

/// 测试不同维度的观察空间
#[test]
fn test_various_obs_dimensions() {
    Python::attach(|py| {
        // 2 维
        let env = GymEnv::new(py, "MountainCarContinuous-v0");
        assert_eq!(env.get_obs_prop()[0].shape_vec, vec![2]);
        assert_eq!(env.get_flatten_observation_len(), 2);
        env.close();

        // 3 维
        let env = GymEnv::new(py, "Pendulum-v1");
        assert_eq!(env.get_obs_prop()[0].shape_vec, vec![3]);
        assert_eq!(env.get_flatten_observation_len(), 3);
        env.close();

        // 8 维
        let env = GymEnv::new(py, "LunarLander-v3");
        assert_eq!(env.get_obs_prop()[0].shape_vec, vec![8]);
        assert_eq!(env.get_flatten_observation_len(), 8);
        env.close();

        // 24 维
        let env = GymEnv::new(py, "BipedalWalker-v3");
        assert_eq!(env.get_obs_prop()[0].shape_vec, vec![24]);
        assert_eq!(env.get_flatten_observation_len(), 24);
        env.close();
    });
}

/// 测试 reset 返回的观察值
#[test]
fn test_reset_observation() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");

        // 不带 seed
        let obs1 = env.reset(None);
        assert_eq!(obs1.len(), 1);
        assert_eq!(obs1[0].len(), 4);

        // 带 seed（结果应可复现）
        let obs2 = env.reset(Some(42));
        let obs3 = env.reset(Some(42));
        assert_eq!(obs2, obs3);

        env.close();
    });
}

/// 测试 step 返回的观察值
#[test]
fn test_step_observation() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");
        let _obs = env.reset(Some(42));

        let action = vec![0.0];
        let (next_obs, _reward, _done) = env.step(&action);

        assert_eq!(next_obs.len(), 1);
        assert_eq!(next_obs[0].len(), 4);

        env.close();
    });
}

/// 测试环境信息打印（不 panic 即可）
#[test]
fn test_print_env_info() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");
        env.print_env_basic_info();
        env.close();
    });
}
