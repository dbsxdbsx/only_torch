//! GymEnv 环境覆盖测试
//!
//! 通过代表性环境验证 `GymEnv` 封装的核心功能：
//! - 离散动作环境（CartPole）
//! - 单维连续动作环境（Pendulum）
//! - 多维连续动作环境（BipedalWalker）
//! - 高维连续动作环境（MuJoCo Ant）
//! - 图像观察环境（Atari Breakout）
//! - 混合动作环境（Platform-v0）
//! - 自定义环境（五子棋 Gomoku）
//! - terminated / truncated 分离验证
//!
//! 注意：所有测试使用 `#[serial]` 确保串行执行，避免 Python 模块导入竞争

use crate::rl::{ActionType, GymEnv, ObsType};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use serial_test::serial;

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
#[serial]
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
        let (next_obs, reward, _terminated, _truncated) = env.step(&action);
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
#[serial]
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
        let (next_obs, reward, _terminated, _truncated) = env.step(&action);
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
#[serial]
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
        let (next_obs, _reward, _terminated, _truncated) = env.step(&action);
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
#[serial]
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

        let (next_obs, _reward, _terminated, _truncated) = env.step(&action);
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
#[serial]
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
        let (next_obs, _reward, _terminated, _truncated) = env.step(&action);
        assert_eq!(next_obs[0].len(), 210 * 160 * 3);

        // 采样验证
        let sampled = env.sample_action();
        assert_eq!(sampled.len(), 1);
        assert!(sampled[0] >= 0.0 && sampled[0] < 4.0);

        env.close();
    });
}

// ============================================================================
// 辅助功能测试
// ============================================================================

/// 测试 reset 的 seed 可复现性
///
/// 验证相同 seed 产生相同的初始观察
#[test]
#[serial]
fn test_reset_seed_reproducibility() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");

        // 带 seed 的 reset 应该可复现
        let obs1 = env.reset(Some(42));
        let obs2 = env.reset(Some(42));
        assert_eq!(obs1, obs2, "相同 seed 应产生相同观察");

        // 不同 seed 应该不同（大概率）
        let obs3 = env.reset(Some(123));
        assert_ne!(obs1, obs3, "不同 seed 应产生不同观察");

        env.close();
    });
}

/// 测试环境信息打印（不 panic 即可）
#[test]
#[serial]
fn test_print_env_info() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");
        env.print_env_basic_info();
        env.close();
    });
}

/// 测试多步执行循环
///
/// 验证 GymEnv 在多步交互中的稳定性
#[test]
#[serial]
fn test_multiple_steps() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");
        let _obs = env.reset(Some(42));
        let mut total_reward = 0.0;

        for _ in 0..10 {
            let action = env.sample_action();
            let (_next_obs, reward, terminated, truncated) = env.step(&action);
            total_reward += reward;

            if terminated || truncated {
                break;
            }
        }

        assert!(total_reward > 0.0);
        env.close();
    });
}

// ============================================================================
// 终止语义测试
// ============================================================================

/// 测试 step 透出 terminated / truncated（不合并为 done）
///
/// CartPole-v1 撞 500 步为 truncated（非 terminated）
#[test]
#[serial]
fn test_step_terminated_truncated_separation() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");
        let _obs = env.reset(Some(42));

        let mut saw_termination = false;
        for _ in 0..600 {
            let action = env.sample_action();
            let (_next_obs, _reward, terminated, truncated) = env.step(&action);

            if terminated {
                saw_termination = true;
                break;
            }
            if truncated {
                // CartPole 撞步数上限 → truncated=true, terminated=false
                assert!(!terminated, "truncated 时 terminated 应为 false");
                break;
            }
        }

        // CartPole 随机动作通常很快 terminated（杆倒了）
        assert!(saw_termination, "随机动作应很快导致 terminated");
        env.close();
    });
}

/// 测试 get_module_name 始终返回 "gymnasium"
#[test]
#[serial]
fn test_module_name_is_gymnasium() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");
        assert_eq!(env.get_module_name(), "gymnasium");
        env.close();
    });
}

// ============================================================================
// 混合动作环境（Platform-v0）
// ============================================================================

/// 测试 Platform-v0 混合动作空间结构
///
/// Platform-v0 动作空间：Tuple(Discrete(3), Tuple(Box(1,), Box(1,), Box(1,)))
/// 观察空间：Tuple(Box(9,), Discrete(200))
#[test]
#[serial]
fn test_platform_hybrid_action_space() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "Platform-v0");

        assert_eq!(env.get_action_type(), ActionType::Mix);
        assert_eq!(env.get_module_name(), "gymnasium");

        // 观察空间：Tuple → 扁平化后 10 维（9 + 1）
        assert_eq!(env.get_flatten_observation_len(), 10);

        // 动作空间：1 离散(3选1) + 3 连续参数 = 4 维
        assert_eq!(env.get_action_num_for_each_step(), 4);

        let action_ranges = env.get_all_action_valid_range();
        assert_eq!(action_ranges.len(), 4, "混合动作应展平为 4 个维度");

        // [0] 离散：Discrete(3)
        assert!(action_ranges[0].is_discrete_action());
        assert_eq!(action_ranges[0].get_discrete_action_selectable_num(), 3);

        // [1..=3] 连续参数
        for i in 1..=3 {
            assert!(!action_ranges[i].is_discrete_action());
        }

        env.close();
    });
}

/// 测试 Platform-v0 的 Tuple obs flatten
#[test]
#[serial]
fn test_platform_tuple_obs_flatten() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "Platform-v0");

        let obs_vec = env.reset(Some(42));

        // Tuple obs 应返回 2 个子空间
        assert_eq!(obs_vec.len(), 2, "Platform obs 应有 2 个子空间");
        assert_eq!(obs_vec[0].len(), 9, "Box(9,) 子空间长度应为 9");
        assert_eq!(obs_vec[1].len(), 1, "Discrete(200) 子空间长度应为 1");

        // flatten 后应为 10 维
        let flat = env.flatten_obs(&obs_vec);
        assert_eq!(flat.len(), 10);
        assert_eq!(&flat[..9], &obs_vec[0][..]);
        assert_eq!(flat[9], obs_vec[1][0]);

        env.close();
    });
}

/// 测试 Platform-v0 采样和执行
#[test]
#[serial]
fn test_platform_sample_and_step() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "Platform-v0");
        let _obs = env.reset(Some(42));

        for _ in 0..3 {
            let sampled = env.sample_action();
            assert_eq!(sampled.len(), 4);

            // 离散部分：0, 1, 或 2
            assert!(sampled[0] >= 0.0 && sampled[0] < 3.0);
            assert_eq!(sampled[0], sampled[0].floor());

            let (next_obs, _reward, terminated, truncated) = env.step(&sampled);
            // Tuple obs: 2 子空间
            assert_eq!(next_obs.len(), 2);

            if terminated || truncated {
                break;
            }
        }

        env.close();
    });
}

// ============================================================================
// 自定义环境（五子棋）
// ============================================================================

/// 注册五子棋自定义环境
fn register_gomoku_envs(py: Python<'_>) {
    // 添加项目根目录到 sys.path，确保可以导入 tests.python.custom_envs
    let sys = py.import("sys").expect("import sys 失败");
    let path = sys.getattr("path").expect("获取 sys.path 失败");

    // 获取当前工作目录（项目根目录）
    let os = py.import("os").expect("import os 失败");
    let cwd = os
        .call_method0("getcwd")
        .expect("获取当前工作目录失败")
        .extract::<String>()
        .expect("提取路径字符串失败");

    // 添加到 sys.path（如果不存在）
    let contains: bool = path
        .call_method1("__contains__", (&cwd,))
        .expect("检查路径失败")
        .extract()
        .expect("提取布尔值失败");

    if !contains {
        path.call_method1("insert", (0, &cwd))
            .expect("添加路径失败");
    }

    // 导入自定义环境模块（触发注册）
    // 优先使用新包 gym_env（v0.22，默认 9×9），回退到旧路径
    if py.import("gym_env").is_err() {
        py.import("tests.python.custom_envs")
            .expect("导入五子棋自定义环境模块失败");
    }
}

/// 测试五子棋环境基本功能（Gomoku-naive2-v0，默认 9×9）
///
/// 验证点：
/// - 动作类型：SingleDiscrete
/// - 观察空间：(3, 9, 9) 三通道
/// - 动作空间：81 个离散动作（9×9 棋盘）
#[test]
#[serial]
fn test_gomoku_env_basic() {
    Python::attach(|py| {
        register_gomoku_envs(py);

        let env = GymEnv::new(py, "Gomoku-naive2-v0");

        assert_eq!(env.get_action_type(), ActionType::SingleDiscrete);
        assert_eq!(env.get_obs_type(), ObsType::ChannelFirst);

        let obs_prop = env.get_obs_prop();
        assert_eq!(obs_prop.len(), 1);
        assert_eq!(obs_prop[0].shape_vec, vec![3, 9, 9]);

        let action_ranges = env.get_all_action_valid_range();
        assert_eq!(action_ranges.len(), 1);
        assert!(action_ranges[0].is_discrete_action());
        assert_eq!(action_ranges[0].get_discrete_action_selectable_num(), 81);

        assert_eq!(env.get_flatten_observation_len(), 3 * 9 * 9);

        env.close();
    });
}

/// 测试五子棋环境 reset 和 step（9×9 默认棋盘）
#[test]
#[serial]
fn test_gomoku_env_reset_step() {
    Python::attach(|py| {
        register_gomoku_envs(py);

        let env = GymEnv::new(py, "Gomoku-naive2-v0");

        let obs = env.reset(Some(42));
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].len(), 3 * 9 * 9);

        // 在中心位置落子 (4, 4) -> action = 4*9 + 4 = 40
        let action = vec![40.0];
        let (next_obs, reward, terminated, truncated) = env.step(&action);
        assert_eq!(next_obs[0].len(), 3 * 9 * 9);
        assert!(reward == 0.0 || reward == 1.0 || reward == -1.0);

        if !(terminated || truncated) {
            let sampled = env.sample_action();
            assert_eq!(sampled.len(), 1);
            assert!(sampled[0] >= 0.0 && sampled[0] < 81.0);
            let (_obs, _reward, _terminated, _truncated) = env.step(&sampled);
        }

        env.close();
    });
}

/// 测试五子棋环境多步对局
///
/// 验证环境在多步交互中的稳定性
#[test]
#[serial]
fn test_gomoku_env_multiple_steps() {
    Python::attach(|py| {
        register_gomoku_envs(py);

        // 使用 random 对手，更容易获胜
        let env = GymEnv::new(py, "Gomoku-random-v0");

        let _obs = env.reset(Some(42));
        let mut total_steps = 0;
        let mut game_ended = false;

        // 执行多步随机动作（最多 100 步）
        for _ in 0..100 {
            let action = env.sample_action();
            assert_eq!(action.len(), 1);

            let (_next_obs, reward, terminated, truncated) = env.step(&action);
            total_steps += 1;

            if terminated || truncated {
                game_ended = true;
                // 游戏结束，验证奖励
                assert!(
                    reward == 1.0 || reward == -1.0 || reward == 0.0,
                    "游戏结束时奖励应为 1.0（胜）、-1.0（负）或 0.0（平局），实际为 {}",
                    reward
                );
                break;
            }
        }

        assert!(total_steps > 0, "至少应执行一步");
        // 随机对手下，100 步内游戏应该结束
        assert!(game_ended, "100 步内游戏应该结束");

        env.close();
    });
}

/// 测试五子棋环境信息打印
#[test]
#[serial]
fn test_gomoku_env_print_info() {
    Python::attach(|py| {
        register_gomoku_envs(py);

        let env = GymEnv::new(py, "Gomoku-naive2-v0");
        env.print_env_basic_info();
        env.close();
    });
}
