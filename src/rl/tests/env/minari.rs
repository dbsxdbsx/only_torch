//! Minari 离线 RL 数据集测试
//!
//! 对应 Python 测试：`tests/python/gym/test_06_minari.py`
//!
//! 测试 `MinariDataset` 封装的功能，包括：
//! - 数据集列表获取
//! - 数据集加载
//! - Episode 采样
//! - Episode 数据结构解析
//!
//! 注意：所有测试使用 `#[serial]` 确保串行执行，避免 Python 模块导入竞争

use crate::rl::MinariDataset;
use pyo3::Python;
use serial_test::serial;

// ============================================================================
// 数据集列表测试
// ============================================================================

#[test]
#[serial]
fn test_list_local_datasets() {
    Python::attach(|py| {
        let datasets = MinariDataset::list_local(py);

        // 本地数据集可能为空，只验证返回类型正确
        println!("本地数据集数量: {}", datasets.len());
        for ds in &datasets {
            println!("  - {}", ds);
        }

        // 验证返回的是 Vec<String>
        assert!(
            datasets
                .iter()
                .all(|s| !s.is_empty() || datasets.is_empty())
        );
    });
}

#[test]
#[serial]
fn test_list_remote_datasets() {
    Python::attach(|py| {
        let datasets = MinariDataset::list_remote(py);

        // 远程数据集应该有很多
        println!("远程数据集数量: {}", datasets.len());
        assert!(
            datasets.len() > 100,
            "远程数据集数量应该超过 100，实际: {}",
            datasets.len()
        );

        // 验证包含 D4RL 系列数据集
        let has_d4rl = datasets.iter().any(|s| s.contains("D4RL"));
        assert!(has_d4rl, "应该包含 D4RL 系列数据集");

        // 打印前 5 个数据集
        println!("前 5 个数据集:");
        for ds in datasets.iter().take(5) {
            println!("  - {}", ds);
        }
    });
}

// ============================================================================
// 数据集加载测试
// ============================================================================

#[test]
#[serial]
fn test_load_dataset() {
    Python::attach(|py| {
        let dataset_name = "D4RL/pointmaze/umaze-v2";
        let dataset = MinariDataset::load(py, dataset_name);

        // 验证基本属性
        assert_eq!(dataset.get_name(), dataset_name);
        assert!(dataset.total_episodes() > 0, "数据集应该包含 episode");
        assert!(dataset.total_steps() > 0, "数据集应该包含 step");

        println!("数据集: {}", dataset.get_name());
        println!("  总 episode 数: {}", dataset.total_episodes());
        println!("  总 step 数: {}", dataset.total_steps());
    });
}

#[test]
#[serial]
fn test_dataset_print_info() {
    Python::attach(|py| {
        let dataset = MinariDataset::load(py, "D4RL/pointmaze/umaze-v2");
        dataset.print_info();
    });
}

// ============================================================================
// Episode 采样测试
// ============================================================================

#[test]
#[serial]
fn test_sample_single_episode() {
    Python::attach(|py| {
        let dataset = MinariDataset::load(py, "D4RL/pointmaze/umaze-v2");

        // 采样 1 个 episode
        let episodes = dataset.sample_episodes(1);
        assert_eq!(episodes.len(), 1, "应该采样到 1 个 episode");

        let ep = &episodes[0];
        println!("Episode 数据:");
        println!("  observations 数量: {}", ep.observations.len());
        println!("  actions 数量: {}", ep.actions.len());
        println!("  rewards 数量: {}", ep.rewards.len());
        println!("  terminations 数量: {}", ep.terminations.len());
        println!("  truncations 数量: {}", ep.truncations.len());

        // 验证数据非空
        assert!(!ep.rewards.is_empty(), "episode 应该有 rewards");
    });
}

#[test]
#[serial]
fn test_sample_multiple_episodes() {
    Python::attach(|py| {
        let dataset = MinariDataset::load(py, "D4RL/pointmaze/umaze-v2");

        // 采样 3 个 episodes
        let episodes = dataset.sample_episodes(3);
        assert_eq!(episodes.len(), 3, "应该采样到 3 个 episodes");

        // 验证每个 episode
        for (i, ep) in episodes.iter().enumerate() {
            println!("Episode {} rewards 长度: {}", i, ep.rewards.len());
            assert!(!ep.rewards.is_empty(), "每个 episode 应该有 rewards");
        }
    });
}

// ============================================================================
// Episode 结构验证测试
// ============================================================================

#[test]
#[serial]
fn test_episode_structure() {
    Python::attach(|py| {
        let dataset = MinariDataset::load(py, "D4RL/pointmaze/umaze-v2");
        let episodes = dataset.sample_episodes(1);
        let ep = &episodes[0];

        // 验证各字段长度一致性
        // rewards, terminations, truncations 长度应该相同
        assert_eq!(
            ep.rewards.len(),
            ep.terminations.len(),
            "rewards 和 terminations 长度应该相同"
        );
        assert_eq!(
            ep.rewards.len(),
            ep.truncations.len(),
            "rewards 和 truncations 长度应该相同"
        );

        // observations 比 actions 多一个（包含最终状态）
        // 或者相等（取决于数据集格式）
        let obs_len = ep.observations.len();
        let act_len = ep.actions.len();
        assert!(
            obs_len == act_len || obs_len == act_len + 1,
            "observations 长度 ({}) 应该等于或比 actions 长度 ({}) 多 1",
            obs_len,
            act_len
        );

        println!("Episode 结构验证通过");
        println!("  observations: {}", obs_len);
        println!("  actions: {}", act_len);
        println!("  rewards: {}", ep.rewards.len());
    });
}

#[test]
#[serial]
fn test_episode_data_values() {
    Python::attach(|py| {
        let dataset = MinariDataset::load(py, "D4RL/pointmaze/umaze-v2");
        let episodes = dataset.sample_episodes(1);
        let ep = &episodes[0];

        // 验证 observations 非空且有实际值
        if !ep.observations.is_empty() {
            let first_obs = &ep.observations[0];
            println!("第一个 observation 维度: {}", first_obs.len());
            assert!(!first_obs.is_empty(), "observation 应该有维度");
        }

        // 验证 actions 非空且有实际值
        if !ep.actions.is_empty() {
            let first_action = &ep.actions[0];
            println!("第一个 action 维度: {}", first_action.len());
            assert!(!first_action.is_empty(), "action 应该有维度");
        }

        // 验证 rewards 是有效数值
        for reward in &ep.rewards {
            assert!(reward.is_finite(), "reward 应该是有效数值");
        }

        println!("Episode 数据值验证通过");
    });
}
