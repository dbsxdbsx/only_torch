//! Minari 离线 RL 数据集封装
//!
//! 提供与 `GymEnv` 风格一致的 Rust 接口，封装所有 pyo3 调用细节。
//!
//! ## 示例
//!
//! ```ignore
//! use pyo3::Python;
//! use only_torch::rl::MinariDataset;
//!
//! Python::attach(|py| {
//!     // 列出可用数据集
//!     let local = MinariDataset::list_local(py);
//!     let remote = MinariDataset::list_remote(py);
//!
//!     // 加载数据集
//!     let dataset = MinariDataset::load(py, "D4RL/pointmaze/umaze-v2");
//!     println!("总 episodes: {}", dataset.total_episodes());
//!
//!     // 采样 episode
//!     let episodes = dataset.sample_episodes(3);
//!     for ep in &episodes {
//!         println!("奖励数: {}", ep.rewards.len());
//!     }
//! });
//! ```

use pyo3::prelude::*;

// ============================================================================
// Episode 数据结构
// ============================================================================

/// 单个 Episode 的数据
///
/// 包含一个完整轨迹的观察、动作、奖励等信息。
#[derive(Debug, Clone)]
pub struct Episode {
    /// 观察序列（扁平化为 f32 向量）
    pub observations: Vec<Vec<f32>>,
    /// 动作序列
    pub actions: Vec<Vec<f32>>,
    /// 奖励序列
    pub rewards: Vec<f32>,
    /// 终止标志序列
    pub terminations: Vec<bool>,
    /// 截断标志序列
    pub truncations: Vec<bool>,
}

// ============================================================================
// MinariDataset 核心实现
// ============================================================================

/// Minari 离线数据集封装
///
/// 提供离线 RL 数据集的加载、采样等功能。
///
/// ## 与 GymEnv 的区别
///
/// - `GymEnv`：在线交互式环境（reset → step → done 循环）
/// - `MinariDataset`：离线数据集（load → sample → 读取轨迹数据）
pub struct MinariDataset<'py> {
    /// 数据集对象
    dataset: Bound<'py, PyAny>,
    /// 数据集名称
    name: String,
    /// 总 episode 数
    total_episodes: usize,
    /// 总 step 数
    total_steps: usize,
}

impl<'py> MinariDataset<'py> {
    /// 加载 Minari 数据集
    ///
    /// 如果数据集不存在，会自动下载。
    ///
    /// # 参数
    /// - `py`: Python 解释器引用
    /// - `dataset_name`: 数据集名称，如 "D4RL/pointmaze/umaze-v2"
    ///
    /// # 示例
    /// ```ignore
    /// let dataset = MinariDataset::load(py, "D4RL/pointmaze/umaze-v2");
    /// ```
    pub fn load(py: Python<'py>, dataset_name: &str) -> Self {
        let minari = py.import("minari").expect("import minari 失败");

        // 下载数据集（如果本地不存在）
        let download_fn = minari
            .getattr("download_dataset")
            .expect("获取 download_dataset 失败");
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("force_download", false).unwrap();
        let _ = download_fn.call((dataset_name,), Some(&kwargs));

        // 加载数据集
        let dataset = minari
            .call_method1("load_dataset", (dataset_name,))
            .expect("加载数据集失败");

        // 获取元信息
        let total_episodes: usize = dataset
            .getattr("total_episodes")
            .expect("获取 total_episodes 失败")
            .extract()
            .expect("解析 total_episodes 失败");

        let total_steps: usize = dataset
            .getattr("total_steps")
            .expect("获取 total_steps 失败")
            .extract()
            .expect("解析 total_steps 失败");

        Self {
            dataset,
            name: dataset_name.to_string(),
            total_episodes,
            total_steps,
        }
    }

    /// 获取数据集名称
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// 获取总 episode 数量
    pub fn total_episodes(&self) -> usize {
        self.total_episodes
    }

    /// 获取总 step 数量
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// 采样指定数量的 episodes
    ///
    /// # 参数
    /// - `n`: 要采样的 episode 数量
    ///
    /// # 返回
    /// Episode 数据列表
    pub fn sample_episodes(&self, n: usize) -> Vec<Episode> {
        let episodes_py = self
            .dataset
            .call_method1("sample_episodes", (n,))
            .expect("采样 episode 失败");

        let episodes_len = episodes_py.len().expect("获取采样长度失败");
        let mut episodes = Vec::with_capacity(episodes_len);

        for i in 0..episodes_len {
            let ep_py = episodes_py.get_item(i).expect("获取 episode 失败");
            let episode = self.parse_episode(&ep_py);
            episodes.push(episode);
        }

        episodes
    }

    /// 列出本地已下载的数据集
    ///
    /// # 返回
    /// 数据集名称列表
    pub fn list_local(py: Python<'_>) -> Vec<String> {
        let minari = py.import("minari").expect("import minari 失败");
        let datasets_dict = minari
            .call_method0("list_local_datasets")
            .expect("list_local_datasets 失败");

        Self::extract_dict_keys(py, &datasets_dict)
    }

    /// 列出远程可用的数据集
    ///
    /// # 返回
    /// 数据集名称列表
    pub fn list_remote(py: Python<'_>) -> Vec<String> {
        let minari = py.import("minari").expect("import minari 失败");
        let datasets_dict = minari
            .call_method0("list_remote_datasets")
            .expect("list_remote_datasets 失败");

        Self::extract_dict_keys(py, &datasets_dict)
    }

    /// 打印数据集基本信息
    pub fn print_info(&self) {
        println!("----------------------------");
        println!("数据集: {}", self.name);
        println!("总 episode 数: {}", self.total_episodes);
        println!("总 step 数: {}", self.total_steps);
        println!("----------------------------");
    }

    // ========================================================================
    // 内部方法
    // ========================================================================

    /// 从 Python 字典提取 keys 列表
    fn extract_dict_keys(py: Python<'_>, dict: &Bound<'_, PyAny>) -> Vec<String> {
        let builtins = py.import("builtins").expect("import builtins 失败");
        let list_fn = builtins.getattr("list").expect("获取 list 失败");
        let keys = dict.call_method0("keys").expect("获取 keys 失败");
        let keys_list = list_fn.call1((keys,)).expect("转换为 list 失败");
        keys_list.extract::<Vec<String>>().unwrap_or_default()
    }

    /// 解析单个 Episode
    fn parse_episode(&self, ep_py: &Bound<'py, PyAny>) -> Episode {
        // 解析 observations
        let obs_py = ep_py.getattr("observations").expect("获取 observations 失败");
        let observations = self.parse_observations(&obs_py);

        // 解析 actions
        let actions_py = ep_py.getattr("actions").expect("获取 actions 失败");
        let actions = self.parse_array_2d(&actions_py);

        // 解析 rewards
        let rewards_py = ep_py.getattr("rewards").expect("获取 rewards 失败");
        let rewards: Vec<f32> = rewards_py.extract().unwrap_or_default();

        // 解析 terminations
        let terminations_py = ep_py.getattr("terminations").expect("获取 terminations 失败");
        let terminations: Vec<bool> = terminations_py.extract().unwrap_or_default();

        // 解析 truncations
        let truncations_py = ep_py.getattr("truncations").expect("获取 truncations 失败");
        let truncations: Vec<bool> = truncations_py.extract().unwrap_or_default();

        Episode {
            observations,
            actions,
            rewards,
            terminations,
            truncations,
        }
    }

    /// 解析 observations（可能是 dict 或 ndarray）
    fn parse_observations(&self, obs_py: &Bound<'py, PyAny>) -> Vec<Vec<f32>> {
        // Minari 的 observations 可能是字典（多个 key）或直接的 ndarray
        // 通过检查是否有 keys() 方法来判断是否为字典类型
        if obs_py.hasattr("keys").unwrap_or(false) {
            // 字典类型：合并所有值
            self.parse_dict_observations(obs_py)
        } else {
            // 直接是 ndarray
            self.parse_array_2d(obs_py)
        }
    }

    /// 解析字典类型的 observations
    fn parse_dict_observations(&self, obs_py: &Bound<'py, PyAny>) -> Vec<Vec<f32>> {
        let mut result: Vec<Vec<f32>> = Vec::new();

        // 获取所有 keys
        let keys = match obs_py.call_method0("keys") {
            Ok(k) => k,
            Err(_) => return result,
        };

        let keys_list: Vec<String> = match keys.extract() {
            Ok(k) => k,
            Err(_) => {
                // 尝试通过 list() 转换
                let py = obs_py.py();
                let builtins = py.import("builtins").ok();
                if let Some(builtins) = builtins {
                    if let Ok(list_fn) = builtins.getattr("list") {
                        if let Ok(list_py) = list_fn.call1((keys,)) {
                            list_py.extract().unwrap_or_default()
                        } else {
                            return result;
                        }
                    } else {
                        return result;
                    }
                } else {
                    return result;
                }
            }
        };

        for key in keys_list {
            // 跳过某些特殊 key
            if key == "achieved_goal" || key == "desired_goal" {
                continue;
            }

            let value = match obs_py.get_item(&key) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let array_2d = self.parse_array_2d(&value);

            if result.is_empty() {
                result = array_2d;
            } else {
                // 合并到现有结果
                for (i, row) in array_2d.into_iter().enumerate() {
                    if i < result.len() {
                        result[i].extend(row);
                    }
                }
            }
        }

        result
    }

    /// 解析 2D 数组
    fn parse_array_2d(&self, array_py: &Bound<'py, PyAny>) -> Vec<Vec<f32>> {
        // 尝试直接提取为 Vec<Vec<f32>>
        if let Ok(arr) = array_py.extract::<Vec<Vec<f32>>>() {
            return arr;
        }

        // 尝试通过 tolist() 转换
        if let Ok(list_py) = array_py.call_method0("tolist") {
            if let Ok(arr) = list_py.extract::<Vec<Vec<f32>>>() {
                return arr;
            }
        }

        // 尝试单维数组
        if let Ok(arr) = array_py.extract::<Vec<f32>>() {
            return arr.into_iter().map(|v| vec![v]).collect();
        }

        Vec::new()
    }
}
