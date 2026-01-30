//! Gymnasium/Gym 环境封装
//!
//! 提供与 Python RL 环境交互的 Rust 接口，支持：
//! - gymnasium 环境（推荐，现代标准）
//! - gym 环境（兼容老式环境，如 gym-hybrid）
//! - 自定义环境（注册到 gymnasium）
//!
//! 参考：https://github.com/MrRobb/gym-rs/blob/master/src/lib.rs
//! 基于 RustRL 项目迁移，适配 only_torch

use numpy::ndarray::{array, Array1};
use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

// ============================================================================
// 观察空间类型定义
// ============================================================================

/// 观察空间维度信息
///
/// 存储单个观察的形状信息，如图像输入 `[210, 160, 3]` 或向量输入 `[4]`。
#[derive(Debug, Clone)]
pub struct ObsDim {
    /// 形状向量，如 `[4]` 表示 4 维向量，`[84, 84, 3]` 表示图像
    pub shape_vec: Vec<i64>,
}

/// 观察类型
///
/// 用于区分不同类型的观察空间，主要用于图像输入的通道顺序判断。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObsType {
    /// 无通道维度的图像（灰度图 HW）
    NoChannel,
    /// 通道在前的图像（CHW，如 PyTorch 风格）
    ChannelFirst,
    /// 通道在后的图像（HWC，如 TensorFlow 风格）
    ChannelLast,
    /// 非图像类型，普通向量输入
    Vector,
}

// ============================================================================
// 动作空间类型定义
// ============================================================================

/// 动作空间整体类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    /// 单维离散动作（如 CartPole）
    SingleDiscrete,
    /// 连续动作（单维或多维）
    Continuous,
    /// 混合动作（离散 + 连续，如 Platform）
    Mix,
    /// 未知类型（初始化前的默认值）
    Unknown,
}

impl Default for ActionType {
    fn default() -> Self {
        Self::Unknown
    }
}

/// 单个动作维度的具体类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionDimType {
    /// 整数类型（离散动作）
    Int64,
    /// 浮点类型（连续动作）
    Float64,
    /// NumPy 标量（单元素数组）
    NumpyFloat64Scalar,
    /// NumPy 列表（多元素数组）
    NumpyFloat64List,
    /// Tuple 类型（中间介质，用于混合动作空间）
    Tuple,
    /// 未知类型
    Unknown,
}

impl Default for ActionDimType {
    fn default() -> Self {
        Self::Unknown
    }
}

/// 单个动作维度的属性
///
/// 描述某一个维度的动作属性，包括取值范围和类型。
#[derive(Debug, Clone, Default)]
pub struct ActionDim {
    /// 动作上界（离散动作为最大值，连续动作为上限）
    pub high_v_op: Option<f32>,
    /// 动作下界（离散动作通常为 0，连续动作为下限）
    pub low_v_op: Option<f32>,
    /// 动作类型
    pub action_type: ActionDimType,
    /// 子动作维度（用于 Tuple/NumpyFloat64List 类型）
    pub sub_action_dim_op: Option<Vec<ActionDim>>,
}

/// 动作取值范围
///
/// 返回给模型的实质性动作取值范围，用于确定输出层结构。
#[derive(Debug, Clone, Default)]
pub struct ActionRange {
    /// 动作范围 (low_bound, high_bound)
    action_range: (f32, f32),
    /// 是否为离散动作
    is_discrete: bool,
}

impl ActionRange {
    /// 判断是否为离散动作
    pub fn is_discrete_action(&self) -> bool {
        self.is_discrete
    }

    /// 获取离散动作的可选数量
    ///
    /// # Panics
    /// 如果当前动作不是离散类型
    pub fn get_discrete_action_selectable_num(&self) -> usize {
        assert!(self.is_discrete, "当前动作不是离散类型");
        (self.action_range.1 - self.action_range.0) as usize + 1
    }

    /// 获取连续动作的取值范围 (low, high)
    ///
    /// # Panics
    /// 如果当前动作不是连续类型
    pub fn get_continuous_action_low_high(&self) -> (f32, f32) {
        assert!(!self.is_discrete, "当前动作不是连续类型");
        self.action_range
    }
}

// ============================================================================
// GymEnv 核心实现
// ============================================================================

/// Gymnasium/Gym 环境封装
///
/// 提供与 Python RL 环境交互的 Rust 接口。智能加载策略：
/// 1. 优先尝试 gymnasium 模块
/// 2. 若失败，自动回退到 gym 模块
/// 3. 对用户完全透明
///
/// ## 支持的环境类型
///
/// - **离散动作**：CartPole-v1, Acrobot-v1, LunarLander-v3 等
/// - **连续动作**：Pendulum-v1, MountainCarContinuous-v0 等
/// - **多维连续**：BipedalWalker-v3, Ant-v5, HalfCheetah-v5 等
/// - **混合动作**：Moving-v0, Sliding-v0 等（离散 + 连续，来自 gym-hybrid）
///
/// ## 示例
///
/// ```ignore
/// use pyo3::Python;
/// use only_torch::rl::GymEnv;
///
/// Python::attach(|py| {
///     // 自动选择合适的模块加载
///     let env = GymEnv::new(py, "CartPole-v1");  // gymnasium
///     let hybrid_env = GymEnv::new(py, "Moving-v0");  // 自动回退到 gym
///     
///     let obs = env.reset(Some(42));
///     println!("初始观察: {:?}", obs);
/// });
/// ```
pub struct GymEnv<'py> {
    /// Python 解释器引用
    py: Python<'py>,
    /// 环境对象
    env: Bound<'py, PyAny>,
    /// 是否使用老式 gym 模块（影响 API 调用方式）
    use_legacy_gym: bool,

    // 观察空间相关
    /// 观察空间属性列表
    obs_prop_vec: Vec<ObsDim>,
    /// 观察类型（向量/图像）
    obs_type: ObsType,

    // 动作空间相关
    /// 动作空间对象（用于采样）
    action_space: Bound<'py, PyAny>,
    /// 动作属性列表
    action_prop_vec: Vec<ActionDim>,
    /// 每步所需的动作数量
    action_num: usize,
    /// 动作空间整体类型
    action_type: ActionType,

    // 环境元信息
    /// 环境名称
    env_name: String,
}

impl<'py> GymEnv<'py> {
    /// 创建新的环境（智能加载）
    ///
    /// 自动选择合适的 Python 模块：
    /// 1. 优先尝试 gymnasium（现代标准）
    /// 2. 若失败，自动回退到 gym（兼容老式环境如 gym-hybrid）
    ///
    /// # 参数
    /// - `py`: Python 解释器引用
    /// - `env_name`: 环境名称，如 "CartPole-v1" 或 "Moving-v0"
    ///
    /// # 示例
    /// ```ignore
    /// Python::attach(|py| {
    ///     let env = GymEnv::new(py, "Pendulum-v1");     // gymnasium
    ///     let hybrid = GymEnv::new(py, "Moving-v0");    // 自动回退到 gym
    /// });
    /// ```
    pub fn new(py: Python<'py>, env_name: &str) -> Self {
        // 设置 sys.argv（否则某些环境 render() 会失败）
        let sys = py.import("sys").expect("import sys 失败");
        if let Ok(argv) = sys.getattr("argv") {
            let _ = argv.call_method1("append", ("",));
        }

        // 智能加载：先尝试 gymnasium，失败则回退到 gym
        let (env, use_legacy_gym) = Self::try_make_env(py, env_name);

        // 解析观察空间
        let obs_space = env
            .getattr("observation_space")
            .expect("获取 observation_space 失败");
        let obs_prop_vec = init_obs_prop(&obs_space);

        // 解析动作空间
        let action_space = env
            .getattr("action_space")
            .expect("获取 action_space 失败");
        let action_prop_vec = init_act_prop(&action_space, true);

        // 计算动作类型和数量
        let action_type = check_action_type(&action_prop_vec);
        let action_num = calc_action_real_num(&action_prop_vec);

        // 构建环境对象
        let mut env_obj = Self {
            py,
            env,
            use_legacy_gym,
            obs_prop_vec,
            obs_type: ObsType::Vector,
            action_space,
            action_prop_vec,
            action_num,
            action_type,
            env_name: env_name.to_string(),
        };

        // 检测观察类型
        env_obj.check_obs_type();
        env_obj
    }

    /// 尝试创建环境（gymnasium 优先，gym 回退）
    ///
    /// 返回 (env, use_legacy_gym)
    fn try_make_env(py: Python<'py>, env_name: &str) -> (Bound<'py, PyAny>, bool) {
        // 1. 先尝试 gymnasium
        if let Ok(gymnasium) = py.import("gymnasium") {
            if let Ok(make) = gymnasium.getattr("make") {
                if let Ok(env) = make.call1((env_name,)) {
                    return (env, false);
                }
            }
        }

        // 2. gymnasium 失败，回退到 gym
        // 需要先导入环境注册模块（如 gym_hybrid）
        Self::try_import_gym_env_module(py, env_name);

        let gym = py.import("gym").unwrap_or_else(|_| {
            panic!(
                "无法加载环境 '{}': gymnasium 和 gym 模块均不可用",
                env_name
            )
        });

        // 使用 kwargs 禁用 env_checker（解决 numpy 2.0 兼容性问题）
        // gym 的 env_checker 使用了 np.bool8，在 numpy 2.0 中已移除
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("disable_env_checker", true).unwrap();

        let env = gym
            .getattr("make")
            .expect("获取 gym.make 失败")
            .call((env_name,), Some(&kwargs))
            .unwrap_or_else(|_| {
                panic!(
                    "无法创建环境 '{}': 请确保已安装对应的环境包",
                    env_name
                )
            });

        (env, true)
    }

    /// 尝试导入 gym 环境的注册模块
    ///
    /// 某些环境（如 gym-hybrid）需要先导入其模块才能被 gym.make 识别
    fn try_import_gym_env_module(py: Python<'_>, env_name: &str) {
        // gym-hybrid 环境：Moving-v0, Sliding-v0
        if env_name == "Moving-v0" || env_name == "Sliding-v0" {
            let _ = py.import("gym_hybrid");
        }
        // 未来可在此处添加更多环境的导入逻辑
    }

    /// 获取当前使用的模块类型
    ///
    /// 返回 "gymnasium" 或 "gym"
    pub fn get_module_name(&self) -> &'static str {
        if self.use_legacy_gym {
            "gym"
        } else {
            "gymnasium"
        }
    }

    /// 获取环境名称
    pub fn get_name(&self) -> &str {
        &self.env_name
    }

    /// 获取动作空间类型
    pub fn get_action_type(&self) -> ActionType {
        self.action_type
    }

    /// 获取环境的奖励阈值（达标分数线）
    pub fn get_threshold(&self) -> f32 {
        let spec = self.env.getattr("spec").expect("获取 spec 失败");
        let reward_threshold = spec
            .getattr("reward_threshold")
            .expect("获取 reward_threshold 失败");

        reward_threshold.extract::<f32>().unwrap_or(f32::INFINITY)
    }

    /// 打印环境基本信息
    pub fn print_env_basic_info(&self) {
        println!("----------------------------");
        println!("环境名称: {}", self.get_name());

        // 打印观察空间信息
        for (i, obs) in self.get_obs_prop().iter().enumerate() {
            println!("obs-{} 形状: {:?}", i + 1, obs.shape_vec);
        }

        // 打印动作空间信息
        let action_dims = self.get_flatten_substant_action_dim(None);
        for (i, action_dim) in action_dims.iter().enumerate() {
            let type_str = match action_dim.action_type {
                ActionDimType::Int64 => "离散",
                ActionDimType::Float64 | ActionDimType::NumpyFloat64Scalar => "连续",
                _ => "其他",
            };
            println!(
                "action-{} 类型: <{}> 范围: {}~{}",
                i + 1,
                type_str,
                action_dim.low_v_op.unwrap_or(0.0),
                action_dim.high_v_op.unwrap_or(0.0),
            );
        }

        println!("达标分数线: {:.2}", self.get_threshold());
        println!("----------------------------");
    }

    /// 渲染环境（human 模式）
    pub fn render(&self) {
        let _ = self.env.call_method1("render", ("human",));
    }

    /// 重置环境
    ///
    /// # 参数
    /// - `seed`: 可选的随机种子
    ///
    /// # 返回
    /// 初始观察向量列表（通常只有一个元素）
    pub fn reset(&self, seed: Option<u64>) -> Vec<Vec<f32>> {
        if self.use_legacy_gym {
            // gym API: env.seed(seed) + env.reset()
            if let Some(s) = seed {
                let _ = self.env.call_method1("seed", (s,));
            }
            let result = self.env.call_method0("reset").expect("调用 reset 失败");

            // gym 返回值处理：
            // - 0.26+: (obs, info) tuple
            // - 旧版/某些环境: 直接返回 obs（可能是 list/ndarray）
            let obs = self.extract_gym_reset_obs(&result);
            self.get_obs_vec_from_python(&obs)
        } else {
            // gymnasium API: reset(seed=...)
            let kwargs = pyo3::types::PyDict::new(self.py);
            if let Some(s) = seed {
                kwargs.set_item("seed", s).unwrap();
            }

            let result = self
                .env
                .call_method("reset", (), Some(&kwargs))
                .expect("调用 reset 失败");

            // Gymnasium 返回 (obs, info)
            let obs = result.get_item(0).expect("获取 obs 失败");
            self.get_obs_vec_from_python(&obs)
        }
    }

    /// 从 gym reset 返回值中提取 obs
    ///
    /// gym 的返回值可能是：
    /// - tuple (obs, info) - gym 0.26+
    /// - obs 本身 (list/ndarray) - 旧版或某些环境
    fn extract_gym_reset_obs<'a>(&self, result: &'a Bound<'py, PyAny>) -> Bound<'a, PyAny> {
        let type_name = result
            .get_type()
            .name()
            .map(|s| s.to_string())
            .unwrap_or_default();

        // 如果是 tuple，取第一个元素
        if type_name == "tuple" {
            result.get_item(0).expect("获取 tuple[0] 失败")
        } else {
            // 否则直接返回（list/ndarray 都是 obs 本身）
            result.clone()
        }
    }

    /// 关闭环境
    pub fn close(&self) {
        let _ = self.env.call_method0("close");
    }

    /// 执行一步动作
    ///
    /// # 参数
    /// - `action`: 动作向量
    ///
    /// # 返回
    /// (obs_vec, reward, done) 元组
    pub fn step(&self, action: &[f32]) -> (Vec<Vec<f32>>, f32, bool) {
        // 将 Rust 动作转换为 Python 对象
        let action_py = self.convert_action_to_python(action);

        // 执行 step
        let result = self
            .env
            .call_method1("step", (action_py,))
            .expect("调用 step 失败");

        // 解析返回值
        let obs_py = result.get_item(0).expect("获取 obs 失败");
        let obs_vec = self.get_obs_vec_from_python(&obs_py);

        let reward: f32 = result
            .get_item(1)
            .expect("获取 reward 失败")
            .extract()
            .expect("解析 reward 失败");

        // gym 和 gymnasium 的返回值差异：
        // - gymnasium: (obs, reward, terminated, truncated, info) - 5 元素
        // - gym: (obs, reward, done, info) - 4 元素
        let done = if self.use_legacy_gym {
            // gym API: 第 3 个元素是 done
            result
                .get_item(2)
                .expect("获取 done 失败")
                .extract()
                .expect("解析 done 失败")
        } else {
            // gymnasium API: terminated || truncated
            let terminated: bool = result
                .get_item(2)
                .expect("获取 terminated 失败")
                .extract()
                .expect("解析 terminated 失败");

            let truncated: bool = result
                .get_item(3)
                .expect("获取 truncated 失败")
                .extract()
                .expect("解析 truncated 失败");

            terminated || truncated
        };

        (obs_vec, reward, done)
    }

    /// 采样随机动作
    ///
    /// # 返回
    /// 随机动作向量
    pub fn sample_action(&self) -> Vec<f32> {
        let action_py = self
            .action_space
            .call_method0("sample")
            .expect("采样动作失败");

        self.convert_python_action_to_rust(&action_py)
    }

    /// 获取观察空间属性
    pub fn get_obs_prop(&self) -> &[ObsDim] {
        &self.obs_prop_vec
    }

    /// 获取观察类型
    pub fn get_obs_type(&self) -> ObsType {
        self.obs_type
    }

    /// 获取动作空间属性
    pub fn get_action_prop(&self) -> &[ActionDim] {
        &self.action_prop_vec
    }

    /// 获取每步所需的动作数量
    pub fn get_action_num_for_each_step(&self) -> usize {
        self.action_num
    }

    /// 获取所有动作的有效范围
    pub fn get_all_action_valid_range(&self) -> Vec<ActionRange> {
        let mut ranges = Vec::new();
        for action_dim in self.get_flatten_substant_action_dim(None) {
            match action_dim.action_type {
                ActionDimType::Int64 => {
                    ranges.push(ActionRange {
                        action_range: (0.0, action_dim.high_v_op.unwrap_or(0.0)),
                        is_discrete: true,
                    });
                }
                ActionDimType::Float64 | ActionDimType::NumpyFloat64Scalar => {
                    ranges.push(ActionRange {
                        action_range: (
                            action_dim.low_v_op.unwrap_or(0.0),
                            action_dim.high_v_op.unwrap_or(0.0),
                        ),
                        is_discrete: false,
                    });
                }
                _ => {}
            }
        }
        ranges
    }

    /// 获取扁平化的实质性动作维度列表
    ///
    /// 去除 Tuple、NumpyFloat64List 等中间类型，返回有具体属性的 ActionDim。
    pub fn get_flatten_substant_action_dim(
        &self,
        action_dim_vec_op: Option<&[ActionDim]>,
    ) -> Vec<ActionDim> {
        let mut result = Vec::new();
        let action_dims = action_dim_vec_op.unwrap_or(self.get_action_prop());

        for action_dim in action_dims {
            match action_dim.action_type {
                ActionDimType::Tuple | ActionDimType::NumpyFloat64List => {
                    if let Some(sub_dims) = &action_dim.sub_action_dim_op {
                        let mut sub_result = self.get_flatten_substant_action_dim(Some(sub_dims));
                        result.append(&mut sub_result);
                    }
                }
                ActionDimType::Unknown => {
                    panic!("出现了不该出现的 ActionDimType::Unknown");
                }
                _ => {
                    result.push(action_dim.clone());
                }
            }
        }
        result
    }

    /// 获取扁平化的观察长度
    pub fn get_flatten_observation_len(&self) -> usize {
        let mut total_len = 0;
        for obs_prop in &self.obs_prop_vec {
            if obs_prop.shape_vec.is_empty() {
                total_len += 1;
            } else {
                let mut len = 1;
                for dim in &obs_prop.shape_vec {
                    len *= *dim as usize;
                }
                total_len += len;
            }
        }
        total_len
    }

    // ========================================================================
    // 内部方法
    // ========================================================================

    /// 检测观察类型
    fn check_obs_type(&mut self) {
        if self.obs_prop_vec.len() != 1 {
            // 多元素 obs（如 Platform）视为向量
            self.obs_type = ObsType::Vector;
            return;
        }

        let shape = &self.obs_prop_vec[0].shape_vec;
        self.obs_type = match shape[..] {
            [_, _] => ObsType::NoChannel,
            [c, _, _] if c <= 4 => ObsType::ChannelFirst,
            [_, _, c] if c <= 4 => ObsType::ChannelLast,
            _ => ObsType::Vector,
        };
    }

    /// 从 Python 对象获取观察向量
    fn get_obs_vec_from_python(&self, obs_py: &Bound<'py, PyAny>) -> Vec<Vec<f32>> {
        let mut obs_vec = Vec::new();

        if self.obs_prop_vec.len() == 1 {
            // 常规情况：单个 obs
            let obs = if let Ok(flattened) = obs_py.getattr("flatten") {
                if let Ok(flat_obs) = flattened.call0() {
                    flat_obs
                        .extract::<Vec<f32>>()
                        .unwrap_or_else(|_| vec![0.0])
                } else {
                    obs_py.extract::<Vec<f32>>().unwrap_or_else(|_| vec![0.0])
                }
            } else {
                obs_py.extract::<Vec<f32>>().unwrap_or_else(|_| vec![0.0])
            };
            obs_vec.push(obs);
        } else {
            // Tuple 情况（如 Platform）
            for (i, _obs_prop) in self.obs_prop_vec.iter().enumerate() {
                let cur_obs = obs_py.get_item(i).expect("获取 tuple obs 元素失败");
                let obs = if let Ok(v) = cur_obs.extract::<f32>() {
                    vec![v]
                } else {
                    cur_obs.extract::<Vec<f32>>().unwrap_or_else(|_| vec![0.0])
                };
                obs_vec.push(obs);
            }
        }

        obs_vec
    }

    /// 将 Rust 动作转换为 Python 对象
    fn convert_action_to_python(&self, action: &[f32]) -> Py<PyAny> {
        match self.action_type {
            ActionType::SingleDiscrete => {
                // 离散动作：返回整数
                let action_int = action.first().map(|&a| a as i64).unwrap_or(0);
                action_int.into_py_any(self.py).unwrap()
            }
            ActionType::Continuous => {
                // 连续动作：返回浮点数组
                action.to_vec().into_py_any(self.py).unwrap()
            }
            ActionType::Mix => {
                // 混合动作：递归构建
                let action_py_vec = self.build_mixed_action_py(
                    self.action_prop_vec[0].sub_action_dim_op.as_deref(),
                    action,
                    0,
                );
                action_py_vec.into_py_any(self.py).unwrap()
            }
            ActionType::Unknown => {
                panic!("无法处理 Unknown 类型的动作");
            }
        }
    }

    /// 构建混合动作的 Python 对象
    fn build_mixed_action_py(
        &self,
        sub_action_space_op: Option<&[ActionDim]>,
        action: &[f32],
        start_index: usize,
    ) -> Vec<Py<PyAny>> {
        let sub_action_space = sub_action_space_op.expect("混合动作缺少子动作空间");
        let mut result = Vec::new();

        for (pos, action_dim) in sub_action_space.iter().enumerate() {
            let idx = start_index + pos;
            let action_py: Py<PyAny> = match action_dim.action_type {
                ActionDimType::Int64 => {
                    let v = action.get(idx).map(|&a| a as i64).unwrap_or(0);
                    v.into_py_any(self.py).unwrap()
                }
                ActionDimType::Float64 => {
                    let v = action.get(idx).copied().unwrap_or(0.0);
                    v.into_py_any(self.py).unwrap()
                }
                ActionDimType::NumpyFloat64Scalar => {
                    let v = action.get(idx).copied().unwrap_or(0.0);
                    array![v].to_pyarray(self.py).unbind().into_any()
                }
                ActionDimType::NumpyFloat64List => {
                    let sub_actions =
                        self.build_mixed_action_py(action_dim.sub_action_dim_op.as_deref(), action, idx);
                    let floats: Vec<f32> = sub_actions
                        .iter()
                        .filter_map(|a| a.extract::<f32>(self.py).ok())
                        .collect();
                    Array1::from_vec(floats)
                        .to_pyarray(self.py)
                        .unbind()
                        .into_any()
                }
                ActionDimType::Tuple => {
                    let sub_actions =
                        self.build_mixed_action_py(action_dim.sub_action_dim_op.as_deref(), action, idx);
                    sub_actions.into_py_any(self.py).unwrap()
                }
                ActionDimType::Unknown => {
                    panic!("出现了不该出现的 ActionDimType::Unknown");
                }
            };
            result.push(action_py);
        }

        result
    }

    /// 将 Python 动作转换为 Rust 向量
    fn convert_python_action_to_rust(&self, action_py: &Bound<'py, PyAny>) -> Vec<f32> {
        match self.action_type {
            ActionType::SingleDiscrete => {
                let v: i64 = action_py.extract().unwrap_or(0);
                vec![v as f32]
            }
            ActionType::Continuous => {
                action_py.extract::<Vec<f32>>().unwrap_or_else(|_| {
                    // 尝试单个浮点数
                    if let Ok(v) = action_py.extract::<f32>() {
                        vec![v]
                    } else {
                        vec![0.0]
                    }
                })
            }
            ActionType::Mix => {
                // 递归展开混合动作
                self.flatten_python_action(action_py)
            }
            ActionType::Unknown => vec![0.0],
        }
    }

    /// 扁平化 Python 混合动作
    fn flatten_python_action(&self, action_py: &Bound<'py, PyAny>) -> Vec<f32> {
        let mut result = Vec::new();

        if let Ok(v) = action_py.extract::<f32>() {
            result.push(v);
        } else if let Ok(v) = action_py.extract::<i64>() {
            result.push(v as f32);
        } else if let Ok(vec) = action_py.extract::<Vec<f32>>() {
            result.extend(vec);
        } else if let Ok(len) = action_py.len() {
            for i in 0..len {
                if let Ok(item) = action_py.get_item(i) {
                    result.extend(self.flatten_python_action(&item));
                }
            }
        }

        result
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 初始化观察空间属性
fn init_obs_prop(obs_space: &Bound<'_, PyAny>) -> Vec<ObsDim> {
    let mut obs_prop_vec = Vec::new();

    let shape = obs_space.getattr("shape").expect("获取 obs shape 失败");

    match shape.extract::<Vec<i64>>() {
        Ok(shape_vec) => {
            obs_prop_vec.push(ObsDim { shape_vec });
        }
        Err(_) => {
            // Tuple 类型的观察空间
            if let Ok(tuple_size) = obs_space.len() {
                for i in 0..tuple_size {
                    let cur_obs = obs_space.get_item(i).expect("获取 tuple obs 元素失败");
                    let cur_shape: Vec<i64> = cur_obs
                        .getattr("shape")
                        .expect("获取子 obs shape 失败")
                        .extract()
                        .unwrap_or_default();
                    obs_prop_vec.push(ObsDim {
                        shape_vec: cur_shape,
                    });
                }
            }
        }
    }

    obs_prop_vec
}

/// 初始化动作空间属性
fn init_act_prop(action_space: &Bound<'_, PyAny>, check_type: bool) -> Vec<ActionDim> {
    let mut action_prop_vec = Vec::new();

    let type_name = action_space
        .get_type()
        .name()
        .expect("获取动作空间类型名失败")
        .to_string();

    match type_name.as_str() {
        "Discrete" => {
            // 单维离散动作
            let n: i64 = action_space
                .getattr("n")
                .expect("获取 Discrete.n 失败")
                .extract()
                .expect("解析 n 失败");

            action_prop_vec.push(ActionDim {
                high_v_op: Some((n - 1) as f32),
                low_v_op: Some(0.0),
                action_type: ActionDimType::default(),
                sub_action_dim_op: None,
            });
        }
        "Box" => {
            // 连续动作
            let shape: Vec<usize> = action_space
                .getattr("shape")
                .expect("获取 Box.shape 失败")
                .extract()
                .expect("解析 shape 失败");

            let action_num = shape.first().copied().unwrap_or(1);

            let high_vec: Vec<f32> = action_space
                .getattr("high")
                .expect("获取 Box.high 失败")
                .extract()
                .expect("解析 high 失败");

            let low_vec: Vec<f32> = action_space
                .getattr("low")
                .expect("获取 Box.low 失败")
                .extract()
                .expect("解析 low 失败");

            if check_type || action_num == 1 {
                for i in 0..action_num {
                    action_prop_vec.push(ActionDim {
                        high_v_op: Some(high_vec[i]),
                        low_v_op: Some(low_vec[i]),
                        action_type: ActionDimType::default(),
                        sub_action_dim_op: None,
                    });
                }
            } else {
                // 内层多元素 Box -> NumpyFloat64List
                let mut sub_dims = Vec::new();
                for i in 0..action_num {
                    sub_dims.push(ActionDim {
                        high_v_op: Some(high_vec[i]),
                        low_v_op: Some(low_vec[i]),
                        action_type: ActionDimType::default(),
                        sub_action_dim_op: None,
                    });
                }
                action_prop_vec.push(ActionDim {
                    high_v_op: None,
                    low_v_op: None,
                    action_type: ActionDimType::NumpyFloat64List,
                    sub_action_dim_op: Some(sub_dims),
                });
            }
        }
        "Tuple" => {
            // Tuple 类型（混合动作）
            let tuple_size = action_space.len().expect("获取 Tuple 长度失败");
            let mut sub_dims = Vec::new();

            for i in 0..tuple_size {
                let sub_space = action_space.get_item(i).expect("获取 Tuple 元素失败");
                let mut cur_props = init_act_prop(&sub_space, false);
                sub_dims.append(&mut cur_props);
            }

            action_prop_vec.push(ActionDim {
                high_v_op: None,
                low_v_op: None,
                action_type: ActionDimType::Tuple,
                sub_action_dim_op: Some(sub_dims),
            });
        }
        _ => {
            panic!("不支持的动作空间类型: {}", type_name);
        }
    }

    if !check_type {
        return action_prop_vec;
    }

    // 通过采样确定具体的动作类型
    let sampled_action = action_space
        .call_method0("sample")
        .expect("采样动作失败");
    get_action_type_by_sample(&sampled_action, &mut action_prop_vec);

    action_prop_vec
}

/// 通过采样的动作确定动作类型
fn get_action_type_by_sample(action_py: &Bound<'_, PyAny>, action_dims: &mut [ActionDim]) {
    let dims_len = action_dims.len();

    for (pos, action_dim) in action_dims.iter_mut().enumerate() {
        let cur_action = if dims_len > 1 {
            action_py
                .get_item(pos)
                .expect("获取采样动作元素失败")
        } else {
            action_py.clone()
        };

        let type_name = cur_action
            .get_type()
            .name()
            .expect("获取动作类型名失败")
            .to_string();

        match type_name.to_lowercase().as_str() {
            "int" | "int32" | "int64" => {
                action_dim.action_type = ActionDimType::Int64;
            }
            "float" | "float32" | "float64" => {
                action_dim.action_type = ActionDimType::Float64;
            }
            "ndarray" => {
                let len = cur_action.len().unwrap_or(1);
                if len == 1 {
                    action_dim.action_type = ActionDimType::NumpyFloat64Scalar;
                } else {
                    action_dim.action_type = ActionDimType::NumpyFloat64List;
                    if let Some(sub_dims) = &mut action_dim.sub_action_dim_op {
                        get_action_type_by_sample(&cur_action, sub_dims);
                    }
                }
            }
            "tuple" => {
                action_dim.action_type = ActionDimType::Tuple;
                if let Some(sub_dims) = &mut action_dim.sub_action_dim_op {
                    get_action_type_by_sample(&cur_action, sub_dims);
                }
            }
            _ => {
                // 尝试更多类型判断
                if cur_action.extract::<i64>().is_ok() {
                    action_dim.action_type = ActionDimType::Int64;
                } else if cur_action.extract::<f32>().is_ok() {
                    action_dim.action_type = ActionDimType::Float64;
                }
            }
        }
    }
}

/// 计算动作总数量（最细粒度）
fn calc_action_real_num(action_dims: &[ActionDim]) -> usize {
    let mut count = 0;
    for action_dim in action_dims {
        match action_dim.action_type {
            ActionDimType::Tuple | ActionDimType::NumpyFloat64List => {
                // Tuple 和 NumpyFloat64List 需要递归计算子元素
                if let Some(sub_dims) = &action_dim.sub_action_dim_op {
                    count += calc_action_real_num(sub_dims);
                }
            }
            ActionDimType::Unknown => {
                // 未知类型按 1 计算
                count += 1;
            }
            _ => {
                count += 1;
            }
        }
    }
    count
}

/// 检查动作空间的整体类型
fn check_action_type(action_dims: &[ActionDim]) -> ActionType {
    // 检查是否有 Tuple 类型
    if action_dims
        .iter()
        .any(|d| d.action_type == ActionDimType::Tuple)
    {
        return ActionType::Mix;
    }

    // 检查是否为单维离散
    if action_dims.len() == 1 && action_dims[0].action_type == ActionDimType::Int64 {
        return ActionType::SingleDiscrete;
    }

    // 默认为连续
    ActionType::Continuous
}
