//! MyZero 的稳定输入 / 动作契约。
//!
//! 这里描述“数据代表什么”，不改变计算 dtype：环境观测、模型输入与 latent
//! 进入 [`Tensor`](crate::tensor::Tensor) 后仍统一使用 `f32`。

use crate::rl::mcts::ActionPayload;

/// 模型入口观测的语义与形状。
///
/// `Flat` / `Image` / `Board` 保留历史含义；其余变体用于验证通用输入接缝。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationSchema {
    /// 普通稠密向量。
    Flat(usize),
    /// 历史方图契约；保留 `side` 字段以兼容已有调用。
    Image { channels: usize, side: usize },
    /// 矩形图像，展平顺序为 CHW。
    ImageRect {
        channels: usize,
        height: usize,
        width: usize,
    },
    /// 小棋盘，使用 stride-1 CNN。
    Board { channels: usize, side: usize },
    /// 已预处理图像 + 稠密辅助特征，输入按 `[image_flat, aux]` 拼接。
    ImageDense {
        channels: usize,
        height: usize,
        width: usize,
        aux_dim: usize,
    },
    /// 固定长度 token IDs。ID 以可精确表示整数的 f32 进入计算图。
    Tokens {
        length: usize,
        vocab_size: usize,
        embed_dim: usize,
        pad_id: usize,
    },
}

impl ObservationSchema {
    /// 模型入口的展平 f32 元素数。
    pub const fn dim(self) -> usize {
        match self {
            Self::Flat(dim) => dim,
            Self::Image { channels, side } | Self::Board { channels, side } => {
                channels * side * side
            }
            Self::ImageRect {
                channels,
                height,
                width,
            } => channels * height * width,
            Self::ImageDense {
                channels,
                height,
                width,
                aux_dim,
            } => channels * height * width + aux_dim,
            // 模型输入 = token IDs + 同长 padding mask。
            Self::Tokens { length, .. } => length * 2,
        }
    }

    /// 图像区域的元素数；非图像 schema 返回 `None`。
    pub const fn image_dim(self) -> Option<usize> {
        match self {
            Self::Image { channels, side } | Self::Board { channels, side } => {
                Some(channels * side * side)
            }
            Self::ImageRect {
                channels,
                height,
                width,
            }
            | Self::ImageDense {
                channels,
                height,
                width,
                ..
            } => Some(channels * height * width),
            Self::Flat(_) | Self::Tokens { .. } => None,
        }
    }
}

/// 历史名称兼容层；新代码优先使用 [`ObservationSchema`]。
pub type ObsSpec = ObservationSchema;

/// 有限动作空间在模型中的策略布局。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyLayout {
    /// 单一 categorical head，宽度等于联合动作数。
    Categorical { actions: usize },
    /// 多个 categorical factor；模型输出按各 factor 顺序拼接。
    Factorized { factors: Vec<usize> },
}

impl PolicyLayout {
    pub fn output_dim(&self) -> usize {
        match self {
            Self::Categorical { actions } => *actions,
            Self::Factorized { factors } => factors.iter().sum(),
        }
    }

    pub fn factors(&self) -> Vec<usize> {
        match self {
            Self::Categorical { actions } => vec![*actions],
            Self::Factorized { factors } => factors.clone(),
        }
    }
}

/// 当前可生产执行的有限动作 schema。
///
/// 多维连续在本阶段采用 Sampled MuZero 的 categorical bins；真正无界的连续
/// proposal 留给后续按真实环境需求扩展。
#[derive(Debug, Clone, PartialEq)]
pub enum ActionSchema {
    Discrete {
        actions: usize,
    },
    MultiDiscrete {
        factors: Vec<usize>,
    },
    ContinuousBins {
        ranges: Vec<(f32, f32)>,
        buckets: usize,
    },
    HybridBins {
        discrete: Vec<usize>,
        continuous: Vec<(f32, f32)>,
        buckets: usize,
    },
}

impl ActionSchema {
    pub fn discrete(actions: usize) -> Self {
        assert!(actions > 0, "ActionSchema: 离散动作数必须 > 0");
        Self::Discrete { actions }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Discrete { actions } if *actions == 0 => {
                Err("离散动作数必须 > 0".into())
            }
            Self::MultiDiscrete { factors } if factors.is_empty() || factors.contains(&0) =>
            {
                Err("MultiDiscrete factors 必须非空且每维 > 0".into())
            }
            Self::ContinuousBins { ranges, buckets }
                if ranges.is_empty()
                    || *buckets == 0
                    || ranges
                        .iter()
                        .any(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo > hi) =>
            {
                Err("ContinuousBins 需要非空有限 ranges、low≤high 且 buckets>0".into())
            }
            Self::HybridBins {
                discrete,
                continuous,
                buckets,
            } if discrete.len() != 1
                || discrete[0] == 0
                || continuous.is_empty()
                || *buckets == 0
                || continuous
                    .iter()
                    .any(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo > hi) =>
            {
                Err(
                    "HybridBins 当前要求恰好一个非零离散 factor、非空有限 continuous ranges 与 buckets>0"
                        .into(),
                )
            }
            _ => Ok(()),
        }
    }

    fn assert_valid(&self) {
        if let Err(message) = self.validate() {
            panic!("ActionSchema: {message}");
        }
    }

    /// 各 factor 的 categorical 宽度。
    pub fn factors(&self) -> Vec<usize> {
        self.assert_valid();
        match self {
            Self::Discrete { actions } => vec![*actions],
            Self::MultiDiscrete { factors } => factors.clone(),
            Self::ContinuousBins { ranges, buckets } => vec![*buckets; ranges.len()],
            Self::HybridBins {
                discrete,
                continuous,
                buckets,
            } => {
                let mut out = discrete.clone();
                out.extend(std::iter::repeat_n(*buckets, continuous.len()));
                out
            }
        }
    }

    /// 联合动作总数。溢出时 panic，避免静默构造错误 action ID。
    pub fn action_count(&self) -> usize {
        self.assert_valid();
        self.factors()
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .expect("ActionSchema: 联合动作数溢出 usize")
    }

    pub fn policy_layout(&self) -> PolicyLayout {
        self.assert_valid();
        match self {
            Self::Discrete { actions } => PolicyLayout::Categorical { actions: *actions },
            Self::MultiDiscrete { factors } if factors.len() == 1 => PolicyLayout::Categorical {
                actions: factors[0],
            },
            Self::ContinuousBins { ranges, buckets } if ranges.len() == 1 => {
                PolicyLayout::Categorical { actions: *buckets }
            }
            _ => PolicyLayout::Factorized {
                factors: self.factors(),
            },
        }
    }

    /// mixed-radix 解码：稳定联合 `ActionId` → 各 factor 下标。
    pub fn decode_joint(&self, action_id: usize) -> Vec<usize> {
        let factors = self.factors();
        assert!(
            action_id < self.action_count(),
            "ActionSchema: action_id {action_id} 越界"
        );
        let mut rem = action_id;
        let mut digits = vec![0usize; factors.len()];
        for (digit, radix) in digits.iter_mut().rev().zip(factors.iter().rev()) {
            *digit = rem % *radix;
            rem /= *radix;
        }
        digits
    }

    /// 各 factor 下标 → 稳定联合 `ActionId`。
    pub fn encode_joint(&self, digits: &[usize]) -> usize {
        let factors = self.factors();
        assert_eq!(
            digits.len(),
            factors.len(),
            "ActionSchema: factor 数量不匹配"
        );
        digits
            .iter()
            .zip(factors)
            .fold(0usize, |acc, (&digit, radix)| {
                assert!(
                    digit < radix,
                    "ActionSchema: factor 下标 {digit} 越界 {radix}"
                );
                acc * radix + digit
            })
    }

    /// 联合动作的真实执行载荷。
    pub fn payload(&self, action_id: usize) -> ActionPayload {
        let digits = self.decode_joint(action_id);
        match self {
            Self::Discrete { .. } => ActionPayload::Discrete(action_id),
            Self::MultiDiscrete { .. } => ActionPayload::MultiDiscrete(digits),
            Self::ContinuousBins {
                ranges, buckets, ..
            } => ActionPayload::Continuous(
                ranges
                    .iter()
                    .zip(digits)
                    .map(|(&(lo, hi), idx)| bin_center(idx, lo, hi, *buckets))
                    .collect(),
            ),
            Self::HybridBins {
                discrete,
                continuous,
                buckets,
            } => {
                let split = discrete.len();
                let discrete_values = digits[..split].to_vec();
                assert_eq!(
                    discrete_values.len(),
                    1,
                    "ActionSchema: 当前 Hybrid payload 只支持一个离散 factor"
                );
                let continuous_values = continuous
                    .iter()
                    .zip(digits[split..].iter().copied())
                    .map(|(&(lo, hi), idx)| bin_center(idx, lo, hi, *buckets))
                    .collect();
                ActionPayload::Hybrid {
                    discrete: discrete_values[0],
                    continuous: continuous_values,
                }
            }
        }
    }

    /// 交给 `GymEnv::step` 的扁平动作向量。
    pub fn to_env(&self, action_id: usize) -> Vec<f32> {
        match self.payload(action_id) {
            ActionPayload::Discrete(idx) => vec![idx as f32],
            ActionPayload::MultiDiscrete(values) => values.into_iter().map(|v| v as f32).collect(),
            ActionPayload::Continuous(values) => values,
            ActionPayload::Hybrid {
                discrete,
                continuous,
            } => std::iter::once(discrete as f32).chain(continuous).collect(),
        }
    }

    /// 把联合动作 visit target 投影为模型 policy head 的 target。
    ///
    /// categorical 原样返回；factorized 返回各维边缘分布的顺序拼接。
    pub fn encode_policy_target(&self, joint: &[f32]) -> Vec<f32> {
        assert_eq!(
            joint.len(),
            self.action_count(),
            "ActionSchema: policy target 长度不匹配"
        );
        match self.policy_layout() {
            PolicyLayout::Categorical { .. } => joint.to_vec(),
            PolicyLayout::Factorized { factors } => {
                let mut marginals: Vec<Vec<f32>> = factors.iter().map(|&n| vec![0.0; n]).collect();
                for (action_id, &weight) in joint.iter().enumerate() {
                    for (factor, digit) in self.decode_joint(action_id).into_iter().enumerate() {
                        marginals[factor][digit] += weight;
                    }
                }
                marginals.into_iter().flatten().collect()
            }
        }
    }

    /// factor 概率的顺序拼接 → 联合动作概率。
    pub fn joint_priors(&self, factor_probs: &[f32]) -> Vec<f32> {
        let layout = self.policy_layout();
        match layout {
            PolicyLayout::Categorical { actions } => {
                assert_eq!(factor_probs.len(), actions);
                factor_probs.to_vec()
            }
            PolicyLayout::Factorized { factors } => {
                assert_eq!(factor_probs.len(), factors.iter().sum::<usize>());
                let mut offsets = Vec::with_capacity(factors.len());
                let mut start = 0;
                for &n in &factors {
                    offsets.push((start, start + n));
                    start += n;
                }
                (0..self.action_count())
                    .map(|action_id| {
                        self.decode_joint(action_id)
                            .into_iter()
                            .enumerate()
                            .map(|(factor, digit)| {
                                let (lo, _) = offsets[factor];
                                factor_probs[lo + digit]
                            })
                            .product()
                    })
                    .collect()
            }
        }
    }
}

/// 动作 schema 对模型与环境的稳定编解码契约。
pub trait ActionCodec {
    fn action_count(&self) -> usize;
    fn payload(&self, action_id: usize) -> ActionPayload;
    fn to_env(&self, action_id: usize) -> Vec<f32>;
}

impl ActionCodec for ActionSchema {
    fn action_count(&self) -> usize {
        ActionSchema::action_count(self)
    }

    fn payload(&self, action_id: usize) -> ActionPayload {
        ActionSchema::payload(self, action_id)
    }

    fn to_env(&self, action_id: usize) -> Vec<f32> {
        ActionSchema::to_env(self, action_id)
    }
}

fn bin_center(idx: usize, lo: f32, hi: f32, buckets: usize) -> f32 {
    assert!(buckets > 0, "ActionSchema: buckets 必须 > 0");
    assert!(idx < buckets, "ActionSchema: bucket 下标越界");
    if buckets == 1 {
        return 0.5 * (lo + hi);
    }
    let width = (hi - lo) / buckets as f32;
    (idx as f32 + 0.5).mul_add(width, lo)
}
