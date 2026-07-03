//! 存储态观测（[`StoredObs`]）：replay buffer 内 obs 的存储编码。
//!
//! # 与 f32-only 契约的关系
//!
//! 本项目计算层（`Tensor` / autograd / 算子）**只有 f32** 一种 dtype，该契约不变：
//! 任何数据在进入 `Tensor` 之前都已反量化回 f32。`U8` 变体是数据的**休眠编码**
//! （类比磁盘上的 PNG 本就是 u8，`ToTensor` 后才是 f32），不是新的计算 dtype。
//!
//! # 为什么需要它
//!
//! 图像 obs 的像素源头是 0–255 整数，以 f32 存 buffer 是 4× 内存/带宽膨胀
//! （84² 单帧 28KB → 7KB；Atari 口径 buffer ~800MB → ~200MB）。量化决策由
//! obs 管线在**源头**显式做出（依据 env 观察空间声明，不做数据猜测），对
//! `MyZero::new(...)` 的最终用户完全透明。

/// 存储态观测：`F32` 直通（向量 obs，零开销）/ `U8` 像素量化帧（图像 obs，4× 省内存）。
///
/// # 量化语义（`U8`）
///
/// - **写入**：像素域值 `v ∈ [0, 255]` 四舍五入为 u8（[`quantize_pixels`](Self::quantize_pixels)）
/// - **读取**：反量化为 `[0, 1]` f32（`u8 as f32 / 255.0`）
/// - **往返误差**：≤ 0.5/255（半个量化步长；对 RL 训练可忽略，DQN→MuZero 系标准做法）
///
/// 自定义算法构造 [`SelfPlayStep`](super::SelfPlayStep) 时用 `From<Vec<f32>>`
/// （`obs: my_vec.into()`）即可，默认心智模型仍是 f32 向量；`U8` 仅由图像管线主动构造。
#[derive(Debug, Clone, PartialEq)]
pub enum StoredObs {
    /// f32 向量直通（语义与旧 `Vec<f32>` 字段逐 bit 一致）
    F32(Vec<f32>),
    /// 像素量化帧：0–255 域 round 为 u8，读取反量化为 `[0, 1]` f32
    U8(Vec<u8>),
}

impl StoredObs {
    /// 像素域（0–255）f32 量化为 `U8` 帧（round + clamp）。
    pub fn quantize_pixels(pixels: &[f32]) -> Self {
        Self::U8(
            pixels
                .iter()
                .map(|&v| v.round().clamp(0.0, 255.0) as u8)
                .collect(),
        )
    }

    /// 元素个数（与反量化后 f32 向量长度一致）。
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::U8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 借出 f32 切片（仅 `F32` 变体；量化帧无法零拷贝借出）。
    ///
    /// # Panics
    /// 对 `U8` 变体 panic——调用方属于「仅向量 obs」路径（如 reanalyze），
    /// 图像模式应走堆叠组装（`assemble_stacked_obs`）。
    pub fn as_f32(&self) -> &[f32] {
        match self {
            Self::F32(v) => v,
            Self::U8(_) => {
                panic!("StoredObs::as_f32: 量化帧（U8）不支持借出 f32 切片，该路径仅限向量 obs")
            }
        }
    }

    /// 反量化追加到 f32 缓冲（堆叠组装热路径，避免中间分配）。
    pub fn append_f32_into(&self, out: &mut Vec<f32>) {
        match self {
            Self::F32(v) => out.extend_from_slice(v),
            Self::U8(v) => out.extend(v.iter().map(|&b| f32::from(b) / 255.0)),
        }
    }

    /// 反量化为独立 f32 向量（`F32` 变体等价于 clone）。
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len());
        self.append_f32_into(&mut out);
        out
    }
}

impl From<Vec<f32>> for StoredObs {
    fn from(v: Vec<f32>) -> Self {
        Self::F32(v)
    }
}
