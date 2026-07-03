//! 图像 obs 预处理管线（v0.26 Phase 1，收口规划 §2 / Phase 1 计划 §S1）。
//!
//! **管线**（Atari 标准口径）：HWC/CHW f32 原始帧（0–255）→ BT.601 灰度 →
//! 双线性降采样至 [`OUT_SIDE`]² → **u8 量化存储**（[`StoredObs::U8`]，round，4× 省内存）
//! → 读取时反量化 `[0,1]` → 最近 [`STACK`] 帧堆叠为 CHW。
//!
//! **内存纪律**（收口规划 §2 图像线）：replay buffer 只存**处理后量化单帧**
//! （[`frame_len`] 个 u8），堆叠在两处按需组装（两处吃**同一份量化语义**，数值自洽）：
//! - acting 期：[`ImagePipe`] 维护 episode 内滑窗（[`ImagePipe::stacked`]）；
//! - 训练期：[`assemble_stacked_obs`] 从 `SelfPlayGame.steps` 就地组装
//!   （episode 起点用首帧前向填充，与 acting 期 reset 语义逐 bit 一致）。
//!
//! **量化口径**：resize 输出（0–255 像素域）round 为 u8，反量化 `u8/255`；
//! 相对旧「f32 直存 `v/255`」每像素误差 ≤ 0.5/255（DQN→MuZero 系标准做法）。
//! 计算层 f32-only 契约不变，见 [`StoredObs`] 模块文档。

use super::network::ObsSpec;
use crate::rl::{GymEnv, ObsType, StoredObs};
use std::collections::VecDeque;

/// 预处理输出边长（Atari 社区标准 84×84）
pub const OUT_SIDE: usize = 84;
/// 帧堆叠数（Atari 社区标准 4）
pub const STACK: usize = 4;

/// 处理后单帧长度
pub const fn frame_len() -> usize {
    OUT_SIDE * OUT_SIDE
}

/// 模型输入维度（堆叠后展平）
pub const fn stacked_obs_dim() -> usize {
    STACK * frame_len()
}

/// BT.601 灰度系数（与 `image::to_luma8` / `OpenCV` 一致）
const LUMA_R: f32 = 0.299;
const LUMA_G: f32 = 0.587;
const LUMA_B: f32 = 0.114;

/// 图像 obs 管线：单帧预处理 + episode 内帧堆叠滑窗。
///
/// 由 [`from_env`](Self::from_env) 按 env 观察空间事实构造（非图像 env 返回 `None`）。
pub struct ImagePipe {
    in_h: usize,
    in_w: usize,
    in_c: usize,
    channel_first: bool,
    /// 首帧像素域校验是否已通过（仅在第一次 reset 时检查一次）
    domain_checked: bool,
    /// 最近 STACK 个处理后量化单帧（老 → 新；与 buffer 存储同编码）
    frames: VecDeque<StoredObs>,
}

impl ImagePipe {
    /// 从 env 观察空间事实构造；非图像 obs（普通向量）返回 `None`。
    pub fn from_env(env: &GymEnv) -> Option<Self> {
        let shape = &env.get_obs_prop().first()?.shape_vec;
        let (in_h, in_w, in_c, channel_first) = match env.get_obs_type() {
            ObsType::NoChannel => (shape[0] as usize, shape[1] as usize, 1, false),
            ObsType::ChannelFirst => (
                shape[1] as usize,
                shape[2] as usize,
                shape[0] as usize,
                true,
            ),
            ObsType::ChannelLast => (
                shape[0] as usize,
                shape[1] as usize,
                shape[2] as usize,
                false,
            ),
            ObsType::Vector => return None,
        };
        Some(Self {
            in_h,
            in_w,
            in_c,
            channel_first,
            domain_checked: false,
            frames: VecDeque::with_capacity(STACK),
        })
    }

    /// 首帧像素域校验：管线假定原始值为 **0–255 像素域**（u8 量化前提）。
    ///
    /// 若 env 已把图像归一化到 `[0,1]`（部分非 ALE 包装器会这么做），量化会把
    /// 所有像素压成 0/1 再除以 255，输入尺度静默崩坏——在此显式 panic 拦截。
    ///
    /// # Panics
    /// - 值域超出 `[0, 255]`（未知量纲）；
    /// - 首帧最大值 ≤ 1.0（看似已归一化；真实像素首帧全黑的概率可忽略）。
    pub(crate) fn validate_pixel_domain(raw: &[f32]) {
        let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in raw {
            min = min.min(v);
            max = max.max(v);
        }
        assert!(
            min >= 0.0 && max <= 255.0,
            "ImagePipe: 图像 obs 值域 [{min}, {max}] 超出 0–255 像素域，无法按 u8 量化存储"
        );
        assert!(
            max > 1.0,
            "ImagePipe: 图像 obs 首帧最大值 {max} ≤ 1.0，疑似已归一化到 [0,1]；\
             当前管线假定 0–255 像素域（如 ALE 原始帧），请去掉 env 侧归一化包装"
        );
    }

    /// episode 开始：处理首帧并将滑窗填满该帧（标准前向填充），返回处理后量化单帧。
    pub fn reset(&mut self, raw: &[f32]) -> StoredObs {
        if !self.domain_checked {
            Self::validate_pixel_domain(raw);
            self.domain_checked = true;
        }
        let frame = self.process_frame(raw);
        self.frames.clear();
        for _ in 0..STACK {
            self.frames.push_back(frame.clone());
        }
        frame
    }

    /// episode 中：处理新帧入滑窗（挤出最老帧），返回处理后量化单帧。
    pub fn push(&mut self, raw: &[f32]) -> StoredObs {
        let frame = self.process_frame(raw);
        if self.frames.len() >= STACK {
            self.frames.pop_front();
        }
        self.frames.push_back(frame.clone());
        frame
    }

    /// 当前堆叠观测（反量化 `[0,1]` f32，CHW 展平，老 → 新），供 MCTS 搜索用。
    ///
    /// # Panics
    /// 未 [`reset`](Self::reset) 先调用时 panic。
    pub fn stacked(&self) -> Vec<f32> {
        assert_eq!(
            self.frames.len(),
            STACK,
            "ImagePipe::stacked: 须先 reset 填满滑窗"
        );
        let mut out = Vec::with_capacity(stacked_obs_dim());
        for f in &self.frames {
            f.append_f32_into(&mut out);
        }
        out
    }

    /// 单帧预处理：灰度 → 双线性缩放 → u8 量化（0–255 域 round）。
    ///
    /// 输入为 env 原始展平 f32（0–255，HWC 或 CHW 排布），输出 [`frame_len`] 长度
    /// 量化帧；`[0,1]` 归一在读取（[`stacked`](Self::stacked) / `assemble_stacked_obs`）
    /// 时反量化完成。
    pub fn process_frame(&self, raw: &[f32]) -> StoredObs {
        debug_assert_eq!(
            raw.len(),
            self.in_h * self.in_w * self.in_c,
            "ImagePipe::process_frame: 输入长度与声明形状不符"
        );
        let gray = self.to_gray(raw);
        let resized = bilinear_resize(&gray, self.in_h, self.in_w, OUT_SIDE, OUT_SIDE);
        StoredObs::quantize_pixels(&resized)
    }

    /// 灰度化（BT.601）；单通道原样返回。输出 [h*w]（行主序）。
    fn to_gray(&self, raw: &[f32]) -> Vec<f32> {
        let plane = self.in_h * self.in_w;
        if self.in_c == 1 {
            return raw.to_vec();
        }
        let mut gray = vec![0.0f32; plane];
        if self.channel_first {
            // CHW：plane 偏移取 R/G/B
            let (r, g, b) = (
                &raw[..plane],
                &raw[plane..2 * plane],
                &raw[2 * plane..3 * plane],
            );
            for (i, px) in gray.iter_mut().enumerate() {
                *px = LUMA_R.mul_add(r[i], LUMA_G * g[i]) + LUMA_B * b[i];
            }
        } else {
            // HWC：逐像素连续取 RGB
            for (px, rgb) in gray.iter_mut().zip(raw.chunks_exact(self.in_c)) {
                *px = LUMA_R.mul_add(rgb[0], LUMA_G * rgb[1]) + LUMA_B * rgb[2];
            }
        }
        gray
    }
}

/// 双线性缩放（行主序单通道），align-corners=false（像素中心对齐，OpenCV/PyTorch 默认）。
pub(crate) fn bilinear_resize(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    debug_assert_eq!(src.len(), src_h * src_w);
    let mut out = vec![0.0f32; dst_h * dst_w];
    let scale_y = src_h as f32 / dst_h as f32;
    let scale_x = src_w as f32 / dst_w as f32;
    for dy in 0..dst_h {
        // 像素中心映射：src_y = (dy + 0.5)·scale − 0.5
        let sy = (dy as f32 + 0.5).mul_add(scale_y, -0.5).max(0.0);
        let y0 = (sy as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = sy - y0 as f32;
        for dx in 0..dst_w {
            let sx = (dx as f32 + 0.5).mul_add(scale_x, -0.5).max(0.0);
            let x0 = (sx as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let fx = sx - x0 as f32;
            let p00 = src[y0 * src_w + x0];
            let p01 = src[y0 * src_w + x1];
            let p10 = src[y1 * src_w + x0];
            let p11 = src[y1 * src_w + x1];
            let top = (p01 - p00).mul_add(fx, p00);
            let bot = (p11 - p10).mul_add(fx, p10);
            out[dy * dst_w + dx] = (bot - top).mul_add(fy, top);
        }
    }
    out
}

/// 训练期堆叠组装的**语义参照实现**（现仅测试使用）。
///
/// 生产路径已融合进 batch 组装（`ObsSource::Stacked::append_into` 直接写入最终
/// flat buffer，零中间 Vec），本函数保留为逐 bit 等价的守门基准：
/// episode 起点（`t < STACK-1`）用 `saturating_sub` 前向填充首帧，
/// 与 acting 期 [`ImagePipe::reset`] 填满滑窗的语义逐 bit 一致
/// （共用 [`StoredObs::append_f32_into`] 的反量化口径）。
#[cfg(test)]
pub(crate) fn assemble_stacked_obs(frames: &[&StoredObs], t: usize, stack: usize) -> Vec<f32> {
    let flen = frames[t].len();
    let mut out = Vec::with_capacity(stack * flen);
    for i in (0..stack).rev() {
        let idx = t.saturating_sub(i);
        frames[idx].append_f32_into(&mut out);
    }
    out
}

// ============================================================================
// ObsAdapter：runner 的 obs I/O 统一入口（Flat 直通 / Image 走管线）
// ============================================================================

/// obs 适配器：把 env 原生观察桥接为「模型搜索 obs + buffer 存储 obs」。
///
/// 与 [`ActionAdapter`](super::action::ActionAdapter) 对偶——obs 是图像还是向量是
/// **env 事实**（[`resolve`](Self::resolve) 自动检测），预处理规格（84²×4 灰度堆叠）
/// 是**算法选择**（本模块常量）。
///
/// | 模式 | 搜索 obs（MCTS 输入） | 存储 obs（`SelfPlayStep.obs`） |
/// |------|----------------------|------------------------------|
/// | `Flat` | 原始展平 obs | 同左（[`StoredObs::F32`] 直通） |
/// | `Image` | [`STACK`] 帧堆叠（[`stacked_obs_dim`]） | 处理后**量化单帧**（[`StoredObs::U8`]，[`frame_len`] 个 u8，f32 整局的 1/16） |
pub(crate) enum ObsAdapter {
    Flat,
    Image(ImagePipe),
}

impl ObsAdapter {
    /// 按 env 观察空间事实解析（图像 obs → 图像管线）。
    pub fn resolve(env: &GymEnv) -> Self {
        match ImagePipe::from_env(env) {
            Some(pipe) => Self::Image(pipe),
            None => Self::Flat,
        }
    }

    /// 模型入口 obs 规格
    pub fn model_obs_spec(&self, env: &GymEnv) -> ObsSpec {
        match self {
            Self::Flat => ObsSpec::Flat(env.get_flatten_observation_len()),
            Self::Image(_) => ObsSpec::Image {
                channels: STACK,
                side: OUT_SIDE,
            },
        }
    }

    /// 训练期堆叠参数（Flat → `None`，Image → `Some(STACK)`），供 batch 组装。
    pub const fn image_stack(&self) -> Option<usize> {
        match self {
            Self::Flat => None,
            Self::Image(_) => Some(STACK),
        }
    }

    /// `env.reset` → `(搜索 obs, 存储 obs)`
    pub fn reset(&mut self, env: &GymEnv, seed: Option<u64>) -> (Vec<f32>, StoredObs) {
        let raw = env.reset(seed);
        match self {
            Self::Flat => {
                let o = raw[0].clone();
                (o.clone(), o.into())
            }
            Self::Image(pipe) => {
                let frame = pipe.reset(&raw[0]);
                (pipe.stacked(), frame)
            }
        }
    }

    /// `env.step` → `(搜索 obs, 存储 obs, reward, terminated, truncated)`
    pub fn step(&mut self, env: &GymEnv, action: &[f32]) -> (Vec<f32>, StoredObs, f32, bool, bool) {
        let (raw, reward, terminated, truncated) = {
            crate::prof_scope!("env.step");
            env.step(action)
        };
        match self {
            Self::Flat => {
                let o = raw[0].clone();
                (o.clone(), o.into(), reward, terminated, truncated)
            }
            Self::Image(pipe) => {
                let frame = pipe.push(&raw[0]);
                (pipe.stacked(), frame, reward, terminated, truncated)
            }
        }
    }
}
