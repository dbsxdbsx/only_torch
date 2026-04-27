//! 合成数据集使用的轻量确定性随机数工具。
//!
//! 这个类型只面向 examples、tests 和小型 synthetic dataset。模型参数初始化仍应使用
//! `Graph::new_with_seed`，演化过程仍应使用 `Evolution::with_seed`。

use std::ops::Range;

const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// 用于合成数据生成的确定性伪随机数生成器。
///
/// 它不依赖全局状态，适合按 `seed + sample_idx + field` 派生稳定样本。该生成器不具备
/// 密码学安全性，也不用于模型训练过程内部随机性。
#[derive(Clone, Debug)]
pub struct SyntheticRng {
    state: u64,
}

impl SyntheticRng {
    /// 使用固定种子创建生成器。
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// 从主 seed 和若干派生字段创建独立随机流。
    ///
    /// 常见用法是把 `sample_idx`、`object_idx`、像素坐标等放入 `parts`，从而让每个样本
    /// 或对象拥有稳定且互不明显相关的随机流。
    pub fn from_seed_parts(seed: u64, parts: &[u64]) -> Self {
        let mut state = mix(seed);
        for &part in parts {
            state = mix(state ^ mix(part));
        }
        Self { state }
    }

    /// 从当前随机流派生一个子流，不消耗当前流状态。
    pub fn fork(&self, stream: u64) -> Self {
        Self::from_seed_parts(self.state, &[stream])
    }

    /// 生成下一个 `u64`。
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX_GAMMA);
        mix(self.state)
    }

    /// 生成 `[0, 1)` 范围内的 `f32`。
    pub fn next_f32(&mut self) -> f32 {
        const SCALE: f32 = 1.0 / ((1u32 << 24) as f32);
        ((self.next_u64() >> 40) as u32) as f32 * SCALE
    }

    /// 生成指定半开区间 `[start, end)` 内的 `usize`。
    ///
    /// # Panics
    ///
    /// 当 range 为空时 panic。
    pub fn usize_range(&mut self, range: Range<usize>) -> usize {
        assert!(
            range.start < range.end,
            "SyntheticRng::usize_range 需要非空 range"
        );
        let span = range.end - range.start;
        range.start + (self.next_u64() % span as u64) as usize
    }

    /// 生成指定半开区间 `[start, end)` 内的 `isize`。
    ///
    /// # Panics
    ///
    /// 当 range 为空时 panic。
    pub fn isize_range(&mut self, range: Range<isize>) -> isize {
        assert!(
            range.start < range.end,
            "SyntheticRng::isize_range 需要非空 range"
        );
        let span = (range.end - range.start) as u64;
        range.start + (self.next_u64() % span) as isize
    }

    /// 生成指定半开区间 `[start, end)` 内的 `f32`。
    ///
    /// # Panics
    ///
    /// 当 range 为空或边界不是有限数时 panic。
    pub fn f32_range(&mut self, range: Range<f32>) -> f32 {
        assert!(
            range.start.is_finite() && range.end.is_finite() && range.start < range.end,
            "SyntheticRng::f32_range 需要有限且非空 range"
        );
        range.start + (range.end - range.start) * self.next_f32()
    }

    /// 生成布尔值。
    pub fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

#[cfg(test)]
mod tests {
    use super::SyntheticRng;

    #[test]
    fn test_same_seed_reproducible() {
        let mut a = SyntheticRng::new(42);
        let mut b = SyntheticRng::new(42);

        let values_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let values_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();

        assert_eq!(values_a, values_b);
    }

    #[test]
    fn test_seed_parts_are_reproducible_and_distinct() {
        let mut a = SyntheticRng::from_seed_parts(42, &[7, 3, 1]);
        let mut b = SyntheticRng::from_seed_parts(42, &[7, 3, 1]);
        let mut c = SyntheticRng::from_seed_parts(42, &[7, 3, 2]);

        assert_eq!(a.next_u64(), b.next_u64());
        assert_ne!(a.next_u64(), c.next_u64());
    }

    #[test]
    fn test_ranges_stay_inside_bounds() {
        let mut rng = SyntheticRng::new(u64::MAX);

        for _ in 0..256 {
            let u = rng.usize_range(3..9);
            let i = rng.isize_range(-4..5);
            let f = rng.f32_range(-2.0..2.0);

            assert!((3..9).contains(&u));
            assert!((-4..5).contains(&i));
            assert!((-2.0..2.0).contains(&f));
        }
    }

    #[test]
    fn test_fork_does_not_consume_parent_stream() {
        let parent = SyntheticRng::new(123);
        let mut child_a = parent.fork(1);
        let mut child_b = parent.fork(1);
        let mut parent_a = parent.clone();
        let mut parent_b = parent;

        assert_eq!(child_a.next_u64(), child_b.next_u64());
        assert_eq!(parent_a.next_u64(), parent_b.next_u64());
    }
}
