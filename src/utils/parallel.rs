/*
 * @Author       : 老董
 * @Date         : 2026-07-04
 * @Description  : Rayon 小任务并行阈值分流（数据并行统一入口）
 *
 * 背景：op 级 per-sample / per-row 并行在 batch=1 或小张量场景下，rayon
 * fork-join 调度开销（微秒级）会盖过并行收益（热路径审计 Reviewer 方向 2）。
 * 本模块提供带阈值的统一调度入口：低于阈值走串行，执行体（闭包）完全相同，
 * 每元素运算与写入位置逐 bit 一致——仅改变"由哪个线程执行"。
 *
 * ⚠️ 适用边界：**仅限 map-only 场景**（各 chunk 独立写、无跨 chunk 归约）。
 * 跨样本归约（如 CE 前向 par sum、conv dK 累加）的串行/并行切换会改变
 * f32 累加顺序，不得使用本入口做分流。
 *
 * 阈值定标：benches/rayon_threshold.rs（结论见 .doc/design/threading_model.md）。
 */

use rayon::prelude::*;

/// 并行调度经验阈值：预估串行总工作量（近似"元素数 × 每元素基本运算数"）
/// 低于该值时走串行。由 `benches/rayon_threshold.rs` 定标（2026-07-04，
/// release+MKL：work≈8k 时串行胜 2~5×，work≈33k 起并行稳定胜 1.5×+，
/// 交叉点在 16k~33k 区间，取 32768 保守偏串行）。
pub(crate) const PAR_MIN_WORK: usize = 1 << 15;

/// 是否值得并行：多于一个 chunk 且总工作量达到阈值
#[inline]
pub(crate) fn should_par(n_chunks: usize, total_work: usize) -> bool {
    n_chunks > 1 && total_work >= PAR_MIN_WORK
}

/// chunk 级并行 `for_each`（带阈值分流）
///
/// - `chunk_size`：每个并行单元的元素数（如单样本 / 单行的展平长度）
/// - `total_work`：调用方预估的串行总工作量（用于阈值判断，见 [`PAR_MIN_WORK`]）
/// - `f(chunk_idx, chunk)`：对第 `chunk_idx` 个 chunk 的独立写入体
#[inline]
pub(crate) fn for_each_chunk_mut<F>(data: &mut [f32], chunk_size: usize, total_work: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync + Send,
{
    let n_chunks = data.len().div_ceil(chunk_size.max(1));
    if should_par(n_chunks, total_work) {
        data.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
    } else {
        data.chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
    }
}

/// 双缓冲 zip 版 chunk 级并行 `for_each`（带阈值分流）
///
/// 用于同时直写两个输出 buffer 的场景（如 max_pool2d 前向的 输出+索引）。
#[inline]
pub(crate) fn for_each_chunk_mut2<F>(
    data_a: &mut [f32],
    data_b: &mut [f32],
    chunk_size: usize,
    total_work: usize,
    f: F,
) where
    F: Fn(usize, &mut [f32], &mut [f32]) + Sync + Send,
{
    debug_assert_eq!(data_a.len(), data_b.len());
    let n_chunks = data_a.len().div_ceil(chunk_size.max(1));
    if should_par(n_chunks, total_work) {
        data_a
            .par_chunks_mut(chunk_size)
            .zip(data_b.par_chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (ca, cb))| f(i, ca, cb));
    } else {
        data_a
            .chunks_mut(chunk_size)
            .zip(data_b.chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (ca, cb))| f(i, ca, cb));
    }
}

/// 索引映射并行 collect（带阈值分流），产物顺序恒为 `0..n`
///
/// 用于"每样本独立产出一个中间结果再统一消费"的场景（如 conv dK 的
/// per-sample 部分梯度）。串行/并行产物逐 bit 相同且顺序一致。
#[inline]
pub(crate) fn map_indexed<T, F>(n: usize, total_work: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    if should_par(n, total_work) {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 串行与并行路径产物逐位一致() {
        let n = 8;
        let chunk = 1024;
        let mut serial = vec![0.0f32; n * chunk];
        let mut par = vec![0.0f32; n * chunk];
        let body = |i: usize, c: &mut [f32]| {
            for (j, v) in c.iter_mut().enumerate() {
                *v = (i * chunk + j) as f32 * 1.000_1 + 0.5;
            }
        };
        // total_work=0 强制串行；total_work=usize::MAX 强制并行
        for_each_chunk_mut(&mut serial, chunk, 0, body);
        for_each_chunk_mut(&mut par, chunk, usize::MAX, body);
        assert_eq!(serial, par);

        let sm = map_indexed(n, 0, |i| i * 2);
        let pm = map_indexed(n, usize::MAX, |i| i * 2);
        assert_eq!(sm, pm);
    }

    #[test]
    fn 阈值判断边界() {
        assert!(!should_par(1, usize::MAX), "单 chunk 永远串行");
        assert!(!should_par(8, PAR_MIN_WORK - 1));
        assert!(should_par(2, PAR_MIN_WORK));
    }
}
