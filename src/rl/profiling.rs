//! 细粒度计时 profiler（诊断用，`PROFILE` 环境变量开启）。
//!
//! 用于把 MCTS 搜索 / 训练的 wall-clock 按命名桶拆开，定位真实热点。
//! 纯计时、零行为改变：未开 `PROFILE` 时 [`Scope::new`] 不调用 `Instant::now`，近零开销。
//!
//! # 分配画像（可选）
//! 编译时开启 `alloc-profile` feature 后，每个命名桶额外累计 **alloc 次数 / 字节**
//! （由全局计数 allocator 提供，见 `utils::alloc_profile`）。全局计数覆盖 rayon
//! worker 线程的分配（发生在 scope 时间窗内即计入）；若多个 scope 并发存在，
//! delta 会互相包含——主流程单线程模型（`Python::attach` 内）下不构成问题。
//!
//! # 用法
//! ```ignore
//! use crate::prof_scope;
//! prof_scope!("mcts.recurrent_fwd"); // 作用域结束自动累加
//! ```
//! 训练结束调用 [`print_report`] 打印各桶累计耗时 + 调用次数（+ 分配量）。
//!
//! # 线程模型
//! MyZero 训练与 MCTS 均单线程（`Python::attach` 内），故用 `thread_local` 累加即可。

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// 单个命名桶的累计统计
#[derive(Default, Clone, Copy)]
struct BucketStat {
    dur: Duration,
    calls: u64,
    #[cfg(feature = "alloc-profile")]
    allocs: u64,
    #[cfg(feature = "alloc-profile")]
    alloc_bytes: u64,
}

thread_local! {
    static PROF: RefCell<BTreeMap<&'static str, BucketStat>> =
        const { RefCell::new(BTreeMap::new()) };
}

/// 是否开启 profiling（读一次 `PROFILE` 环境变量后缓存）。
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PROFILE").is_ok())
}

/// RAII 计时守卫：构造记录起点，析构累加到对应桶。
///
/// 未开 profiling 时 `start = None`，drop 直接返回，不产生任何计时开销。
pub struct Scope {
    name: &'static str,
    start: Option<Instant>,
    #[cfg(feature = "alloc-profile")]
    alloc_start: (u64, u64),
}

impl Scope {
    #[inline]
    pub fn new(name: &'static str) -> Self {
        let start = if enabled() {
            Some(Instant::now())
        } else {
            None
        };
        Self {
            name,
            start,
            #[cfg(feature = "alloc-profile")]
            alloc_start: if start.is_some() {
                crate::utils::alloc_profile::snapshot()
            } else {
                (0, 0)
            },
        }
    }
}

impl Drop for Scope {
    #[inline]
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let dt = start.elapsed();
            #[cfg(feature = "alloc-profile")]
            let (alloc_count, alloc_bytes) = {
                let (c, b) = crate::utils::alloc_profile::snapshot();
                (
                    c.saturating_sub(self.alloc_start.0),
                    b.saturating_sub(self.alloc_start.1),
                )
            };
            PROF.with(|p| {
                let mut m = p.borrow_mut();
                let e = m.entry(self.name).or_default();
                e.dur += dt;
                e.calls += 1;
                #[cfg(feature = "alloc-profile")]
                {
                    e.allocs += alloc_count;
                    e.alloc_bytes += alloc_bytes;
                }
            });
        }
    }
}

/// 在当前作用域插入一个命名计时桶（见模块文档）。
#[macro_export]
macro_rules! prof_scope {
    ($name:expr) => {
        let _prof_guard = $crate::rl::profiling::Scope::new($name);
    };
}

/// 清空累计（每个 seed 训练开始时调用，避免跨 seed 混淆）。
pub fn reset() {
    PROF.with(|p| p.borrow_mut().clear());
}

/// 打印各桶累计耗时 + 调用次数（按耗时降序）。仅 profiling 开启时输出。
///
/// `alloc-profile` feature 开启时额外打印每桶累计 alloc 次数 / 字节 / 每调用均值。
pub fn print_report() {
    if !enabled() {
        return;
    }
    let mut rows: Vec<(&'static str, BucketStat)> =
        PROF.with(|p| p.borrow().iter().map(|(k, v)| (*k, *v)).collect());
    if rows.is_empty() {
        return;
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.1.dur));
    println!("[PROFILE-fine] 命名桶累计（降序）：");
    for (name, stat) in rows {
        let secs = stat.dur.as_secs_f32();
        let per_call_us = if stat.calls > 0 {
            stat.dur.as_secs_f64() * 1e6 / stat.calls as f64
        } else {
            0.0
        };
        #[cfg(not(feature = "alloc-profile"))]
        println!(
            "  {name:<24} {secs:8.2}s  calls={:>9}  {per_call_us:8.2}us/call",
            stat.calls
        );
        #[cfg(feature = "alloc-profile")]
        {
            let allocs_per_call = if stat.calls > 0 {
                stat.allocs as f64 / stat.calls as f64
            } else {
                0.0
            };
            let kb = stat.alloc_bytes as f64 / 1024.0;
            println!(
                "  {name:<24} {secs:8.2}s  calls={:>9}  {per_call_us:8.2}us/call  \
                 allocs={:>12} ({allocs_per_call:9.1}/call)  {kb:14.1} KiB",
                stat.calls, stat.allocs
            );
        }
    }
}
