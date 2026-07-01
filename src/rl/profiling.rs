//! 细粒度计时 profiler（诊断用，`PROFILE` 环境变量开启）。
//!
//! 用于把 MCTS 搜索 / 训练的 wall-clock 按命名桶拆开，定位真实热点。
//! 纯计时、零行为改变：未开 `PROFILE` 时 [`Scope::new`] 不调用 `Instant::now`，近零开销。
//!
//! # 用法
//! ```ignore
//! use crate::prof_scope;
//! prof_scope!("mcts.recurrent_fwd"); // 作用域结束自动累加
//! ```
//! 训练结束调用 [`print_report`] 打印各桶累计耗时 + 调用次数。
//!
//! # 线程模型
//! MyZero 训练与 MCTS 均单线程（`Python::attach` 内），故用 `thread_local` 累加即可。

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

thread_local! {
    static PROF: RefCell<BTreeMap<&'static str, (Duration, u64)>> =
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
}

impl Scope {
    #[inline]
    pub fn new(name: &'static str) -> Self {
        let start = if enabled() {
            Some(Instant::now())
        } else {
            None
        };
        Self { name, start }
    }
}

impl Drop for Scope {
    #[inline]
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let dt = start.elapsed();
            PROF.with(|p| {
                let mut m = p.borrow_mut();
                let e = m.entry(self.name).or_insert((Duration::ZERO, 0));
                e.0 += dt;
                e.1 += 1;
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
pub fn print_report() {
    if !enabled() {
        return;
    }
    let mut rows: Vec<(&'static str, Duration, u64)> =
        PROF.with(|p| p.borrow().iter().map(|(k, v)| (*k, v.0, v.1)).collect());
    if rows.is_empty() {
        return;
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    println!("[PROFILE-fine] 命名桶累计（降序）：");
    for (name, dur, count) in rows {
        let secs = dur.as_secs_f32();
        let per_call_us = if count > 0 {
            dur.as_secs_f64() * 1e6 / count as f64
        } else {
            0.0
        };
        println!("  {name:<24} {secs:8.2}s  calls={count:>9}  {per_call_us:8.2}us/call");
    }
}
