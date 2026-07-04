/*
 * @Author       : 老董
 * @Date         : 2026-07-04
 * @Description  : 分配画像计数 allocator（`alloc-profile` feature，默认关）
 *
 * 设计边界（Reviewer 裁决的缩版设计，刻意最小化）：
 * - allocator 内**只做两个全局 atomic 计数**（次数 / 字节），不加锁、不分配、
 *   不写任何 map ——否则会递归分配或死锁。
 * - 归因交给外层：`rl::profiling::Scope` 在 enter/drop 读取全局计数做 delta，
 *   把分配量挂到已有的 PROFILE 命名桶上。全局计数天然覆盖 rayon worker 线程
 *   的分配（发生在 scope 时间窗内即被计入），主流程单线程模型下归因近似成立。
 * - 不做 per-allocation 归因、不做 outstanding bytes（存活量）——那需要
 *   dealloc 侧记账，跨线程释放会造假，YAGNI。
 * - `realloc` 按 new_size 全额计入（诊断口径求简单一致，不追增量精确）。
 */

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// 进程级累计分配次数
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
/// 进程级累计分配字节数
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

/// 计数 allocator：转发 `System`，仅在分配路径上累加两个 atomic
pub struct CountingAllocator;

// SAFETY: 完全转发 System allocator，只附加无副作用的 Relaxed 计数
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(new_size as u64, Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL_COUNTING_ALLOCATOR: CountingAllocator = CountingAllocator;

/// 读取当前累计（次数, 字节）快照，供 Scope 做 delta
#[inline]
pub fn snapshot() -> (u64, u64) {
    (ALLOC_COUNT.load(Relaxed), ALLOC_BYTES.load(Relaxed))
}

#[cfg(test)]
mod tests {
    use super::snapshot;

    #[test]
    fn 计数器随堆分配单调增长() {
        let (c0, b0) = snapshot();
        let v: Vec<u8> = Vec::with_capacity(4096);
        let (c1, b1) = snapshot();
        assert!(c1 > c0, "分配次数应增长");
        assert!(b1 - b0 >= 4096, "分配字节应至少计入 4096");
        drop(v);
    }
}
