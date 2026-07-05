//! PerPriorities（位置级 PER 伴生采样器）单元测试（纯 Rust，无 pyo3）

use crate::rl::buffer::{PER_EPS, PerPriorities};
use rand::SeedableRng;
use rand::rngs::StdRng;

// ============================================================================
// FIFO 镜像
// ============================================================================

/// 满容淘汰最老局：槽位应与伴生 buffer 的 FIFO 下标恒一致。
#[test]
fn per_fifo_mirror() {
    let mut per = PerPriorities::new(2, 1.0);
    per.push_game(vec![0.0, 0.0]); // 局 A（将被淘汰）
    per.push_game(vec![0.0, 0.0]); // 局 B → 槽 0
    per.push_game(vec![100.0]); // 局 C → 槽 1（唯一高优先级）
    assert_eq!(per.len_games(), 2);

    let mut rng = StdRng::seed_from_u64(42);
    let picks = per.sample(64, &mut rng);
    assert_eq!(picks.len(), 64);
    // 局 C 在槽 1 位置 0，优先级 100 vs 局 B 两个 ~ε → 绝大多数样本应落在 (1, 0)
    let c_hits = picks.iter().filter(|&&(s, p)| s == 1 && p == 0).count();
    assert!(c_hits > 56, "高优先级位置应主导采样，实际 {c_hits}/64");
    // 槽 0（局 B）仍可被采到（ε 下限保证非零概率），且位置越界不允许
    assert!(
        picks
            .iter()
            .all(|&(s, p)| (s == 0 && p < 2) || (s == 1 && p == 0))
    );
}

// ============================================================================
// 采样分布
// ============================================================================

/// α=1 全比例：优先级 3:1 的两个位置，采样频次应接近 3:1（±统计噪声）。
#[test]
fn per_proportional_sampling() {
    let mut per = PerPriorities::new(4, 1.0);
    per.push_game(vec![3.0 - PER_EPS, 1.0 - PER_EPS]); // 加 ε 后恰为 3.0 / 1.0
    let mut rng = StdRng::seed_from_u64(7);
    let picks = per.sample(4000, &mut rng);
    let hi = picks.iter().filter(|&&(_, p)| p == 0).count() as f32;
    let ratio = hi / (picks.len() as f32 - hi);
    assert!(
        (2.4..=3.6).contains(&ratio),
        "3:1 优先级的采样比应近 3（实际 {ratio:.2}）"
    );
}

/// α=0 退化均匀：优先级差异不影响采样分布。
#[test]
fn per_alpha_zero_uniform() {
    let mut per = PerPriorities::new(4, 0.0);
    per.push_game(vec![1000.0, 0.0]);
    let mut rng = StdRng::seed_from_u64(11);
    let picks = per.sample(4000, &mut rng);
    let hi = picks.iter().filter(|&&(_, p)| p == 0).count() as f32;
    let frac = hi / picks.len() as f32;
    assert!(
        (0.45..=0.55).contains(&frac),
        "α=0 应均匀采样（位置 0 占比 {frac:.3}）"
    );
}

// ============================================================================
// 边界
// ============================================================================

/// 空采样器 / batch=0 / 全空局 → 返回空 Vec（调用方按无样本处理）。
#[test]
fn per_empty_cases() {
    let per = PerPriorities::new(4, 0.6);
    let mut rng = StdRng::seed_from_u64(1);
    assert!(per.sample(8, &mut rng).is_empty(), "无局应返回空");

    let mut per2 = PerPriorities::new(4, 0.6);
    per2.push_game(vec![1.0]);
    assert!(per2.sample(0, &mut rng).is_empty(), "batch=0 应返回空");

    let mut per3 = PerPriorities::new(4, 0.6);
    per3.push_game(Vec::new()); // 空局（<2 步训练路径跳过）
    assert!(per3.sample(8, &mut rng).is_empty(), "全空局零质量应返回空");
}

/// 全零原始优先级：ε 下限兜底 → 均匀采样而非除零/空返回。
#[test]
fn per_all_zero_priorities_fallback_uniform() {
    let mut per = PerPriorities::new(4, 0.6);
    per.push_game(vec![0.0, 0.0, 0.0, 0.0]);
    let mut rng = StdRng::seed_from_u64(3);
    let picks = per.sample(400, &mut rng);
    assert_eq!(picks.len(), 400);
    // 四个位置都应被采到（均匀 ~100 次；宽松下限防偶发）
    for pos in 0..4 {
        let n = picks.iter().filter(|&&(_, p)| p == pos).count();
        assert!(n > 50, "位置 {pos} 应被均匀采到（实际 {n}/400）");
    }
}
