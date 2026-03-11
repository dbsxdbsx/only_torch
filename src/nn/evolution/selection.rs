/*
 * @Author       : 老董
 * @Date         : 2026-03-11
 * @Description  : NSGA-II 多目标选择与 Pareto Archive 管理
 *
 * 核心功能：
 * - 双目标 Pareto 支配判定（maximize primary + minimize inference_cost）
 * - 快速非支配排序（pareto_rank）
 * - 拥挤度距离（crowding_distance）
 * - NSGA-II 环境选择（nsga2_select）
 * - 全局 Pareto Archive 维护（update_archive / archive_changed）
 *
 * 设计原则：
 * - 通过 ObjectivePoint 抽象目标提取，便于未来扩展为 FLOPs / latency 等
 * - tiebreak_loss 仅在 objective 完全打平时介入，不作为正式第三目标
 * - inference_cost 为 None 时自动退化为单目标排序
 */

use super::task::FitnessScore;

// ==================== ObjectivePoint ====================

/// 目标向量：统一为"越大越好"的规范化表示
///
/// 内部把 inference_cost（越低越好）取负，使所有维度统一为 maximize。
/// 未来扩展 FLOPs / latency 时只需修改此处提取逻辑。
#[derive(Clone, Debug)]
pub(crate) struct ObjectivePoint {
    /// 规范化目标值（全部越大越好）
    values: Vec<f32>,
}

impl ObjectivePoint {
    /// 目标维度数
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

/// 从 FitnessScore 提取目标向量
///
/// - primary: 越高越好，直接取值
/// - inference_cost: 越低越好，取负（使其越大越好）
/// - inference_cost 为 None 时退化为单目标
pub(crate) fn objective_point(score: &FitnessScore) -> ObjectivePoint {
    let mut values = vec![score.primary];
    if let Some(cost) = score.inference_cost {
        values.push(-cost); // 越低越好 → 取负后越大越好
    }
    ObjectivePoint { values }
}

// ==================== Pareto 支配 ====================

/// Pareto 支配判定：a 是否支配 b
///
/// a 支配 b 当且仅当：a 在所有目标上不劣于 b，且至少一个目标严格更优。
/// 目标方向：primary 越高越好，inference_cost 越低越好。
pub(crate) fn dominates(a: &FitnessScore, b: &FitnessScore) -> bool {
    let pa = objective_point(a);
    let pb = objective_point(b);
    dominates_points(&pa, &pb)
}

fn dominates_points(a: &ObjectivePoint, b: &ObjectivePoint) -> bool {
    assert_eq!(a.dim(), b.dim(), "目标维度不一致");
    let mut better_in_any = false;
    for (av, bv) in a.values.iter().zip(b.values.iter()) {
        if av < bv {
            return false; // a 在某个目标上劣于 b
        }
        if av > bv {
            better_in_any = true;
        }
    }
    better_in_any
}

// ==================== 快速非支配排序 ====================

/// 快速非支配排序（NSGA-II），返回每个个体的 rank（0 = 第一前沿）
///
/// 时间复杂度 O(M * N²)，M = 目标数，N = 种群大小。
/// 对当前规模（N ≤ 64, M = 2）完全足够。
pub(crate) fn pareto_rank(scores: &[FitnessScore]) -> Vec<usize> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }

    let points: Vec<ObjectivePoint> = scores.iter().map(objective_point).collect();

    let mut dominated_count = vec![0usize; n];
    let mut dominates_list: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut ranks = vec![0usize; n];

    // 计算支配关系
    for i in 0..n {
        for j in (i + 1)..n {
            if dominates_points(&points[i], &points[j]) {
                dominates_list[i].push(j);
                dominated_count[j] += 1;
            } else if dominates_points(&points[j], &points[i]) {
                dominates_list[j].push(i);
                dominated_count[i] += 1;
            }
        }
    }

    // 逐层剥离前沿
    let mut current_front: Vec<usize> = (0..n)
        .filter(|&i| dominated_count[i] == 0)
        .collect();
    let mut front_idx = 0;

    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &p in &current_front {
            ranks[p] = front_idx;
            for &q in &dominates_list[p] {
                dominated_count[q] -= 1;
                if dominated_count[q] == 0 {
                    next_front.push(q);
                }
            }
        }
        front_idx += 1;
        current_front = next_front;
    }

    ranks
}

// ==================== 拥挤度距离 ====================

/// 拥挤度距离（NSGA-II）
///
/// 对每个目标维度：排序后边界点获得 +∞ 距离，内部点获得归一化距离贡献。
/// 用于同 rank 内选择更分散的个体。
pub(crate) fn crowding_distance(scores: &[FitnessScore]) -> Vec<f32> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    if n <= 2 {
        return vec![f32::INFINITY; n];
    }

    let points: Vec<ObjectivePoint> = scores.iter().map(objective_point).collect();
    let m = points[0].dim();
    if m == 0 {
        return vec![0.0; n];
    }

    let mut distances = vec![0.0f32; n];
    let mut indices: Vec<usize> = (0..n).collect();

    for dim in 0..m {
        indices.sort_unstable_by(|&a, &b| {
            points[a].values[dim]
                .partial_cmp(&points[b].values[dim])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let min_val = points[indices[0]].values[dim];
        let max_val = points[indices[n - 1]].values[dim];
        let range = max_val - min_val;

        if !range.is_finite() || range == 0.0 {
            continue;
        }

        // 边界点获得无穷距离
        distances[indices[0]] = f32::INFINITY;
        distances[indices[n - 1]] = f32::INFINITY;

        // 内部点：归一化距离
        for k in 1..(n - 1) {
            let prev = points[indices[k - 1]].values[dim];
            let next = points[indices[k + 1]].values[dim];
            distances[indices[k]] += (next - prev).abs() / range;
        }
    }

    distances
}

// ==================== tiebreak 比较 ====================

/// 同 rank 同 crowding distance 时的确定性 tie-breaker
///
/// 使用 tiebreak_loss（越低越好）；都为 None 时视为相等。
fn tiebreak_cmp(a: &FitnessScore, b: &FitnessScore) -> std::cmp::Ordering {
    match (a.tiebreak_loss, b.tiebreak_loss) {
        (Some(la), Some(lb)) => la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal),
        (Some(_), None) => std::cmp::Ordering::Less, // 有 tiebreak 的更精确，优先
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

// ==================== NSGA-II 选择 ====================

/// NSGA-II 环境选择：从 pool 中选出 count 个最优个体
///
/// 排序规则：
/// 1. rank 越小越好（更优前沿）
/// 2. 同 rank 时 crowding distance 越大越好（更分散）
/// 3. 同 rank 同 distance 时用 tiebreak_loss 决胜
pub(crate) fn nsga2_select<T>(
    pool: Vec<(T, FitnessScore)>,
    count: usize,
) -> Vec<(T, FitnessScore)> {
    if pool.len() <= count {
        return pool;
    }

    let scores: Vec<FitnessScore> = pool.iter().map(|(_, s)| s.clone()).collect();
    let ranks = pareto_rank(&scores);
    let distances = crowding_distance(&scores);

    // 为每个个体构造排序键
    let mut indices: Vec<usize> = (0..pool.len()).collect();
    indices.sort_by(|&a, &b| {
        // 1. rank 升序
        let rank_cmp = ranks[a].cmp(&ranks[b]);
        if rank_cmp != std::cmp::Ordering::Equal {
            return rank_cmp;
        }
        // 2. crowding distance 降序
        let dist_cmp = distances[b]
            .partial_cmp(&distances[a])
            .unwrap_or(std::cmp::Ordering::Equal);
        if dist_cmp != std::cmp::Ordering::Equal {
            return dist_cmp;
        }
        // 3. tiebreak_loss
        tiebreak_cmp(&scores[a], &scores[b])
    });

    // 按排序顺序提取前 count 个
    // 从后往前 swap_remove 以保持索引稳定
    let selected_set: std::collections::HashSet<usize> =
        indices.iter().take(count).copied().collect();
    let mut selected = Vec::with_capacity(count);
    let mut remaining = Vec::with_capacity(pool.len() - count);

    for (i, item) in pool.into_iter().enumerate() {
        if selected_set.contains(&i) {
            selected.push(item);
        } else {
            remaining.push(item);
        }
    }
    drop(remaining);

    selected
}

// ==================== Pareto Archive ====================

/// 更新全局 Pareto archive：合并新候选，保留所有非支配解
///
/// archive 中只保留互不支配的个体。新候选如果被 archive 中已有个体支配则丢弃，
/// 如果支配了 archive 中的某些个体则替换之。
pub(crate) fn update_archive<T: Clone>(
    archive: &mut Vec<(T, FitnessScore)>,
    candidates: Vec<(T, FitnessScore)>,
) {
    for candidate in candidates {
        let mut dominated_by_archive = false;
        let mut to_remove = Vec::new();

        for (i, (_, existing_score)) in archive.iter().enumerate() {
            if dominates(existing_score, &candidate.1) {
                dominated_by_archive = true;
                break;
            }
            if dominates(&candidate.1, existing_score) {
                to_remove.push(i);
            }
        }

        if dominated_by_archive {
            continue;
        }

        // 检查是否与 archive 中某个体 objective 完全相同（避免重复）
        if to_remove.is_empty() {
            let dup = archive.iter().any(|(_, s)| {
                let pa = objective_point(s);
                let pb = objective_point(&candidate.1);
                pa.values == pb.values
            });
            if dup {
                continue;
            }
        }

        // 从后往前移除被支配的
        to_remove.sort_unstable();
        for &i in to_remove.iter().rev() {
            archive.swap_remove(i);
        }

        archive.push(candidate);
    }
}

/// 判断 archive 是否发生了实质变化（收敛检测用）
///
/// 比较两个 archive 快照的 objective 值集合是否一致（容忍浮点误差）。
pub(crate) fn archive_changed(
    prev_scores: &[FitnessScore],
    next_scores: &[FitnessScore],
    tolerance: f32,
) -> bool {
    if prev_scores.len() != next_scores.len() {
        return true;
    }
    if prev_scores.is_empty() {
        return false;
    }

    // 将两个集合按 primary 排序后逐个比较
    let mut prev: Vec<ObjectivePoint> = prev_scores.iter().map(objective_point).collect();
    let mut next: Vec<ObjectivePoint> = next_scores.iter().map(objective_point).collect();

    let sort_key = |p: &ObjectivePoint| -> Vec<i64> {
        p.values.iter().map(|v| (v * 1e6) as i64).collect()
    };
    prev.sort_by_key(sort_key);
    next.sort_by_key(sort_key);

    for (p, n) in prev.iter().zip(next.iter()) {
        if p.dim() != n.dim() {
            return true;
        }
        for (pv, nv) in p.values.iter().zip(n.values.iter()) {
            if (pv - nv).abs() > tolerance {
                return true;
            }
        }
    }

    false
}

/// 提取 Pareto 前沿的索引（rank=0 的个体）
pub(crate) fn pareto_front_indices(scores: &[FitnessScore]) -> Vec<usize> {
    let ranks = pareto_rank(scores);
    ranks
        .iter()
        .enumerate()
        .filter(|&(_, r)| *r == 0)
        .map(|(i, _)| i)
        .collect()
}
