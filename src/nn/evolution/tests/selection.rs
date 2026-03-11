use crate::nn::evolution::selection::*;
use crate::nn::evolution::task::FitnessScore;

// ==================== 辅助构造 ====================

fn score(primary: f32, cost: Option<f32>) -> FitnessScore {
    FitnessScore {
        primary,
        inference_cost: cost,
        tiebreak_loss: None,
    }
}

fn score_with_tiebreak(primary: f32, cost: Option<f32>, tiebreak: Option<f32>) -> FitnessScore {
    FitnessScore {
        primary,
        inference_cost: cost,
        tiebreak_loss: tiebreak,
    }
}

// ==================== dominates ====================

#[test]
fn test_dominates_basic() {
    // A: 更高 primary + 更低 cost → 支配 B
    let a = score(0.9, Some(100.0));
    let b = score(0.8, Some(200.0));
    assert!(dominates(&a, &b));
    assert!(!dominates(&b, &a));
}

#[test]
fn test_dominates_equal_is_not_dominance() {
    let a = score(0.9, Some(100.0));
    let b = score(0.9, Some(100.0));
    assert!(!dominates(&a, &b));
    assert!(!dominates(&b, &a));
}

#[test]
fn test_dominates_tradeoff_no_dominance() {
    // A: 更高 primary 但更高 cost → 互不支配
    let a = score(0.95, Some(300.0));
    let b = score(0.85, Some(100.0));
    assert!(!dominates(&a, &b));
    assert!(!dominates(&b, &a));
}

#[test]
fn test_dominates_one_dimension_better_one_equal() {
    let a = score(0.9, Some(100.0));
    let b = score(0.9, Some(200.0));
    // a 的 primary 相同，cost 更低 → a 支配 b
    assert!(dominates(&a, &b));
    assert!(!dominates(&b, &a));
}

// ==================== tiebreak_loss 不参与支配 ====================

#[test]
fn test_tiebreak_loss_only_used_after_objective_tie() {
    // 两个个体 objective 相同，只有 tiebreak_loss 不同
    let a = score_with_tiebreak(0.9, Some(100.0), Some(0.1));
    let b = score_with_tiebreak(0.9, Some(100.0), Some(0.5));
    // 不应该形成支配关系
    assert!(!dominates(&a, &b));
    assert!(!dominates(&b, &a));
}

// ==================== pareto_rank ====================

#[test]
fn test_pareto_rank_two_objectives() {
    let scores = vec![
        score(0.9, Some(300.0)), // 0: 高 primary, 高 cost
        score(0.7, Some(100.0)), // 1: 低 primary, 低 cost
        score(0.8, Some(200.0)), // 2: 中 primary, 中 cost
        score(0.6, Some(400.0)), // 3: 低 primary, 高 cost → 被 0,1,2 支配
    ];
    let ranks = pareto_rank(&scores);
    // 0, 1, 2 互不支配 → rank 0
    assert_eq!(ranks[0], 0);
    assert_eq!(ranks[1], 0);
    assert_eq!(ranks[2], 0);
    // 3 被支配 → rank > 0
    assert!(ranks[3] > 0);
}

#[test]
fn test_pareto_rank_all_on_front() {
    // 所有个体互不支配
    let scores = vec![
        score(0.9, Some(300.0)),
        score(0.7, Some(100.0)),
        score(0.8, Some(200.0)),
    ];
    let ranks = pareto_rank(&scores);
    assert!(ranks.iter().all(|&r| r == 0));
}

#[test]
fn test_pareto_rank_empty() {
    let ranks = pareto_rank(&[]);
    assert!(ranks.is_empty());
}

#[test]
fn test_pareto_rank_single() {
    let ranks = pareto_rank(&[score(0.5, Some(100.0))]);
    assert_eq!(ranks, vec![0]);
}

// ==================== crowding_distance ====================

#[test]
fn test_crowding_distance_boundary_infinite() {
    let scores = vec![
        score(0.9, Some(300.0)),
        score(0.7, Some(100.0)),
        score(0.8, Some(200.0)),
    ];
    let distances = crowding_distance(&scores);
    // 在各维度上的极端值应获得无穷距离
    // primary: min=0.7(idx1), max=0.9(idx0) → 0, 1 获得 inf
    // cost: min=-300(idx0), max=-100(idx1) → 0, 1 获得 inf
    assert!(distances[0].is_infinite());
    assert!(distances[1].is_infinite());
    // 中间点获得有限距离
    assert!(distances[2].is_finite());
    assert!(distances[2] > 0.0);
}

#[test]
fn test_crowding_distance_two_points() {
    let scores = vec![score(0.9, Some(100.0)), score(0.7, Some(200.0))];
    let distances = crowding_distance(&scores);
    assert!(distances[0].is_infinite());
    assert!(distances[1].is_infinite());
}

#[test]
fn test_crowding_distance_empty() {
    let distances = crowding_distance(&[]);
    assert!(distances.is_empty());
}

#[test]
fn test_crowding_distance_is_computed_per_front() {
    let scores = vec![
        score(0.9, Some(500.0)), // front 0
        score(0.8, Some(300.0)), // front 0
        score(0.7, Some(100.0)), // front 0
        score(0.6, Some(450.0)), // front 1
        score(0.5, Some(250.0)), // front 1
        score(0.4, Some(150.0)), // front 1
    ];
    let ranks = pareto_rank(&scores);
    assert_eq!(ranks, vec![0, 0, 0, 1, 1, 1]);

    let distances = crowding_distance(&scores);
    assert!(
        distances[3].is_infinite(),
        "第二前沿的边界点应在其 front 内获得无穷距离"
    );
    assert!(
        distances[5].is_infinite(),
        "第二前沿的另一端边界点应在其 front 内获得无穷距离"
    );
    assert!(distances[4].is_finite());
}

// ==================== nsga2_select ====================

#[test]
fn test_nsga2_select_preserves_front() {
    let pool: Vec<(usize, FitnessScore)> = vec![
        (0, score(0.9, Some(300.0))), // front
        (1, score(0.7, Some(100.0))), // front
        (2, score(0.8, Some(200.0))), // front
        (3, score(0.6, Some(400.0))), // dominated
        (4, score(0.5, Some(500.0))), // dominated
    ];
    let selected = nsga2_select(pool, 3);
    assert_eq!(selected.len(), 3);
    // 前沿的 3 个个体都应被选中
    let selected_ids: Vec<usize> = selected.iter().map(|(id, _)| *id).collect();
    assert!(selected_ids.contains(&0));
    assert!(selected_ids.contains(&1));
    assert!(selected_ids.contains(&2));
}

#[test]
fn test_nsga2_select_count_exceeds_pool() {
    let pool: Vec<(usize, FitnessScore)> = vec![
        (0, score(0.9, Some(100.0))),
        (1, score(0.8, Some(200.0))),
    ];
    let selected = nsga2_select(pool, 5);
    assert_eq!(selected.len(), 2); // 不能超过 pool 大小
}

#[test]
fn test_nsga2_select_tiebreak_used_for_equal_objectives() {
    // 所有个体 objective 相同，只有 tiebreak_loss 不同
    let pool: Vec<(usize, FitnessScore)> = vec![
        (0, score_with_tiebreak(0.9, Some(100.0), Some(0.5))),
        (1, score_with_tiebreak(0.9, Some(100.0), Some(0.1))), // 最低 tiebreak
        (2, score_with_tiebreak(0.9, Some(100.0), Some(0.8))),
    ];
    let selected = nsga2_select(pool, 1);
    assert_eq!(selected.len(), 1);
    // tiebreak_loss 最低的应被选中
    assert_eq!(selected[0].0, 1);
}

// ==================== update_archive ====================

#[test]
fn test_update_archive_keeps_non_dominated_history() {
    let mut archive: Vec<(String, FitnessScore)> = Vec::new();

    // 第一批：加入 A（高 primary, 高 cost）
    update_archive(
        &mut archive,
        vec![("A".into(), score(0.9, Some(300.0)))],
    );
    assert_eq!(archive.len(), 1);

    // 第二批：加入 B（低 primary, 低 cost）→ 互不支配，都保留
    update_archive(
        &mut archive,
        vec![("B".into(), score(0.7, Some(100.0)))],
    );
    assert_eq!(archive.len(), 2);

    // 第三批：加入 C，被 A 支配 → 不加入
    update_archive(
        &mut archive,
        vec![("C".into(), score(0.8, Some(350.0)))],
    );
    assert_eq!(archive.len(), 2);

    // 第四批：加入 D，支配 A → 替换 A
    update_archive(
        &mut archive,
        vec![("D".into(), score(0.95, Some(250.0)))],
    );
    assert_eq!(archive.len(), 2);
    let names: Vec<&str> = archive.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"B"));
    assert!(names.contains(&"D"));
    assert!(!names.contains(&"A")); // A 被 D 替换
}

#[test]
fn test_update_archive_rejects_duplicates() {
    let mut archive: Vec<(usize, FitnessScore)> = Vec::new();
    update_archive(&mut archive, vec![(0, score(0.9, Some(100.0)))]);
    update_archive(&mut archive, vec![(1, score(0.9, Some(100.0)))]);
    // 完全相同的 objective → 不重复加入
    assert_eq!(archive.len(), 1);
}

// ==================== archive_changed ====================

#[test]
fn test_archive_changed_detects_difference() {
    let prev = vec![score(0.9, Some(100.0)), score(0.7, Some(50.0))];
    let next = vec![score(0.95, Some(100.0)), score(0.7, Some(50.0))];
    assert!(archive_changed(&prev, &next, 1e-4));
}

#[test]
fn test_archive_changed_same_within_tolerance() {
    let prev = vec![score(0.9, Some(100.0))];
    let next = vec![score(0.9 + 1e-7, Some(100.0 + 1e-7))];
    assert!(!archive_changed(&prev, &next, 1e-4));
}

#[test]
fn test_archive_changed_different_sizes() {
    let prev = vec![score(0.9, Some(100.0))];
    let next = vec![
        score(0.9, Some(100.0)),
        score(0.7, Some(50.0)),
    ];
    assert!(archive_changed(&prev, &next, 1e-4));
}

// ==================== 单目标退化 ====================

#[test]
fn test_single_objective_fallback() {
    // inference_cost 全为 None → 退化为单目标
    let scores = vec![
        score(0.9, None),
        score(0.8, None),
        score(0.7, None),
    ];
    let ranks = pareto_rank(&scores);
    // 单目标下 0.9 支配 0.8 支配 0.7
    assert_eq!(ranks[0], 0);
    assert_eq!(ranks[1], 1);
    assert_eq!(ranks[2], 2);
}

#[test]
fn test_nsga2_select_single_objective() {
    let pool: Vec<(usize, FitnessScore)> = vec![
        (0, score(0.9, None)),
        (1, score(0.8, None)),
        (2, score(0.7, None)),
    ];
    let selected = nsga2_select(pool, 2);
    let ids: Vec<usize> = selected.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&0));
    assert!(ids.contains(&1));
    assert!(!ids.contains(&2));
}

// ==================== pareto_front_indices ====================

#[test]
fn test_pareto_front_indices() {
    let scores = vec![
        score(0.9, Some(300.0)), // front
        score(0.7, Some(100.0)), // front
        score(0.6, Some(400.0)), // dominated
    ];
    let indices = pareto_front_indices(&scores);
    assert!(indices.contains(&0));
    assert!(indices.contains(&1));
    assert!(!indices.contains(&2));
}
