use crate::nn::evolution::selection::nsga2_select;
use crate::nn::evolution::task::{FitnessScore, ProxyKind, compute_loss_slope_proxy};

#[test]
fn loss_slope_monotone_decreasing_is_positive() {
    let curve: Vec<f32> = (0..20).map(|i| 1.0 - (i as f32) * 0.04).collect();
    let slope = compute_loss_slope_proxy(&curve).expect("应返回 Some");
    assert!(slope > 0.0);
}

#[test]
fn loss_slope_flat_curve_is_near_zero() {
    let curve = vec![0.5_f32; 16];
    let slope = compute_loss_slope_proxy(&curve).expect("应返回 Some");
    assert!(slope.abs() < 1e-6);
}

#[test]
fn loss_slope_increasing_is_negative() {
    let curve: Vec<f32> = (0..10).map(|i| 0.1 + (i as f32) * 0.05).collect();
    let slope = compute_loss_slope_proxy(&curve).expect("应返回 Some");
    assert!(slope < 0.0);
}

#[test]
fn loss_slope_invalid_inputs_return_none() {
    assert!(compute_loss_slope_proxy(&[]).is_none());
    assert!(compute_loss_slope_proxy(&[1.0]).is_none());
    assert!(compute_loss_slope_proxy(&[1.0, 0.8, f32::NAN, 0.5]).is_none());
    assert!(compute_loss_slope_proxy(&[1.0, f32::INFINITY, 0.5]).is_none());
}

#[test]
fn fitness_score_deserialize_without_primary_proxy() {
    let json = r#"{"primary":0.8,"inference_cost":null,"tiebreak_loss":0.1}"#;
    let score: FitnessScore = serde_json::from_str(json).expect("旧版 JSON 应能反序列化");

    assert_eq!(score.primary, 0.8);
    assert_eq!(score.tiebreak_loss, Some(0.1));
    assert!(score.primary_proxy.is_none());
}

#[test]
fn nsga2_tiebreak_prefers_higher_proxy_on_plateau() {
    let pool = vec![
        ("a", score_with_proxy(Some(0.01), 0.2)),
        ("b", score_with_proxy(Some(0.05), 0.2)),
        ("c", score_with_proxy(None, 0.2)),
    ];
    let selected = nsga2_select(pool, 2);
    let mut labels: Vec<&str> = selected.iter().map(|(label, _)| *label).collect();
    labels.sort();

    assert_eq!(labels, vec!["a", "b"]);
}

#[test]
fn nsga2_proxy_takes_precedence_over_tiebreak_loss() {
    let pool = vec![
        ("hi_proxy_hi_loss", score_with_proxy(Some(0.10), 0.5)),
        ("lo_proxy_lo_loss", score_with_proxy(Some(0.01), 0.1)),
    ];
    let selected = nsga2_select(pool, 1);

    assert_eq!(selected[0].0, "hi_proxy_hi_loss");
}

#[test]
fn proxy_kind_copy_and_eq() {
    let a = ProxyKind::LossSlope;
    let b = a;
    assert_eq!(a, b);
}

fn score_with_proxy(primary_proxy: Option<f32>, tiebreak_loss: f32) -> FitnessScore {
    FitnessScore {
        primary: 0.9,
        inference_cost: Some(1.0),
        tiebreak_loss: Some(tiebreak_loss),
        primary_proxy,
        report: Default::default(),
    }
}
