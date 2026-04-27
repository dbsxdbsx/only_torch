// F3: 学习速度代理（LossSlope）与 plateau tiebreak 的单元测试
//
// 覆盖：
// - `compute_loss_slope_proxy` 数值正确性（单调下降 / 平坦 / 过短 / NaN）
// - `FitnessScore::primary_proxy` 的 serde 向后兼容（字段缺失时默认 None）
// - NSGA-II plateau tiebreak：primary 相同、proxy 不同时，proxy 更高者胜出

use crate::nn::evolution::task::{FitnessScore, ProxyKind, compute_loss_slope_proxy};

// ---------- compute_loss_slope_proxy ----------

#[test]
fn loss_slope_monotone_decreasing_is_positive() {
    // 单调下降曲线 → (head - tail) / n > 0
    let curve: Vec<f32> = (0..20).map(|i| 1.0 - (i as f32) * 0.04).collect();
    let slope = compute_loss_slope_proxy(&curve).expect("应返回 Some");
    assert!(slope > 0.0, "下降曲线 slope 应为正: {slope}");
}

#[test]
fn loss_slope_flat_curve_is_near_zero() {
    let curve = vec![0.5_f32; 16];
    let slope = compute_loss_slope_proxy(&curve).expect("应返回 Some");
    assert!(slope.abs() < 1e-6, "平坦曲线 slope 应接近 0: {slope}");
}

#[test]
fn loss_slope_increasing_is_negative() {
    // 发散曲线 → slope 为负
    let curve: Vec<f32> = (0..10).map(|i| 0.1 + (i as f32) * 0.05).collect();
    let slope = compute_loss_slope_proxy(&curve).expect("应返回 Some");
    assert!(slope < 0.0, "上升曲线 slope 应为负: {slope}");
}

#[test]
fn loss_slope_too_short_returns_none() {
    assert!(compute_loss_slope_proxy(&[]).is_none());
    assert!(compute_loss_slope_proxy(&[1.0]).is_none());
    assert!(compute_loss_slope_proxy(&[1.0, 0.5]).is_none());
}

#[test]
fn loss_slope_with_nan_returns_none() {
    let curve = vec![1.0_f32, 0.8, f32::NAN, 0.5, 0.3];
    assert!(compute_loss_slope_proxy(&curve).is_none());
    let curve = vec![1.0_f32, 0.8, f32::INFINITY, 0.5, 0.3];
    assert!(compute_loss_slope_proxy(&curve).is_none());
}

#[test]
fn loss_slope_window_is_n_over_4() {
    // n=12 → window=3；head = avg(前3)，tail = avg(后3)
    let curve: Vec<f32> = vec![
        1.0, 0.9, 0.8, // head avg = 0.9
        0.7, 0.6, 0.5, 0.4, 0.35, 0.3, // 中段
        0.2, 0.1, 0.0, // tail avg = 0.1
    ];
    // slope = (0.9 - 0.1) / 12 ≈ 0.06667
    let slope = compute_loss_slope_proxy(&curve).expect("应返回 Some");
    assert!((slope - 0.8_f32 / 12.0).abs() < 1e-5, "slope = {slope}");
}

// ---------- FitnessScore serde 向后兼容 ----------

#[test]
fn fitness_score_deserialize_without_primary_proxy() {
    // 旧版 JSON（没有 primary_proxy 字段）应能反序列化，默认 None
    let json = r#"{"primary":0.8,"inference_cost":null,"tiebreak_loss":0.1}"#;
    let score: FitnessScore = serde_json::from_str(json).expect("旧版 JSON 应能反序列化");
    assert_eq!(score.primary, 0.8);
    assert_eq!(score.tiebreak_loss, Some(0.1));
    assert!(score.primary_proxy.is_none());
}

// ---------- plateau tiebreak ----------

#[test]
fn nsga2_tiebreak_prefers_higher_proxy_on_plateau() {
    use crate::nn::evolution::selection::nsga2_select;
    // 三个候选：primary 相同、inference_cost 相同 → 同 rank 同 crowding
    // proxy: a=0.01, b=0.05, c=None
    // 期望：b 最优（proxy 最高），a 次之（proxy 有值），c 最后（proxy 为 None）
    let pool = vec![
        (
            "a",
            FitnessScore {
                primary: 0.9,
                inference_cost: Some(1.0),
                tiebreak_loss: Some(0.2),
                primary_proxy: Some(0.01),
                report: Default::default(),
            },
        ),
        (
            "b",
            FitnessScore {
                primary: 0.9,
                inference_cost: Some(1.0),
                tiebreak_loss: Some(0.2),
                primary_proxy: Some(0.05),
                report: Default::default(),
            },
        ),
        (
            "c",
            FitnessScore {
                primary: 0.9,
                inference_cost: Some(1.0),
                tiebreak_loss: Some(0.2),
                primary_proxy: None,
                report: Default::default(),
            },
        ),
    ];
    let selected = nsga2_select(pool, 2);
    let mut labels: Vec<&str> = selected.iter().map(|(l, _)| *l).collect();
    labels.sort();
    assert_eq!(labels, vec!["a", "b"], "proxy 为 None 的 c 应被淘汰");
}

#[test]
fn nsga2_proxy_takes_precedence_over_tiebreak_loss() {
    // primary 相同、proxy 相反、tiebreak_loss 相反
    // proxy 优先级高于 tiebreak_loss：proxy 高者胜出，即使其 tiebreak_loss 更差
    use crate::nn::evolution::selection::nsga2_select;
    let pool = vec![
        (
            "hi_proxy_hi_loss",
            FitnessScore {
                primary: 0.9,
                inference_cost: Some(1.0),
                tiebreak_loss: Some(0.5),  // 更差
                primary_proxy: Some(0.10), // 更好
                report: Default::default(),
            },
        ),
        (
            "lo_proxy_lo_loss",
            FitnessScore {
                primary: 0.9,
                inference_cost: Some(1.0),
                tiebreak_loss: Some(0.1),  // 更好
                primary_proxy: Some(0.01), // 更差
                report: Default::default(),
            },
        ),
    ];
    let selected = nsga2_select(pool, 1);
    assert_eq!(
        selected[0].0, "hi_proxy_hi_loss",
        "proxy 优先级应高于 tiebreak_loss"
    );
}

// ---------- ProxyKind 基本属性 ----------

#[test]
fn proxy_kind_copy_and_eq() {
    let a = ProxyKind::LossSlope;
    let b = a; // Copy
    assert_eq!(a, b);
}

// ---------- SupervisedTask 启用 proxy 的端到端 ----------

#[test]
fn supervised_task_records_proxy_when_enabled() {
    use crate::nn::evolution::builder::BuildResult;
    use crate::nn::evolution::convergence::*;
    use crate::nn::evolution::gene::*;
    use crate::nn::evolution::task::{EvolutionTask, SupervisedTask};
    use crate::tensor::Tensor;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let data: (Vec<Tensor>, Vec<Tensor>) = (
        vec![
            Tensor::new(&[0.0, 0.0], &[2]),
            Tensor::new(&[0.0, 1.0], &[2]),
            Tensor::new(&[1.0, 0.0], &[2]),
            Tensor::new(&[1.0, 1.0], &[2]),
        ],
        vec![
            Tensor::new(&[0.0], &[1]),
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[0.0], &[1]),
        ],
    );
    let mut task = SupervisedTask::new(data.clone(), data, TaskMetric::Accuracy).unwrap();
    task.configure_proxy(Some(ProxyKind::LossSlope));

    let mut genome = NetworkGenome::minimal(2, 1);
    let inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    let inn_act = genome.next_innovation_number();
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: inn_act,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );

    let mut rng = StdRng::seed_from_u64(42);
    let build: BuildResult = genome.build(&mut rng).unwrap();
    genome.restore_weights(&build).unwrap();

    let convergence = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(10),
        ..Default::default()
    };
    let outcome = task.train(&genome, &build, &convergence, &mut rng).unwrap();
    assert!(outcome.final_loss.is_finite());
    assert!(
        outcome.proxy.is_some(),
        "启用 LossSlope 后 proxy 应有值，实际 None"
    );
}

#[test]
fn supervised_task_proxy_none_when_disabled() {
    use crate::nn::evolution::builder::BuildResult;
    use crate::nn::evolution::convergence::*;
    use crate::nn::evolution::gene::*;
    use crate::nn::evolution::task::{EvolutionTask, SupervisedTask};
    use crate::tensor::Tensor;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let data: (Vec<Tensor>, Vec<Tensor>) = (
        vec![
            Tensor::new(&[0.0, 0.0], &[2]),
            Tensor::new(&[1.0, 1.0], &[2]),
        ],
        vec![Tensor::new(&[0.0], &[1]), Tensor::new(&[0.0], &[1])],
    );
    let task = SupervisedTask::new(data.clone(), data, TaskMetric::Accuracy).unwrap();
    // 不启用 proxy
    let genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(1);
    let build: BuildResult = genome.build(&mut rng).unwrap();
    genome.restore_weights(&build).unwrap();

    let convergence = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(5),
        ..Default::default()
    };
    let outcome = task.train(&genome, &build, &convergence, &mut rng).unwrap();
    assert!(outcome.proxy.is_none(), "默认未启用时 proxy 应为 None");
}
