use crate::nn::evolution::gene::*;
use crate::nn::evolution::mutation::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn constraints() -> SizeConstraints {
    SizeConstraints::default()
}

/// 构造含隐藏层的基因组：Input(2) → Linear(4) → ReLU → [Linear(1)]
fn genome_with_hidden() -> NetworkGenome {
    let mut g = NetworkGenome::minimal(2, 1);
    let i1 = g.next_innovation_number();
    let i2 = g.next_innovation_number();
    g.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    g.layers.insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    g
}

// ==================== InsertLayerMutation ====================

#[test]
fn test_insert_layer_happy_path() {
    let mut g = NetworkGenome::minimal(2, 1);
    let mut r = rng();
    let m = InsertLayerMutation::default();
    let c = constraints();

    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    assert!(g.layer_count() >= 2);
    assert!(g.resolve_dimensions().is_ok());
    // 输出头仍是最后一个 enabled 层
    let last = g.layers.iter().rev().find(|l| l.enabled).unwrap();
    assert_eq!(last.layer_config, LayerConfig::Linear { out_features: 1 });
}

#[test]
fn test_insert_layer_max_layers_reached() {
    let g = genome_with_hidden();
    let c = SizeConstraints {
        max_layers: 3,
        ..constraints()
    };
    let m = InsertLayerMutation::default();
    assert!(!m.is_applicable(&g, &c));
}

#[test]
fn test_insert_layer_no_consecutive_activation() {
    // 从 minimal 出发插入多次，不应出现连续 Activation
    let mut g = NetworkGenome::minimal(2, 1);
    let mut r = rng();
    let m = InsertLayerMutation::default();
    let c = constraints();

    for _ in 0..20 {
        if m.is_applicable(&g, &c) {
            let _ = m.apply(&mut g, &c, &mut r);
        }
    }

    let enabled: Vec<&LayerConfig> = g
        .layers
        .iter()
        .filter(|l| l.enabled)
        .map(|l| &l.layer_config)
        .collect();
    for w in enabled.windows(2) {
        let both_act = matches!(w[0], LayerConfig::Activation { .. })
            && matches!(w[1], LayerConfig::Activation { .. });
        assert!(!both_act, "发现连续 Activation: {:?} {:?}", w[0], w[1]);
    }
}

// ==================== RemoveLayerMutation ====================

#[test]
fn test_remove_layer_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = RemoveLayerMutation;
    let c = constraints();

    let before = g.layer_count();
    m.apply(&mut g, &c, &mut r).unwrap();
    assert_eq!(g.layer_count(), before - 1);
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_remove_layer_minimal_not_applicable() {
    let g = NetworkGenome::minimal(2, 1);
    let m = RemoveLayerMutation;
    assert!(!m.is_applicable(&g, &constraints()));
}

#[test]
fn test_remove_layer_preserves_output_head() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = RemoveLayerMutation;
    let c = constraints();

    // 连续删除直到只剩输出头
    while m.is_applicable(&g, &c) {
        m.apply(&mut g, &c, &mut r).unwrap();
    }

    assert_eq!(g.layer_count(), 1);
    let last = g.layers.iter().find(|l| l.enabled).unwrap();
    assert_eq!(
        last.layer_config,
        LayerConfig::Linear {
            out_features: g.output_dim
        }
    );
}

// ==================== ReplaceLayerTypeMutation ====================

#[test]
fn test_replace_layer_type_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = ReplaceLayerTypeMutation::default();
    let c = constraints();

    // genome_with_hidden 有一个 ReLU
    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    // 确认 Activation 层被替换为不同的类型
    let act_layer = g
        .layers
        .iter()
        .find(|l| matches!(l.layer_config, LayerConfig::Activation { .. }))
        .unwrap();
    // 可能替换为 ReLU 以外的任何激活
    assert!(g.resolve_dimensions().is_ok());
    assert_ne!(
        act_layer.layer_config,
        LayerConfig::Activation {
            activation_type: ActivationType::ReLU,
        }
    );
}

#[test]
fn test_replace_layer_type_no_activation_not_applicable() {
    // 只有 Linear 层（无 Activation），不可替换
    let mut g = NetworkGenome::minimal(2, 1);
    let inn = g.next_innovation_number();
    g.layers.insert(
        0,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let m = ReplaceLayerTypeMutation::default();
    assert!(!m.is_applicable(&g, &constraints()));
}

#[test]
fn test_replace_no_alternative_returns_error() {
    let mut g = genome_with_hidden(); // 含一个 ReLU 层
    let mut r = rng();
    // 可用列表只有 ReLU，与当前层相同 → 无替代
    let m = ReplaceLayerTypeMutation::new(vec![ActivationType::ReLU]);
    let c = constraints();

    assert!(m.is_applicable(&g, &c));
    assert!(m.apply(&mut g, &c, &mut r).is_err());
}

#[test]
fn test_replace_does_not_touch_output_head() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = ReplaceLayerTypeMutation::default();
    let c = constraints();

    for _ in 0..20 {
        if m.is_applicable(&g, &c) {
            let _ = m.apply(&mut g, &c, &mut r);
        }
    }

    let last = g.layers.iter().rev().find(|l| l.enabled).unwrap();
    assert_eq!(last.layer_config, LayerConfig::Linear { out_features: 1 });
}

// ==================== GrowHiddenSizeMutation ====================

#[test]
fn test_grow_hidden_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = GrowHiddenSizeMutation;
    let c = constraints();

    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    let linear = g
        .layers
        .iter()
        .find(|l| {
            l.enabled
                && matches!(l.layer_config, LayerConfig::Linear { out_features } if out_features != g.output_dim)
        })
        .unwrap();
    if let LayerConfig::Linear { out_features } = linear.layer_config {
        assert!(out_features > 4);
    }
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_grow_hidden_max_reached_not_applicable() {
    let mut g = genome_with_hidden();
    // 设置隐藏层已达上限
    g.layers[0].layer_config = LayerConfig::Linear { out_features: 64 };
    let c = SizeConstraints {
        max_hidden_size: 64,
        ..constraints()
    };
    let m = GrowHiddenSizeMutation;
    assert!(!m.is_applicable(&g, &c));
}

#[test]
fn test_grow_hidden_max_params_violation() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = GrowHiddenSizeMutation;
    // 非常低的 max_total_params，增长必定超标
    let c = SizeConstraints {
        max_total_params: 1,
        ..constraints()
    };
    let result = m.apply(&mut g, &c, &mut r);
    // 要么 is_applicable=false（因为增长后必超标），要么 apply 返回 ConstraintViolation
    // 不管哪种情况，基因组不应被修改
    assert!(result.is_err() || !m.is_applicable(&g, &c));
}

#[test]
fn test_grow_does_not_touch_output_head() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = GrowHiddenSizeMutation;
    let c = constraints();

    for _ in 0..20 {
        if m.is_applicable(&g, &c) {
            let _ = m.apply(&mut g, &c, &mut r);
        }
    }

    let last = g.layers.iter().rev().find(|l| l.enabled).unwrap();
    assert_eq!(last.layer_config, LayerConfig::Linear { out_features: 1 });
}

// ==================== ShrinkHiddenSizeMutation ====================

#[test]
fn test_shrink_hidden_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = ShrinkHiddenSizeMutation;
    let c = constraints();

    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    let linear = g
        .layers
        .iter()
        .find(|l| {
            l.enabled
                && matches!(l.layer_config, LayerConfig::Linear { out_features } if out_features != g.output_dim)
        })
        .unwrap();
    if let LayerConfig::Linear { out_features } = linear.layer_config {
        assert!(out_features < 4);
    }
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_shrink_hidden_min_reached_not_applicable() {
    let mut g = genome_with_hidden();
    g.layers[0].layer_config = LayerConfig::Linear { out_features: 1 };
    let c = SizeConstraints {
        min_hidden_size: 1,
        ..constraints()
    };
    let m = ShrinkHiddenSizeMutation;
    assert!(!m.is_applicable(&g, &c));
}

#[test]
fn test_shrink_does_not_touch_output_head() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = ShrinkHiddenSizeMutation;
    let c = constraints();

    for _ in 0..20 {
        if m.is_applicable(&g, &c) {
            let _ = m.apply(&mut g, &c, &mut r);
        }
    }

    let last = g.layers.iter().rev().find(|l| l.enabled).unwrap();
    assert_eq!(last.layer_config, LayerConfig::Linear { out_features: 1 });
}

// ==================== MutateLayerParamMutation ====================

#[test]
fn test_mutate_layer_param_leaky_relu() {
    let mut g = NetworkGenome::minimal(2, 1);
    let inn = g.next_innovation_number();
    g.layers.insert(
        0,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::LeakyReLU { alpha: 0.01 },
            },
            enabled: true,
        },
    );

    let mut r = rng();
    let m = MutateLayerParamMutation;
    let c = constraints();

    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    if let LayerConfig::Activation {
        activation_type: ActivationType::LeakyReLU { alpha },
    } = g.layers[0].layer_config
    {
        assert!((0.001..=0.5).contains(&alpha));
    } else {
        panic!("层类型应保持为 LeakyReLU");
    }
}

#[test]
fn test_mutate_layer_param_not_applicable_without_parameterized() {
    let g = genome_with_hidden(); // 只有 ReLU（无参数可变异）
    let m = MutateLayerParamMutation;
    // ReLU 不是参数化的激活函数
    assert!(!m.is_applicable(&g, &constraints()));
}

// ==================== MutateLossFunctionMutation ====================

#[test]
fn test_mutate_loss_function_happy_path() {
    let mut g = NetworkGenome::minimal(2, 1); // binary classification
    let mut r = rng();
    let m = MutateLossFunctionMutation {
        task_metric: TaskMetric::Accuracy,
    };
    let c = constraints();

    // Accuracy + output_dim=1 → [BCE, MSE]，可变异
    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    assert!(g.training_config.loss_override.is_some());
    let loss = g.training_config.loss_override.unwrap();
    // 默认推断是 BCE，变异后应该不同（MSE）
    assert_ne!(loss, LossType::BCE);
    assert_eq!(loss, LossType::MSE);
}

#[test]
fn test_mutate_loss_not_applicable_single_loss() {
    let g = NetworkGenome::minimal(2, 3); // multiclass
    let m = MutateLossFunctionMutation {
        task_metric: TaskMetric::Accuracy,
    };
    // Accuracy + output_dim=3 → 只有 [CrossEntropy]，无法变异
    assert!(!m.is_applicable(&g, &constraints()));
}

#[test]
fn test_mutate_loss_not_applicable_r2() {
    let g = NetworkGenome::minimal(2, 1);
    let m = MutateLossFunctionMutation {
        task_metric: TaskMetric::R2,
    };
    // R2 → 只有 [MSE]，无法变异
    assert!(!m.is_applicable(&g, &constraints()));
}

// ==================== MutationRegistry ====================

#[test]
fn test_default_registry_has_7_mutations() {
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy);
    assert_eq!(reg.len(), 7);
}

#[test]
fn test_registry_apply_random_no_applicable() {
    let mut g = NetworkGenome::minimal(2, 1);
    let mut r = rng();
    // 只注册一个不可能成功的变异
    let mut reg = MutationRegistry::new();
    reg.register(1.0, RemoveLayerMutation);
    // minimal 只有输出头，RemoveLayer 不适用
    let result = reg.apply_random(&mut g, &constraints(), &mut r);
    assert!(result.is_err());
}

#[test]
fn test_registry_retries_on_apply_failure() {
    let mut g = genome_with_hidden();
    let mut r = rng();

    let mut reg = MutationRegistry::new();
    reg.register(1000.0, GrowHiddenSizeMutation);
    reg.register(0.001, ShrinkHiddenSizeMutation);

    // max_total_params=20 使 GrowHidden 的 apply 必定失败（当前 17，任何增长都超 20）
    // 但 is_applicable 仍返回 true（out_features=4 < max_hidden_size=64）
    // 重试后应选中 ShrinkHidden 并成功
    let c = SizeConstraints {
        max_total_params: 20,
        ..constraints()
    };

    let name = reg.apply_random(&mut g, &c, &mut r).unwrap();
    assert_eq!(name, "ShrinkHiddenSize");
}

#[test]
fn test_registry_apply_random_returns_name() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy);
    let name = reg.apply_random(&mut g, &constraints(), &mut r).unwrap();
    assert!(!name.is_empty());
    assert_eq!(g.generated_by, name);
}

// ==================== 输出头保护 ====================

#[test]
fn test_insert_never_after_output_head() {
    let mut g = NetworkGenome::minimal(2, 1);
    let mut r = rng();
    let m = InsertLayerMutation::default();
    let c = constraints();

    for _ in 0..50 {
        if m.is_applicable(&g, &c) {
            m.apply(&mut g, &c, &mut r).unwrap();
        }
    }

    let last = g.layers.iter().rev().find(|l| l.enabled).unwrap();
    assert_eq!(
        last.layer_config,
        LayerConfig::Linear {
            out_features: g.output_dim
        }
    );
}

#[test]
fn test_minimal_only_insert_applicable() {
    let g = NetworkGenome::minimal(2, 1);
    let c = constraints();

    assert!(InsertLayerMutation::default().is_applicable(&g, &c));
    assert!(!RemoveLayerMutation.is_applicable(&g, &c));
    assert!(!ReplaceLayerTypeMutation::default().is_applicable(&g, &c));
    assert!(!GrowHiddenSizeMutation.is_applicable(&g, &c));
    assert!(!ShrinkHiddenSizeMutation.is_applicable(&g, &c));
    assert!(!MutateLayerParamMutation.is_applicable(&g, &c));
}

// ==================== 组合鲁棒性 ====================

#[test]
fn test_random_mutations_keep_genome_valid() {
    let mut g = NetworkGenome::minimal(2, 1);
    let mut r = rng();
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy);
    let c = constraints();

    for _ in 0..50 {
        let _ = reg.apply_random(&mut g, &c, &mut r);

        // 每次变异后，基因组必须合法
        assert!(g.resolve_dimensions().is_ok(), "维度链断裂: {g}");
        assert!(g.total_params().is_ok(), "参数量计算失败: {g}");
        assert!(g.layer_count() >= 1, "层数为零: {g}");

        // 输出头完整
        let last = g.layers.iter().rev().find(|l| l.enabled).unwrap();
        assert_eq!(
            last.layer_config,
            LayerConfig::Linear {
                out_features: g.output_dim
            },
            "输出头被破坏: {g}"
        );

        // 创新号唯一
        let inns: Vec<u64> = g.layers.iter().map(|l| l.innovation_number).collect();
        let unique: std::collections::HashSet<u64> = inns.iter().copied().collect();
        assert_eq!(inns.len(), unique.len(), "创新号重复: {g}");
    }
}

#[test]
fn test_seed_reproducibility() {
    let run = |seed: u64| -> String {
        let mut g = NetworkGenome::minimal(2, 1);
        let mut r = StdRng::seed_from_u64(seed);
        let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy);
        let c = constraints();

        for _ in 0..20 {
            let _ = reg.apply_random(&mut g, &c, &mut r);
        }
        format!("{g}")
    };

    assert_eq!(run(123), run(123));
    assert_eq!(run(999), run(999));
}
