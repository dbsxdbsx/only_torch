use crate::nn::evolution::gene::*;
use crate::tensor::Tensor;
use std::collections::HashMap;

// ==================== 基本构造 ====================

#[test]
fn test_minimal_creates_correct_genome() {
    let genome = NetworkGenome::minimal(2, 1);

    assert_eq!(genome.input_dim, 2);
    assert_eq!(genome.output_dim, 1);
    assert_eq!(genome.layers().len(), 1);
    assert_eq!(genome.skip_edges().len(), 0);

    let output_head = &genome.layers()[0];
    assert_eq!(output_head.innovation_number, 1);
    assert!(output_head.enabled);
    assert_eq!(
        output_head.layer_config,
        LayerConfig::Linear { out_features: 1 }
    );
}

#[test]
fn test_minimal_output_dim_matches() {
    for out in [1, 3, 10] {
        let genome = NetworkGenome::minimal(4, out);
        if let LayerConfig::Linear { out_features } = &genome.layers().last().unwrap().layer_config {
            assert_eq!(*out_features, out);
        } else {
            panic!("输出头必须是 Linear");
        }
    }
}

#[test]
fn test_minimal_total_params() {
    // minimal(2, 1): Linear in=2, out=1 → W(2×1) + b(1) = 3
    let genome = NetworkGenome::minimal(2, 1);
    assert_eq!(genome.total_params().unwrap(), 3);

    // minimal(3, 2): Linear in=3, out=2 → W(3×2) + b(2) = 8
    let genome = NetworkGenome::minimal(3, 2);
    assert_eq!(genome.total_params().unwrap(), 8);
}

#[test]
fn test_minimal_layer_count() {
    let genome = NetworkGenome::minimal(2, 1);
    assert_eq!(genome.layer_count(), 1);
}

#[test]
#[should_panic(expected = "input_dim 不能为零")]
fn test_minimal_zero_input_panics() {
    NetworkGenome::minimal(0, 1);
}

#[test]
#[should_panic(expected = "output_dim 不能为零")]
fn test_minimal_zero_output_panics() {
    NetworkGenome::minimal(1, 0);
}

// ==================== 创新号 ====================

#[test]
fn test_innovation_numbers_monotonic() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let a = genome.next_innovation_number();
    let b = genome.next_innovation_number();
    let c = genome.next_innovation_number();

    assert!(a < b);
    assert!(b < c);
    assert_eq!(a, 2); // 0=INPUT, 1=输出头, next=2
    assert_eq!(b, 3);
    assert_eq!(c, 4);
}

#[test]
fn test_innovation_numbers_unique() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let mut seen = std::collections::HashSet::new();
    // 输出头的创新号
    seen.insert(genome.layers()[0].innovation_number);
    for _ in 0..100 {
        let id = genome.next_innovation_number();
        assert!(seen.insert(id), "创新号 {id} 重复");
    }
}

// ==================== disabled 层语义 ====================

#[test]
fn test_disabled_layer_not_in_params() {
    let mut genome = NetworkGenome::minimal(2, 1);

    // 在输出头之前插入一个 disabled 的隐藏层
    let hidden = LayerGene {
        innovation_number: genome.next_innovation_number(),
        layer_config: LayerConfig::Linear { out_features: 4 },
        enabled: false,
    };
    genome.layers_mut().insert(0, hidden);

    // disabled 层不计入参数量（仍然只有输出头的参数）
    assert_eq!(genome.total_params().unwrap(), 3); // 2*1 + 1
    assert_eq!(genome.layer_count(), 1);
}

#[test]
fn test_disabled_layer_enabled_comparison() {
    let mut genome = NetworkGenome::minimal(2, 1);

    let hidden = LayerGene {
        innovation_number: genome.next_innovation_number(),
        layer_config: LayerConfig::Linear { out_features: 4 },
        enabled: true,
    };
    genome.layers_mut().insert(0, hidden);

    // enabled: Input(2) → Linear(4) → Linear(1)
    // Linear(4): W(2×4)+b(4)=12, Linear(1): W(4×1)+b(1)=5 → 17
    assert_eq!(genome.total_params().unwrap(), 17);
    assert_eq!(genome.layer_count(), 2);

    // disable 隐藏层
    genome.layers_mut()[0].enabled = false;
    // 现在只有输出头 Linear(1): W(2×1)+b(1)=3
    assert_eq!(genome.total_params().unwrap(), 3);
    assert_eq!(genome.layer_count(), 1);
}

// ==================== 维度推导 ====================

#[test]
fn test_resolve_simple_linear_chain() {
    let genome = NetworkGenome::minimal(2, 1);
    let resolved = genome.resolve_dimensions().unwrap();

    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].in_dim, 2);
    assert_eq!(resolved[0].out_dim, 1);
}

#[test]
fn test_resolve_with_hidden_layers() {
    // Input(2) → Linear(4) → ReLU → Linear(1)
    let mut genome = NetworkGenome::minimal(2, 1);
    let inn1 = genome.next_innovation_number();
    let inn2 = genome.next_innovation_number();

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: inn1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: inn2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );

    let resolved = genome.resolve_dimensions().unwrap();
    assert_eq!(resolved.len(), 3);

    // Linear(4): in=2, out=4
    assert_eq!(resolved[0].in_dim, 2);
    assert_eq!(resolved[0].out_dim, 4);
    // ReLU: in=4, out=4（透传）
    assert_eq!(resolved[1].in_dim, 4);
    assert_eq!(resolved[1].out_dim, 4);
    // Linear(1): in=4, out=1（输出头）
    assert_eq!(resolved[2].in_dim, 4);
    assert_eq!(resolved[2].out_dim, 1);
}

#[test]
fn test_resolve_skips_disabled() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Linear { out_features: 8 },
            enabled: false,
        },
    );

    let resolved = genome.resolve_dimensions().unwrap();
    // disabled 层被跳过，只有输出头
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].in_dim, 2); // 直接从 input_dim
    assert_eq!(resolved[0].out_dim, 1);
}

#[test]
fn test_resolve_activation_passthrough() {
    // Activation 和 Dropout 不改变维度
    let mut genome = NetworkGenome::minimal(4, 2);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::Tanh,
            },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Dropout { p: 0.5 },
            enabled: true,
        },
    );

    let resolved = genome.resolve_dimensions().unwrap();
    // Tanh: 4→4, Dropout: 4→4, Linear(2): 4→2
    assert_eq!(resolved[0].out_dim, 4);
    assert_eq!(resolved[1].out_dim, 4);
    assert_eq!(resolved[2].in_dim, 4);
}

// ==================== Skip edge 聚合维度 ====================

#[test]
fn test_resolve_skip_add_same_dim() {
    // Input(4) → Linear(4) → [Linear(2)]
    // skip edge: INPUT → Linear(2), strategy=Add
    // Add 要求 main_path(4) == skip_source(4) ✓
    let mut genome = NetworkGenome::minimal(4, 2);
    let i1 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    let resolved = genome.resolve_dimensions().unwrap();
    // Linear(4): in=4, out=4
    // Linear(2)(输出头): Add(main=4, skip=4) → in=4, out=2
    assert_eq!(resolved[1].in_dim, 4);
    assert_eq!(resolved[1].out_dim, 2);
}

#[test]
fn test_resolve_skip_add_different_dim_error() {
    // Input(2) → Linear(4) → [Linear(1)]
    // skip edge: INPUT(dim=2) → Linear(1), strategy=Add
    // Add 要求 main_path(4) == skip_source(2)，不兼容 → Err
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    let result = genome.resolve_dimensions();
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("维度不兼容"));
}

#[test]
fn test_resolve_skip_concat() {
    // Input(2) → Linear(4) → [Linear(1)]
    // skip edge: INPUT(dim=2) → 输出头, strategy=Concat
    // Concat: 输入维度 = 4 + 2 = 6
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Concat { dim: -1 },
        enabled: true,
    });

    let resolved = genome.resolve_dimensions().unwrap();
    // 输出头: Concat(main=4, skip=2) → in=6, out=1
    assert_eq!(resolved[1].in_dim, 6);
    assert_eq!(resolved[1].out_dim, 1);
}

#[test]
fn test_resolve_skip_mean_same_dim() {
    // 与 Add 相同规则
    let mut genome = NetworkGenome::minimal(4, 2);
    let i1 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Mean,
        enabled: true,
    });

    let resolved = genome.resolve_dimensions().unwrap();
    assert_eq!(resolved[1].in_dim, 4);
}

#[test]
fn test_resolve_skip_max_different_dim_error() {
    // Max 与 Add 相同约束
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Max,
        enabled: true,
    });

    assert!(genome.resolve_dimensions().is_err());
}

#[test]
fn test_resolve_disabled_skip_edge_ignored() {
    // disabled 的 skip edge 不参与维度计算
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Add, // 维度不兼容(2≠4)，但 disabled
        enabled: false,
    });

    // disabled skip edge 被忽略，不触发维度检查
    assert!(genome.resolve_dimensions().is_ok());
}

// ==================== total_params 基于 resolve_dimensions ====================

#[test]
fn test_total_params_with_hidden_layers() {
    // Input(2) → Linear(4) → ReLU → [Linear(1)]
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );

    // Linear(4): 2*4+4=12, ReLU: 0, Linear(1): 4*1+1=5 → 17
    assert_eq!(genome.total_params().unwrap(), 17);
}

#[test]
fn test_total_params_concat_affects_downstream() {
    // Input(2) → Linear(4) → [Linear(1)]
    // skip: INPUT(2) → 输出头, Concat
    // 输出头 in_dim = 4+2 = 6, params = 6*1+1 = 7
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Concat { dim: -1 },
        enabled: true,
    });

    // Linear(4): 2*4+4=12, Linear(1) with Concat(in=6): 6*1+1=7 → 19
    assert_eq!(genome.total_params().unwrap(), 19);
}

// ==================== 权重快照 ====================

#[test]
fn test_new_genome_empty_snapshots() {
    let genome = NetworkGenome::minimal(2, 1);
    assert!(!genome.has_weight_snapshots());
    assert!(genome.weight_snapshots().is_empty());
}

#[test]
fn test_clone_weight_snapshots_independent() {
    let mut genome = NetworkGenome::minimal(2, 1);

    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let mut snapshots = HashMap::new();
    snapshots.insert(1u64, vec![tensor]);
    genome.set_weight_snapshots(snapshots);

    let cloned = genome.clone();

    // 修改原件的快照
    genome.set_weight_snapshots(HashMap::new());

    // 克隆体不受影响
    assert!(cloned.has_weight_snapshots());
    assert!(!genome.has_weight_snapshots());
}

// ==================== Display ====================

#[test]
fn test_display_minimal() {
    let genome = NetworkGenome::minimal(2, 1);
    assert_eq!(format!("{genome}"), "Input(2) → [Linear(1)]");
}

#[test]
fn test_display_with_hidden_layers() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );

    assert_eq!(
        format!("{genome}"),
        "Input(2) → Linear(4) → ReLU → [Linear(1)]"
    );
}

#[test]
fn test_display_disabled_not_shown() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Linear { out_features: 8 },
            enabled: false,
        },
    );

    // disabled 层不出现在 Display 中
    assert_eq!(format!("{genome}"), "Input(2) → [Linear(1)]");
}

#[test]
fn test_display_various_layer_types() {
    let mut genome = NetworkGenome::minimal(4, 2);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();
    let i3 = genome.next_innovation_number();

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 8 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::LeakyReLU { alpha: 0.01 },
            },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: i3,
            layer_config: LayerConfig::Dropout { p: 0.5 },
            enabled: true,
        },
    );

    assert_eq!(
        format!("{genome}"),
        "Input(4) → Linear(8) → LeakyReLU(0.01) → Dropout(0.5) → [Linear(2)]"
    );
}

// ==================== Display: skip edge 注解 ====================

#[test]
fn test_display_single_skip_edge() {
    // Input(2) → Linear(4) → [Linear(1)]
    // skip: Input(2) ──(Add)──→ Linear(4)  (维度兼容：input_dim=2, Linear(4) main_in=2)
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: i1,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Concat { dim: -1 },
        enabled: true,
    });

    let display = format!("{genome}");
    assert_eq!(
        display,
        "Input(2) → Linear(4) → [Linear(1)]\n\
         \x20 └─ skip: Linear(4) ──(Concat)──→ [Linear(1)]"
    );
}

#[test]
fn test_display_multiple_skip_edges() {
    // Input(2) → Linear(2) → ReLU → [Linear(1)]
    // skip1: Input(2) ──(Add)──→ ReLU  (维度 2==2 OK)
    // skip2: Linear(2) ──(Concat)──→ [Linear(1)]
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 2 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );

    let s1 = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: s1,
        from_innovation: INPUT_INNOVATION,
        to_innovation: i2,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });
    let s2 = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: s2,
        from_innovation: i1,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Concat { dim: -1 },
        enabled: true,
    });

    let display = format!("{genome}");
    assert!(
        display.contains("├─ skip:"),
        "多条 skip edge 时非末尾应用 ├─，实际: {display}"
    );
    assert!(
        display.contains("└─ skip:"),
        "末尾 skip edge 应用 └─，实际: {display}"
    );
    assert!(
        display.contains("(Add)"),
        "应包含 Add 策略，实际: {display}"
    );
    assert!(
        display.contains("(Concat)"),
        "应包含 Concat 策略，实际: {display}"
    );
}

#[test]
fn test_display_disabled_skip_edge_not_shown() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Add,
        enabled: false, // disabled
    });

    // disabled skip edge 不出现 → 单行输出
    assert_eq!(format!("{genome}"), "Input(2) → Linear(4) → [Linear(1)]");
}

#[test]
fn test_main_path_summary_ignores_skip_edges() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: i1,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Concat { dim: -1 },
        enabled: true,
    });

    // main_path_summary 始终单行，不含 skip
    assert_eq!(
        genome.main_path_summary(),
        "Input(2) → Linear(4) → [Linear(1)]"
    );
    // Display 包含 skip
    assert!(format!("{genome}").contains("skip:"));
}

// ==================== Display: 重名消歧 ====================

#[test]
fn test_display_duplicate_hidden_layers_disambiguated() {
    // Input(2) → Linear(4)#1 → ReLU → Linear(4)#2 → [Linear(1)]
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();
    let i3 = genome.next_innovation_number();

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: i3,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    assert_eq!(
        format!("{genome}"),
        "Input(2) → Linear(4)#1 → ReLU → Linear(4)#2 → [Linear(1)]"
    );
}

#[test]
fn test_display_skip_edge_uses_disambiguated_names() {
    // Input(2) → Linear(4)#1 → ReLU#1 → Linear(4)#2 → ReLU#2 → [Linear(1)]
    // skip: Linear(4)#1 ──(Add)──→ ReLU#2
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();
    let i3 = genome.next_innovation_number();
    let i4 = genome.next_innovation_number();

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: i3,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        3,
        LayerGene {
            innovation_number: i4,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );

    let s = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: s,
        from_innovation: i1,
        to_innovation: i4,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    let display = format!("{genome}");
    assert!(
        display.contains("Linear(4)#1 →"),
        "第一个 Linear(4) 应有 #1: {display}"
    );
    assert!(
        display.contains("Linear(4)#2 →"),
        "第二个 Linear(4) 应有 #2: {display}"
    );
    assert!(
        display.contains("skip: Linear(4)#1 ──(Add)──→ ReLU#2"),
        "skip edge 应使用消歧名称: {display}"
    );
}

#[test]
fn test_display_same_config_hidden_vs_output_no_suffix() {
    // Hidden Linear(4) 与 output head [Linear(4)] 显示名不同，无需 #N
    let mut genome = NetworkGenome::minimal(2, 4);
    let i1 = genome.next_innovation_number();

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    assert_eq!(format!("{genome}"), "Input(2) → Linear(4) → [Linear(4)]");
}

#[test]
fn test_main_path_summary_duplicate_names_disambiguated() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    assert_eq!(
        genome.main_path_summary(),
        "Input(2) → Linear(4)#1 → Linear(4)#2 → [Linear(1)]"
    );
}

// ==================== TrainingConfig ====================

#[test]
fn test_training_config_default() {
    let config = TrainingConfig::default();
    assert_eq!(config.optimizer_type, OptimizerType::Adam);
    assert!((config.learning_rate - 0.01).abs() < 1e-6);
    assert_eq!(config.batch_size, None);
    assert!((config.weight_decay - 0.0).abs() < 1e-6);
    assert_eq!(config.loss_override, None);
}

#[test]
fn test_minimal_uses_default_training_config() {
    let genome = NetworkGenome::minimal(2, 1);
    let default = TrainingConfig::default();
    assert_eq!(
        genome.training_config.optimizer_type,
        default.optimizer_type
    );
    assert!((genome.training_config.learning_rate - default.learning_rate).abs() < 1e-6);
}

// ==================== 边界条件与组合 ====================

#[test]
fn test_all_layers_disabled_returns_error() {
    let mut genome = NetworkGenome::minimal(2, 1);
    genome.layers_mut()[0].enabled = false;

    let result = genome.resolve_dimensions();
    assert!(result.is_err());
}

#[test]
fn test_large_network_params() {
    // 验证多层叠加的参数量计算
    // Input(10) → Linear(20) → Linear(30) → [Linear(5)]
    let mut genome = NetworkGenome::minimal(10, 5);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 20 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Linear { out_features: 30 },
            enabled: true,
        },
    );

    // Linear(20): 10*20+20=220
    // Linear(30): 20*30+30=630
    // Linear(5):  30*5+5=155
    // Total: 1005
    assert_eq!(genome.total_params().unwrap(), 1005);
}

#[test]
fn test_generated_by_field() {
    let genome = NetworkGenome::minimal(2, 1);
    assert_eq!(genome.generated_by, "minimal");
}

// ==================== skip edge 补强：中间层源 + 悬空引用 + 多路汇聚 ====================

#[test]
fn test_resolve_skip_from_intermediate_layer() {
    // Input(2) → Linear(4) → Linear(3) → [Linear(1)]
    // skip: Linear(4) → 输出头, Concat
    // 输出头: Concat(main=3, skip=4) → in=7
    // 与现有 Concat 测试不同：源是中间层而非 INPUT，验证 dim_map 对中间层的查找
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Linear { out_features: 3 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: i1,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Concat { dim: -1 },
        enabled: true,
    });

    let resolved = genome.resolve_dimensions().unwrap();
    // 无 skip 时 in_dim=3；有 skip 时 in_dim=7 —— 只有 dim_map 正确查找中间层才能得到 7
    assert_eq!(resolved[2].in_dim, 7);
    assert_eq!(resolved[2].out_dim, 1);
}

#[test]
fn test_resolve_skip_add_from_intermediate_different_dim_error() {
    // Input(2) → Linear(4) → Linear(3) → [Linear(1)]
    // skip: Linear(4)(out=4) → 输出头(main=3), Add
    // Add 要求 3==4，不兼容 → Err
    // 验证 dim_map 确实查找到中间层的维度(4)，而非跳过检查
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Linear { out_features: 3 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: i1,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    assert!(genome.resolve_dimensions().is_err());
}

#[test]
fn test_resolve_skip_invalid_from_innovation() {
    // skip edge 的 from_innovation 指向不存在的创新号 → InvalidSkipEdge
    let mut genome = NetworkGenome::minimal(2, 1);
    let output_inn = genome.layers().last().unwrap().innovation_number;

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: 999,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    let result = genome.resolve_dimensions();
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("无效 skip edge"));
}

#[test]
fn test_resolve_multiple_skip_edges_concat() {
    // Input(2) → Linear(3) → Linear(4) → [Linear(1)]
    // skip1: INPUT(2) → 输出头, Concat
    // skip2: Linear(3) → 输出头, Concat
    // 输出头: Concat(main=4, INPUT=2, Linear(3)=3) → in=9
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 3 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let s1 = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: s1,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Concat { dim: -1 },
        enabled: true,
    });

    let s2 = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: s2,
        from_innovation: i1,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Concat { dim: -1 },
        enabled: true,
    });

    let resolved = genome.resolve_dimensions().unwrap();
    // 无 skip: in_dim=4; 有 2 条 skip: Concat(4,2,3) = 9
    assert_eq!(resolved[2].in_dim, 9);
}

#[test]
fn test_resolve_multiple_skip_edges_add_mixed_dims_error() {
    // Input(2) → Linear(3) → Linear(4) → [Linear(1)]
    // skip1: INPUT(2) → 输出头, Add
    // skip2: Linear(3) → 输出头, Add
    // Add 要求全部维度相同: main=4, skip1=2, skip2=3 → 不兼容 → Err
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();
    let output_inn = genome.layers().last().unwrap().innovation_number;

    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 3 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let s1 = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: s1,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    let s2 = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: s2,
        from_innovation: i1,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    assert!(genome.resolve_dimensions().is_err());
}

// ==================== TaskMetric + loss 推断 ====================

#[test]
fn test_compatible_losses_accuracy_binary() {
    let losses = compatible_losses(&TaskMetric::Accuracy, 1);
    assert_eq!(losses, vec![LossType::BCE, LossType::MSE]);
}

#[test]
fn test_compatible_losses_accuracy_multiclass() {
    let losses = compatible_losses(&TaskMetric::Accuracy, 3);
    assert_eq!(losses, vec![LossType::CrossEntropy]);
}

#[test]
fn test_compatible_losses_r2() {
    let losses = compatible_losses(&TaskMetric::R2, 1);
    assert_eq!(losses, vec![LossType::MSE]);
}

#[test]
fn test_compatible_losses_multi_label() {
    let losses = compatible_losses(&TaskMetric::MultiLabelAccuracy, 5);
    assert_eq!(losses, vec![LossType::BCE]);
}

#[test]
fn test_effective_loss_default_infer() {
    let genome = NetworkGenome::minimal(2, 1);
    assert_eq!(genome.effective_loss(&TaskMetric::Accuracy), LossType::BCE);

    let genome = NetworkGenome::minimal(2, 3);
    assert_eq!(
        genome.effective_loss(&TaskMetric::Accuracy),
        LossType::CrossEntropy
    );

    let genome = NetworkGenome::minimal(2, 1);
    assert_eq!(genome.effective_loss(&TaskMetric::R2), LossType::MSE);
}

#[test]
fn test_effective_loss_override() {
    let mut genome = NetworkGenome::minimal(2, 1);
    genome.training_config.loss_override = Some(LossType::MSE);
    assert_eq!(genome.effective_loss(&TaskMetric::Accuracy), LossType::MSE);
}

#[test]
fn test_task_metric_is_discrete() {
    assert!(TaskMetric::Accuracy.is_discrete());
    assert!(TaskMetric::MultiLabelAccuracy.is_discrete());
    assert!(!TaskMetric::R2.is_discrete());
}

// ==================== 序列 Genome / 域验证 ====================

#[test]
fn test_minimal_sequential_genome() {
    let genome = NetworkGenome::minimal_sequential(3, 2);

    assert_eq!(genome.input_dim, 3);
    assert_eq!(genome.output_dim, 2);
    assert_eq!(genome.seq_len, Some(0)); // 占位值
    assert_eq!(genome.layers().len(), 2);
    assert_eq!(genome.generated_by, "minimal_sequential");

    // 第一层是 Rnn
    assert_eq!(
        genome.layers()[0].layer_config,
        LayerConfig::Rnn { hidden_size: 2 }
    );
    // 第二层（输出头）是 Linear
    assert_eq!(
        genome.layers()[1].layer_config,
        LayerConfig::Linear { out_features: 2 }
    );
}

#[test]
#[should_panic(expected = "input_dim 不能为零")]
fn test_minimal_sequential_zero_input_panics() {
    NetworkGenome::minimal_sequential(0, 1);
}

#[test]
#[should_panic(expected = "output_dim 不能为零")]
fn test_minimal_sequential_zero_output_panics() {
    NetworkGenome::minimal_sequential(1, 0);
}

#[test]
fn test_resolve_dimensions_with_rnn() {
    // 构造序列 genome: Rnn(4) → [Linear(1)]
    let mut genome = NetworkGenome::minimal_sequential(3, 1);
    genome.layers_mut()[0].layer_config = LayerConfig::Rnn { hidden_size: 4 };
    genome.seq_len = Some(5);

    let resolved = genome.resolve_dimensions().unwrap();
    assert_eq!(resolved.len(), 2);
    // Rnn: in=3, out=4
    assert_eq!(resolved[0].in_dim, 3);
    assert_eq!(resolved[0].out_dim, 4);
    // Linear(输出头): in=4, out=1
    assert_eq!(resolved[1].in_dim, 4);
    assert_eq!(resolved[1].out_dim, 1);
}

#[test]
fn test_resolve_dimensions_with_lstm() {
    let mut genome = NetworkGenome::minimal_sequential(5, 2);
    genome.layers_mut()[0].layer_config = LayerConfig::Lstm { hidden_size: 8 };
    genome.seq_len = Some(10);

    let resolved = genome.resolve_dimensions().unwrap();
    assert_eq!(resolved[0].in_dim, 5);
    assert_eq!(resolved[0].out_dim, 8);
    assert_eq!(resolved[1].in_dim, 8);
    assert_eq!(resolved[1].out_dim, 2);
}

#[test]
fn test_total_params_rnn_genome() {
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.layers_mut()[0].layer_config = LayerConfig::Rnn { hidden_size: 3 };
    genome.seq_len = Some(4);

    // Rnn(in=2, hidden=3): W_ih(2*3) + W_hh(3*3) + b_h(3) = 6+9+3 = 18
    // Linear(in=3, out=1): W(3*1) + b(1) = 4
    // Total: 22
    assert_eq!(genome.total_params().unwrap(), 22);

    // 换成 Lstm
    genome.layers_mut()[0].layer_config = LayerConfig::Lstm { hidden_size: 3 };
    // Lstm: 4 * (2*3 + 3*3 + 3) = 4 * 18 = 72
    // Linear: 4
    assert_eq!(genome.total_params().unwrap(), 76);

    // 换成 Gru
    genome.layers_mut()[0].layer_config = LayerConfig::Gru { hidden_size: 3 };
    // Gru: 3 * (2*3 + 3*3 + 3) = 3 * 18 = 54
    // Linear: 4
    assert_eq!(genome.total_params().unwrap(), 58);
}

#[test]
fn test_domain_valid_single_rnn() {
    let mut genome = NetworkGenome::minimal_sequential(3, 1);
    genome.seq_len = Some(5);
    // Rnn → Linear：Sequence→Flat→Flat，合法
    assert!(genome.is_domain_valid());
}

#[test]
fn test_domain_valid_stacked_rnn() {
    // Rnn → Lstm → Gru → [Linear(1)]
    let mut genome = NetworkGenome::minimal_sequential(3, 1);
    genome.seq_len = Some(5);

    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Lstm { hidden_size: 4 },
            enabled: true,
        },
    );
    genome.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Gru { hidden_size: 4 },
            enabled: true,
        },
    );

    assert!(genome.is_domain_valid());
}

#[test]
fn test_domain_invalid_linear_in_seq() {
    // seq genome: Linear(4) → Rnn(3) → [Linear(1)]
    // Linear 在 Sequence 域中，非法
    let mut genome = NetworkGenome::minimal_sequential(3, 1);
    genome.seq_len = Some(5);
    let inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    assert!(!genome.is_domain_valid());
}

#[test]
fn test_domain_invalid_no_rnn() {
    // seq genome 无 RNN 层：Linear → [Linear]
    let mut genome = NetworkGenome::minimal(3, 1);
    genome.seq_len = Some(5); // 强制序列模式但无 RNN
    // 域初始为 Sequence，但 Linear 要求 Flat → 非法
    assert!(!genome.is_domain_valid());
}

#[test]
fn test_domain_valid_flat_genome_always_true() {
    // 平坦模式（seq_len=None）直接返回 true
    let genome = NetworkGenome::minimal(3, 1);
    assert!(genome.is_domain_valid());
}

#[test]
fn test_is_recurrent_helper() {
    assert!(NetworkGenome::is_recurrent(&LayerConfig::Rnn {
        hidden_size: 4
    }));
    assert!(NetworkGenome::is_recurrent(&LayerConfig::Lstm {
        hidden_size: 4
    }));
    assert!(NetworkGenome::is_recurrent(&LayerConfig::Gru {
        hidden_size: 4
    }));
    assert!(!NetworkGenome::is_recurrent(&LayerConfig::Linear {
        out_features: 4
    }));
    assert!(!NetworkGenome::is_recurrent(&LayerConfig::Activation {
        activation_type: ActivationType::ReLU,
    }));
}

#[test]
fn test_sequential_main_path_summary() {
    let genome = NetworkGenome::minimal_sequential(3, 1);
    let summary = genome.main_path_summary();
    assert!(
        summary.contains("Input(seq×3)"),
        "序列模式应显示 seq×: {summary}"
    );
    assert!(summary.contains("RNN"), "应包含 RNN: {summary}");
}

// ==================== compute_domain_map ====================

#[test]
fn test_domain_map_flat_genome() {
    // 平坦模式：所有节点均为 Flat
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

    let map = genome.compute_domain_map();
    assert_eq!(map[&INPUT_INNOVATION], ShapeDomain::Flat);
    assert_eq!(map[&inn], ShapeDomain::Flat);
    assert_eq!(map[&1], ShapeDomain::Flat); // 输出头
}

#[test]
fn test_domain_map_single_rnn() {
    // Input(Seq) → Rnn(Flat) → Linear(Flat)
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.seq_len = Some(5);
    let rnn_inn = genome.layers()[0].innovation_number;
    let out_inn = genome.layers()[1].innovation_number;

    let map = genome.compute_domain_map();
    assert_eq!(map[&INPUT_INNOVATION], ShapeDomain::Sequence);
    assert_eq!(map[&rnn_inn], ShapeDomain::Flat); // RNN 后无循环层 → Flat
    assert_eq!(map[&out_inn], ShapeDomain::Flat);
}

#[test]
fn test_domain_map_stacked_rnn() {
    // Input(Seq) → Rnn(Seq) → Lstm(Flat) → Linear(Flat)
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.seq_len = Some(5);
    let rnn_inn = genome.layers()[0].innovation_number;

    let lstm_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: lstm_inn,
            layer_config: LayerConfig::Lstm { hidden_size: 4 },
            enabled: true,
        },
    );
    let out_inn = genome.layers()[2].innovation_number;

    let map = genome.compute_domain_map();
    assert_eq!(map[&INPUT_INNOVATION], ShapeDomain::Sequence);
    assert_eq!(map[&rnn_inn], ShapeDomain::Sequence); // 下一个实质层是 LSTM → Seq
    assert_eq!(map[&lstm_inn], ShapeDomain::Flat); // 下一个实质层是 Linear → Flat
    assert_eq!(map[&out_inn], ShapeDomain::Flat);
}

#[test]
fn test_domain_map_rnn_with_activation() {
    // Input(Seq) → Rnn(Flat) → Tanh(Flat) → Linear(Flat)
    // Activation 保持 RNN 输出后的 Flat 域
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.seq_len = Some(5);
    let rnn_inn = genome.layers()[0].innovation_number;

    let act_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: act_inn,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::Tanh,
            },
            enabled: true,
        },
    );
    let out_inn = genome.layers()[2].innovation_number;

    let map = genome.compute_domain_map();
    assert_eq!(map[&INPUT_INNOVATION], ShapeDomain::Sequence);
    assert_eq!(map[&rnn_inn], ShapeDomain::Flat);
    assert_eq!(map[&act_inn], ShapeDomain::Flat); // Activation 透传 Flat
    assert_eq!(map[&out_inn], ShapeDomain::Flat);
}

#[test]
fn test_domain_map_activation_between_rnns() {
    // Input(Seq) → Rnn(Seq) → Tanh(Seq) → Gru(Flat) → Linear(Flat)
    // Activation 在两个 RNN 之间应保持 Sequence 域
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.seq_len = Some(5);
    let rnn_inn = genome.layers()[0].innovation_number;

    let act_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: act_inn,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::Tanh,
            },
            enabled: true,
        },
    );
    let gru_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: gru_inn,
            layer_config: LayerConfig::Gru { hidden_size: 4 },
            enabled: true,
        },
    );

    let map = genome.compute_domain_map();
    assert_eq!(map[&rnn_inn], ShapeDomain::Sequence); // 跳过 Tanh 后见 Gru
    assert_eq!(map[&act_inn], ShapeDomain::Sequence); // 透传 Sequence
    assert_eq!(map[&gru_inn], ShapeDomain::Flat); // 最后一个 RNN
}

// ==================== Spatial 模式测试 ====================

#[test]
fn test_minimal_spatial_creates_correct_genome() {
    let genome = NetworkGenome::minimal_spatial(3, 10, (28, 28));

    assert_eq!(genome.input_dim, 3); // in_channels
    assert_eq!(genome.output_dim, 10);
    assert_eq!(genome.input_spatial, Some((28, 28)));
    assert!(genome.is_spatial());
    assert_eq!(genome.layers().len(), 4); // Conv2d, Pool2d, Flatten, Linear

    assert!(matches!(
        genome.layers()[0].layer_config,
        LayerConfig::Conv2d { out_channels: 8, kernel_size: 3 }
    ));
    assert!(matches!(
        genome.layers()[1].layer_config,
        LayerConfig::Pool2d { pool_type: PoolType::Max, kernel_size: 2, stride: 2 }
    ));
    assert!(matches!(
        genome.layers()[2].layer_config,
        LayerConfig::Flatten
    ));
    assert!(matches!(
        genome.layers()[3].layer_config,
        LayerConfig::Linear { out_features: 10 }
    ));
}

#[test]
#[should_panic(expected = "input_channels 不能为零")]
fn test_minimal_spatial_zero_channels_panics() {
    NetworkGenome::minimal_spatial(0, 10, (28, 28));
}

#[test]
#[should_panic(expected = "spatial (H, W) 不能为零")]
fn test_minimal_spatial_zero_hw_panics() {
    NetworkGenome::minimal_spatial(3, 10, (0, 28));
}

#[test]
fn test_resolve_dimensions_spatial_minimal() {
    // Conv2d(3→8,k=3) → Pool2d(Max,2,2) → Flatten → Linear(10)
    // 输入: 3 channels, 28×28
    let genome = NetworkGenome::minimal_spatial(3, 10, (28, 28));
    let resolved = genome.resolve_dimensions().unwrap();

    assert_eq!(resolved.len(), 4);
    // Conv2d: in=3(channels), out=8(out_channels)
    assert_eq!(resolved[0].in_dim, 3);
    assert_eq!(resolved[0].out_dim, 8);
    // Pool2d: in=8, out=8 (channels 不变)
    assert_eq!(resolved[1].in_dim, 8);
    assert_eq!(resolved[1].out_dim, 8);
    // Flatten: in=8(channels), out=8*14*14=1568 (28/2=14 after pool)
    assert_eq!(resolved[2].in_dim, 8);
    assert_eq!(resolved[2].out_dim, 8 * 14 * 14);
    // Linear: in=1568, out=10
    assert_eq!(resolved[3].in_dim, 1568);
    assert_eq!(resolved[3].out_dim, 10);
}

#[test]
fn test_resolve_dimensions_spatial_with_extra_pool() {
    // 基线 minimal: Conv2d(1→8,k=3) → Pool2d → Flatten → Linear(10)
    // 在首层 Pool2d 后再插入一层 MaxPool，使 8×8 经两次池化降到 2×2 后接 Flatten
    // 输入: 1 channel, 8×8
    let mut genome = NetworkGenome::minimal_spatial(1, 10, (8, 8));

    let p_extra_inn = genome.next_innovation_number();
    // index 2 = 第一个 Pool2d 之后、Flatten 之前
    genome.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: p_extra_inn,
            layer_config: LayerConfig::Pool2d {
                pool_type: PoolType::Max,
                kernel_size: 2,
                stride: 2,
            },
            enabled: true,
        },
    );

    let resolved = genome.resolve_dimensions().unwrap();
    assert_eq!(resolved.len(), 5);
    // Conv2d: in=1, out=8, spatial=8×8
    assert_eq!(resolved[0].in_dim, 1);
    assert_eq!(resolved[0].out_dim, 8);
    // Pool2d #1: 8×8 → 4×4, channels=8
    assert_eq!(resolved[1].in_dim, 8);
    assert_eq!(resolved[1].out_dim, 8);
    // Pool2d #2: 4×4 → 2×2, channels=8
    assert_eq!(resolved[2].in_dim, 8);
    assert_eq!(resolved[2].out_dim, 8);
    // Flatten: in=8, out=8*2*2=32
    assert_eq!(resolved[3].in_dim, 8);
    assert_eq!(resolved[3].out_dim, 8 * 2 * 2);
    // Linear: in=32, out=10
    assert_eq!(resolved[4].in_dim, 32);
    assert_eq!(resolved[4].out_dim, 10);
}

#[test]
fn test_compute_layer_params_conv2d() {
    // Conv2d: out_ch * in_ch * k * k + out_ch (bias)
    // 由于 compute_layer_params 是私有的，通过 total_params 间接测试
    // 手动构建含 Conv2d 的空间 genome
    let mut test_genome = NetworkGenome::minimal_spatial(3, 10, (8, 8));
    let conv_inn = test_genome.next_innovation_number();
    test_genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: conv_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 16,
                kernel_size: 3,
            },
            enabled: true,
        },
    );
    let total = test_genome.total_params().unwrap();
    // 层序: Conv#1(3→16) → Conv#2(16→8) → Pool(8×8→4×4) → Flatten(8*4*4) → Linear(10)
    // 448 + 1160 + 0 + (128*10+10) = 2898
    assert_eq!(total, 2898);
}

#[test]
fn test_compute_layer_params_pool2d_and_flatten_zero() {
    // Pool2d 和 Flatten 均无可学习参数
    // 手动构建含 Conv2d + Pool2d 的空间 genome
    let mut genome = NetworkGenome::minimal_spatial(1, 2, (4, 4));
    let conv_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: conv_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 1,
                kernel_size: 1,
            },
            enabled: true,
        },
    );
    let pool_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: pool_inn,
            layer_config: LayerConfig::Pool2d {
                pool_type: PoolType::Max,
                kernel_size: 2,
                stride: 2,
            },
            enabled: true,
        },
    );

    let total = genome.total_params().unwrap();
    // 在 minimal 前插入: Conv2d(1, k=1) 与 MaxPool(2) 后:
    // Conv#1(1, k1 in=1): 2; Conv#2(8, k3 in=1)（种子）: 8*1*9+8=80; Pool/Flatten/Linear(2)
    // Flatten 后: 8 ch, 1×1 → 8 维; Linear(2): 8*2+2=18
    assert_eq!(total, 2 + 0 + 80 + 0 + 0 + 18);
}

#[test]
fn test_is_domain_valid_spatial_minimal() {
    // minimal_spatial：Conv → Pool → Flatten → Linear，域链合法
    let genome = NetworkGenome::minimal_spatial(3, 10, (28, 28));
    assert!(genome.is_domain_valid());
}

#[test]
fn test_is_domain_valid_spatial_flatten_only() {
    // 验证「Conv+Pool+Flatten+Linear」最小 CNN 主路径的域是合法的
    let genome = NetworkGenome::minimal_spatial(1, 10, (28, 28));
    assert!(genome.is_domain_valid());
    assert_eq!(genome.layers().len(), 4);
}

#[test]
fn test_is_domain_valid_spatial_conv_then_flatten() {
    // Conv2d → Flatten → [Linear]：合法（演化插入 Conv2d 后）
    let mut genome = NetworkGenome::minimal_spatial(1, 10, (28, 28));
    let conv_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: conv_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 4,
                kernel_size: 3,
            },
            enabled: true,
        },
    );
    assert!(genome.is_domain_valid());
}

#[test]
fn test_is_domain_valid_spatial_conv_pool_flatten_linear() {
    // Conv2d → Pool2d → Flatten → Linear：合法
    let mut genome = NetworkGenome::minimal_spatial(1, 2, (8, 8));
    let conv_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: conv_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 4,
                kernel_size: 3,
            },
            enabled: true,
        },
    );
    let pool_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: pool_inn,
            layer_config: LayerConfig::Pool2d {
                pool_type: PoolType::Max,
                kernel_size: 2,
                stride: 2,
            },
            enabled: true,
        },
    );
    assert!(genome.is_domain_valid());
}

#[test]
fn test_is_domain_valid_spatial_missing_flatten() {
    // 去掉 Flatten 后 [Conv, Pool, …, Linear]：非法（空间模式未回到 Flat 域前出现 Linear/末态不合法）
    let mut genome = NetworkGenome::minimal_spatial(3, 10, (28, 28));
    // Flatten 在 minimal_spatial 中为第 3 层（0-based 索引 2）
    genome.layers_mut().remove(2);
    assert!(!genome.is_domain_valid());
}

#[test]
fn test_is_domain_valid_spatial_linear_before_flatten() {
    // Linear → Flatten → Linear：非法（Spatial 域下不能有 Linear）
    let mut genome = NetworkGenome::minimal_spatial(3, 10, (28, 28));
    let lin_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: lin_inn,
            layer_config: LayerConfig::Linear { out_features: 8 },
            enabled: true,
        },
    );
    assert!(!genome.is_domain_valid());
}

#[test]
fn test_is_domain_valid_spatial_activation_in_spatial_domain() {
    // Conv2d → ReLU → Flatten → Linear：合法（Activation 在 Spatial 域透传）
    let mut genome = NetworkGenome::minimal_spatial(3, 10, (28, 28));
    let conv_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: conv_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 4,
                kernel_size: 3,
            },
            enabled: true,
        },
    );
    let act_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: act_inn,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    assert!(genome.is_domain_valid());
}

#[test]
fn test_is_domain_valid_spatial_rnn_illegal() {
    // Rnn → Flatten → Linear：非法（Spatial 域下不能有 RNN）
    let mut genome = NetworkGenome::minimal_spatial(3, 10, (28, 28));
    let rnn_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: rnn_inn,
            layer_config: LayerConfig::Rnn { hidden_size: 8 },
            enabled: true,
        },
    );
    assert!(!genome.is_domain_valid());
}

#[test]
fn test_domain_map_spatial_genome() {
    // Input(Spatial) → Conv/Pool(Spatial) → Flatten(Flat) → Linear(Flat)
    let genome = NetworkGenome::minimal_spatial(3, 10, (28, 28));
    let map = genome.compute_domain_map();

    assert_eq!(map[&INPUT_INNOVATION], ShapeDomain::Spatial);
    assert_eq!(map[&genome.layers()[0].innovation_number], ShapeDomain::Spatial);
    assert_eq!(map[&genome.layers()[1].innovation_number], ShapeDomain::Spatial);
    assert_eq!(map[&genome.layers()[2].innovation_number], ShapeDomain::Flat); // Flatten
    assert_eq!(map[&genome.layers()[3].innovation_number], ShapeDomain::Flat); // Linear
}

#[test]
fn test_spatial_map_minimal() {
    // Input(8×8) → Conv(8×8) → Pool(4×4) → Flatten(None) → Linear(None)
    let genome = NetworkGenome::minimal_spatial(1, 2, (8, 8));

    let smap = genome.compute_spatial_map();
    assert_eq!(smap[&INPUT_INNOVATION], Some((8, 8)));
    assert_eq!(smap[&genome.layers()[0].innovation_number], Some((8, 8)));
    assert_eq!(smap[&genome.layers()[1].innovation_number], Some((4, 4)));
    assert_eq!(smap[&genome.layers()[2].innovation_number], None); // Flatten
    assert_eq!(smap[&genome.layers()[3].innovation_number], None); // Linear
}

#[test]
fn test_validate_skip_edge_spatial_same_hw() {
    // Conv2d(#1) → Conv2d(#2) → Flatten → Linear
    // Skip: #1 → #2（同 H/W）→ 合法
    let mut genome = NetworkGenome::minimal_spatial(1, 2, (8, 8));
    let conv1_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: conv1_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 1,
                kernel_size: 3,
            },
            enabled: true,
        },
    );
    let conv2_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: conv2_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 1,
                kernel_size: 3,
            },
            enabled: true,
        },
    );
    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: conv1_inn,
        to_innovation: conv2_inn,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    assert!(genome.validate_skip_edge_domains());
}

#[test]
fn test_validate_skip_edge_spatial_different_hw() {
    // Conv2d(#1, 8×8) → Pool2d(4×4) → Conv2d(#2, 4×4) → Flatten → Linear
    // Skip: #1(8×8) → #2(4×4)（H/W 不同）→ 非法
    let mut genome = NetworkGenome::minimal_spatial(1, 2, (8, 8));
    let conv1_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: conv1_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 1,
                kernel_size: 3,
            },
            enabled: true,
        },
    );
    let pool_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: pool_inn,
            layer_config: LayerConfig::Pool2d {
                pool_type: PoolType::Max,
                kernel_size: 2,
                stride: 2,
            },
            enabled: true,
        },
    );
    let conv2_inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: conv2_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 1,
                kernel_size: 3,
            },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: conv1_inn,
        to_innovation: conv2_inn,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    assert!(!genome.validate_skip_edge_domains());
}
