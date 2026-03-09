use crate::nn::evolution::gene::*;
use crate::tensor::Tensor;
use std::collections::HashMap;

// ==================== 基本构造 ====================

#[test]
fn test_minimal_creates_correct_genome() {
    let genome = NetworkGenome::minimal(2, 1);

    assert_eq!(genome.input_dim, 2);
    assert_eq!(genome.output_dim, 1);
    assert_eq!(genome.layers.len(), 1);
    assert_eq!(genome.skip_edges.len(), 0);

    let output_head = &genome.layers[0];
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
        if let LayerConfig::Linear { out_features } = &genome.layers.last().unwrap().layer_config {
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
    seen.insert(genome.layers[0].innovation_number);
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
    genome.layers.insert(0, hidden);

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
    genome.layers.insert(0, hidden);

    // enabled: Input(2) → Linear(4) → Linear(1)
    // Linear(4): W(2×4)+b(4)=12, Linear(1): W(4×1)+b(1)=5 → 17
    assert_eq!(genome.total_params().unwrap(), 17);
    assert_eq!(genome.layer_count(), 2);

    // disable 隐藏层
    genome.layers[0].enabled = false;
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

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: inn1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers.insert(
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
    genome.layers.insert(
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

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::Tanh,
            },
            enabled: true,
        },
    );
    genome.layers.insert(
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers.insert(
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers.insert(
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
    genome.layers.insert(
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

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 8 },
            enabled: true,
        },
    );
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::LeakyReLU { alpha: 0.01 },
            },
            enabled: true,
        },
    );
    genome.layers.insert(
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 2 },
            enabled: true,
        },
    );
    genome.layers.insert(
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
    genome.skip_edges.push(SkipEdge {
        innovation_number: s1,
        from_innovation: INPUT_INNOVATION,
        to_innovation: i2,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });
    let s2 = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Add,
        enabled: false, // disabled
    });

    // disabled skip edge 不出现 → 单行输出
    assert_eq!(
        format!("{genome}"),
        "Input(2) → Linear(4) → [Linear(1)]"
    );
}

#[test]
fn test_main_path_summary_ignores_skip_edges() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    genome.layers.insert(
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

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    genome.layers.insert(
        2,
        LayerGene {
            innovation_number: i3,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers.insert(
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
    genome.skip_edges.push(SkipEdge {
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

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    assert_eq!(
        format!("{genome}"),
        "Input(2) → Linear(4) → [Linear(4)]"
    );
}

#[test]
fn test_main_path_summary_duplicate_names_disambiguated() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let i1 = genome.next_innovation_number();
    let i2 = genome.next_innovation_number();

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers.insert(
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
    genome.layers[0].enabled = false;

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

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 20 },
            enabled: true,
        },
    );
    genome.layers.insert(
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Linear { out_features: 3 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Linear { out_features: 3 },
            enabled: true,
        },
    );

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    let skip_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 3 },
            enabled: true,
        },
    );
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let s1 = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
        innovation_number: s1,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Concat { dim: -1 },
        enabled: true,
    });

    let s2 = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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
    let output_inn = genome.layers.last().unwrap().innovation_number;

    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 3 },
            enabled: true,
        },
    );
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let s1 = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
        innovation_number: s1,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    let s2 = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
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
