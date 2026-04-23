/*
 * @Author       : 老董
 * @Date         : 2026-03-09
 * @Description  : 演化模型 save/load (.otm) 测试
 */

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::gene::{
    ActivationType, LayerConfig, LayerGene, NetworkGenome, TaskMetric,
};
use crate::nn::evolution::mutation::{AddConnectionMutation, Mutation, SizeConstraints};
use crate::nn::evolution::node_ops::find_removable_skip_connections;
use crate::nn::evolution::{Evolution, EvolutionResult};
use crate::tensor::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// XOR 数据集（复用 evolution_xor 示例的数据）
fn xor_data() -> (Vec<Tensor>, Vec<Tensor>) {
    (
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
    )
}

// ==================== NetworkGenome serde 往返测试 ====================

#[test]
fn test_genome_serde_roundtrip_minimal() {
    let genome = NetworkGenome::minimal(4, 3);
    let json = serde_json::to_string(&genome).expect("序列化失败");
    let restored: NetworkGenome = serde_json::from_str(&json).expect("反序列化失败");

    assert_eq!(restored.input_dim, 4);
    assert_eq!(restored.output_dim, 3);
    assert_eq!(restored.layers().len(), genome.layers().len());
    assert_eq!(restored.skip_edges().len(), 0);
}

#[test]
fn test_genome_serde_roundtrip_with_weights() {
    let mut genome = NetworkGenome::minimal(2, 1);
    // 模拟有权重快照
    let mut snapshots = std::collections::HashMap::new();
    snapshots.insert(
        1,
        vec![
            Tensor::new(&[0.1, 0.2], &[2, 1]),
            Tensor::new(&[0.3], &[1, 1]),
        ],
    );
    genome.set_weight_snapshots(snapshots);

    let json = serde_json::to_string(&genome).expect("序列化失败");
    let restored: NetworkGenome = serde_json::from_str(&json).expect("反序列化失败");

    assert!(restored.has_weight_snapshots());
    let ws = restored.weight_snapshots();
    assert!(ws.contains_key(&1));
    assert_eq!(ws[&1].len(), 2);
    // 验证权重值
    assert!((ws[&1][0].to_vec()[0] - 0.1).abs() < 1e-6);
    assert!((ws[&1][0].to_vec()[1] - 0.2).abs() < 1e-6);
    assert!((ws[&1][1].to_vec()[0] - 0.3).abs() < 1e-6);
}

#[test]
fn test_nodelevel_genome_serde_roundtrip_with_skip_connection() {
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
    genome
        .migrate_to_node_level()
        .expect("迁移到 NodeLevel 应成功");

    let mut rng = StdRng::seed_from_u64(42);
    AddConnectionMutation
        .apply(&mut genome, &Default::default(), &mut rng)
        .expect("应能添加 NodeLevel 跳跃连接");

    let json = serde_json::to_string(&genome).expect("序列化失败");
    let restored: NetworkGenome = serde_json::from_str(&json).expect("反序列化失败");

    assert!(restored.is_node_level(), "恢复后应保持 NodeLevel 表示");
    assert!(
        !find_removable_skip_connections(&restored).is_empty(),
        "恢复后应仍包含可识别的跳跃聚合节点"
    );
    assert!(
        restored
            .nodes()
            .iter()
            .any(|n| n.block_id.is_none() && matches!(n.node_type, NodeTypeDescriptor::Add)),
        "恢复后应仍存在跳跃聚合 Add 节点"
    );

    let analysis = restored.analyze();
    assert!(analysis.is_valid, "恢复后图应合法: {:?}", analysis.errors);

    let mut build_rng = StdRng::seed_from_u64(7);
    assert!(
        restored.build(&mut build_rng).is_ok(),
        "恢复后的 NodeLevel 跳跃连接图应可构建"
    );
}

#[test]
fn test_nodelevel_sequential_genome_serde_roundtrip() {
    let mut genome = NetworkGenome::minimal_sequential(3, 2);
    genome.seq_len = Some(5);
    genome
        .migrate_to_node_level()
        .expect("迁移到 NodeLevel 应成功");

    let json = serde_json::to_string(&genome).expect("序列化失败");
    let restored: NetworkGenome = serde_json::from_str(&json).expect("反序列化失败");

    assert!(restored.is_node_level(), "恢复后应保持 NodeLevel 表示");
    assert_eq!(restored.seq_len, genome.seq_len, "seq_len 应保持一致");
    assert_eq!(
        restored.nodes().len(),
        genome.nodes().len(),
        "节点数应保持一致"
    );

    for (lhs, rhs) in genome.nodes().iter().zip(restored.nodes().iter()) {
        assert_eq!(lhs.innovation_number, rhs.innovation_number);
        assert_eq!(lhs.output_shape, rhs.output_shape);
        assert_eq!(lhs.parents, rhs.parents);
        assert_eq!(lhs.block_id, rhs.block_id);
    }

    assert!(
        restored
            .nodes()
            .iter()
            .any(|n| matches!(n.node_type, NodeTypeDescriptor::CellRnn { .. })),
        "恢复后应仍存在 CellRnn 节点"
    );

    let analysis = restored.analyze();
    assert!(
        analysis.is_valid,
        "恢复后的序列 NodeLevel 图应合法: {:?}",
        analysis.errors
    );

    let mut build_rng = StdRng::seed_from_u64(11);
    assert!(
        restored.build(&mut build_rng).is_ok(),
        "恢复后的序列 NodeLevel 图应可构建"
    );
}

// ==================== EvolutionResult save/load 测试 ====================

#[test]
fn test_evolution_result_save_load_roundtrip() {
    let temp_path = "test_evolution_save_load_roundtrip";

    // 1. 运行演化
    let data = xor_data();
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_target_metric(0.75) // 低阈值，确保快速收敛
        .with_seed(42)
        .with_verbose(false)
        .run()
        .expect("演化失败");

    let original_fitness = result.fitness.primary;
    let original_generations = result.generations;
    let original_arch = result.architecture().to_string();

    // 收集原始预测
    let test_input = Tensor::new(&[1.0, 0.0], &[2]);
    let original_pred = result.predict(&test_input).expect("原始推理失败");
    let original_pred_vec = original_pred.to_vec();

    // 2. 保存
    result.save(temp_path).expect("保存失败");

    // 验证文件存在
    assert!(
        std::path::Path::new("test_evolution_save_load_roundtrip.otm").exists(),
        ".otm 文件应该存在"
    );

    // 3. 加载
    let loaded = EvolutionResult::load(temp_path).expect("加载失败");

    // 4. 验证元数据
    assert!(
        (loaded.fitness.primary - original_fitness).abs() < 1e-6,
        "fitness 应一致"
    );
    assert_eq!(loaded.generations, original_generations, "代数应一致");
    assert_eq!(loaded.architecture(), original_arch, "架构描述应一致");

    // 5. 验证推理结果一致
    let loaded_pred = loaded.predict(&test_input).expect("加载后推理失败");
    let loaded_pred_vec = loaded_pred.to_vec();

    assert_eq!(
        original_pred_vec.len(),
        loaded_pred_vec.len(),
        "预测维度应一致"
    );
    for (i, (a, b)) in original_pred_vec
        .iter()
        .zip(loaded_pred_vec.iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-5,
            "预测值[{i}]不一致: 原始={a}, 加载后={b}"
        );
    }

    // 清理
    std::fs::remove_file("test_evolution_save_load_roundtrip.otm").ok();
}

#[test]
fn test_evolution_result_save_load_visualize() {
    let temp_path = "test_evolution_save_load_vis";

    let data = xor_data();
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_target_metric(0.75)
        .with_seed(42)
        .with_verbose(false)
        .run()
        .expect("演化失败");

    result.save(temp_path).expect("保存失败");

    let loaded = EvolutionResult::load(temp_path).expect("加载失败");

    // 加载后应该可以可视化（snapshot 已在 load 中自动拍摄）
    let vis = loaded
        .visualize("test_evolution_save_load_vis_graph")
        .expect("可视化失败");
    assert!(vis.dot_path.exists(), "DOT 文件应该存在");

    // 清理
    std::fs::remove_file("test_evolution_save_load_roundtrip.otm").ok();
    std::fs::remove_file("test_evolution_save_load_vis.otm").ok();
    std::fs::remove_file(&vis.dot_path).ok();
    if let Some(img) = &vis.image_path {
        std::fs::remove_file(img).ok();
    }
}

#[test]
fn test_load_invalid_file() {
    let temp_path = "test_load_invalid_otm";
    let file_path = "test_load_invalid_otm.otm";

    // 写入无效数据
    std::fs::write(file_path, b"NOT_A_VALID_OTM_FILE").expect("写入测试文件失败");

    let result = EvolutionResult::load(temp_path);
    assert!(result.is_err(), "加载无效文件应该失败");
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => unreachable!(),
    };
    assert!(
        err_msg.contains("魔数不匹配"),
        "错误信息应提到魔数: {}",
        err_msg
    );

    // 清理
    std::fs::remove_file(file_path).ok();
}

#[test]
fn test_load_nonexistent_file() {
    let result = EvolutionResult::load("nonexistent_model");
    assert!(result.is_err(), "加载不存在的文件应该失败");
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => unreachable!(),
    };
    assert!(
        err_msg.contains("无法打开文件"),
        "错误信息应提到文件不存在: {}",
        err_msg
    );
}

#[test]
fn test_save_creates_parent_directories() {
    let temp_path = "test_otm_nested_dir/subdir/model";

    let data = xor_data();
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_max_generations(1)
        .with_seed(42)
        .with_verbose(false)
        .run()
        .expect("演化失败");

    result.save(temp_path).expect("保存到嵌套目录应成功");

    assert!(
        std::path::Path::new("test_otm_nested_dir/subdir/model.otm").exists(),
        "文件应该在嵌套目录中创建"
    );

    // 清理
    std::fs::remove_dir_all("test_otm_nested_dir").ok();
}

// ==================== NodeLevel 持久化格式验收测试 ====================

/// 简单空间数据（4 个 1-通道 8×8 假图像，二分类标签）
/// 4×4 在 Conv+Pool+变异链路过短时易出现非法空间下采样；8×8 为 CNN 起种子留出余量
fn spatial_data() -> (Vec<Tensor>, Vec<Tensor>) {
    let n = 8 * 8;
    let pos: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let neg: Vec<f32> = (0..n).map(|i| -(i as f32) / n as f32).collect();
    (
        vec![
            Tensor::new(&pos, &[1, 8, 8]),
            Tensor::new(&neg, &[1, 8, 8]),
            Tensor::new(&pos, &[1, 8, 8]),
            Tensor::new(&neg, &[1, 8, 8]),
        ],
        vec![
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[0.0], &[1]),
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[0.0], &[1]),
        ],
    )
}

/// 简单序列数据（4 条长度 2、特征维 1 的序列，对应 XOR）
fn seq_data() -> (Vec<Tensor>, Vec<Tensor>) {
    (
        vec![
            Tensor::new(&[0.0, 0.0], &[2, 1]),
            Tensor::new(&[1.0, 0.0], &[2, 1]),
            Tensor::new(&[0.0, 1.0], &[2, 1]),
            Tensor::new(&[1.0, 1.0], &[2, 1]),
        ],
        vec![
            Tensor::new(&[0.0], &[1]),
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[0.0], &[1]),
        ],
    )
}

/// Flat（XOR）演化产出 NodeLevel genome，保存/加载往返一致
///
/// 验证：
/// - 演化后 genome.is_node_level() == true
/// - 保存的 .otm 再次加载后 genome 也是 NodeLevel
/// - 加载后推理结果与保存前一致
#[test]
fn test_phase6_flat_evolution_produces_nodelevel_genome() {
    let temp_path = "test_phase6_flat_nodelevel";

    let data = xor_data();
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_target_metric(0.75)
        .with_seed(42)
        .with_verbose(false)
        .run()
        .expect("演化失败");

    // 演化结果的 genome 必须是 NodeLevel
    assert!(
        result.genome.is_node_level(),
        "Flat 模式演化产出的 genome 应为 NodeLevel（演化主循环已在 run() 中迁移）"
    );

    // 保存
    result.save(temp_path).expect("保存失败");

    // 加载并验证 NodeLevel
    let loaded = EvolutionResult::load(temp_path).expect("加载失败");
    assert!(
        loaded.genome.is_node_level(),
        "加载后的 genome 应为 NodeLevel"
    );

    // 推理结果一致性
    let test_input = Tensor::new(&[1.0, 0.0], &[2]);
    let pred_before = result.predict(&test_input).expect("保存前推理失败");
    let pred_after = loaded.predict(&test_input).expect("加载后推理失败");
    for (a, b) in pred_before.to_vec().iter().zip(pred_after.to_vec().iter()) {
        assert!((a - b).abs() < 1e-5, "加载前后推理结果不一致: {a} vs {b}");
    }

    std::fs::remove_file(format!("{temp_path}.otm")).ok();
}

/// Spatial 演化产出 NodeLevel genome，保存/加载往返一致
///
/// 验证：
/// - Spatial 模式演化后 genome.is_node_level() == true
/// - save/load 往返，加载后 genome 也是 NodeLevel
/// - 加载后推理正常（输入张量 [C, H, W] = [1, 8, 8]）
#[test]
fn test_phase6_spatial_nodelevel_save_load_roundtrip() {
    let temp_path = "test_phase6_spatial_nodelevel";

    let data = spatial_data();
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_max_generations(2) // 只跑 2 代，快速验证格式
        .with_seed(42)
        // 不依赖 SizeConstraints::auto 的大参数量级；用默认紧凑上界降低非法结构出现概率
        .with_constraints(SizeConstraints::default())
        .with_parallelism(1)
        .with_verbose(false)
        .run()
        .expect("Spatial 演化失败");

    // Spatial 演化后 genome 必须是 NodeLevel
    assert!(
        result.genome.is_node_level(),
        "Spatial 模式演化产出的 genome 应为 NodeLevel"
    );

    // 保存
    result.save(temp_path).expect("Spatial 保存失败");

    // 加载并验证 NodeLevel
    let loaded = EvolutionResult::load(temp_path).expect("Spatial 加载失败");
    assert!(
        loaded.genome.is_node_level(),
        "加载后的 Spatial genome 应为 NodeLevel"
    );

    // 加载后推理成功（不要求结果一致，只验证前向传播不崩溃）
    let n = 8 * 8;
    let test_img = Tensor::new(
        &(0..n).map(|i| i as f32 / n as f32).collect::<Vec<_>>(),
        &[1, 8, 8],
    );
    let pred = loaded.predict(&test_img).expect("Spatial 加载后推理失败");
    assert_eq!(pred.shape()[1], 1, "输出维度应为 1");

    std::fs::remove_file(format!("{temp_path}.otm")).ok();
}

/// 手写 GraphDescriptor → NetworkGenome → 变异 → 构建闭环
///
/// 模拟"手写训练模型作为演化种子"的全链路：
/// 1. 手写建立 MLP（Input(2) → FC(4) → ReLU → FC(1)）
/// 2. 提取 GraphDescriptor
/// 3. NetworkGenome::from_graph_descriptor() 创建 NodeLevel genome
/// 4. 验证 genome 维度、分析结果、可构图
/// 5. 执行变异后仍可构图且输出维度不变
/// 6. to_graph_descriptor() → from_graph_descriptor() 往返节点数一致
#[test]
fn test_phase6_from_graph_descriptor_handwritten_seed() {
    use crate::nn::evolution::mutation::{MutationRegistry, SizeConstraints};
    use crate::nn::{Graph, Linear, VarActivationOps};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    // 1. 手写 MLP: Input(2) → FC(4) → ReLU → FC(1)
    let graph = Graph::new_with_seed(42);
    let input = graph.input_shape(&[1, 2], Some("input")).unwrap();
    let fc1 = Linear::new(&graph, 2, 4, true, "fc1").unwrap();
    let h = fc1.forward(&input).relu();
    let fc2 = Linear::new(&graph, 4, 1, true, "fc2").unwrap();
    let output = fc2.forward(&h);

    // 2. 提取 GraphDescriptor（trace from output var）
    let desc = crate::nn::var::Var::vars_to_graph_descriptor(&[&output], "handwritten_mlp");

    // 3. 转换为 NodeLevel NetworkGenome
    let genome =
        NetworkGenome::from_graph_descriptor(&desc).expect("from_graph_descriptor 不应失败");

    // 4a. 验证 genome 是 NodeLevel
    assert!(
        genome.is_node_level(),
        "从 GraphDescriptor 创建的 genome 应为 NodeLevel"
    );
    assert_eq!(genome.input_dim, 2, "input_dim 应为 2");
    assert_eq!(genome.output_dim, 1, "output_dim 应为 1");
    assert!(genome.seq_len.is_none(), "应为 Flat 模式（非序列）");
    assert!(genome.input_spatial.is_none(), "应为 Flat 模式（非空间）");

    // 4b. 验证 GenomeAnalysis 通过
    let analysis = genome.analyze();
    assert!(
        analysis.is_valid,
        "GenomeAnalysis 应通过：{:?}",
        analysis.errors
    );
    assert!(analysis.param_count > 0, "应有可训练参数");

    // 5a. 构建并前向传播
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).expect("genome.build 不应失败");
    let test_input = Tensor::new(&[0.5, -0.3], &[1, 2]);
    build.input.set_value(&test_input).unwrap();
    build.graph.forward(&build.output).unwrap();
    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 1], "输出形状应为 [1, 1]");

    // 5b. 执行变异后仍可构建
    let constraints = SizeConstraints::default();
    let registry = MutationRegistry::default_registry(&TaskMetric::Accuracy, false, false);
    let mut mutated = genome.clone();
    let mut mut_rng = StdRng::seed_from_u64(99);
    let mut ok = false;
    for _ in 0..20 {
        if registry
            .apply_random(&mut mutated, &constraints, &mut mut_rng)
            .is_ok()
        {
            ok = true;
            break;
        }
    }
    assert!(ok, "至少应有一次变异成功");

    let mut build_rng = StdRng::seed_from_u64(123);
    let build2 = mutated
        .build(&mut build_rng)
        .expect("变异后 build 不应失败");
    build2.input.set_value(&test_input).unwrap();
    build2.graph.forward(&build2.output).unwrap();
    let out2 = build2.output.value().unwrap().unwrap();
    assert_eq!(out2.shape()[1], 1, "变异后输出维度仍应为 1");

    // 6. to_graph_descriptor() → from_graph_descriptor() 往返节点数一致
    let desc2 = genome
        .to_graph_descriptor()
        .expect("to_graph_descriptor 不应失败");
    let genome2 = NetworkGenome::from_graph_descriptor(&desc2).expect("descriptor 往返不应失败");
    assert_eq!(
        genome.nodes().len(),
        genome2.nodes().len(),
        "descriptor 往返后节点数量应一致"
    );
}

/// 手写 .otm → 演化种子 → 变异 → 重新保存 → Graph::load_model 后继续手写训练
#[test]
fn test_phase6_triangle_interop_handwritten_to_evolution_to_manual_train() {
    use crate::nn::graph::model_save;
    use crate::nn::optimizer::{Optimizer, SGD};
    use crate::nn::{Graph, Linear, Var, VarActivationOps, VarLossOps};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let hand_path = "test_phase6_triangle_handwritten";
    let evolved_path = "test_phase6_triangle_evolved";

    // 1. 手写模型并保存为 .otm
    let graph = Graph::new_with_seed(7);
    let input = graph.input_shape(&[1, 2], Some("input")).unwrap();
    let fc1 = Linear::new(&graph, 2, 4, true, "fc1").unwrap();
    let h = fc1.forward(&input).relu();
    let fc2 = Linear::new(&graph, 4, 1, true, "fc2").unwrap();
    let output = fc2.forward(&h);
    graph.save_model(hand_path, &[&output]).unwrap();

    // 2. 从手写 .otm 读取 GraphDescriptor，导入为演化种子
    let (metadata, _) = model_save::read_otm_file(hand_path).unwrap();
    let genome = NetworkGenome::from_graph_descriptor(&metadata.graph).unwrap();
    assert!(
        genome.is_node_level(),
        "手写 .otm 导入后应为 NodeLevel genome"
    );

    // 3. 执行一次变异并重建图
    use crate::nn::evolution::mutation::{MutationRegistry, SizeConstraints};
    let registry = MutationRegistry::default_registry(&TaskMetric::Accuracy, false, false);
    let constraints = SizeConstraints::default();
    let mut mutated = genome.clone();
    let mut mut_rng = StdRng::seed_from_u64(99);
    for _ in 0..20 {
        if registry
            .apply_random(&mut mutated, &constraints, &mut mut_rng)
            .is_ok()
        {
            break;
        }
    }

    let mut build_rng = StdRng::seed_from_u64(123);
    let build = mutated.build(&mut build_rng).unwrap();
    build
        .graph
        .save_model(evolved_path, &[&build.output])
        .unwrap();

    // 4. 作为普通 Graph 模型加载，并继续手写训练
    let loaded = Graph::load_model(evolved_path).unwrap();
    loaded.graph.train();
    let param_vars = &loaded.parameters;
    assert!(!param_vars.is_empty(), "演化后模型应仍可提取参数并继续训练");

    let params_before: Vec<Vec<f32>> = param_vars
        .iter()
        .map(|p| p.node().value().unwrap().to_vec())
        .collect();

    let target_var = loaded.graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let mut optimizer = SGD::new(&loaded.graph, param_vars, 0.01);
    let loss = loaded.outputs[0].mse_loss(&target_var).unwrap();
    for _ in 0..3 {
        loaded.inputs[0]
            .1
            .set_value(&Tensor::new(&[1.0, 0.0], &[1, 2]))
            .unwrap();
        target_var.set_value(&Tensor::new(&[1.0], &[1, 1])).unwrap();
        let loss_val = optimizer.minimize(&loss).unwrap();
        assert!(loss_val.is_finite(), "继续训练时 loss 应为有限值");
    }

    let params_after: Vec<Vec<f32>> = param_vars
        .iter()
        .map(|p| p.node().value().unwrap().to_vec())
        .collect();
    assert!(
        params_before
            .iter()
            .zip(params_after.iter())
            .any(|(before, after)| {
                before
                    .iter()
                    .zip(after.iter())
                    .any(|(a, b)| (a - b).abs() > 1e-10)
            }),
        "继续训练后至少一个参数应发生变化"
    );

    std::fs::remove_file(format!("{hand_path}.otm")).ok();
    std::fs::remove_file(format!("{evolved_path}.otm")).ok();
}

/// Sequential 演化路径不被误伤
///
/// - Sequential 演化仍然保存/加载正常（genome 为 NodeLevel 格式）
/// - 加载后推理成功
#[test]
fn test_phase6_sequential_save_load_unaffected() {
    let temp_path = "test_phase6_sequential_unaffected";

    let data = seq_data();
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_max_generations(1) // 极短，只验证格式
        .with_seed(99)
        .with_verbose(false)
        .run()
        .expect("Sequential 演化失败");

    // Sequential genome 已由主循环迁移为 NodeLevel
    assert!(
        result.genome.is_node_level(),
        "Sequential 演化产出的 genome 应为 NodeLevel"
    );

    // 保存/加载应成功（Sequential 与 Flat 一样写入 NodeLevel .otm）
    result.save(temp_path).expect("Sequential 保存失败");
    let loaded = EvolutionResult::load(temp_path).expect("Sequential 加载失败");

    // 加载后推理不崩溃
    let test_input = Tensor::new(&[0.5, 0.3], &[2, 1]);
    let pred = loaded.predict(&test_input).expect("Sequential 推理失败");
    assert_eq!(pred.shape()[1], 1, "Sequential 输出维度应为 1");

    std::fs::remove_file(format!("{temp_path}.otm")).ok();
}

/// 旧格式 Flat/Spatial LayerLevel genome 加载时返回明确错误
///
/// 模拟加载一个在 NodeLevel 成为唯一支持的序列化格式之前生成的旧格式 .otm：
/// - is_node_level = false（LayerLevel）
/// - seq_len = null（Flat 或 Spatial）
/// - 此组合应被 into_genome() 拒绝，返回包含说明的错误信息
#[test]
fn test_phase6_old_layerlevel_flat_genome_load_rejected() {
    use crate::nn::evolution::model_io::GenomeSerialized;

    // 通过 JSON 反序列化构造一个旧格式 GenomeSerialized（is_node_level=false, seq_len=null）
    // 这模拟从旧版 .otm 文件中解析到的 genome 元数据
    let json = serde_json::json!({
        "layers": [{"innovation_number": 1, "layer_config": {"Linear": {"out_features": 1}}, "enabled": true}],
        "skip_edges": [],
        "input_dim": 2,
        "output_dim": 1,
        "seq_len": null,
        "input_spatial": null,
        "training_config": {
            "optimizer_type": "Adam",
            "learning_rate": 0.01,
            "batch_size": null,
            "weight_decay": 0.0,
            "loss_override": null
        },
        "generated_by": "legacy_evolution",
        "next_innovation": 2,
        "nodes": [],
        "is_node_level": false
    });

    let old_genome: GenomeSerialized =
        serde_json::from_value(json).expect("构造旧格式 GenomeSerialized 失败");

    // into_genome() 应拒绝旧格式 Flat genome 并返回明确错误
    let result = old_genome.into_genome();
    assert!(
        result.is_err(),
        "旧格式 Flat LayerLevel genome 应被拒绝加载"
    );
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("旧格式") || err_msg.contains("LayerLevel") || err_msg.contains("已停止支持"),
        "错误信息应明确指出是旧格式问题，实际：{err_msg}"
    );
}

// ==================== Graph.save_weights/load_weights 测试 ====================

#[test]
fn test_graph_save_load_weights() {
    use crate::nn::{Graph, Linear};

    let temp_path = "test_graph_save_load_weights";

    // 1. 创建并训练模型
    let graph = Graph::new_with_seed(42);
    let fc = Linear::new(&graph, 2, 3, true, "fc").expect("创建层失败");

    // 2. 保存权重
    graph.save_weights(temp_path).expect("保存权重失败");

    // 验证 .bin 文件存在
    assert!(
        std::path::Path::new("test_graph_save_load_weights.bin").exists(),
        ".bin 文件应该存在"
    );

    // 3. 创建相同结构的新图并加载
    let graph2 = Graph::new_with_seed(99); // 不同 seed → 不同初始权重
    let fc2 = Linear::new(&graph2, 2, 3, true, "fc").expect("创建层失败");

    graph2.load_weights(temp_path).expect("加载权重失败");

    // 4. 验证权重一致
    let w1 = fc.weights().value().unwrap().unwrap();
    let w2 = fc2.weights().value().unwrap().unwrap();

    for (a, b) in w1.to_vec().iter().zip(w2.to_vec().iter()) {
        assert!((a - b).abs() < 1e-6, "权重应一致: {} vs {}", a, b);
    }

    // 清理
    std::fs::remove_file("test_graph_save_load_weights.bin").ok();
}

// ==================== 循环边 model_io 测试 ====================

#[test]
fn test_recurrent_genome_save_load_roundtrip() {
    use crate::nn::evolution::mutation::{AddRecurrentEdgeMutation, SizeConstraints};
    use crate::nn::evolution::node_gene::RecurrentEdge;

    // 创建不含 cell-based 循环的序列 NodeLevel 基因组
    let mut g = NetworkGenome::minimal(2, 1);
    let i1 = g.next_innovation_number();
    let i2 = g.next_innovation_number();
    g.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    g.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    g.seq_len = Some(5);
    g.migrate_to_node_level().unwrap();

    // 添加循环边
    let c = SizeConstraints::default();
    let mut rng = StdRng::seed_from_u64(42);
    AddRecurrentEdgeMutation.apply(&mut g, &c, &mut rng).unwrap();

    // 序列化 → 反序列化
    let json = serde_json::to_string(&g).unwrap();
    let g2: NetworkGenome = serde_json::from_str(&json).unwrap();

    // 验证循环边保留
    let has_rec_1 = g.nodes().iter().any(|n| !n.recurrent_parents.is_empty());
    let has_rec_2 = g2.nodes().iter().any(|n| !n.recurrent_parents.is_empty());
    assert!(has_rec_1, "原始基因组应有循环边");
    assert!(has_rec_2, "反序列化后应保留循环边");

    // 验证循环边内容一致
    for (n1, n2) in g.nodes().iter().zip(g2.nodes().iter()) {
        assert_eq!(
            n1.recurrent_parents.len(),
            n2.recurrent_parents.len(),
            "节点 {} 的循环边数量应一致",
            n1.innovation_number
        );
        for (e1, e2) in n1.recurrent_parents.iter().zip(n2.recurrent_parents.iter()) {
            assert_eq!(e1.source_id, e2.source_id, "循环边 source_id 应一致");
            assert_eq!(
                e1.weight_param_id, e2.weight_param_id,
                "循环边 weight_param_id 应一致"
            );
        }
    }

    // 验证两者都能 build
    let mut r1 = StdRng::seed_from_u64(100);
    let mut r2 = StdRng::seed_from_u64(100);
    assert!(g.build(&mut r1).is_ok(), "原始基因组 build 应成功");
    assert!(g2.build(&mut r2).is_ok(), "反序列化基因组 build 应成功");
}

#[test]
fn test_recurrent_genome_has_recurrent_flag() {
    use crate::nn::evolution::mutation::AddRecurrentEdgeMutation;

    let mut g = NetworkGenome::minimal(2, 1);
    let i1 = g.next_innovation_number();
    g.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    g.seq_len = Some(5);
    g.migrate_to_node_level().unwrap();

    let c = crate::nn::evolution::mutation::SizeConstraints::default();
    let mut rng = StdRng::seed_from_u64(42);
    AddRecurrentEdgeMutation
        .apply(&mut g, &c, &mut rng)
        .unwrap();

    let analysis = g.analyze();
    assert!(analysis.is_valid, "应合法: {:?}", analysis.errors);
    assert!(
        analysis.has_recurrent_edges,
        "应标记 has_recurrent_edges"
    );

    // 确认含循环边的节点存在
    let recurrent_count: usize = g
        .nodes()
        .iter()
        .filter(|n| !n.recurrent_parents.is_empty())
        .count();
    assert!(
        recurrent_count > 0,
        "应有含循环边的节点，实际 0"
    );
}
