/*
 * @Author       : 老董
 * @Date         : 2026-03-09
 * @Description  : 演化模型 save/load (.otm) 测试
 */

use crate::nn::evolution::gene::{NetworkGenome, TaskMetric};
use crate::nn::evolution::{Evolution, EvolutionResult};
use crate::tensor::Tensor;

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
    assert_eq!(restored.layers.len(), genome.layers.len());
    assert_eq!(restored.skip_edges.len(), 0);
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
        assert!(
            (a - b).abs() < 1e-6,
            "权重应一致: {} vs {}",
            a,
            b
        );
    }

    // 清理
    std::fs::remove_file("test_graph_save_load_weights.bin").ok();
}
