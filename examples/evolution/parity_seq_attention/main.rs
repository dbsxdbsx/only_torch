/*
 * @Author       : 老董
 * @Date         : 2026-05-02
 * @Description  : 奇偶性检测 — 引入 MultiHeadAttention 的序列演化示例（固定长度）
 *
 * 本示例与 `examples/evolution/parity_seq/main.rs` 数据完全一致，唯一差别是
 * `with_sequence_ops(SequenceOpSet::RecurrentWithAttention)`：让演化的 InsertLayer
 * 在序列任务中**同时**采样 RNN/LSTM/GRU 与 MultiHeadAttention，由 Pareto 选择决定
 * 谁更适合这个任务。
 *
 * 注意点：
 * 1. parity 是经典 attention 不擅长的任务（counting / parity 没有 chain-of-thought
 *    时 vanilla transformer 通常 60–70%）。这里 target 设到 ≥85%，让搜索过程
 *    既能享受混合算子带来的探索多样性，也不强迫一定选 attention。
 * 2. 与 `parity_seq` 一样使用 8 长度二进制序列，run 起来约几分钟。
 * 3. `attention_num_heads_candidates` 默认 `[2, 4, 8]`，与默认 `min_hidden_size=8` /
 *    `AlignTo(8)` 兼容。
 *
 * ## 运行
 * ```bash
 * cargo run --example evolution_parity_seq_attention
 * ```
 */

use only_torch::data::SyntheticRng;
use only_torch::nn::evolution::{
    Evolution, EvolutionResult, SequenceOpSet, SizeConstraints, SizeStrategy, TaskMetric,
};
use only_torch::tensor::Tensor;
use std::path::Path;

fn generate_parity_data(n: usize, seq_len: usize, seed: u64) -> (Vec<Tensor>, Vec<Tensor>) {
    let mut inputs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for i in 0..n {
        let mut rng = SyntheticRng::from_seed_parts(seed, &[i as u64]);

        let data: Vec<f32> = (0..seq_len)
            .map(|_| if rng.next_bool() { 1.0 } else { 0.0 })
            .collect();
        let ones_count: f32 = data.iter().sum();
        let parity = ones_count as usize % 2;

        let one_hot = if parity == 0 { [1.0, 0.0] } else { [0.0, 1.0] };

        inputs.push(Tensor::new(&data, &[seq_len, 1]));
        labels.push(Tensor::new(&one_hot, &[2]));
    }

    (inputs, labels)
}

fn main() {
    println!("=== 奇偶性检测 — 序列演化（RNN/LSTM/GRU + Attention 混合）===\n");
    println!("起始结构: Input(seq×1) → MemoryCell(hidden≥4) → [Linear(2)]");
    println!("候选算子: RNN / LSTM / GRU / MultiHeadAttention");
    println!("attention_num_heads ∈ {{2, 4, 8}}（embed_dim 必须能整除）");
    println!("目标: ≥85% 准确率（混合搜索目标，attention 在 parity 上不一定占优）\n");

    let seq_len = 8;
    let n_train = 500;
    let n_test = 100;

    let train = generate_parity_data(n_train, seq_len, 42);
    let test = generate_parity_data(n_test, seq_len, 99);

    // 让 attention/RNN 候选共用同一套尺寸约束。AlignTo(8) 让 hidden_size
    // 自然落在 8/16/24/32 等可以被 num_heads ∈ {2,4,8} 整除的值上。
    let constraints = SizeConstraints {
        max_layers: 8,
        max_hidden_size: 64,
        max_total_params: 100_000,
        min_hidden_size: 8,
        size_strategy: SizeStrategy::AlignTo(8),
        sequence_ops: SequenceOpSet::RecurrentWithAttention,
        attention_num_heads_candidates: vec![2, 4, 8],
    };

    let result = Evolution::supervised(train, test, TaskMetric::Accuracy)
        .with_target_metric(0.85)
        .with_seed(42)
        .with_constraints(constraints)
        .with_max_generations(40)
        .run()
        .expect("演化过程出错");

    println!("\n=== 演化结果 ===");
    println!("状态: {:?}", result.status);
    println!("代数: {}", result.generations);
    println!("准确率: {:.2}%", result.fitness.primary * 100.0);
    println!("最终架构: {}", result.architecture());

    // 可视化演化后的计算图
    match result.visualize("examples/evolution/parity_seq_attention/evolution_parity_seq_attention")
    {
        Ok(vis) => {
            println!("\n计算图已保存: {}", vis.dot_path.display());
            if let Some(img) = &vis.image_path {
                println!("可视化图像: {}", img.display());
            }
        }
        Err(err) => {
            eprintln!("\n可视化保存失败（best-effort，不阻塞示例）：{err}");
        }
    }

    // 模型保存与回放
    let model_path = "examples/evolution/parity_seq_attention/parity_seq_attention_model";
    result.save(model_path).expect("保存模型失败");
    println!("\n模型已保存: {model_path}.otm");

    let loaded = EvolutionResult::load(model_path).expect("加载模型失败");
    println!("从磁盘加载后架构: {}", loaded.architecture());

    let sample = Tensor::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[seq_len, 1]);
    let pred = loaded.predict(&sample).expect("推理失败");
    println!(
        "样本 [1,0,1,0,1,1,0,0]（4个1，偶数）预测: {:?}",
        pred.to_vec()
    );

    let _ = std::fs::remove_file(Path::new(model_path).with_extension("otm"));

    println!("\n演化引擎在 RNN 系列与注意力之间挑出了适合 parity 的架构。");
}
