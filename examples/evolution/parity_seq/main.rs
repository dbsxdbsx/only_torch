/*
 * @Author       : 老董
 * @Date         : 2026-03-09
 * @Description  : 奇偶性检测 — 序列神经架构演化示例（固定长度）
 *
 * 展示 Evolution API 在固定长度序列数据上的零模型代码用法。
 * 系统自动决定使用何种记忆单元（RNN/LSTM/GRU）及网络拓扑。
 *
 * 数据：每个样本为 [seq_len, 1] 的二进制序列，
 * 标签为序列中 1 的个数的奇偶性（0 或 1）。
 * 唯一与 XOR 演化示例的区别：数据从 [2] 变成 [8, 1]。
 *
 * ## 运行
 * ```bash
 * cargo run --example evolution_parity_seq
 * ```
 */

use only_torch::nn::evolution::gene::TaskMetric;
use only_torch::nn::evolution::{Evolution, EvolutionResult};
use only_torch::tensor::Tensor;
use std::path::Path;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// 生成固定长度奇偶性检测数据
///
/// 每个样本: [seq_len, 1] 二进制序列
/// 标签: [2] one-hot（偶数=[1,0]，奇数=[0,1]）
///
/// 使用 one-hot + softmax 交叉熵比单标量 + MSE 更利于梯度优化，
/// 也与 `examples/traditional/parity_rnn_fixed_len` 保持一致，
/// 便于两种方案公平对比。
fn generate_parity_data(n: usize, seq_len: usize, seed: u64) -> (Vec<Tensor>, Vec<Tensor>) {
    let mut inputs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for i in 0..n {
        let mut hasher = DefaultHasher::new();
        (seed, i as u64).hash(&mut hasher);
        let hash = hasher.finish();

        let data: Vec<f32> = (0..seq_len)
            .map(|j| ((hash >> (j % 64)) & 1) as f32)
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
    println!("=== 奇偶性检测 — 序列演化示例（固定长度）===\n");
    println!("起始结构: Input(seq×1) → MemoryCell(hidden≥4) → [Linear(2)]");
    println!("标签: [2] one-hot（偶数=[1,0]，奇数=[0,1]）");
    println!("目标: 自动演化到 ≥90% 准确率\n");

    let seq_len = 8;
    let n_train = 200;
    let n_test = 50;

    let train = generate_parity_data(n_train, seq_len, 42);
    let test = generate_parity_data(n_test, seq_len, 99);

    let result = Evolution::supervised(train, test, TaskMetric::Accuracy)
        .with_target_metric(0.90)
        .with_seed(42)
        .run()
        .expect("演化过程出错");

    println!("\n=== 演化结果 ===");
    println!("状态: {:?}", result.status);
    println!("代数: {}", result.generations);
    println!("准确率: {:.0}%", result.fitness.primary * 100.0);
    println!("最终架构: {}", result.architecture());

    // 可视化演化后的计算图
    let vis = result
        .visualize("examples/evolution/parity_seq/evolution_parity_seq")
        .expect("可视化失败");
    println!("\n计算图已保存: {}", vis.dot_path.display());
    if let Some(img) = &vis.image_path {
        println!("可视化图像: {}", img.display());
    }

    // ==================== 模型保存/加载 ====================
    let model_path = "examples/evolution/parity_seq/parity_seq_model";
    result.save(model_path).expect("保存模型失败");
    println!("\n模型已保存: {model_path}.otm");

    let loaded = EvolutionResult::load(model_path).expect("加载模型失败");
    println!("从磁盘加载后架构: {}", loaded.architecture());

    // 推理验证
    let sample = Tensor::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[seq_len, 1]);
    let pred = loaded.predict(&sample).expect("推理失败");
    println!(
        "从磁盘加载后 [1,0,1,0,1,1,0,0]（4个1，偶数）预测: {:?}",
        pred.to_vec()
    );

    // 清理临时模型文件
    let _ = std::fs::remove_file(Path::new(model_path).with_extension("otm"));

    println!("\n✅ 系统自动发现了解决奇偶性检测问题的序列架构！");
}
