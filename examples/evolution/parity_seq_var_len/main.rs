/*
 * @Author       : 老董
 * @Date         : 2026-03-09
 * @Description  : 奇偶性检测 — 序列演化示例（变长序列）
 *
 * 与固定长度版本的写法完全相同——唯一区别是数据 seq_len 不一致，
 * SupervisedTask 自动 zero-pad 到 max_len。
 * 系统自动决定使用何种记忆单元（RNN/LSTM/GRU）及网络拓扑。
 *
 * ## 运行
 * ```bash
 * cargo run --example evolution_parity_seq_var_len
 * ```
 */

use only_torch::nn::evolution::gene::TaskMetric;
use only_torch::nn::evolution::{Evolution, EvolutionResult};
use only_torch::tensor::Tensor;
use std::path::Path;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// 生成变长奇偶性检测数据
///
/// 每个样本 seq_len ∈ [min_len, max_len]，形状 [seq_len_i, 1]。
/// 标签: [2] one-hot（偶数=[1,0]，奇数=[0,1]），与固定长度版本保持一致。
fn generate_var_len_parity_data(
    n: usize,
    min_len: usize,
    max_len: usize,
    seed: u64,
) -> (Vec<Tensor>, Vec<Tensor>) {
    let mut inputs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for i in 0..n {
        let mut hasher = DefaultHasher::new();
        (seed, i as u64).hash(&mut hasher);
        let hash = hasher.finish();

        let seq_len = min_len + (hash as usize % (max_len - min_len + 1));

        let mut hasher2 = DefaultHasher::new();
        (seed, i as u64, 0xCAFEu64).hash(&mut hasher2);
        let hash2 = hasher2.finish();

        let data: Vec<f32> = (0..seq_len)
            .map(|j| ((hash2 >> (j % 64)) & 1) as f32)
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
    println!("=== 奇偶性检测 — 序列演化示例（变长序列）===\n");
    println!("序列长度: 4~12（自动 zero-pad）");
    println!("目标: 自动演化到 ≥85% 准确率\n");

    let train = generate_var_len_parity_data(200, 4, 12, 42);
    let test = generate_var_len_parity_data(50, 4, 12, 99);

    let result = Evolution::supervised(train, test, TaskMetric::Accuracy)
        .with_target_metric(0.85)
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
        .visualize("examples/evolution/parity_seq_var_len/evolution_parity_seq_var_len")
        .expect("可视化失败");
    println!("\n计算图已保存: {}", vis.dot_path.display());
    if let Some(img) = &vis.image_path {
        println!("可视化图像: {}", img.display());
    }

    // ==================== 模型保存/加载 ====================
    let model_path = "examples/evolution/parity_seq_var_len/parity_seq_var_len_model";
    result.save(model_path).expect("保存模型失败");
    println!("\n模型已保存: {model_path}.otm");

    let loaded = EvolutionResult::load(model_path).expect("加载模型失败");
    println!("从磁盘加载后架构: {}", loaded.architecture());
    println!("加载模型准确率: {:.0}%", loaded.fitness.primary * 100.0);

    // 清理临时模型文件
    let _ = std::fs::remove_file(Path::new(model_path).with_extension("otm"));

    println!("\n✅ 变长序列数据也能自动演化！");
}
