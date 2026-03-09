/*
 * @Author       : 老董
 * @Date         : 2026-03-09
 * @Description  : XOR 神经架构演化示例（零模型代码）
 *
 * 与 `examples/xor`（手动定义网络 + 训练循环）不同，
 * 本示例展示 **Evolution API**——只提供数据和目标，
 * 系统从最小结构 `Input(2) → [Linear(1)]` 出发，
 * 通过层级变异自动发现能解决 XOR 问题的最优架构。
 *
 * ## 运行
 * ```bash
 * cargo run --example evolution_xor
 * ```
 */

use only_torch::nn::evolution::gene::TaskMetric;
use only_torch::nn::evolution::Evolution;
use only_torch::tensor::Tensor;

/// XOR 数据集（二分类标量标签）
fn xor_data() -> (Vec<Tensor>, Vec<Tensor>) {
    (
        vec![
            Tensor::new(&[0.0, 0.0], &[2]),
            Tensor::new(&[0.0, 1.0], &[2]),
            Tensor::new(&[1.0, 0.0], &[2]),
            Tensor::new(&[1.0, 1.0], &[2]),
        ],
        vec![
            Tensor::new(&[0.0], &[1]), // XOR(0,0) = 0
            Tensor::new(&[1.0], &[1]), // XOR(0,1) = 1
            Tensor::new(&[1.0], &[1]), // XOR(1,0) = 1
            Tensor::new(&[0.0], &[1]), // XOR(1,1) = 0
        ],
    )
}

fn main() {
    println!("=== XOR 神经架构演化示例 ===\n");
    println!("起始结构: Input(2) → [Linear(1)]（仅输出头，无隐藏层）");
    println!("目标: 通过层级变异自动演化到 100% XOR 准确率\n");

    let data = xor_data();

    // Evolution API：只需提供数据、指标、目标——零模型代码
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_target_metric(1.0)
        .with_seed(42)
        .run()
        .expect("演化过程出错");

    println!("\n=== 演化结果 ===");
    println!("状态: {:?}", result.status);
    println!("代数: {}", result.generations);
    println!("准确率: {:.0}%", result.fitness.primary * 100.0);
    println!("最终架构: {}", result.architecture_summary);
    println!("\n✅ 系统自动发现了解决 XOR 问题的网络架构！");
}
