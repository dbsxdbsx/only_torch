/*
 * @Author       : 老董
 * @Date         : 2026-03-09
 * @Description  : Iris 鸢尾花神经架构演化示例（零模型代码 + mini-batch）
 *
 * 与 `examples/iris`（手动定义 MLP + 训练循环）不同，
 * 本示例展示 **Evolution API**——只提供数据和目标，
 * 系统从最小结构 `Input(4) → [Linear(3)]` 出发，
 * 通过层级变异自动发现能分类 Iris 数据集的最优架构。
 *
 * 关键特性：
 * - 150 样本 → 自动选择 mini-batch 训练（batch_size=64）
 * - 三分类 → 自动推断 CrossEntropy loss + argmax accuracy
 * - 可通过 `.with_batch_size()` 显式覆盖 batch size
 *
 * ## 运行
 * ```bash
 * cargo run --example evolution_iris
 * ```
 */

mod data;

use data::{load_iris, CLASS_NAMES};
use only_torch::nn::evolution::gene::TaskMetric;
use only_torch::nn::evolution::Evolution;

fn main() {
    println!("=== Iris 鸢尾花神经架构演化示例 ===\n");
    println!("数据: 150 样本，4 特征，3 类别 ({:?})", CLASS_NAMES);
    println!("起始结构: Input(4) → [Linear(3)]（仅输出头，无隐藏层）");
    println!("训练模式: mini-batch（150 样本 → auto batch_size=64）");
    println!("目标: 通过层级变异自动演化到 ≥95% 准确率\n");

    let data = load_iris();

    // Evolution API：只需提供数据、指标、目标——零模型代码
    // 150 样本 > 128 → 自动启用 mini-batch（batch_size=64）
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_target_metric(0.95)
        .with_seed(42)
        .run()
        .expect("演化过程出错");

    println!("\n=== 演化结果 ===");
    println!("状态: {:?}", result.status);
    println!("代数: {}", result.generations);
    println!("准确率: {:.1}%", result.fitness.primary * 100.0);
    println!("最终架构: {}", result.architecture_summary);
    println!("\n✅ 系统自动发现了解决 Iris 三分类问题的网络架构！");
}
