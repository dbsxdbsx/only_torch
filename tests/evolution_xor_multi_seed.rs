/*
 * @Author       : 老董
 * @Date         : 2026-03-07
 * @Description  : XOR 多种子稳定性测试（Phase 7A 验收）
 *
 * 10 次独立运行（不同种子），成功率须 > 90%。
 * 每次运行输出演化日志，便于分析中性漂移和阶梯状轨迹。
 */

use only_torch::nn::evolution::gene::TaskMetric;
use only_torch::nn::evolution::{Evolution, EvolutionStatus};
use only_torch::tensor::Tensor;

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

#[test]
fn test_xor_multi_seed_stability() {
    let seeds: Vec<u64> = (0..10).collect();
    let mut successes = 0;
    let mut results = Vec::new();

    for &seed in &seeds {
        let data = xor_data();

        println!("\n{}", "=".repeat(60));
        println!("===== Seed {seed} =====");

        let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
            .with_target_metric(1.0)
            .with_seed(seed)
            .with_max_generations(500)
            .run()
            .unwrap();

        let ok = result.status == EvolutionStatus::TargetReached;
        if ok {
            successes += 1;
        }

        println!(
            "Seed {seed}: status={:?}, generations={}, fitness={:.3}, arch={}",
            result.status, result.generations, result.fitness.primary, result.architecture_summary
        );

        results.push((seed, ok, result.generations, result.architecture_summary));
    }

    println!("\n==================== 汇总 ====================");
    for (seed, ok, gens, arch) in &results {
        let mark = if *ok { "PASS" } else { "FAIL" };
        println!("[{mark}] seed={seed}, generations={gens}, arch={arch}");
    }
    println!(
        "\n成功率: {successes}/{} = {:.0}%",
        seeds.len(),
        100.0 * successes as f64 / seeds.len() as f64
    );

    assert!(
        successes as f64 / seeds.len() as f64 > 0.9,
        "10 次独立运行成功率应 > 90%，实际 {successes}/{}",
        seeds.len()
    );
}
