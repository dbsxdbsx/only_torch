/*
 * @Author       : 老董
 * @Date         : 2026-03-07
 * @Description  : 演化过程回调接口
 *
 * EvolutionCallback trait 提供可观测性 + 外部控制：
 * - on_generation: 每代训练+评估后调用
 * - on_new_best: primary 严格提升时调用
 * - on_mutation: 变异执行后调用
 * - should_stop: 每代开始前检查是否终止
 *
 * DefaultCallback 实现默认日志输出 + 最大代数限制。
 */

use super::gene::NetworkGenome;
use super::task::FitnessScore;

// ==================== EvolutionCallback trait ====================

/// 演化过程回调（可观测性 + 外部控制）
pub trait EvolutionCallback {
    /// 每代训练+评估完成后恰好调用一次
    fn on_generation(
        &mut self,
        _generation: usize,
        _genome: &NetworkGenome,
        _loss: f32,
        _score: &FitnessScore,
    ) {
    }

    /// 发现新的全局最优（仅 primary **严格提升**时触发）
    ///
    /// tiebreak_loss 改善或中性漂移均不触发。
    fn on_new_best(
        &mut self,
        _generation: usize,
        _genome: &NetworkGenome,
        _score: &FitnessScore,
    ) {
    }

    /// 变异执行后、下一代训练前调用
    fn on_mutation(
        &mut self,
        _generation: usize,
        _mutation_name: &str,
        _genome: &NetworkGenome,
    ) {
    }

    /// 种群评估完成后调用（每代一次，种群模式专用）
    ///
    /// 默认空操作，DefaultCallback 实现种群级日志输出。
    fn on_population_evaluated(
        &mut self,
        _generation: usize,
        _population_size: usize,
        _offspring_evaluated: usize,
        _archive_size: usize,
        _pareto_front_size: usize,
        _best_primary: f32,
        _best_cost: f32,
    ) {
    }

    /// 每代开始前检查，返回 true 则终止演化
    fn should_stop(&self, _generation: usize) -> bool {
        false
    }
}

// ==================== DefaultCallback ====================

/// 默认回调：日志输出 + 最大代数限制
///
/// `verbose=true` 时 `on_generation` 输出格式：
/// ```text
/// [Gen  0] Input(2) -> [Linear(1)]                 | fitness=0.501 | 初始
/// [Gen  5] Input(2) -> Linear(4) -> ReLU -> [Linear(1)] | fitness=1.000 | GrowHidden *
/// ```
/// `*` 标记表示 `on_new_best` 在本代触发（primary 严格提升）。
pub struct DefaultCallback {
    pub max_generations: usize,
    pub verbose: bool,
    /// 内部 flag：on_new_best 设置，on_generation 消费后清除
    is_new_best: bool,
}

impl DefaultCallback {
    pub fn new(max_generations: usize, verbose: bool) -> Self {
        Self {
            max_generations,
            verbose,
            is_new_best: false,
        }
    }
}

impl Default for DefaultCallback {
    fn default() -> Self {
        Self::new(100, true)
    }
}

impl EvolutionCallback for DefaultCallback {
    fn on_generation(
        &mut self,
        _generation: usize,
        _genome: &NetworkGenome,
        _loss: f32,
        _score: &FitnessScore,
    ) {
        // 种群模式下日志输出移至 on_population_evaluated
    }

    fn on_new_best(
        &mut self,
        _generation: usize,
        _genome: &NetworkGenome,
        _score: &FitnessScore,
    ) {
        self.is_new_best = true;
    }

    fn on_mutation(
        &mut self,
        _generation: usize,
        _mutation_name: &str,
        _genome: &NetworkGenome,
    ) {
    }

    fn on_population_evaluated(
        &mut self,
        generation: usize,
        population_size: usize,
        offspring_evaluated: usize,
        archive_size: usize,
        _pareto_front_size: usize,
        best_primary: f32,
        best_cost: f32,
    ) {
        if !self.verbose {
            return;
        }
        let star = if self.is_new_best { " *" } else { "" };
        self.is_new_best = false;
        println!(
            "[Gen {:>3}] pop={} | off={} | archive={} | best={:.3} | cost={:.0}{}",
            generation, population_size, offspring_evaluated, archive_size,
            best_primary, best_cost, star
        );
    }

    fn should_stop(&self, generation: usize) -> bool {
        generation >= self.max_generations
    }
}
