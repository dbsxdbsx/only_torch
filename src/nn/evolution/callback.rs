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
    fn on_new_best(&mut self, _generation: usize, _genome: &NetworkGenome, _score: &FitnessScore) {}

    /// 变异执行后、下一代训练前调用
    fn on_mutation(&mut self, _generation: usize, _mutation_name: &str, _genome: &NetworkGenome) {}

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

    /// 每代权重继承统计（仅 NodeLevel genome 有意义）
    ///
    /// `inherited`：本代 offspring 中权重继承（全量或部分）的参数节点数之和。
    /// `reinitialized`：无法继承、保留随机初始化的参数节点数之和。
    ///
    /// 在 verbose 调试时可借此监控 Grow/Shrink 变异后权重复用率是否符合预期。
    fn on_inherit_stats(&mut self, _generation: usize, _inherited: usize, _reinitialized: usize) {}

    /// 每代开始前检查，返回 true 则终止演化
    fn should_stop(&self, _generation: usize) -> bool {
        false
    }
}

// ==================== DefaultCallback ====================

/// 默认回调：日志输出 + 最大代数限制
///
/// `verbose=true` 时输出两行：
/// ```text
/// [Gen   5] arch=nodes=9 active=9 params=4 | metrics=accuracy=0.875 f1=0.867
/// [Gen   5] pop=8 | off=8 | archive=6 | best=1.000 | cost=42 *
/// ```
/// `*` 标记表示 `on_new_best` 在本代触发（primary 严格提升）。
pub struct DefaultCallback {
    pub max_generations: usize,
    pub verbose: bool,
    /// 内部 flag：on_new_best 设置，on_generation 消费后清除
    is_new_best: bool,
    last_arch_summary: Option<String>,
}

impl DefaultCallback {
    pub fn new(max_generations: usize, verbose: bool) -> Self {
        Self {
            max_generations,
            verbose,
            is_new_best: false,
            last_arch_summary: None,
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
        generation: usize,
        genome: &NetworkGenome,
        _loss: f32,
        _score: &FitnessScore,
    ) {
        if !self.verbose {
            return;
        }
        let summary = genome.main_path_summary();
        self.last_arch_summary = Some(summary.clone());
        let report = _score.report.format_compact();
        if report.is_empty() {
            println!("[Gen {:>3}] arch={summary}", generation);
        } else {
            println!("[Gen {:>3}] arch={summary} | metrics={report}", generation);
        }
    }

    fn on_new_best(&mut self, _generation: usize, _genome: &NetworkGenome, _score: &FitnessScore) {
        self.is_new_best = true;
    }

    fn on_mutation(&mut self, _generation: usize, _mutation_name: &str, _genome: &NetworkGenome) {}

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
        let arch = self
            .last_arch_summary
            .take()
            .unwrap_or_else(|| "<unknown>".to_string());
        println!(
            "[Gen {:>3}] pop={} | off={} | archive={} | best={:.3} | cost={:.0} | arch={}{}",
            generation,
            population_size,
            offspring_evaluated,
            archive_size,
            best_primary,
            best_cost,
            arch,
            star
        );
    }

    fn on_inherit_stats(&mut self, generation: usize, inherited: usize, reinitialized: usize) {
        if !self.verbose {
            return;
        }
        let total = inherited + reinitialized;
        let pct = if total > 0 {
            inherited * 100 / total
        } else {
            0
        };
        println!(
            "         inherit={}/{} ({}%) reinit={}",
            inherited, total, pct, reinitialized
        );
        let _ = generation; // 仅在 verbose 模式打印，generation 暂不使用
    }

    fn should_stop(&self, generation: usize) -> bool {
        generation >= self.max_generations
    }
}
