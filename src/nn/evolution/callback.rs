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

// ==================== EvaluationTimingSummary ====================

/// 一批候选评估的阶段耗时汇总。
///
/// `wall_secs` 是批次真实墙钟时间；其他字段是所有候选在对应阶段的累计耗时。
/// 并行评估时，候选累计耗时可能大于墙钟时间，用于定位 build/train/evaluate 等阶段占比。
#[derive(Clone, Debug, Default)]
pub struct EvaluationTimingSummary {
    pub batches: usize,
    pub candidates: usize,
    pub wall_secs: f64,
    pub candidate_secs: f64,
    pub build_secs: f64,
    pub restore_secs: f64,
    pub train_secs: f64,
    pub capture_secs: f64,
    pub evaluate_secs: f64,
    pub cost_secs: f64,
    pub train_setup_secs: f64,
    pub train_shuffle_secs: f64,
    pub train_batch_slice_secs: f64,
    pub train_set_value_secs: f64,
    pub train_zero_grad_secs: f64,
    pub train_backward_secs: f64,
    pub train_step_secs: f64,
    pub train_grad_norm_secs: f64,
    pub primary_min: Option<f32>,
    pub primary_max: Option<f32>,
    pub primary_sum: f32,
    pub primary_count: usize,
    pub cost_min: Option<f32>,
    pub cost_max: Option<f32>,
    pub cost_sum: f32,
    pub cost_count: usize,
}

impl EvaluationTimingSummary {
    pub(crate) fn merge(&mut self, other: &Self) {
        self.batches += other.batches;
        self.candidates += other.candidates;
        self.wall_secs += other.wall_secs;
        self.candidate_secs += other.candidate_secs;
        self.build_secs += other.build_secs;
        self.restore_secs += other.restore_secs;
        self.train_secs += other.train_secs;
        self.capture_secs += other.capture_secs;
        self.evaluate_secs += other.evaluate_secs;
        self.cost_secs += other.cost_secs;
        self.train_setup_secs += other.train_setup_secs;
        self.train_shuffle_secs += other.train_shuffle_secs;
        self.train_batch_slice_secs += other.train_batch_slice_secs;
        self.train_set_value_secs += other.train_set_value_secs;
        self.train_zero_grad_secs += other.train_zero_grad_secs;
        self.train_backward_secs += other.train_backward_secs;
        self.train_step_secs += other.train_step_secs;
        self.train_grad_norm_secs += other.train_grad_norm_secs;
        self.primary_min = merge_min(self.primary_min, other.primary_min);
        self.primary_max = merge_max(self.primary_max, other.primary_max);
        self.primary_sum += other.primary_sum;
        self.primary_count += other.primary_count;
        self.cost_min = merge_min(self.cost_min, other.cost_min);
        self.cost_max = merge_max(self.cost_max, other.cost_max);
        self.cost_sum += other.cost_sum;
        self.cost_count += other.cost_count;
    }

    pub fn avg_candidate_secs(&self) -> f64 {
        if self.candidates == 0 {
            0.0
        } else {
            self.candidate_secs / self.candidates as f64
        }
    }

    pub fn avg_primary(&self) -> Option<f32> {
        if self.primary_count == 0 {
            None
        } else {
            Some(self.primary_sum / self.primary_count as f32)
        }
    }

    pub fn avg_cost(&self) -> Option<f32> {
        if self.cost_count == 0 {
            None
        } else {
            Some(self.cost_sum / self.cost_count as f32)
        }
    }

    pub(crate) fn replace_quality_with(&mut self, other: &Self) {
        self.primary_min = other.primary_min;
        self.primary_max = other.primary_max;
        self.primary_sum = other.primary_sum;
        self.primary_count = other.primary_count;
        self.cost_min = other.cost_min;
        self.cost_max = other.cost_max;
        self.cost_sum = other.cost_sum;
        self.cost_count = other.cost_count;
    }
}

fn merge_min(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

fn merge_max(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

// ==================== CandidatePrefilterSummary ====================

/// P5-lite 候选预筛阶段的轻量统计。
#[derive(Clone, Debug, Default)]
pub struct CandidatePrefilterSummary {
    pub generated: usize,
    pub kept: usize,
    pub score_min: Option<f32>,
    pub score_max: Option<f32>,
    pub score_sum: f32,
    pub flops_min: Option<f32>,
    pub flops_max: Option<f32>,
    pub flops_sum: f32,
    pub flops_count: usize,
}

impl CandidatePrefilterSummary {
    pub(crate) fn observe(&mut self, score: f32, flops: Option<f32>) {
        self.generated += 1;
        self.score_min = Some(self.score_min.map_or(score, |v| v.min(score)));
        self.score_max = Some(self.score_max.map_or(score, |v| v.max(score)));
        self.score_sum += score;
        if let Some(flops) = flops {
            self.flops_min = Some(self.flops_min.map_or(flops, |v| v.min(flops)));
            self.flops_max = Some(self.flops_max.map_or(flops, |v| v.max(flops)));
            self.flops_sum += flops;
            self.flops_count += 1;
        }
    }

    pub fn avg_score(&self) -> Option<f32> {
        if self.generated == 0 {
            None
        } else {
            Some(self.score_sum / self.generated as f32)
        }
    }

    pub fn avg_flops(&self) -> Option<f32> {
        if self.flops_count == 0 {
            None
        } else {
            Some(self.flops_sum / self.flops_count as f32)
        }
    }
}

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

    /// 一批候选评估完成后的阶段耗时。
    ///
    /// `phase` 当前可能是 `init`、`offspring` 或 `asha`。
    fn on_evaluation_timing(
        &mut self,
        _generation: usize,
        _phase: &str,
        _summary: &EvaluationTimingSummary,
    ) {
    }

    /// P5-lite 候选预筛完成后调用。
    fn on_candidate_prefilter(&mut self, _generation: usize, _summary: &CandidatePrefilterSummary) {
    }

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

    fn on_evaluation_timing(
        &mut self,
        _generation: usize,
        phase: &str,
        summary: &EvaluationTimingSummary,
    ) {
        if !self.verbose || summary.candidates == 0 {
            return;
        }

        println!(
            "         timing({phase}): wall={:.2}s cand={} avg={:.2}s | build={:.2}s restore={:.2}s train={:.2}s capture={:.2}s eval={:.2}s cost={:.2}s",
            summary.wall_secs,
            summary.candidates,
            summary.avg_candidate_secs(),
            summary.build_secs,
            summary.restore_secs,
            summary.train_secs,
            summary.capture_secs,
            summary.evaluate_secs,
            summary.cost_secs,
        );
        println!(
            "         train-detail: setup={:.2}s shuffle={:.2}s slice={:.2}s set={:.2}s zero={:.2}s backward={:.2}s step={:.2}s grad_norm={:.2}s",
            summary.train_setup_secs,
            summary.train_shuffle_secs,
            summary.train_batch_slice_secs,
            summary.train_set_value_secs,
            summary.train_zero_grad_secs,
            summary.train_backward_secs,
            summary.train_step_secs,
            summary.train_grad_norm_secs,
        );
        if let (Some(p_min), Some(p_avg), Some(p_max)) = (
            summary.primary_min,
            summary.avg_primary(),
            summary.primary_max,
        ) {
            println!(
                "         eval-detail: primary[min/avg/max]={:.3}/{:.3}/{:.3} | cost[min/avg/max]={:.0}/{:.0}/{:.0}",
                p_min,
                p_avg,
                p_max,
                summary.cost_min.unwrap_or(f32::NAN),
                summary.avg_cost().unwrap_or(f32::NAN),
                summary.cost_max.unwrap_or(f32::NAN),
            );
        }
    }

    fn on_candidate_prefilter(&mut self, generation: usize, summary: &CandidatePrefilterSummary) {
        if !self.verbose || summary.generated == 0 {
            return;
        }

        println!(
            "         p5-lite(gen={generation}): kept={}/{} | score[min/avg/max]={:.3}/{:.3}/{:.3} | flops[min/avg/max]={:.0}/{:.0}/{:.0}",
            summary.kept,
            summary.generated,
            summary.score_min.unwrap_or(f32::NAN),
            summary.avg_score().unwrap_or(f32::NAN),
            summary.score_max.unwrap_or(f32::NAN),
            summary.flops_min.unwrap_or(f32::NAN),
            summary.avg_flops().unwrap_or(f32::NAN),
            summary.flops_max.unwrap_or(f32::NAN),
        );
    }

    fn should_stop(&self, generation: usize) -> bool {
        generation >= self.max_generations
    }
}
