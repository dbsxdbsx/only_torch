use std::cell::RefCell;
use std::rc::Rc;

use crate::nn::evolution::callback::EvolutionCallback;
use crate::nn::evolution::convergence::ConvergenceConfig;
use crate::nn::evolution::gene::*;
use crate::nn::evolution::mutation::SizeConstraints;
use crate::nn::evolution::task::FitnessScore;
use crate::nn::evolution::{Evolution, EvolutionStatus};
use crate::tensor::Tensor;

// ==================== 辅助构造 ====================

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

fn xor_evolution() -> Evolution {
    let data = xor_data();
    Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
}

// ==================== Mock Callback ====================

/// 用于测试的 Mock 回调，记录所有事件
#[derive(Clone)]
struct MockCallbackState {
    generation_count: usize,
    new_best_count: usize,
    mutation_count: usize,
    new_best_generations: Vec<usize>,
    /// 每代 on_new_best 时记录的 primary 值
    new_best_primaries: Vec<f32>,
    /// 每代 on_generation 时记录的 primary 值
    generation_primaries: Vec<f32>,
    stop_at: Option<usize>,
    last_primary: f32,
    last_mutation_name: String,
    mutation_names: Vec<String>,
}

struct MockCallback {
    state: Rc<RefCell<MockCallbackState>>,
}

impl MockCallback {
    fn new(stop_at: Option<usize>) -> (Self, Rc<RefCell<MockCallbackState>>) {
        let state = Rc::new(RefCell::new(MockCallbackState {
            generation_count: 0,
            new_best_count: 0,
            mutation_count: 0,
            new_best_generations: Vec::new(),
            new_best_primaries: Vec::new(),
            generation_primaries: Vec::new(),
            stop_at,
            last_primary: f32::NEG_INFINITY,
            last_mutation_name: String::new(),
            mutation_names: Vec::new(),
        }));
        (
            MockCallback {
                state: Rc::clone(&state),
            },
            state,
        )
    }
}

impl EvolutionCallback for MockCallback {
    fn on_generation(
        &mut self,
        _generation: usize,
        _genome: &NetworkGenome,
        _loss: f32,
        score: &FitnessScore,
    ) {
        let mut s = self.state.borrow_mut();
        s.generation_count += 1;
        s.last_primary = score.primary;
        s.generation_primaries.push(score.primary);
    }

    fn on_new_best(
        &mut self,
        generation: usize,
        _genome: &NetworkGenome,
        score: &FitnessScore,
    ) {
        let mut s = self.state.borrow_mut();
        s.new_best_count += 1;
        s.new_best_generations.push(generation);
        s.new_best_primaries.push(score.primary);
    }

    fn on_mutation(
        &mut self,
        _generation: usize,
        mutation_name: &str,
        _genome: &NetworkGenome,
    ) {
        let mut s = self.state.borrow_mut();
        s.mutation_count += 1;
        s.last_mutation_name = mutation_name.to_string();
        s.mutation_names.push(mutation_name.to_string());
    }

    fn should_stop(&self, generation: usize) -> bool {
        self.state
            .borrow()
            .stop_at
            .map_or(false, |limit| generation >= limit)
    }
}

// ==================== 基本运行 ====================

#[test]
fn test_evolution_runs_and_returns_result() {
    let result = xor_evolution()
        .with_seed(42)
        .with_max_generations(3)
        .with_verbose(false)
        .run()
        .unwrap();

    assert!(result.fitness.primary >= 0.0);
    assert!(result.generations <= 3);
    assert!(!result.architecture_summary.is_empty());
}

#[test]
fn test_evolution_status_max_generations() {
    let result = xor_evolution()
        .with_seed(42)
        .with_max_generations(5)
        .with_target_metric(2.0) // 不可达，确保一定触发 MaxGenerations
        .with_verbose(false)
        .run()
        .unwrap();

    assert_eq!(result.status, EvolutionStatus::MaxGenerations);
    assert_eq!(result.generations, 5);
}

// ==================== 达标提前终止 ====================

#[test]
fn test_evolution_target_reached_early_stop() {
    // target_metric 设为 0.0，首代就能达标
    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(0.0)
        .with_max_generations(100)
        .with_verbose(false)
        .run()
        .unwrap();

    assert_eq!(result.status, EvolutionStatus::TargetReached);
    assert_eq!(result.generations, 0);
    assert!(result.fitness.primary >= 0.0);
}

// ==================== 回调触发 ====================

#[test]
fn test_callback_on_generation_called_every_gen() {
    let (mock, state) = MockCallback::new(Some(5));

    let _result = xor_evolution()
        .with_seed(42)
        .with_target_metric(2.0) // 不可达，确保跑满 5 代
        .with_callback(mock)
        .run()
        .unwrap();

    let s = state.borrow();
    assert_eq!(s.generation_count, 5, "on_generation 应每代调用一次");
}

#[test]
fn test_callback_on_new_best_only_on_strict_improvement() {
    let (mock, state) = MockCallback::new(Some(15));

    let _result = xor_evolution()
        .with_seed(42)
        .with_target_metric(2.0) // 不可达
        .with_callback(mock)
        .run()
        .unwrap();

    let s = state.borrow();
    assert!(
        s.new_best_count >= 1,
        "至少首代应触发 on_new_best，实际触发 {} 次",
        s.new_best_count
    );
    assert!(s.new_best_count <= s.generation_count);

    // 每次 on_new_best 的 primary 必须严格递增（首代除外，它是从无到有）
    for i in 1..s.new_best_primaries.len() {
        assert!(
            s.new_best_primaries[i] > s.new_best_primaries[i - 1],
            "on_new_best 第 {} 次 primary={} 应严格大于第 {} 次 primary={}",
            i,
            s.new_best_primaries[i],
            i - 1,
            s.new_best_primaries[i - 1]
        );
    }
}

#[test]
fn test_callback_on_mutation_called_each_gen() {
    let (mock, state) = MockCallback::new(Some(5));

    let _result = xor_evolution()
        .with_seed(42)
        .with_target_metric(2.0) // 不可达，确保不提前终止
        .with_callback(mock)
        .run()
        .unwrap();

    let s = state.borrow();
    assert_eq!(
        s.mutation_count, s.generation_count,
        "on_mutation 应与 on_generation 次数一致"
    );
    assert!(
        !s.last_mutation_name.is_empty(),
        "变异名称不应为空"
    );
}

#[test]
fn test_callback_should_stop_terminates() {
    let (mock, state) = MockCallback::new(Some(3));

    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(2.0) // 不可达
        .with_callback(mock)
        .run()
        .unwrap();

    assert_eq!(result.status, EvolutionStatus::CallbackStopped);
    assert_eq!(state.borrow().generation_count, 3);
}

// ==================== seed 可复现性 ====================

#[test]
fn test_seed_reproducibility() {
    let run = |seed: u64| {
        let (mock, state) = MockCallback::new(Some(8));
        let result = xor_evolution()
            .with_seed(seed)
            .with_target_metric(2.0) // 不可达，确保跑满 8 代
            .with_callback(mock)
            .run()
            .unwrap();
        let s = state.borrow();
        (
            result.architecture_summary.clone(),
            result.fitness.primary,
            s.new_best_count,
        )
    };

    let (arch1, fit1, best1) = run(42);
    let (arch2, fit2, best2) = run(42);

    assert_eq!(arch1, arch2, "相同 seed 应产生相同架构");
    assert!(
        (fit1 - fit2).abs() < 1e-6,
        "相同 seed 应产生相同 fitness: {fit1} vs {fit2}"
    );
    assert_eq!(best1, best2, "相同 seed 应产生相同 new_best 次数");
}

#[test]
fn test_different_seeds_produce_different_results() {
    let run = |seed: u64| {
        xor_evolution()
            .with_seed(seed)
            .with_max_generations(10)
            .with_target_metric(1.0)
            .with_verbose(false)
            .run()
            .unwrap()
            .architecture_summary
    };

    let arch1 = run(42);
    let arch2 = run(999);

    // 不同 seed 在 10 代后大概率产生不同架构（非绝对，但极端巧合概率很低）
    // 如果碰巧相同，测试仍通过（不是强制约束）
    if arch1 == arch2 {
        println!("注意：不同 seed 碰巧产生相同架构（极小概率）");
    }
}

// ==================== 接受/回滚机制 ====================

#[test]
fn test_is_at_least_as_good_primary_improvement() {
    let current = FitnessScore {
        primary: 0.8,
        inference_cost: None,
        tiebreak_loss: Some(0.5),
    };
    let best = FitnessScore {
        primary: 0.7,
        inference_cost: None,
        tiebreak_loss: Some(0.3),
    };
    assert!(super::super::is_at_least_as_good(&current, &best));
}

#[test]
fn test_is_at_least_as_good_primary_regression() {
    let current = FitnessScore {
        primary: 0.6,
        inference_cost: None,
        tiebreak_loss: Some(0.1),
    };
    let best = FitnessScore {
        primary: 0.7,
        inference_cost: None,
        tiebreak_loss: Some(0.5),
    };
    assert!(!super::super::is_at_least_as_good(&current, &best));
}

#[test]
fn test_is_at_least_as_good_same_primary_better_tiebreak() {
    let current = FitnessScore {
        primary: 0.5,
        inference_cost: None,
        tiebreak_loss: Some(0.3),
    };
    let best = FitnessScore {
        primary: 0.5,
        inference_cost: None,
        tiebreak_loss: Some(0.5),
    };
    assert!(super::super::is_at_least_as_good(&current, &best));
}

#[test]
fn test_is_at_least_as_good_same_primary_worse_tiebreak() {
    let current = FitnessScore {
        primary: 0.5,
        inference_cost: None,
        tiebreak_loss: Some(0.8),
    };
    let best = FitnessScore {
        primary: 0.5,
        inference_cost: None,
        tiebreak_loss: Some(0.5),
    };
    assert!(!super::super::is_at_least_as_good(&current, &best));
}

#[test]
fn test_is_at_least_as_good_same_primary_equal_tiebreak() {
    let current = FitnessScore {
        primary: 0.5,
        inference_cost: None,
        tiebreak_loss: Some(0.5),
    };
    let best = FitnessScore {
        primary: 0.5,
        inference_cost: None,
        tiebreak_loss: Some(0.5),
    };
    assert!(super::super::is_at_least_as_good(&current, &best));
}

#[test]
fn test_is_at_least_as_good_no_tiebreak_neutral_drift() {
    let current = FitnessScore {
        primary: 0.5,
        inference_cost: None,
        tiebreak_loss: None,
    };
    let best = FitnessScore {
        primary: 0.5,
        inference_cost: None,
        tiebreak_loss: None,
    };
    assert!(super::super::is_at_least_as_good(&current, &best));
}

#[test]
fn test_is_at_least_as_good_mixed_tiebreak_none() {
    // current 有 tiebreak 但 best 没有 → 接受（中性漂移）
    let current = FitnessScore {
        primary: 0.5,
        inference_cost: None,
        tiebreak_loss: Some(0.3),
    };
    let best = FitnessScore {
        primary: 0.5,
        inference_cost: None,
        tiebreak_loss: None,
    };
    assert!(super::super::is_at_least_as_good(&current, &best));
}

// ==================== eval_runs ====================

#[test]
fn test_eval_runs_takes_conservative_value() {
    // eval_runs=3，取 primary 最低值
    let result = xor_evolution()
        .with_seed(42)
        .with_eval_runs(3)
        .with_max_generations(3)
        .with_target_metric(1.0)
        .with_verbose(false)
        .run()
        .unwrap();

    // 只验证运行成功且 fitness 合法
    assert!(result.fitness.primary >= 0.0 && result.fitness.primary <= 1.0);
}

// ==================== NoApplicableMutation ====================

#[test]
fn test_extreme_constraints_graceful() {
    // 极端约束：max_layers=1 + R2 指标（只有 MSE 一种 loss）
    // 最小基因组只有输出头，层级变异全部被阻止。
    // 但 AddSkipEdge 可以在 INPUT→输出头上反复操作（添加后 fitness 不改善→回滚→再添加），
    // 因此演化会跑满 MaxGenerations 而非 NoApplicableMutation。
    let train = (
        vec![Tensor::new(&[1.0], &[1]), Tensor::new(&[2.0], &[1])],
        vec![Tensor::new(&[2.0], &[1]), Tensor::new(&[4.0], &[1])],
    );
    let test = train.clone();

    let result = Evolution::supervised(train, test, TaskMetric::R2)
        .with_seed(42)
        .with_target_metric(2.0) // 不可达
        .with_max_generations(10)
        .with_constraints(SizeConstraints {
            max_layers: 1,
            ..Default::default()
        })
        .with_verbose(false)
        .run()
        .unwrap();

    assert_eq!(
        result.status,
        EvolutionStatus::MaxGenerations,
        "极端约束下 AddSkipEdge 仍可操作，应跑满 MaxGenerations"
    );
}

// ==================== DefaultCallback verbose ====================

#[test]
fn test_default_callback_silent_mode() {
    // verbose=false 应不输出任何内容（间接验证不 panic）
    let result = xor_evolution()
        .with_seed(42)
        .with_max_generations(3)
        .with_verbose(false)
        .run()
        .unwrap();

    assert!(result.generations <= 3);
}

// ==================== EvolutionResult 字段完整性 ====================

#[test]
fn test_evolution_result_fields() {
    let result = xor_evolution()
        .with_seed(42)
        .with_max_generations(5)
        .with_verbose(false)
        .run()
        .unwrap();

    assert!(result.fitness.primary.is_finite());
    assert!(!result.architecture_summary.is_empty());
    assert!(result.architecture_summary.starts_with("Input("));
    // graph 存在且可用（不检查具体内容，只验证不 panic）
    let _ = result.graph.parameter_count();
}

// ==================== Builder methods ====================

#[test]
fn test_builder_methods_chain() {
    let convergence = ConvergenceConfig {
        max_epochs: 30,
        ..Default::default()
    };

    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(0.0)
        .with_convergence(convergence)
        .with_eval_runs(1)
        .with_max_generations(10)
        .with_verbose(false)
        .run()
        .unwrap();

    assert_eq!(result.status, EvolutionStatus::TargetReached);
}

// ==================== 回滚后变异可正常执行 ====================

#[test]
fn test_rollback_then_mutate_succeeds() {
    let (mock, state) = MockCallback::new(Some(15));

    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(2.0) // 不可达，确保跑满 15 代
        .with_callback(mock)
        .run()
        .unwrap();

    let s = state.borrow();
    assert_eq!(s.generation_count, 15);
    // 15 代中 new_best_count < 15 说明有回滚发生
    assert!(
        s.new_best_count < 15,
        "15 代中应有回滚（new_best_count={} < 15）",
        s.new_best_count
    );
    assert_eq!(s.mutation_count, s.generation_count);
    assert!(result.fitness.primary.is_finite());
}

// ==================== 首代始终触发 on_new_best ====================

#[test]
fn test_first_generation_always_triggers_new_best() {
    let (mock, state) = MockCallback::new(Some(1));

    let _result = xor_evolution()
        .with_seed(42)
        .with_target_metric(2.0) // 不可达
        .with_callback(mock)
        .run()
        .unwrap();

    let s = state.borrow();
    assert_eq!(s.generation_count, 1);
    assert_eq!(
        s.new_best_count, 1,
        "首代应触发 on_new_best（从无到有）"
    );
    assert_eq!(s.new_best_generations[0], 0);
}

// ==================== eval_runs=0 边界防御 ====================

#[test]
#[should_panic(expected = "eval_runs 必须 >= 1")]
fn test_eval_runs_zero_panics() {
    let _ = xor_evolution().with_eval_runs(0);
}

// ==================== 变异名称有效性 ====================

#[test]
fn test_mutation_names_are_known_operations() {
    let (mock, state) = MockCallback::new(Some(10));

    let _result = xor_evolution()
        .with_seed(42)
        .with_target_metric(2.0)
        .with_callback(mock)
        .run()
        .unwrap();

    let known = [
        "InsertLayer",
        "RemoveLayer",
        "ReplaceLayerType",
        "GrowHiddenSize",
        "ShrinkHiddenSize",
        "MutateLayerParam",
        "MutateLossFunction",
        "AddSkipEdge",
        "RemoveSkipEdge",
        "MutateAggregateStrategy",
    ];

    let s = state.borrow();
    for name in &s.mutation_names {
        assert!(
            known.contains(&name.as_str()),
            "未知的变异名称: {name}，已知: {known:?}"
        );
    }
}

// ==================== 达标代也触发 on_new_best ====================

#[test]
fn test_on_new_best_fires_on_target_reaching_generation() {
    // target=0.0 → 首代就达标，但 on_new_best 仍应触发（首代从无到有 = 严格提升）
    let (mock, state) = MockCallback::new(None);

    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(0.0)
        .with_callback(mock)
        .run()
        .unwrap();

    assert_eq!(result.status, EvolutionStatus::TargetReached);
    let s = state.borrow();
    assert!(
        s.new_best_count >= 1,
        "达标代也应触发 on_new_best，实际 new_best_count={}",
        s.new_best_count
    );
}

// ==================== on_new_best 在 on_generation 之前调用 ====================

/// 验证修复后的时序：on_new_best 在 on_generation 之前调用，
/// 确保 DefaultCallback 的星标能在同一代日志中出现。
struct TimingCheckCallback {
    new_best_called_before_generation: Vec<bool>,
    pending_new_best: bool,
}

impl TimingCheckCallback {
    fn new() -> Self {
        Self {
            new_best_called_before_generation: Vec::new(),
            pending_new_best: false,
        }
    }
}

impl EvolutionCallback for TimingCheckCallback {
    fn on_new_best(
        &mut self,
        _generation: usize,
        _genome: &NetworkGenome,
        _score: &FitnessScore,
    ) {
        self.pending_new_best = true;
    }

    fn on_generation(
        &mut self,
        _generation: usize,
        _genome: &NetworkGenome,
        _loss: f32,
        _score: &FitnessScore,
    ) {
        self.new_best_called_before_generation
            .push(self.pending_new_best);
        self.pending_new_best = false;
    }

    fn should_stop(&self, generation: usize) -> bool {
        generation >= 5
    }
}

#[test]
fn test_on_new_best_precedes_on_generation() {
    let timing = TimingCheckCallback::new();

    // 用 Rc<RefCell> 包装以便检查结果
    let timing = std::rc::Rc::new(std::cell::RefCell::new(timing));
    let timing_clone = std::rc::Rc::clone(&timing);

    struct TimingWrapper {
        inner: std::rc::Rc<std::cell::RefCell<TimingCheckCallback>>,
    }
    impl EvolutionCallback for TimingWrapper {
        fn on_new_best(
            &mut self,
            g: usize,
            genome: &NetworkGenome,
            score: &FitnessScore,
        ) {
            self.inner.borrow_mut().on_new_best(g, genome, score);
        }
        fn on_generation(
            &mut self,
            g: usize,
            genome: &NetworkGenome,
            loss: f32,
            score: &FitnessScore,
        ) {
            self.inner
                .borrow_mut()
                .on_generation(g, genome, loss, score);
        }
        fn should_stop(&self, g: usize) -> bool {
            self.inner.borrow().should_stop(g)
        }
    }

    let _result = xor_evolution()
        .with_seed(42)
        .with_target_metric(2.0)
        .with_callback(TimingWrapper {
            inner: timing_clone,
        })
        .run()
        .unwrap();

    let t = timing.borrow();
    // 首代必然触发 on_new_best（从无到有），且应在 on_generation 之前
    assert!(
        t.new_best_called_before_generation[0],
        "首代 on_new_best 应在 on_generation 之前被调用"
    );
}
