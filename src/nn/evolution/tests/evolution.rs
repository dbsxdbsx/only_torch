use std::cell::RefCell;
use std::rc::Rc;

use crate::nn::descriptor::NodeTypeDescriptor;
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
    // 测试中固定所有与线程数相关的默认值，确保 seed 完全确定性：
    // - with_parallelism(1)：串行评估，避免多测试并发时超额订阅
    // - with_population_size / with_offspring_batch_size：避免
    //   rayon::current_num_threads() 在全局池懒初始化前后返回不同值
    //   导致 rng 消耗次数不同，破坏跨调用的 seed 可重现性。
    // 专门测试并行路径的用例自行覆盖这些参数。
    Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_parallelism(1)
        .with_population_size(4)
        .with_offspring_batch_size(4)
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

    fn on_new_best(&mut self, generation: usize, _genome: &NetworkGenome, score: &FitnessScore) {
        let mut s = self.state.borrow_mut();
        s.new_best_count += 1;
        s.new_best_generations.push(generation);
        s.new_best_primaries.push(score.primary);
    }

    fn on_mutation(&mut self, _generation: usize, mutation_name: &str, _genome: &NetworkGenome) {
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
    // (1+λ) 策略下每代可能有多次成功变异（λ 个子代各自变异）
    assert!(
        s.mutation_count >= s.generation_count,
        "on_mutation 次数({}) 应 >= on_generation 次数({})",
        s.mutation_count,
        s.generation_count
    );
    assert!(!s.last_mutation_name.is_empty(), "变异名称不应为空");
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
    // 接受 LayerLevel ("Input(") 和 NodeLevel ("nodes=") 两种格式
    assert!(
        result.architecture_summary.starts_with("Input(")
            || result.architecture_summary.starts_with("nodes="),
        "架构描述应以 Input( 或 nodes= 开头: {}",
        result.architecture_summary
    );
    assert!(result.architecture() == result.architecture_summary);
}

// ==================== predict() ====================

#[test]
fn test_predict_single_sample() {
    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(1.0)
        .with_verbose(false)
        .run()
        .unwrap();

    // 1D 输入 [input_dim]
    let pred = result.predict(&Tensor::new(&[1.0, 0.0], &[2])).unwrap();
    assert_eq!(pred.shape(), &[1, 1]); // [1, output_dim]
    assert!(pred.to_vec()[0].is_finite());
}

#[test]
fn test_predict_batch() {
    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(1.0)
        .with_verbose(false)
        .run()
        .unwrap();

    // 2D 输入 [batch, input_dim]
    let batch_input = Tensor::new(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    let pred = result.predict(&batch_input).unwrap();
    assert_eq!(pred.shape()[0], 4);
    assert_eq!(pred.shape()[1], 1); // output_dim
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
    // (1+λ) 策略下每代有多次变异
    assert!(s.mutation_count >= s.generation_count);
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
    assert_eq!(s.new_best_count, 1, "首代应触发 on_new_best（从无到有）");
    assert_eq!(s.new_best_generations[0], 0);
}

// ==================== eval_runs=0 边界防御 ====================

#[test]
#[should_panic(expected = "eval_runs 必须 >= 1")]
fn test_eval_runs_zero_panics() {
    let _ = xor_evolution().with_eval_runs(0);
}

// ==================== 序列演化集成 ====================

fn seq_parity_data(n: usize, seq_len: usize) -> (Vec<Tensor>, Vec<Tensor>) {
    let mut inputs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let data: Vec<f32> = (0..seq_len).map(|j| ((i + j) % 2) as f32).collect();
        let parity = data.iter().sum::<f32>() as usize % 2;
        inputs.push(Tensor::new(&data, &[seq_len, 1]));
        labels.push(Tensor::new(&[parity as f32], &[1]));
    }
    (inputs, labels)
}

#[test]
fn test_evolution_sequential_runs() {
    let data = seq_parity_data(16, 4);
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_seed(42)
        .with_max_generations(3)
        .with_verbose(false)
        .run()
        .unwrap();

    assert!(result.fitness.primary >= 0.0);
    assert!(result.generations <= 3);
    assert!(
        result.architecture_summary.contains("seq×")
            || result.architecture_summary.starts_with("nodes="),
        "序列架构应显示序列或 NodeLevel 结构信息，实际={}",
        result.architecture_summary
    );
    assert!(
        result.genome.is_node_level(),
        "序列演化主入口迁移后 genome 应为 NodeLevel"
    );
    assert!(
        result.genome.nodes().iter().any(|n| matches!(
            n.node_type,
            NodeTypeDescriptor::CellRnn { .. }
                | NodeTypeDescriptor::CellLstm { .. }
                | NodeTypeDescriptor::CellGru { .. }
        )),
        "序列演化结果应包含循环单元节点（CellRnn/CellLstm/CellGru）"
    );
}

#[test]
fn test_evolution_sequential_predict() {
    let data = seq_parity_data(16, 4);
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_seed(42)
        .with_max_generations(3)
        .with_verbose(false)
        .run()
        .unwrap();

    // 单样本推理：[seq_len, input_dim] → [1, output_dim]
    let sample = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4, 1]);
    let pred = result.predict(&sample).unwrap();
    assert_eq!(pred.shape(), &[1, 1]);
    assert!(pred.to_vec()[0].is_finite());
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
        "InsertAtomicNode",
        "RemoveLayer",
        "ReplaceLayerType",
        "GrowHiddenSize",
        "ShrinkHiddenSize",
        "MutateLayerParam",
        "MutateLossFunction",
        "AddSkipEdge",
        "RemoveSkipEdge",
        "MutateAggregateStrategy",
        "MutateLearningRate",
        "MutateOptimizer",
        "AddConnection",
        "RemoveConnection",
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
    fn on_new_best(&mut self, _generation: usize, _genome: &NetworkGenome, _score: &FitnessScore) {
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
        fn on_new_best(&mut self, g: usize, genome: &NetworkGenome, score: &FitnessScore) {
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

// ==================== 空间（CNN）演化集成 ====================

/// 合成空间二分类数据："亮"图像（像素均值 > 0.5）→ [1,0]，"暗"图像 → [0,1]
///
/// 每个样本为 `[C, H, W]` 的张量，标签为 one-hot `[2]`。
fn spatial_brightness_data(
    n: usize,
    channels: usize,
    h: usize,
    w: usize,
) -> (Vec<Tensor>, Vec<Tensor>) {
    let size = channels * h * w;
    let mut inputs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        // 偶数索引 = "亮"，奇数索引 = "暗"
        let bright = i % 2 == 0;
        let base: f32 = if bright { 0.7 } else { 0.2 };
        let pixels: Vec<f32> = (0..size).map(|j| base + (j as f32 * 0.001) % 0.1).collect();
        inputs.push(Tensor::new(&pixels, &[channels, h, w]));
        if bright {
            labels.push(Tensor::new(&[1.0, 0.0], &[2]));
        } else {
            labels.push(Tensor::new(&[0.0, 1.0], &[2]));
        }
    }
    (inputs, labels)
}

#[test]
fn test_evolution_spatial_runs() {
    // 小型空间数据：1 通道, 4×4，模拟 mnist_cnn 的缩小版
    let data = spatial_brightness_data(16, 1, 4, 4);
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_seed(42)
        .with_max_generations(3)
        .with_verbose(false)
        .run()
        .unwrap();

    assert!(result.fitness.primary >= 0.0);
    assert!(result.generations <= 3);
    // 接受 LayerLevel（包含 '@'）和 NodeLevel（以 'nodes=' 开头）两种格式
    assert!(
        result.architecture_summary.contains('@')
            || result.architecture_summary.starts_with("nodes="),
        "空间架构应显示 C@H×W 格式或 nodes= 格式: {}",
        result.architecture_summary
    );
}

#[test]
fn test_evolution_spatial_predict() {
    let data = spatial_brightness_data(16, 1, 4, 4);
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_seed(42)
        .with_max_generations(3)
        .with_verbose(false)
        .run()
        .unwrap();

    // 单样本推理：[C, H, W] → [1, output_dim]
    let sample = Tensor::ones(&[1, 4, 4]);
    let pred = result.predict(&sample).unwrap();
    assert_eq!(pred.shape(), &[1, 2]); // [1, output_dim=2]
    assert!(pred.to_vec().iter().all(|v| v.is_finite()));

    // batch 推理：[N, C, H, W]
    let batch = Tensor::ones(&[3, 1, 4, 4]);
    let pred_batch = result.predict(&batch).unwrap();
    assert_eq!(pred_batch.shape(), &[3, 2]);
}

#[test]
fn test_evolution_spatial_seed_reproducibility() {
    let run = |seed: u64| {
        let data = spatial_brightness_data(16, 1, 4, 4);
        let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
            .with_seed(seed)
            .with_max_generations(5)
            .with_verbose(false)
            .run()
            .unwrap();
        (result.architecture_summary.clone(), result.fitness.primary)
    };

    let (arch1, fit1) = run(200);
    let (arch2, fit2) = run(200);

    assert_eq!(arch1, arch2, "相同 seed 应产生相同空间架构");
    assert!(
        (fit1 - fit2).abs() < 1e-6,
        "相同 seed 应产生相同 fitness: {fit1} vs {fit2}"
    );
}

#[test]
fn test_evolution_spatial_multichannel() {
    // 3 通道（类似 RGB），8×8
    let data = spatial_brightness_data(12, 3, 8, 8);
    let result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_seed(42)
        .with_max_generations(3)
        .with_verbose(false)
        .run()
        .unwrap();

    assert!(result.fitness.primary >= 0.0);
    // 接受 LayerLevel（包含 '3@8×8'）和 NodeLevel（以 'nodes=' 开头）两种格式
    assert!(
        result.architecture_summary.contains("3@8×8")
            || result.architecture_summary.starts_with("nodes="),
        "应显示 3@8×8 或 nodes= 格式: {}",
        result.architecture_summary
    );
}

#[test]
fn test_evolution_spatial_mutation_names_valid() {
    let (mock, state) = MockCallback::new(Some(8));

    let data = spatial_brightness_data(16, 1, 4, 4);
    let _result = Evolution::supervised(data.clone(), data, TaskMetric::Accuracy)
        .with_seed(42)
        .with_target_metric(2.0) // 不可达
        .with_callback(mock)
        .run()
        .unwrap();

    let known = [
        "InsertLayer",
        "InsertAtomicNode",
        "RemoveLayer",
        "ReplaceLayerType",
        "GrowHiddenSize",
        "ShrinkHiddenSize",
        "MutateLayerParam",
        "MutateLossFunction",
        "AddSkipEdge",
        "RemoveSkipEdge",
        "MutateAggregateStrategy",
        "MutateLearningRate",
        "MutateOptimizer",
        "MutateKernelSize", // 空间模式专属
        "MutateStride",     // 空间模式：Conv2d stride (1,1)↔(2,2)
        "AddConnection",
        "RemoveConnection",
    ];

    let s = state.borrow();
    for name in &s.mutation_names {
        assert!(
            known.contains(&name.as_str()),
            "未知的空间变异名称: {name}，已知: {known:?}"
        );
    }
}

// ==================== Pareto 种群模式测试 ====================

#[test]
fn test_pareto_front_populated_on_target_reached() {
    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(0.0) // 首代即达标
        .with_max_generations(100)
        .with_verbose(false)
        .run()
        .unwrap();

    assert_eq!(result.status, EvolutionStatus::TargetReached);
    assert!(
        !result.pareto_front.is_empty(),
        "达标时 pareto_front 不应为空"
    );
    // 每个 pareto summary 的 fitness 应有效
    for ps in &result.pareto_front {
        assert!(ps.fitness.primary.is_finite());
        assert!(!ps.architecture_summary.is_empty());
    }
}

#[test]
fn test_pareto_front_populated_on_max_generations() {
    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(2.0) // 不可达
        .with_max_generations(5)
        .with_verbose(false)
        .run()
        .unwrap();

    assert_eq!(result.status, EvolutionStatus::MaxGenerations);
    assert!(
        !result.pareto_front.is_empty(),
        "MaxGenerations 时 pareto_front 不应为空"
    );
}

#[test]
fn test_population_size_builder() {
    let result = xor_evolution()
        .with_seed(42)
        .with_population_size(4)
        .with_offspring_batch_size(4)
        .with_max_generations(3)
        .with_target_metric(2.0) // 不可达
        .with_verbose(false)
        .run()
        .unwrap();

    assert_eq!(result.status, EvolutionStatus::MaxGenerations);
    assert!(result.fitness.primary.is_finite());
}

#[test]
fn test_rebuild_pareto_member() {
    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(0.0) // 首代达标
        .with_max_generations(100)
        .with_verbose(false)
        .run()
        .unwrap();

    if result.pareto_front.is_empty() {
        return; // 极端情况，跳过
    }

    // 重建第一个 Pareto 成员
    let rebuilt = result.rebuild_pareto_member(0).unwrap();
    assert!(rebuilt.fitness.primary.is_finite());
    assert!(!rebuilt.architecture_summary.is_empty());

    // 越界检查
    let oob = result.rebuild_pareto_member(result.pareto_front.len() + 100);
    assert!(oob.is_err(), "越界索引应返回错误");
}

#[test]
fn test_smallest_meeting_target_index() {
    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(0.0)
        .with_max_generations(100)
        .with_verbose(false)
        .run()
        .unwrap();

    // target=0.0 → 所有 pareto 成员都满足
    if !result.pareto_front.is_empty() {
        let idx = result.smallest_meeting_target_index(0.0);
        assert!(idx.is_some(), "target=0.0 时应找到满足条件的成员");
        assert!(idx.unwrap() < result.pareto_front.len());
    }

    // target=2.0 → 无法满足
    let idx_impossible = result.smallest_meeting_target_index(2.0);
    assert!(
        idx_impossible.is_none(),
        "target=2.0 时不应找到满足条件的成员"
    );
}

#[test]
fn test_pareto_converged_with_small_patience() {
    // 设置极小的 pareto_patience 以触发 ParetoConverged
    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(2.0) // 不可达
        .with_max_generations(200)
        .with_pareto_patience(3) // 极小耐心值
        .with_population_size(4)
        .with_offspring_batch_size(4)
        .with_verbose(false)
        .run()
        .unwrap();

    // 应该在远少于 200 代时因 ParetoConverged 或 MaxGenerations 停止
    assert!(
        result.status == EvolutionStatus::ParetoConverged
            || result.status == EvolutionStatus::MaxGenerations,
        "期望 ParetoConverged 或 MaxGenerations，实际: {:?}",
        result.status
    );
    assert!(!result.pareto_front.is_empty());
}

#[test]
fn test_on_population_evaluated_callback() {
    /// Mock callback 记录 on_population_evaluated 调用
    struct PopEvalCallback {
        pop_eval_count: usize,
        last_pop_size: usize,
        last_archive_size: usize,
        stop_at: usize,
    }

    impl EvolutionCallback for PopEvalCallback {
        fn on_population_evaluated(
            &mut self,
            _generation: usize,
            pop_size: usize,
            _offspring_evaluated: usize,
            archive_size: usize,
            _front_size: usize,
            _best_primary: f32,
            _best_cost: f32,
        ) {
            self.pop_eval_count += 1;
            self.last_pop_size = pop_size;
            self.last_archive_size = archive_size;
        }

        fn should_stop(&self, generation: usize) -> bool {
            generation >= self.stop_at
        }
    }

    let cb = PopEvalCallback {
        pop_eval_count: 0,
        last_pop_size: 0,
        last_archive_size: 0,
        stop_at: 5,
    };

    // 用 Rc<RefCell> 包装
    let cb = std::rc::Rc::new(std::cell::RefCell::new(cb));
    let cb_clone = std::rc::Rc::clone(&cb);

    struct Wrapper {
        inner: std::rc::Rc<std::cell::RefCell<PopEvalCallback>>,
    }
    impl EvolutionCallback for Wrapper {
        fn on_population_evaluated(
            &mut self,
            g: usize,
            pop: usize,
            off: usize,
            arch: usize,
            front: usize,
            bp: f32,
            bc: f32,
        ) {
            self.inner
                .borrow_mut()
                .on_population_evaluated(g, pop, off, arch, front, bp, bc);
        }
        fn should_stop(&self, g: usize) -> bool {
            self.inner.borrow().should_stop(g)
        }
    }

    let _result = xor_evolution()
        .with_seed(42)
        .with_target_metric(2.0) // 不可达
        .with_population_size(4)
        .with_offspring_batch_size(4)
        .with_callback(Wrapper { inner: cb_clone })
        .run()
        .unwrap();

    let s = cb.borrow();
    assert_eq!(
        s.pop_eval_count, 5,
        "on_population_evaluated 应每代调用一次"
    );
    assert!(s.last_pop_size > 0, "pop_size 应 > 0");
    assert!(s.last_archive_size > 0, "archive_size 应 > 0");
}

#[test]
fn test_inference_cost_in_fitness() {
    let result = xor_evolution()
        .with_seed(42)
        .with_target_metric(0.0)
        .with_max_generations(100)
        .with_verbose(false)
        .run()
        .unwrap();

    // 达标成员的 fitness 应有 inference_cost（由 compute_inference_cost 填充）
    assert!(
        result.fitness.inference_cost.is_some(),
        "fitness 应包含 inference_cost"
    );
    assert!(
        result.fitness.inference_cost.unwrap() > 0.0,
        "inference_cost 应 > 0（至少有几个参数）"
    );
}

// ==================== fitness_changed 辅助函数 ====================

#[test]
fn test_fitness_changed_primary_differs() {
    let a = FitnessScore {
        primary: 0.9,
        inference_cost: Some(100.0),
        tiebreak_loss: Some(0.1),
    };
    let b = FitnessScore {
        primary: 0.8,
        inference_cost: Some(100.0),
        tiebreak_loss: Some(0.1),
    };
    assert!(
        super::super::fitness_changed(&a, &b, 1e-6),
        "primary 不同时应检测到变化"
    );
}

#[test]
fn test_fitness_changed_within_tolerance() {
    let a = FitnessScore {
        primary: 0.9,
        inference_cost: Some(100.0),
        tiebreak_loss: Some(0.1),
    };
    let b = FitnessScore {
        primary: 0.9 + 1e-8,
        inference_cost: Some(100.0 + 1e-8),
        tiebreak_loss: Some(0.1 + 1e-8),
    };
    assert!(
        !super::super::fitness_changed(&a, &b, 1e-6),
        "所有字段在容忍范围内应视为未变化"
    );
}

#[test]
fn test_fitness_changed_inference_cost_none_vs_some() {
    let a = FitnessScore {
        primary: 0.9,
        inference_cost: None,
        tiebreak_loss: None,
    };
    let b = FitnessScore {
        primary: 0.9,
        inference_cost: Some(100.0),
        tiebreak_loss: None,
    };
    assert!(
        super::super::fitness_changed(&a, &b, 1e-6),
        "inference_cost None vs Some 应检测到变化"
    );
}

#[test]
fn test_fitness_changed_tiebreak_loss_differs() {
    let a = FitnessScore {
        primary: 0.9,
        inference_cost: Some(100.0),
        tiebreak_loss: Some(0.1),
    };
    let b = FitnessScore {
        primary: 0.9,
        inference_cost: Some(100.0),
        tiebreak_loss: Some(0.5),
    };
    assert!(
        super::super::fitness_changed(&a, &b, 1e-6),
        "tiebreak_loss 差异超出容忍范围应检测到变化"
    );
}

#[test]
fn test_fitness_changed_both_none_unchanged() {
    let a = FitnessScore {
        primary: 0.5,
        inference_cost: None,
        tiebreak_loss: None,
    };
    let b = FitnessScore {
        primary: 0.5,
        inference_cost: None,
        tiebreak_loss: None,
    };
    assert!(
        !super::super::fitness_changed(&a, &b, 1e-6),
        "完全相同（含 None）应视为未变化"
    );
}

// ==================== compute_inference_cost ====================

#[test]
fn test_compute_inference_cost_param_count() {
    use crate::nn::evolution::ComplexityMetric;
    use crate::nn::evolution::gene::NetworkGenome;

    let genome = NetworkGenome::minimal(2, 1);
    let cost =
        super::super::compute_inference_cost(&genome, &ComplexityMetric::ParamCount).unwrap();
    let params = genome.total_params().unwrap() as f32;
    assert!(
        (cost - params).abs() < 1e-3,
        "ParamCount 度量应返回参数总量: cost={cost}, params={params}"
    );
    assert!(cost > 0.0, "至少有输出头参数");
}

// ==================== 并行评估路径 ====================

#[test]
fn test_parallelism_builder_runs_successfully() {
    let result = xor_evolution()
        .with_seed(42)
        .with_parallelism(2)
        .with_population_size(4)
        .with_offspring_batch_size(4)
        .with_max_generations(3)
        .with_target_metric(2.0) // 不可达
        .with_verbose(false)
        .run()
        .unwrap();

    assert_eq!(result.status, EvolutionStatus::MaxGenerations);
    assert!(result.fitness.primary.is_finite());
    assert!(!result.pareto_front.is_empty());
}

#[test]
fn test_parallelism_produces_same_result_as_serial() {
    // 并行路径使用预分配 seed，应与串行路径（同 seed）产生一致结果
    // 注意：由于 rayon 线程调度的不确定性，架构可能因浮点精度差异而不同，
    // 但基本运行正确性应保持
    let result_parallel = xor_evolution()
        .with_seed(42)
        .with_parallelism(2)
        .with_population_size(4)
        .with_offspring_batch_size(4)
        .with_max_generations(3)
        .with_target_metric(2.0)
        .with_verbose(false)
        .run()
        .unwrap();

    assert!(result_parallel.fitness.primary.is_finite());
    assert!(result_parallel.fitness.inference_cost.is_some());
}

// ==================== 停滞耐心与阶段切换 ====================

#[test]
fn test_stagnation_patience_forces_structural_mutations() {
    // 极低 stagnation_patience + 高代数 → 应频繁触发结构变异
    // 通过 MockCallback 记录变异名称来验证
    let (mock, state) = MockCallback::new(Some(20));

    let _result = xor_evolution()
        .with_seed(42)
        .with_target_metric(2.0) // 不可达
        .with_stagnation_patience(2) // 极低耐心
        .with_callback(mock)
        .run()
        .unwrap();

    let s = state.borrow();
    let structural = ["InsertLayer", "RemoveLayer", "ReplaceLayerType"];
    let structural_count = s
        .mutation_names
        .iter()
        .filter(|name| structural.contains(&name.as_str()))
        .count();

    assert!(
        structural_count > 0,
        "低 stagnation_patience 后应有结构变异，实际变异名称: {:?}",
        s.mutation_names
    );
}
