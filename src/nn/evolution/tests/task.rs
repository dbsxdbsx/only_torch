use crate::nn::evolution::builder::BuildResult;
use crate::nn::evolution::convergence::*;
use crate::nn::evolution::gene::*;
use crate::nn::evolution::task::*;
use crate::tensor::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;

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

fn xor_task() -> SupervisedTask {
    let data = xor_data();
    SupervisedTask::new(data.clone(), data, TaskMetric::Accuracy).unwrap()
}

fn tiny_segmentation_data() -> (Vec<Tensor>, Vec<Tensor>) {
    (
        vec![
            Tensor::new(&[0.0, 0.2, 0.8, 1.0], &[1, 2, 2]),
            Tensor::new(&[1.0, 0.8, 0.2, 0.0], &[1, 2, 2]),
        ],
        vec![
            Tensor::new(
                &[
                    1.0, 1.0, 0.0, 0.0, // class 0
                    0.0, 0.0, 1.0, 1.0, // class 1
                ],
                &[2, 2, 2],
            ),
            Tensor::new(
                &[
                    0.0, 0.0, 1.0, 1.0, // class 0
                    1.0, 1.0, 0.0, 0.0, // class 1
                ],
                &[2, 2, 2],
            ),
        ],
    )
}

fn genome_with_hidden(input_dim: usize, output_dim: usize) -> NetworkGenome {
    let mut genome = NetworkGenome::minimal(input_dim, output_dim);
    let inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Linear { out_features: 8 },
            enabled: true,
        },
    );
    let inn_act = genome.next_innovation_number();
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: inn_act,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    genome
}

fn short_convergence() -> ConvergenceConfig {
    ConvergenceConfig {
        budget: TrainingBudget::UntilConverged,
        patience: 5,
        loss_tolerance: 1e-4,
        grad_tolerance: 1e-5,
        max_epochs: 50,
    }
}

fn build_and_restore(genome: &NetworkGenome, rng: &mut StdRng) -> BuildResult {
    let build = genome.build(rng).unwrap();
    genome.restore_weights(&build).unwrap();
    build
}

// ==================== 构造与数据堆叠 ====================

#[test]
fn test_supervised_task_construction_ok() {
    let task = xor_task();
    assert_eq!(*task.metric(), TaskMetric::Accuracy);

    // 验证内部 stack 正确：通过 evaluate 调用间接验证数据形状
    // （evaluate 内部 set_value 时若形状不兼容会报错）
    let genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);
    assert!(task.evaluate(&genome, &build, &mut rng).is_ok());
}

#[test]
fn test_supervised_task_metric_accessor() {
    let task = xor_task();
    assert_eq!(*task.metric(), TaskMetric::Accuracy);
}

#[test]
fn test_empty_train_data_returns_err() {
    let result = SupervisedTask::new((vec![], vec![]), xor_data(), TaskMetric::Accuracy);
    match result {
        Err(e) => assert!(
            e.to_string().contains("训练输入不能为空"),
            "unexpected error: {e}"
        ),
        Ok(_) => panic!("应返回 Err"),
    }
}

#[test]
fn test_train_data_count_mismatch_returns_err() {
    let (inputs, mut labels) = xor_data();
    labels.truncate(2);
    let result = SupervisedTask::new((inputs, labels), xor_data(), TaskMetric::Accuracy);
    match result {
        Err(e) => assert!(
            e.to_string().contains("数量不匹配"),
            "unexpected error: {e}"
        ),
        Ok(_) => panic!("应返回 Err"),
    }
}

#[test]
fn test_empty_test_data_returns_err() {
    let result = SupervisedTask::new(xor_data(), (vec![], vec![]), TaskMetric::Accuracy);
    match result {
        Err(e) => assert!(
            e.to_string().contains("测试输入不能为空"),
            "unexpected error: {e}"
        ),
        Ok(_) => panic!("应返回 Err"),
    }
}

#[test]
fn test_test_data_count_mismatch_returns_err() {
    let (inputs, mut labels) = xor_data();
    labels.truncate(2);
    let result = SupervisedTask::new(xor_data(), (inputs, labels), TaskMetric::Accuracy);
    match result {
        Err(e) => assert!(
            e.to_string().contains("数量不匹配"),
            "unexpected error: {e}"
        ),
        Ok(_) => panic!("应返回 Err"),
    }
}

// ==================== 基本训练 ====================

#[test]
fn test_basic_training_returns_finite_loss() {
    let task = xor_task();
    let genome = genome_with_hidden(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let final_loss = task
        .train(&genome, &build, &short_convergence(), &mut rng)
        .unwrap();

    assert!(
        final_loss.is_finite(),
        "训练后 loss 应为有限值: {final_loss}"
    );
}

#[test]
fn test_training_uses_all_parameters() {
    let task = xor_task();
    let genome = genome_with_hidden(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    // 记录训练前的参数
    let params_before: Vec<Tensor> = build
        .all_parameters()
        .iter()
        .map(|p| p.value().unwrap().unwrap())
        .collect();

    task.train(&genome, &build, &short_convergence(), &mut rng)
        .unwrap();

    // 训练后参数应发生变化
    let params_after: Vec<Tensor> = build
        .all_parameters()
        .iter()
        .map(|p| p.value().unwrap().unwrap())
        .collect();

    let any_changed = params_before
        .iter()
        .zip(params_after.iter())
        .any(|(a, b)| a != b);
    assert!(any_changed, "训练后至少应有一个参数发生变化");
}

// ==================== 评估 ====================

#[test]
fn test_evaluate_returns_valid_fitness_score() {
    let task = xor_task();
    let genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let score = task.evaluate(&genome, &build, &mut rng).unwrap();

    assert!(
        score.primary >= 0.0 && score.primary <= 1.0,
        "Accuracy 应在 [0, 1] 范围: {}",
        score.primary
    );
    assert!(score.inference_cost.is_none());
}

#[test]
fn test_evaluate_after_training() {
    let task = xor_task();
    let genome = genome_with_hidden(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    task.train(&genome, &build, &short_convergence(), &mut rng)
        .unwrap();

    let score = task.evaluate(&genome, &build, &mut rng).unwrap();

    assert!(score.primary >= 0.0 && score.primary <= 1.0);
    assert!(score.tiebreak_loss.is_some());
    assert!(
        score.tiebreak_loss.unwrap().is_finite(),
        "tiebreak_loss 应为有限值"
    );
}

// ==================== 二值解码 ====================

#[test]
fn test_binary_decode_bce_threshold() {
    // BCE: logit >= 0.0 → 1
    let logits = Tensor::new(&[-1.0, -0.1, 0.0, 0.1, 1.0], &[5, 1]);
    let labels = binary_decode(&logits, &LossType::BCE);
    assert_eq!(labels, vec![0, 0, 1, 1, 1]);
}

#[test]
fn test_binary_decode_mse_threshold() {
    // MSE: pred >= 0.5 → 1
    let preds = Tensor::new(&[0.0, 0.49, 0.5, 0.51, 1.0], &[5, 1]);
    let labels = binary_decode(&preds, &LossType::MSE);
    assert_eq!(labels, vec![0, 0, 1, 1, 1]);
}

#[test]
fn test_binary_decode_cross_entropy_threshold() {
    // CrossEntropy 与 MSE 共享 0.5 阈值
    let preds = Tensor::new(&[0.3, 0.5, 0.7], &[3, 1]);
    let labels = binary_decode(&preds, &LossType::CrossEntropy);
    assert_eq!(labels, vec![0, 1, 1]);
}

#[test]
fn test_binary_decode_labels_use_mse_rule() {
    // 标签侧统一用 >= 0.5（标签值域为 {0, 1}）
    let labels_tensor = Tensor::new(&[0.0, 1.0, 0.0, 1.0], &[4, 1]);
    let decoded = binary_decode(&labels_tensor, &LossType::MSE);
    assert_eq!(decoded, vec![0, 1, 0, 1]);
}

// ==================== Tiebreak 语义 ====================

#[test]
fn test_tiebreak_present_for_accuracy() {
    let task = xor_task();
    let genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let score = task.evaluate(&genome, &build, &mut rng).unwrap();

    assert!(
        score.tiebreak_loss.is_some(),
        "Accuracy（离散指标）应有 tiebreak_loss"
    );
}

#[test]
fn test_tiebreak_absent_for_r2() {
    // R2 是连续指标，不需要 tiebreak
    let train = (
        vec![
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[2.0], &[1]),
            Tensor::new(&[3.0], &[1]),
        ],
        vec![
            Tensor::new(&[2.0], &[1]),
            Tensor::new(&[4.0], &[1]),
            Tensor::new(&[6.0], &[1]),
        ],
    );
    let test = train.clone();
    let task = SupervisedTask::new(train, test, TaskMetric::R2).unwrap();

    let genome = NetworkGenome::minimal(1, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let score = task.evaluate(&genome, &build, &mut rng).unwrap();

    assert!(
        score.tiebreak_loss.is_none(),
        "R2（连续指标）不应有 tiebreak_loss"
    );
}

#[test]
fn test_tiebreak_present_for_multilabel_accuracy() {
    let train = (
        vec![
            Tensor::new(&[1.0, 0.0], &[2]),
            Tensor::new(&[0.0, 1.0], &[2]),
        ],
        vec![
            Tensor::new(&[1.0, 0.0], &[2]),
            Tensor::new(&[0.0, 1.0], &[2]),
        ],
    );
    let test = train.clone();
    let task = SupervisedTask::new(train, test, TaskMetric::MultiLabelAccuracy).unwrap();

    let genome = NetworkGenome::minimal(2, 2);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let score = task.evaluate(&genome, &build, &mut rng).unwrap();

    assert!(
        score.tiebreak_loss.is_some(),
        "MultiLabelAccuracy（离散指标）应有 tiebreak_loss"
    );
}

// ==================== Loss 函数选择 ====================

#[test]
fn test_loss_auto_infer_bce_for_binary() {
    // Accuracy + output_dim=1 → BCE
    let genome = NetworkGenome::minimal(2, 1);
    let loss = genome.effective_loss(&TaskMetric::Accuracy);
    assert_eq!(loss, LossType::BCE);
}

#[test]
fn test_loss_auto_infer_ce_for_multiclass() {
    // Accuracy + output_dim>1 → CrossEntropy
    let genome = NetworkGenome::minimal(2, 3);
    let loss = genome.effective_loss(&TaskMetric::Accuracy);
    assert_eq!(loss, LossType::CrossEntropy);
}

#[test]
fn test_loss_auto_infer_mse_for_r2() {
    let genome = NetworkGenome::minimal(2, 1);
    let loss = genome.effective_loss(&TaskMetric::R2);
    assert_eq!(loss, LossType::MSE);
}

#[test]
fn test_loss_auto_infer_bce_for_segmentation_metrics() {
    let genome = NetworkGenome::minimal_spatial_segmentation(1, 2, (2, 2));

    assert_eq!(genome.effective_loss(&TaskMetric::BinaryIoU), LossType::BCE);
    assert_eq!(genome.effective_loss(&TaskMetric::MeanIoU), LossType::BCE);
}

#[test]
fn test_loss_explicit_override() {
    let mut genome = NetworkGenome::minimal(2, 1);
    genome.training_config.loss_override = Some(LossType::MSE);
    // 即使 Accuracy + dim=1 会推断 BCE，显式设置 MSE 优先
    let loss = genome.effective_loss(&TaskMetric::Accuracy);
    assert_eq!(loss, LossType::MSE);
}

// ==================== Batch size 与训练配置约束 ====================

#[test]
fn test_genome_batch_size_used_when_set() {
    // genome.training_config.batch_size 不再 panic，
    // 而是作为最高优先级的 batch_size 使用
    let task = xor_task();
    let mut genome = genome_with_hidden(2, 1);
    genome.training_config.batch_size = Some(2);

    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let loss = task
        .train(&genome, &build, &short_convergence(), &mut rng)
        .unwrap();
    assert!(loss.is_finite(), "genome batch_size=2 应正常训练");
}

#[test]
#[should_panic(expected = "weight_decay")]
fn test_weight_decay_panics() {
    let task = xor_task();
    let mut genome = NetworkGenome::minimal(2, 1);
    genome.training_config.weight_decay = 0.01;

    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let _ = task.train(&genome, &build, &short_convergence(), &mut rng);
}

// ==================== compute_grad_norm ====================

#[test]
fn test_compute_grad_norm_after_training() {
    let task = xor_task();
    let genome = genome_with_hidden(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    // 训练一步以产生梯度
    let convergence = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(1),
        ..Default::default()
    };
    task.train(&genome, &build, &convergence, &mut rng).unwrap();

    let params = build.all_parameters();
    let grad_norm = compute_grad_norm(&params).unwrap();

    assert!(grad_norm.is_finite(), "梯度范数应为有限值: {grad_norm}");
    assert!(grad_norm >= 0.0, "梯度范数应非负: {grad_norm}");
}

#[test]
fn test_compute_grad_norm_no_grad_is_zero() {
    // 未训练（无梯度）时，梯度范数应为 0.0
    let genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let params = build.all_parameters();
    let grad_norm = compute_grad_norm(&params).unwrap();

    assert_eq!(grad_norm, 0.0, "未训练时梯度范数应为 0.0");
}

// ==================== primary 纯净性 ====================

#[test]
fn test_primary_is_pure_metric_value() {
    let task = xor_task();
    let genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let score = task.evaluate(&genome, &build, &mut rng).unwrap();

    // primary 应为精确的 accuracy 值（4 个样本：0/4, 1/4, 2/4, 3/4, 4/4 之一）
    let valid_values = [0.0, 0.25, 0.5, 0.75, 1.0];
    assert!(
        valid_values
            .iter()
            .any(|&v| (score.primary - v).abs() < 1e-6),
        "primary 应为精确的 accuracy 值（4 个样本），实际: {}",
        score.primary
    );
}

// ==================== 训练 + 评估集成 ====================

#[test]
fn test_train_then_evaluate_integration() {
    let task = xor_task();
    let genome = genome_with_hidden(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let convergence = ConvergenceConfig {
        max_epochs: 200,
        patience: 10,
        ..Default::default()
    };

    let loss = task.train(&genome, &build, &convergence, &mut rng).unwrap();
    let score = task.evaluate(&genome, &build, &mut rng).unwrap();

    assert!(loss.is_finite());
    assert!(score.primary >= 0.0 && score.primary <= 1.0);
    assert!(score.tiebreak_loss.unwrap().is_finite());

    println!(
        "集成测试: loss={loss:.4}, accuracy={:.1}%, tiebreak_loss={:.4}",
        score.primary * 100.0,
        score.tiebreak_loss.unwrap()
    );
}

// ==================== MultiLabelAccuracy BCE logit 阈值回归 ====================

#[test]
fn test_multilabel_accuracy_bce_logit_threshold() {
    // 构造已知 logit 输出，验证 BCE 路径下阈值为 0.0 而非 0.5
    //
    // logit 0.3 在 sigmoid 后 ≈ 0.574 > 0.5，属于正例。
    // 如果错误地使用 0.5 阈值（概率语义），0.3 < 0.5 会被误判为负例。
    // 正确行为：BCE 下 logit >= 0.0 → 正例。
    let predictions = Tensor::new(
        &[
            0.3, -0.2, // 样本1: logit [0.3, -0.2] → 正确判定 [1, 0]
            -0.1, 0.8, // 样本2: logit [-0.1, 0.8] → 正确判定 [0, 1]
        ],
        &[2, 2],
    );
    let labels = Tensor::new(
        &[
            1.0, 0.0, // 样本1: [1, 0]
            0.0, 1.0, // 样本2: [0, 1]
        ],
        &[2, 2],
    );

    // BCE 路径：阈值 0.0，所有 logit 判定正确 → accuracy = 1.0
    let primary_bce = compute_primary_metric(
        &TaskMetric::MultiLabelAccuracy,
        &predictions,
        &labels,
        2,
        &LossType::BCE,
    );
    assert!(
        (primary_bce - 1.0).abs() < 1e-6,
        "BCE logit 阈值 0.0 下应全部正确，实际 accuracy = {primary_bce}"
    );

    // 如果错误地用 0.5 阈值，logit 0.3 < 0.5 会被误判 → accuracy < 1.0
    // 这个断言确保 bug 不会回归
}

#[test]
fn test_multilabel_accuracy_bce_logit_boundary() {
    // 边界值测试：logit 恰好在 0.0
    let predictions = Tensor::new(&[0.0, -0.0001], &[1, 2]);
    let labels = Tensor::new(&[1.0, 0.0], &[1, 2]);

    let primary = compute_primary_metric(
        &TaskMetric::MultiLabelAccuracy,
        &predictions,
        &labels,
        2,
        &LossType::BCE,
    );
    assert!(
        (primary - 1.0).abs() < 1e-6,
        "logit=0.0 应判正，logit<0 应判负，实际 accuracy = {primary}"
    );
}

// ==================== loss_override 贯穿 train/evaluate ====================

#[test]
fn test_loss_override_flows_through_train_and_evaluate() {
    // 验证显式 loss_override 真的被 train()/evaluate() 使用，
    // 而不仅是 effective_loss() 辅助函数返回正确值。
    //
    // 策略：用 MSE override（而非默认 BCE）训练 XOR，
    // 确认能正常训练且评估给出合理分数。
    let task = xor_task();
    let mut genome = genome_with_hidden(2, 1);
    genome.training_config.loss_override = Some(LossType::MSE);

    assert_eq!(
        genome.effective_loss(&TaskMetric::Accuracy),
        LossType::MSE,
        "override 应优先于自动推断"
    );

    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let loss = task
        .train(&genome, &build, &short_convergence(), &mut rng)
        .unwrap();
    assert!(loss.is_finite(), "MSE override 训练应产生有限 loss");

    let score = task.evaluate(&genome, &build, &mut rng).unwrap();
    assert!(
        score.primary >= 0.0 && score.primary <= 1.0,
        "MSE override 评估应产生合法 accuracy: {}",
        score.primary
    );
}

// ==================== 同 primary 不同 tiebreak_loss ====================

#[test]
fn test_same_primary_different_tiebreak_loss() {
    // 两个不同参数初始化的网络在同一数据上可能得到相同 accuracy
    // 但 tiebreak_loss 应不同（不同权重 → 不同 loss）。
    // 这里用最小网络（未训练）验证：primary 相同时 tiebreak_loss 字段存在且可比较。
    let task = xor_task();

    let genome = NetworkGenome::minimal(2, 1);
    let mut rng1 = StdRng::seed_from_u64(100);
    let build1 = build_and_restore(&genome, &mut rng1);
    let score1 = task.evaluate(&genome, &build1, &mut rng1).unwrap();

    let mut rng2 = StdRng::seed_from_u64(200);
    let build2 = build_and_restore(&genome, &mut rng2);
    let score2 = task.evaluate(&genome, &build2, &mut rng2).unwrap();

    // 两个未训练的最小网络，在 4 样本 XOR 上 accuracy 很可能相同（大概率 0.5）
    // 但即使 primary 不同，核心断言是 tiebreak_loss 存在且有限
    assert!(score1.tiebreak_loss.is_some());
    assert!(score2.tiebreak_loss.is_some());
    assert!(score1.tiebreak_loss.unwrap().is_finite());
    assert!(score2.tiebreak_loss.unwrap().is_finite());

    // 不同 seed 的参数初始化 → tiebreak_loss 不同
    if (score1.primary - score2.primary).abs() < 1e-6 {
        assert!(
            (score1.tiebreak_loss.unwrap() - score2.tiebreak_loss.unwrap()).abs() > 1e-10,
            "同 primary 下，不同初始化的 tiebreak_loss 应有差异"
        );
    }
}

// ==================== 二分类 [batch,1] 防 argmax 回归 ====================

#[test]
fn test_binary_classification_does_not_use_argmax() {
    // 核心回归测试：output_dim==1 时 accuracy 不应退化到 argmax 行为。
    //
    // argmax 在 [batch, 1] 上永远返回 0，导致假性 100% 或假性 0%。
    // 正确行为：显式二值解码（BCE: logit>=0 → 1, MSE: pred>=0.5 → 1）。
    //
    // 构造场景：预测 logits 全为正（模型"认为"全是类别 1），
    // 标签为 [0, 1, 1, 0]，正确 accuracy 应为 50%。
    let predictions = Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[4, 1]);
    let labels = Tensor::new(&[0.0, 1.0, 1.0, 0.0], &[4, 1]);

    let primary = compute_primary_metric(
        &TaskMetric::Accuracy,
        &predictions,
        &labels,
        1,
        &LossType::BCE,
    );

    // BCE: logit >= 0.0 → 类别 1，所以所有预测都是 1
    // 标签: [0, 1, 1, 0]，命中 2/4 = 0.5
    assert!(
        (primary - 0.5).abs() < 1e-6,
        "output_dim=1 下 accuracy 应为 0.5（2/4 命中），实际: {primary}。\
         如果为 1.0 或 0.0，说明 argmax 回归了"
    );
}

#[test]
fn test_binary_classification_all_negative_logits() {
    // 所有 logit < 0 → BCE 解码全为类别 0
    let predictions = Tensor::new(&[-1.0, -0.5, -2.0], &[3, 1]);
    let labels = Tensor::new(&[0.0, 0.0, 1.0], &[3, 1]);

    let primary = compute_primary_metric(
        &TaskMetric::Accuracy,
        &predictions,
        &labels,
        1,
        &LossType::BCE,
    );

    // 预测全 0，标签 [0, 0, 1] → 命中 2/3
    assert!(
        (primary - 2.0 / 3.0).abs() < 1e-6,
        "全负 logit 下 accuracy 应为 2/3，实际: {primary}"
    );
}

#[test]
fn test_segmentation_primary_metrics() {
    let binary_predictions = Tensor::new(&[0.9, 0.8, 0.2, 0.1], &[1, 1, 2, 2]);
    let binary_labels = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[1, 1, 2, 2]);
    let semantic_predictions = Tensor::new(
        &[
            0.9, 0.2, 0.8, 0.4, // class 0
            0.1, 0.8, 0.2, 0.6, // class 1
        ],
        &[1, 2, 2, 2],
    );
    let semantic_labels = Tensor::new(
        &[
            1.0, 0.0, 1.0, 1.0, // class 0
            0.0, 1.0, 0.0, 0.0, // class 1
        ],
        &[1, 2, 2, 2],
    );

    let binary_iou = compute_primary_metric(
        &TaskMetric::BinaryIoU,
        &binary_predictions,
        &binary_labels,
        1,
        &LossType::MSE,
    );
    let mean_iou = compute_primary_metric(
        &TaskMetric::MeanIoU,
        &semantic_predictions,
        &semantic_labels,
        2,
        &LossType::BCE,
    );

    assert!((binary_iou - 1.0 / 3.0).abs() < 1e-6);
    assert!((mean_iou - (2.0 / 3.0 + 0.5) / 2.0).abs() < 1e-6);
}

#[test]
fn test_supervised_segmentation_task_evaluates_dense_output() {
    let data = tiny_segmentation_data();
    let task = SupervisedTask::new(data.clone(), data, TaskMetric::MeanIoU).unwrap();
    let mut genome = NetworkGenome::minimal_spatial_segmentation(1, 2, (2, 2));
    genome.migrate_to_node_level().unwrap();
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let score = task.evaluate(&genome, &build, &mut rng).unwrap();

    assert!(score.primary >= 0.0 && score.primary <= 1.0);
    assert!(score.tiebreak_loss.is_some());
    assert!(score.report.get(ReportMetric::MeanIoU).is_some());
    assert!(score.report.get(ReportMetric::PixelAccuracy).is_some());
}

// ==================== MetricReport ====================

#[test]
fn test_default_report_metrics_for_accuracy_task() {
    let task = xor_task();
    assert_eq!(
        task.report_metrics(),
        &[
            ReportMetric::Accuracy,
            ReportMetric::Precision,
            ReportMetric::Recall,
            ReportMetric::F1,
        ]
    );
}

#[test]
fn test_default_report_metrics_for_segmentation_tasks() {
    let data = tiny_segmentation_data();
    let binary_task =
        SupervisedTask::new(data.clone(), data.clone(), TaskMetric::BinaryIoU).unwrap();
    let mean_iou_task = SupervisedTask::new(data.clone(), data, TaskMetric::MeanIoU).unwrap();

    assert_eq!(
        binary_task.report_metrics(),
        &[
            ReportMetric::PixelAccuracy,
            ReportMetric::BinaryIoU,
            ReportMetric::Dice,
        ]
    );
    assert_eq!(
        mean_iou_task.report_metrics(),
        &[ReportMetric::PixelAccuracy, ReportMetric::MeanIoU]
    );
}

#[test]
fn test_report_metric_config_dedup_and_ignore_incompatible() {
    let data = xor_data();
    let task = SupervisedTask::new(data.clone(), data, TaskMetric::Accuracy)
        .unwrap()
        .with_report_metrics([ReportMetric::F1, ReportMetric::R2, ReportMetric::Accuracy]);

    assert_eq!(
        task.report_metrics(),
        &[
            ReportMetric::Accuracy,
            ReportMetric::Precision,
            ReportMetric::Recall,
            ReportMetric::F1,
        ],
        "重复指标应去重，不兼容当前任务的 R2 应被忽略"
    );
}

#[test]
fn test_compute_metric_report_binary_classification() {
    let predictions = Tensor::new(&[0.5, -0.5, 0.5, -0.5], &[4, 1]);
    let labels = Tensor::new(&[0.0, 1.0, 1.0, 0.0], &[4, 1]);
    let report = compute_metric_report(
        &TaskMetric::Accuracy,
        &[
            ReportMetric::Accuracy,
            ReportMetric::Precision,
            ReportMetric::Recall,
            ReportMetric::F1,
        ],
        &predictions,
        &labels,
        1,
        &LossType::BCE,
    );

    assert!((report.value(ReportMetric::Accuracy).unwrap() - 0.5).abs() < 1e-6);
    assert!((report.value(ReportMetric::Precision).unwrap() - 0.5).abs() < 1e-6);
    assert!((report.value(ReportMetric::Recall).unwrap() - 0.5).abs() < 1e-6);
    assert!((report.value(ReportMetric::F1).unwrap() - 0.5).abs() < 1e-6);
    assert_eq!(report.get(ReportMetric::Accuracy).unwrap().n_samples, 4);
}

#[test]
fn test_compute_metric_report_regression() {
    let predictions = Tensor::new(&[2.0, 4.0, 6.0], &[3, 1]);
    let labels = Tensor::new(&[1.0, 4.0, 9.0], &[3, 1]);
    let report = compute_metric_report(
        &TaskMetric::R2,
        &[
            ReportMetric::R2,
            ReportMetric::MeanSquaredError,
            ReportMetric::MeanAbsoluteError,
            ReportMetric::RootMeanSquaredError,
        ],
        &predictions,
        &labels,
        1,
        &LossType::MSE,
    );

    assert!((report.value(ReportMetric::R2).unwrap() - 0.6938776).abs() < 1e-5);
    assert!((report.value(ReportMetric::MeanSquaredError).unwrap() - 10.0 / 3.0).abs() < 1e-6);
    assert!((report.value(ReportMetric::MeanAbsoluteError).unwrap() - 4.0 / 3.0).abs() < 1e-6);
    assert!(
        (report.value(ReportMetric::RootMeanSquaredError).unwrap() - (10.0f32 / 3.0).sqrt()).abs()
            < 1e-6
    );
}

#[test]
fn test_compute_metric_report_multilabel_bce_logits() {
    let predictions = Tensor::new(
        &[
            0.2, -0.1, // [1, 0]
            -0.1, 0.2, // [0, 1]
        ],
        &[2, 2],
    );
    let labels = Tensor::new(
        &[
            1.0, 0.0, // [1, 0]
            0.0, 1.0, // [0, 1]
        ],
        &[2, 2],
    );
    let report = compute_metric_report(
        &TaskMetric::MultiLabelAccuracy,
        &[
            ReportMetric::MultiLabelLooseAccuracy,
            ReportMetric::MultiLabelStrictAccuracy,
        ],
        &predictions,
        &labels,
        2,
        &LossType::BCE,
    );

    assert!((report.value(ReportMetric::MultiLabelLooseAccuracy).unwrap() - 1.0).abs() < 1e-6);
    assert!(
        (report
            .value(ReportMetric::MultiLabelStrictAccuracy)
            .unwrap()
            - 1.0)
            .abs()
            < 1e-6
    );
    assert_eq!(
        report
            .get(ReportMetric::MultiLabelStrictAccuracy)
            .unwrap()
            .n_samples,
        2
    );
}

#[test]
fn test_evaluate_report_does_not_change_primary() {
    let task = xor_task();
    let genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let score = task.evaluate(&genome, &build, &mut rng).unwrap();

    assert!(!score.report.is_empty());
    assert!(
        (score.primary - score.report.value(ReportMetric::Accuracy).unwrap()).abs() < 1e-6,
        "报告中的 accuracy 应与 primary 保持一致，但不额外参与选择"
    );
    assert!(score.report.get(ReportMetric::F1).is_some());
}

// ==================== Batch size ====================

#[test]
fn test_auto_batch_size_thresholds() {
    // 小数据集：full-batch
    assert_eq!(auto_batch_size(1), 1);
    assert_eq!(auto_batch_size(4), 4);
    assert_eq!(auto_batch_size(128), 128);
    // 中等数据集：batch_size = 64
    assert_eq!(auto_batch_size(129), 64);
    assert_eq!(auto_batch_size(1000), 64);
    assert_eq!(auto_batch_size(10000), 64);
    // 大数据集：batch_size = 256
    assert_eq!(auto_batch_size(10001), 256);
    assert_eq!(auto_batch_size(60000), 256);
}

/// 创建 n 个样本的合成数据集（二分类，input_dim=2）
fn medium_data(n: usize) -> (Vec<Tensor>, Vec<Tensor>) {
    (
        (0..n)
            .map(|i| {
                let x = (i as f32) / (n as f32);
                Tensor::new(&[x, 1.0 - x], &[2])
            })
            .collect(),
        (0..n)
            .map(|i| Tensor::new(&[if i % 2 == 0 { 1.0 } else { 0.0 }], &[1]))
            .collect(),
    )
}

#[test]
fn test_mini_batch_training_medium_dataset() {
    // 200 样本 → auto_batch_size = 64，触发 mini-batch 路径
    let data = medium_data(200);
    let task = SupervisedTask::new(data.clone(), data, TaskMetric::Accuracy).unwrap();
    let genome = genome_with_hidden(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let loss = task
        .train(&genome, &build, &short_convergence(), &mut rng)
        .unwrap();

    assert!(loss.is_finite(), "mini-batch 训练应产生有限 loss: {loss}");
}

#[test]
fn test_mini_batch_parameters_change() {
    let data = medium_data(200);
    let task = SupervisedTask::new(data.clone(), data, TaskMetric::Accuracy).unwrap();
    let genome = genome_with_hidden(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let params_before: Vec<Tensor> = build
        .all_parameters()
        .iter()
        .map(|p| p.value().unwrap().unwrap())
        .collect();

    task.train(&genome, &build, &short_convergence(), &mut rng)
        .unwrap();

    let params_after: Vec<Tensor> = build
        .all_parameters()
        .iter()
        .map(|p| p.value().unwrap().unwrap())
        .collect();

    let any_changed = params_before
        .iter()
        .zip(params_after.iter())
        .any(|(a, b)| a != b);
    assert!(any_changed, "mini-batch 训练后至少应有一个参数变化");
}

#[test]
fn test_explicit_batch_size_via_configure() {
    let data = xor_data();
    let mut task = SupervisedTask::new(data.clone(), data, TaskMetric::Accuracy).unwrap();
    task.configure_batch_size(Some(2)); // 强制 batch_size=2（4 样本分成 2 个 batch）

    let genome = genome_with_hidden(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    let loss = task
        .train(&genome, &build, &short_convergence(), &mut rng)
        .unwrap();

    assert!(loss.is_finite(), "显式 batch_size=2 应正常训练");
}

#[test]
fn test_genome_batch_size_overrides_task_setting() {
    let data = xor_data();
    let mut task = SupervisedTask::new(data.clone(), data, TaskMetric::Accuracy).unwrap();
    task.configure_batch_size(Some(3)); // task-level: 3

    let mut genome = genome_with_hidden(2, 1);
    genome.training_config.batch_size = Some(2); // genome-level: 2（优先级更高）

    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    // genome 的 batch_size=2 应优先于 task 的 batch_size=3
    let loss = task
        .train(&genome, &build, &short_convergence(), &mut rng)
        .unwrap();

    assert!(loss.is_finite(), "genome batch_size 覆盖应正常训练");
}

#[test]
fn test_mini_batch_evaluate_still_full_batch() {
    // evaluate() 始终使用全量测试集，不受 batch_size 影响
    let data = medium_data(200);
    let mut task = SupervisedTask::new(data.clone(), data, TaskMetric::Accuracy).unwrap();
    task.configure_batch_size(Some(32));

    let genome = genome_with_hidden(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    task.train(&genome, &build, &short_convergence(), &mut rng)
        .unwrap();

    let score = task.evaluate(&genome, &build, &mut rng).unwrap();
    assert!(score.primary >= 0.0 && score.primary <= 1.0);
}

// ==================== mini-batch shuffle ====================

#[test]
fn test_mini_batch_shuffle_produces_different_trajectories() {
    // 验证 mini-batch 训练的 shuffle 产生不同的训练轨迹
    // 策略：用不同的 rng seed 训练同一模型，若 shuffle 生效，
    // 不同 seed 应导致不同的最终 loss
    let data = medium_data(200); // 200 样本，触发 mini-batch 路径
    let task = SupervisedTask::new(data.clone(), data, TaskMetric::Accuracy).unwrap();

    let convergence = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(5),
        ..Default::default()
    };

    let get_loss = |seed: u64| -> f32 {
        let genome = genome_with_hidden(2, 1);
        let mut rng = StdRng::seed_from_u64(seed);
        let build = build_and_restore(&genome, &mut rng);
        task.train(&genome, &build, &convergence, &mut rng)
            .unwrap()
            .final_loss
    };

    let loss_a = get_loss(100);
    let loss_b = get_loss(200);
    let loss_c = get_loss(300);

    // 三个不同 seed 中，至少应有两个产生不同的 loss
    let all_same = (loss_a - loss_b).abs() < 1e-8 && (loss_b - loss_c).abs() < 1e-8;
    assert!(
        !all_same,
        "mini-batch shuffle 应导致不同 seed 产生不同 loss: a={loss_a}, b={loss_b}, c={loss_c}"
    );
}

#[test]
fn test_mini_batch_same_seed_reproducible() {
    // 同一 seed 下 mini-batch shuffle 应可复现
    let data = medium_data(200);
    let task = SupervisedTask::new(data.clone(), data, TaskMetric::Accuracy).unwrap();

    let convergence = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(3),
        ..Default::default()
    };

    let get_loss = |seed: u64| -> f32 {
        let genome = genome_with_hidden(2, 1);
        let mut rng = StdRng::seed_from_u64(seed);
        let build = build_and_restore(&genome, &mut rng);
        task.train(&genome, &build, &convergence, &mut rng)
            .unwrap()
            .final_loss
    };

    let loss_1 = get_loss(42);
    let loss_2 = get_loss(42);
    assert!(
        (loss_1 - loss_2).abs() < 1e-6,
        "同 seed mini-batch 训练应可复现: {loss_1} vs {loss_2}"
    );
}

// ==================== 序列数据检测 ====================

fn seq_data_fixed(n: usize, seq_len: usize, input_dim: usize) -> (Vec<Tensor>, Vec<Tensor>) {
    let inputs: Vec<Tensor> = (0..n)
        .map(|i| {
            let data: Vec<f32> = (0..seq_len * input_dim)
                .map(|j| (i * seq_len * input_dim + j) as f32 * 0.01)
                .collect();
            Tensor::new(&data, &[seq_len, input_dim])
        })
        .collect();
    let labels: Vec<Tensor> = (0..n)
        .map(|i| Tensor::new(&[if i % 2 == 0 { 1.0 } else { 0.0 }], &[1]))
        .collect();
    (inputs, labels)
}

#[test]
fn test_supervised_task_detects_sequential() {
    // 2D 样本 [seq_len, input_dim] → SupervisedTask 自动检测为序列
    let data = seq_data_fixed(4, 5, 2);
    let task = SupervisedTask::new(data.clone(), data, TaskMetric::Accuracy).unwrap();

    // 通过构建序列 genome 并 evaluate 来间接验证
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.seq_len = Some(5);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);
    let score = task.evaluate(&genome, &build, &mut rng).unwrap();
    assert!(score.primary >= 0.0 && score.primary <= 1.0);
}

#[test]
fn test_supervised_task_var_len_auto_pad() {
    // 变长序列：长度 3, 5, 4 → 自动 pad 到 5
    let inputs = vec![
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]),
        Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[5, 2],
        ),
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]),
    ];
    let labels = vec![
        Tensor::new(&[1.0], &[1]),
        Tensor::new(&[0.0], &[1]),
        Tensor::new(&[1.0], &[1]),
    ];

    // 构造应成功（自动 pad）
    let task = SupervisedTask::new(
        (inputs.clone(), labels.clone()),
        (inputs, labels),
        TaskMetric::Accuracy,
    )
    .unwrap();

    // 验证可以正常用于训练/评估
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.seq_len = Some(5);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);
    let score = task.evaluate(&genome, &build, &mut rng).unwrap();
    assert!(score.primary >= 0.0 && score.primary <= 1.0);
}

// ==================== 训练后 loss 下降 ====================

#[test]
fn test_training_loss_actually_decreases() {
    let task = xor_task();
    let genome = genome_with_hidden(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = build_and_restore(&genome, &mut rng);

    // 只训练 1 个 epoch，获取初始 loss
    let convergence_1 = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(1),
        ..Default::default()
    };
    let loss_after_1 = task
        .train(&genome, &build, &convergence_1, &mut rng)
        .unwrap();

    // 继续训练 50 个 epoch
    let convergence_50 = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(50),
        ..Default::default()
    };
    let loss_after_50 = task
        .train(&genome, &build, &convergence_50, &mut rng)
        .unwrap();

    assert!(
        loss_after_50.final_loss < loss_after_1.final_loss,
        "训练 50 epoch 后的 loss ({:.6}) 应低于 1 epoch 后 ({:.6})",
        loss_after_50.final_loss,
        loss_after_1.final_loss
    );
}
