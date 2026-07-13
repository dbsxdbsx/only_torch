//! 当前 learned world model 的逐转移诊断。
//!
//! 本模块只把**真实 transition**与冻结模型 revision 的预测做比较，不参与训练，
//! 也不把 Reference Transition Model 当作 target。Phase 3A0 先逐分量审计这些
//! proxy 是否任务相关、是否可被新增真实数据降低；本轮未过稳定降低门槛，故只保留
//! 诊断能力，不接 `ErrorQ` / Collector。

const PROB_EPS: f32 = 1e-8;

/// 冻结 world model 对一条真实 transition 的原始诊断输出。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TransitionDiagnostics {
    /// dynamics reward categorical distribution。
    pub reward_probs: Vec<f32>,
    /// dynamics continuation probability。
    pub continuation: f32,
    /// `f_policy(g(h(o_t), a_t))`，已解码为联合动作分布。
    pub imagined_next_policy: Vec<f32>,
    /// `f_value(g(h(o_t), a_t))`。
    pub imagined_next_value: f32,
    /// `f_policy(h(o_{t+1}))`。
    pub reencoded_next_policy: Option<Vec<f32>>,
    /// `f_value(h(o_{t+1}))`。
    pub reencoded_next_value: Option<f32>,
}

/// 3A0 单独报告的误差分量。
///
/// reward / continuation 由真实环境直接锚定；policy / value 两项共用当前 `h/f`，
/// 只是 decision-equivalence proxy，不能单独宣称为真实模型误差或 value gap。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ModelErrorComponents {
    /// `KL(two_hot_reward_target || predicted_reward)`；已扣除 target entropy。
    pub reward_kl: f32,
    pub continuation_brier: f32,
    pub policy_jsd: Option<f32>,
    pub value_abs_diff: Option<f32>,
}

impl ModelErrorComponents {
    /// 所有当前可用分量是否有限且非负。
    pub fn is_finite_nonnegative(&self) -> bool {
        let required = [self.reward_kl, self.continuation_brier];
        required
            .into_iter()
            .chain(self.policy_jsd)
            .chain(self.value_abs_diff)
            .all(|x| x.is_finite() && x >= 0.0)
    }
}

/// 用真实 target 给冻结模型的原始诊断打分。
///
/// `next_legal_mask` 是**真实下一状态**的合法动作支撑；仅在 policy JSD 中使用。
/// 缺少真实 next observation 时，policy/value 分量返回 `None`，不能按 0 参与合成。
pub(crate) fn score_transition(
    diagnostics: &TransitionDiagnostics,
    reward_target: &[f32],
    observed_continuation: f32,
    next_legal_mask: Option<&[bool]>,
) -> ModelErrorComponents {
    let reward_kl = categorical_kl(reward_target, &diagnostics.reward_probs);
    let continuation_target = observed_continuation.clamp(0.0, 1.0);
    let continuation_brier = (diagnostics.continuation - continuation_target).powi(2);

    let policy_jsd = diagnostics
        .reencoded_next_policy
        .as_deref()
        .and_then(|real| {
            jensen_shannon_divergence(&diagnostics.imagined_next_policy, real, next_legal_mask)
        });
    let value_abs_diff = diagnostics
        .reencoded_next_value
        .map(|real| (diagnostics.imagined_next_value - real).abs());

    ModelErrorComponents {
        reward_kl,
        continuation_brier,
        policy_jsd,
        value_abs_diff,
    }
}

/// categorical proper loss：`-Σ target_i log(prob_i)`。
pub(crate) fn categorical_nll(target: &[f32], probs: &[f32]) -> f32 {
    assert!(!target.is_empty(), "categorical_nll: target 不能为空");
    assert_eq!(
        target.len(),
        probs.len(),
        "categorical_nll: target/probs 长度不匹配"
    );
    target
        .iter()
        .zip(probs)
        .map(|(&t, &p)| {
            if t <= 0.0 {
                0.0
            } else {
                -t * p.clamp(PROB_EPS, 1.0).ln()
            }
        })
        .sum()
}

/// `KL(target || probs) = CE(target, probs) - H(target)`。
///
/// two-hot reward target 自带类别相关 entropy；扣除后，完美预测在 reward=0/1 上
/// 都严格得到 0，才可把数值解释为模型误差而不是 raw training loss。
pub(crate) fn categorical_kl(target: &[f32], probs: &[f32]) -> f32 {
    let cross_entropy = categorical_nll(target, probs);
    let target_entropy = categorical_nll(target, target);
    let divergence = cross_entropy - target_entropy;
    if divergence.is_finite() {
        divergence.max(0.0)
    } else {
        divergence
    }
}

/// Jensen-Shannon divergence；使用自然对数，范围 `[0, ln 2]`。
///
/// 若提供 mask，先把两个分布限制到同一真实合法支撑，再分别归一化。空支撑、
/// 非有限值或零质量返回 `None`，由上层按“分量缺失”处理。
pub(crate) fn jensen_shannon_divergence(
    p: &[f32],
    q: &[f32],
    mask: Option<&[bool]>,
) -> Option<f32> {
    assert_eq!(p.len(), q.len(), "JSD: 分布长度不匹配");
    if let Some(mask) = mask {
        assert_eq!(p.len(), mask.len(), "JSD: mask 长度不匹配");
    }
    let p = normalize_on_support(p, mask)?;
    let q = normalize_on_support(q, mask)?;
    let mut jsd = 0.0;
    for (&pi, &qi) in p.iter().zip(&q) {
        let m = 0.5 * (pi + qi);
        if pi > 0.0 {
            jsd += 0.5 * pi * (pi / m).ln();
        }
        if qi > 0.0 {
            jsd += 0.5 * qi * (qi / m).ln();
        }
    }
    Some(jsd.max(0.0))
}

fn normalize_on_support(probs: &[f32], mask: Option<&[bool]>) -> Option<Vec<f32>> {
    let mut selected = Vec::with_capacity(probs.len());
    let mut sum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        let included = mask.is_none_or(|m| m[i]);
        let value = if included { p } else { 0.0 };
        if !value.is_finite() || value < 0.0 {
            return None;
        }
        selected.push(value);
        sum += value;
    }
    if !sum.is_finite() || sum <= PROB_EPS {
        return None;
    }
    for p in &mut selected {
        *p /= sum;
    }
    Some(selected)
}

/// Spearman rank correlation（并列值取平均 rank）。
///
/// 非有限输入、长度不足 2 或任一侧 rank 方差为 0 时返回 `None`。
pub(crate) fn spearman_rank_correlation(x: &[f32], y: &[f32]) -> Option<f32> {
    if x.len() != y.len() || x.len() < 2 || x.iter().chain(y).any(|value| !value.is_finite()) {
        return None;
    }
    pearson_correlation(&average_ranks(x), &average_ranks(y))
}

/// 最高 `fraction` 分数样本中的事件率，相对全样本事件率的 lift。
pub(crate) fn top_fraction_event_lift(
    scores: &[f32],
    events: &[bool],
    fraction: f32,
) -> Option<f32> {
    if scores.len() != events.len()
        || scores.is_empty()
        || !(0.0 < fraction && fraction <= 1.0)
        || scores.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let total_events = events.iter().filter(|&&event| event).count();
    if total_events == 0 {
        return None;
    }

    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let requested_top_n = ((scores.len() as f32 * fraction).ceil() as usize)
        .max(1)
        .min(scores.len());
    let threshold = scores[order[requested_top_n - 1]];
    // P90 等阈值语义：边界并列全部纳入，避免按原始记录顺序任意切断 ties。
    let top: Vec<usize> = order
        .iter()
        .copied()
        .take_while(|&index| scores[index] >= threshold)
        .collect();
    let top_events = top.iter().filter(|&&index| events[index]).count();
    let top_rate = top_events as f32 / top.len() as f32;
    let base_rate = total_events as f32 / scores.len() as f32;
    Some(top_rate / base_rate)
}

fn average_ranks(values: &[f32]) -> Vec<f32> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        // rank 使用 1-based；并列区间 [start+1, end] 的平均值。
        let average = (start + 1 + end) as f32 / 2.0;
        for &index in &order[start..end] {
            ranks[index] = average;
        }
        start = end;
    }
    ranks
}

fn pearson_correlation(x: &[f32], y: &[f32]) -> Option<f32> {
    let n = x.len() as f32;
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;
    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    for (&x, &y) in x.iter().zip(y) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    let denominator = (variance_x * variance_y).sqrt();
    (denominator > f32::EPSILON).then_some((covariance / denominator).clamp(-1.0, 1.0))
}
