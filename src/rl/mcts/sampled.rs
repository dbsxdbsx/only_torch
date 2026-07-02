//! Sampled `MuZero` 动作采样与 PUCT 先验修正（Hubert et al. ICML 2021）。
//!
//! - 展开时从 proposal β（默认 = 网络 prior π）无放回采 K 个候选；
//! - PUCT 探索项用 `π̂_β` ∝ (β̂ / β) · π，K ≥ |A| 时退化为标准 `MuZero` prior（无回归）。

use rand::RngCore;

use super::puct::mix_dirichlet_prior;
use super::types::{CandidateSet, MctsConfig};

/// 从完整候选集采样展开候选，返回带 PUCT prior 的候选子集。
///
/// `beta` 与 `pi` 通常相同（β = π）；根节点可在调用前对二者注入 Dirichlet 噪声。
pub fn sample_for_expansion(
    candidates: &CandidateSet,
    k: usize,
    rng: &mut dyn RngCore,
) -> CandidateSet {
    let n = candidates.len();
    if n == 0 {
        return CandidateSet::empty();
    }

    let aligned_beta = align_probs(
        &candidates
            .candidates
            .iter()
            .map(|c| c.proposal_prior.unwrap_or(c.policy_prior))
            .collect::<Vec<_>>(),
        n,
    );
    let aligned_pi = align_probs(&candidates.policy_priors(), n);
    let k = k.max(1).min(n);

    // K 覆盖全动作集 → 与标准 MuZero 一致，不做 sample-based 修正。
    if k >= n {
        let mut out = candidates.clone();
        for (candidate, prior) in out.candidates.iter_mut().zip(aligned_pi) {
            candidate.policy_prior = prior;
        }
        return out;
    }

    let indices = sample_indices_without_replacement(&aligned_beta, k, rng);
    let puct_priors = sampled_puct_priors(&aligned_beta, &aligned_pi, &indices);
    let sampled = indices
        .iter()
        .zip(puct_priors)
        .map(|(&i, prior)| candidates.candidates[i].clone().with_policy_prior(prior))
        .collect();
    CandidateSet {
        candidates: sampled,
    }
}

/// 根节点：Dirichlet 噪声后采样（论文 §5.5：β 与 π 均加噪再采）。
pub fn sample_root_for_expansion(
    candidates: &CandidateSet,
    cfg: &MctsConfig,
    k: usize,
    rng: &mut dyn RngCore,
) -> CandidateSet {
    let n = candidates.len();
    if n == 0 {
        return CandidateSet::empty();
    }
    let mut noisy = align_probs(&candidates.policy_priors(), n);
    mix_dirichlet_prior(&mut noisy, cfg, rng);
    let mut noisy_candidates = candidates.clone();
    for (candidate, prior) in noisy_candidates.candidates.iter_mut().zip(noisy) {
        candidate.policy_prior = prior;
        candidate.proposal_prior = Some(prior);
    }
    sample_for_expansion(&noisy_candidates, k, rng)
}

/// `π̂_β(a)` ∝ (β̂ / β)(a) · π(a)，β̂ 为采样子集上的经验分布（无放回 → 各 1/K）。
pub(in crate::rl) fn sampled_puct_priors(beta: &[f32], pi: &[f32], indices: &[usize]) -> Vec<f32> {
    let k = indices.len().max(1) as f32;
    let mut raw: Vec<f32> = indices
        .iter()
        .map(|&i| {
            let b = beta[i].max(1e-8);
            let p = pi[i].max(1e-8);
            (1.0 / k) * p / b
        })
        .collect();
    normalize_probs(&mut raw);
    raw
}

fn align_probs(probs: &[f32], n: usize) -> Vec<f32> {
    if probs.len() == n {
        probs.to_vec()
    } else {
        vec![1.0 / n as f32; n]
    }
}

fn normalize_probs(v: &mut [f32]) {
    let s: f32 = v.iter().sum();
    if s > 0.0 {
        for x in v {
            *x /= s;
        }
    } else if !v.is_empty() {
        let u = 1.0 / v.len() as f32;
        v.fill(u);
    }
}

/// 按权重无放回采 k 个下标。
fn sample_indices_without_replacement(
    weights: &[f32],
    k: usize,
    rng: &mut dyn RngCore,
) -> Vec<usize> {
    let n = weights.len();
    let k = k.min(n);
    let mut pool: Vec<usize> = (0..n).collect();
    let mut w: Vec<f32> = weights.to_vec();
    let mut out = Vec::with_capacity(k);

    for _ in 0..k {
        let sum: f32 = w.iter().sum();
        if sum <= 0.0 {
            let pick = (rng.next_u32() as usize) % pool.len();
            out.push(pool[pick]);
            pool.remove(pick);
            w.remove(pick);
            continue;
        }
        let mut threshold = (rng.next_u32() as f32 / u32::MAX as f32) * sum;
        let mut pick = 0usize;
        for (i, &wi) in w.iter().enumerate() {
            threshold -= wi;
            if threshold <= 0.0 {
                pick = i;
                break;
            }
            pick = i;
        }
        out.push(pool[pick]);
        pool.remove(pick);
        w.remove(pick);
    }
    out
}
