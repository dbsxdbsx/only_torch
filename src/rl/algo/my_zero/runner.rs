//! MyZero 统一运行入口：self-play + 训练 + greedy eval + 多 seed + SMOKE。

use super::action::ActionAdapter;
use super::checkpoint::BestTracker;
use super::component::Components;
use super::config::{MyZeroConfig, greedy_episode_seed, self_play_episode_seed};
use super::my_zero::MyZero;
use super::n_step::n_step_targets_for_unroll;
use super::network::{MyZeroModel, ObsSource, UnrollItem};
use super::obs_pipeline::ObsAdapter;
use super::reanalyze::reanalyze_unroll_window;
use super::report::TrainReport;
use super::rosmo::{RosmoTargets, rosmo_refresh_window_with_actions};
use super::sampled_params::{
    compute_sampled_k_cfg, format_sampled_log, resolve_sampled_params, sampled_k_effective,
    sampled_would_subsample,
};
use super::search_policy::MyZeroSearchPolicy;
use super::target::mcts_policy_target;
use super::target_net::{TargetConfig, hard_update, sync_target};
use super::value_prefix::reward_prefix_targets;
use crate::nn::{Adam, Graph, GraphError, Optimizer};
use crate::rl::mcts::{
    Dynamics, DynamicsModel, MctsConfig, MctsRecipe, PuctConfig, RootDirichletConfig,
    SampledConfig, SearchBudget, mcts_search,
};
use crate::rl::{GameOutcome, GymEnv, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use pyo3::Python;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeMap, HashMap, VecDeque};

/// 打印已启用的消融组件（全关时不输出）。
fn print_components(c: &Components) {
    let mut tags = Vec::new();
    if c.consistency {
        tags.push("consistency");
    }
    if c.reconstruction {
        tags.push("reconstruction");
    }
    if c.hl_gauss {
        tags.push("HL-Gauss");
    }
    // obs_symlog 是连续观测的 always-on 入口变换，不是消融组件。
    if c.value_prefix {
        tags.push("value_prefix");
    }
    if c.target_net {
        tags.push("target_net");
    }
    if c.sve_enabled() {
        tags.push("SVE");
    }
    if c.gumbel {
        tags.push("Gumbel");
    }
    if c.completed_q_target {
        tags.push("completedQ");
    }
    if c.sampled {
        tags.push("Sampled");
    }
    if c.num_chance_outcomes > 1 {
        tags.push("Stochastic");
    }
    if c.reanalyze {
        tags.push("reanalyze");
    }
    if c.rosmo {
        tags.push("ROSMO");
    }
    if !tags.is_empty() {
        println!("[MyZero] {}", tags.join(" + "));
    }
}

/// target 网在 inference 下评估 `steps[pos]` 的 V，供 n-step 自举尾值。
fn eval_target_bootstrap(
    target: &MyZeroModel,
    steps: &[SelfPlayStep],
    pos: usize,
    image_stack: Option<usize>,
) -> f32 {
    let src = match image_stack {
        None => ObsSource::Single(&steps[pos].obs),
        Some(k) => ObsSource::Stacked {
            steps,
            t: pos,
            stack: k,
        },
    };
    let mut obs = Vec::with_capacity(src.dim());
    src.append_into(&mut obs);
    target
        .graph
        .inference_scope(|_| target.initial_state(&obs).2)
}

/// 非评测（self-play / 训练收集）：原样按 π 或 visit 抽；政策目标不再 `^{1/τ}` 拧尖。
pub(crate) const SELF_PLAY_TEMPERATURE: f32 = 1.0;
/// 评测 / 报分 / `run`：贪心；Gumbel 噪声同步关闭（`temperature < 1e-6`）。
pub(crate) const EVAL_TEMPERATURE: f32 = 0.0;

/// 按温度从概率向量采样动作下标（`T→0` 取 argmax）。
pub(crate) fn sample_from_probs(probs: &[f32], temperature: f32, rng: &mut impl Rng) -> usize {
    assert!(!probs.is_empty(), "sample_from_probs: 空分布");
    if temperature < 1e-6 {
        return probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }
    let inv_t = 1.0 / temperature;
    let weights: Vec<f32> = probs.iter().map(|p| p.max(0.0).powf(inv_t)).collect();
    let sum: f32 = weights.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return rng.gen_range(0..probs.len());
    }
    let mut u = rng.r#gen::<f32>() * sum;
    for (i, w) in weights.iter().enumerate() {
        u -= w;
        if u <= 0.0 {
            return i;
        }
    }
    probs.len() - 1
}

// ============================================================================
// KL 自适应 lr（「用户不调 lr」去旋钮机制；可插拔，默认关）
// ============================================================================

/// KL 自适应 lr 的目标位移（参照 junxiaosong/AlphaZero_Gomoku `kl_targ=0.02`）。
pub(crate) const KL_TARGET: f32 = 0.02;

/// KL 自适应 lr 乘子更新（参照实现口径：KL 超 2× 目标降 1.5×、不足 1/2 升 1.5×，
/// 乘子夹在 [0.1, 10]）——大 batch/高强度训练不再需要手工标 lr 的机制底座。
/// 棋盘域 ⑨ 臂已验证无害（batch 512 自动配平）；单智能体域接线见 `train_one_seed`。
pub(crate) fn kl_lr_multiplier(kl: f32, kl_target: f32, mult: f32) -> f32 {
    if kl > kl_target * 2.0 {
        (mult / 1.5).max(0.1)
    } else if kl < kl_target / 2.0 {
        (mult * 1.5).min(10.0)
    } else {
        mult
    }
}

/// 探针局面上的平均 policy KL：`Σ p_old·ln(p_old/p_new)`（对合法支撑外的微小
/// 概率加 eps 防 ln(0)；探针 = 当局若干 obs，只读不训）。
pub(crate) fn probe_policy_kl(
    model: &MyZeroModel,
    probe_obs: &[Vec<f32>],
    old: &[Vec<f32>],
) -> f32 {
    const EPS: f32 = 1e-10;
    let mut kl_sum = 0.0f32;
    for (obs, p_old) in probe_obs.iter().zip(old) {
        let (_latent, p_new, _v) = Dynamics::initial_state(&model, obs);
        let mut kl = 0.0f32;
        for (po, pn) in p_old.iter().zip(&p_new) {
            if *po > EPS {
                kl += po * ((po + EPS).ln() - (pn + EPS).ln());
            }
        }
        kl_sum += kl;
    }
    kl_sum / probe_obs.len().max(1) as f32
}

/// 提取探针 obs（当局最多 `n` 个局面，均匀间隔）。
///
/// `image_stack = Some(k)` 时按 acting 期同口径就地堆叠最近 k 帧
/// （[`ObsSource::Stacked`] 语义，起点前向填充首帧），保证探针输入与模型
/// 训练/推理输入同构；向量 obs 传 `None`。
pub(crate) fn probe_obs_of(
    steps: &[SelfPlayStep],
    n: usize,
    image_stack: Option<usize>,
) -> Vec<Vec<f32>> {
    let take = steps.len().min(n);
    (0..take)
        .map(|i| {
            let pos = i * steps.len() / take;
            let src = match image_stack {
                None => ObsSource::Single(&steps[pos].obs),
                Some(k) => ObsSource::Stacked {
                    steps,
                    t: pos,
                    stack: k,
                },
            };
            let mut v = Vec::with_capacity(src.dim());
            src.append_into(&mut v);
            v
        })
        .collect()
}

/// 由训练超参 + 组件开关 + joint 候选数 N 构造 MCTS 配置（Sampled 时 K 按公式解析）。
pub(crate) fn my_zero_mcts_config(
    num_simulations: u32,
    temperature: f32,
    discount: f32,
    components: &Components,
    joint_n: usize,
    root_exploration_fraction: f32,
) -> MctsConfig {
    let recipe = MctsRecipe {
        budget: SearchBudget {
            num_simulations,
            temperature,
            discount,
        },
        puct: PuctConfig {
            pb_c_base: MctsConfig::default().pb_c_base,
            pb_c_init: MctsConfig::default().pb_c_init,
        },
        root_dirichlet: RootDirichletConfig {
            alpha: MctsConfig::default().root_dirichlet_alpha,
            exploration_fraction: root_exploration_fraction,
        },
        sampled: if !components.gumbel
            && (components.sampled || sampled_would_subsample(joint_n, num_simulations))
        {
            Some(SampledConfig {
                k: sampled_k_effective(compute_sampled_k_cfg(joint_n, num_simulations), joint_n),
            })
        } else {
            None
        },
        num_chance_outcomes: components.num_chance_outcomes,
    };
    MctsConfig::from_recipe(recipe)
}

#[allow(clippy::too_many_arguments)]
fn self_play_one_episode(
    env: &GymEnv,
    model: &MyZeroModel,
    adapter: &ActionAdapter,
    obs_adapter: &mut ObsAdapter,
    mcts_cfg: &MctsConfig,
    components: &Components,
    gamma: f32,
    cq: Option<(f32, f32)>,
    reset_seed: u64,
    rng: &mut StdRng,
) -> Vec<SelfPlayStep> {
    use super::network::PrecomputedRootDynamics;

    let (mut search_obs, mut store_obs) = obs_adapter.reset(env, Some(reset_seed));
    let mut steps = Vec::new();
    let use_posterior = components.recurrent_posterior && model.posterior.is_some();
    let hidden_size = model.posterior.as_ref().map(|p| p.hidden_size).unwrap_or(0);
    let mut posterior_hidden = vec![0.0f32; hidden_size];
    let mut prev_action_idx: Option<usize> = None;

    loop {
        let (action_idx, policy_target, root_value) = if mcts_cfg.num_simulations <= 1 {
            // 交互默认：一次前向出 π，不搜树。`acting_simulations > 1` 走下面 MCTS。
            let (policy, value) = if use_posterior {
                let repr_latent = model.repr_inference(&search_obs);
                let mut prev_oh = vec![0.0f32; model.action_dim];
                if let Some(a) = prev_action_idx
                    && a < model.action_dim
                {
                    prev_oh[a] = 1.0;
                }
                let (posterior_latent, new_hidden) =
                    model.posterior_step_inference(&repr_latent, &prev_oh, &posterior_hidden);
                posterior_hidden = new_hidden;
                model.pred_inference(&posterior_latent)
            } else {
                let (_z, p, v) = Dynamics::initial_state(&model, &search_obs);
                (p, v)
            };
            let idx = sample_from_probs(&policy, mcts_cfg.temperature, rng);
            (idx, policy, value)
        } else {
            let candidates = adapter.candidates().to_vec();
            let policy = MyZeroSearchPolicy::from_components(components);

            let result = if use_posterior {
                let repr_latent = model.repr_inference(&search_obs);
                let mut prev_oh = vec![0.0f32; model.action_dim];
                if let Some(a) = prev_action_idx
                    && a < model.action_dim
                {
                    prev_oh[a] = 1.0;
                }
                let (posterior_latent, new_hidden) =
                    model.posterior_step_inference(&repr_latent, &prev_oh, &posterior_hidden);
                posterior_hidden = new_hidden;
                let (root_policy, root_value) = model.pred_inference(&posterior_latent);
                let root_dyn =
                    PrecomputedRootDynamics::new(model, posterior_latent, root_policy, root_value);
                let dyn_model = DynamicsModel::new(root_dyn, candidates, gamma);
                mcts_search(&dyn_model, &policy, &search_obs, mcts_cfg, rng)
            } else {
                let dyn_model = DynamicsModel::new(model, candidates, gamma);
                mcts_search(&dyn_model, &policy, &search_obs, mcts_cfg, rng)
            };
            let idx = result.recommended_id.index();
            let root_value = result.root_value();
            let policy_target = mcts_policy_target(&result, cq, adapter.action_dim(), 0);
            (idx, policy_target, root_value)
        };
        prev_action_idx = Some(action_idx);

        steps.push(SelfPlayStep {
            obs: store_obs.clone(),
            action: vec![action_idx as f32],
            policy_target,
            player: 0,
            reward: 0.0,
            root_value: Some(root_value),
            terminated: false,
            truncated: false,
            continuation: 1.0,
            extras: Default::default(),
        });

        let env_action = adapter.to_env(action_idx);
        let (next_search, next_store, reward, terminated, truncated) =
            obs_adapter.step(env, &env_action);
        let last = steps.last_mut().unwrap();
        last.reward = reward;
        last.terminated = terminated;
        last.truncated = truncated;
        last.continuation = if terminated { 0.0 } else { 1.0 };

        if terminated || truncated {
            break;
        }
        search_obs = next_search;
        store_obs = next_store;
    }

    steps
}

/// 给定起点 `start`，计算 K 步 unroll 实际步数（区分 terminated / truncated）。
///
/// **truncated 局刻意排除最后一个 transition**（`len - 1 - start`，而非 `len - start`）：
/// 若纳入末步，unroll 末端位置的 value/policy 目标对应「truncation 之后的状态」——
/// 该状态**非终止**（真值应为 bootstrap 而非 absorbing 0），当前训练器无逐位置 mask，
/// 强行纳入会注入错误 value 监督。代价是末步 reward 少一次监督（与 terminated 局的
/// absorbing padding 轻微不对称），见 `.issue/items/my_zero_truncated_final_step_reward.md`。
pub(crate) fn unroll_len_at(steps: &[SelfPlayStep], start: usize, k_unroll: usize) -> usize {
    let len = steps.len();
    if len == 0 || start >= len {
        return 0;
    }
    if steps[len - 1].terminated {
        k_unroll
    } else {
        k_unroll.min(len - 1 - start)
    }
}

/// 一次训练 batch 中的单条样本：整局 clone + 随机 unroll 起点。
///
/// `buffer_idx = Some(_)` 表示该副本来自 [`ReplayBuffer::sample_indexed`]，
/// train 结束后须 [`writeback_reanalyzed_samples`] 写回标签。
pub(crate) struct TrainBatchItem {
    pub buffer_idx: Option<usize>,
    pub game: SelfPlayGame,
    pub start: usize,
}

/// 训练 batch 的三种来源：
/// - `Borrowed`：非刷新路径，只记 `(buffer 下标, 起点)`，训练时从 buffer
///   借引用——**零整局 clone**（图像 obs 单局可达数十 MB，clone+drop 是实测主瓶颈）。
/// - `Owned`：reanalyze 路径，clone 副本刷标签，train 后写回（语义同旧实现）。
/// - `Refreshed`：ROSMO 路径，零克隆借引用 + 随行现算 target（**不写回**）。
pub(crate) enum PreparedBatch {
    Borrowed(Vec<(usize, usize)>),
    Owned(Vec<TrainBatchItem>),
    Refreshed(Vec<(usize, usize)>, Vec<RosmoTargets>),
}

/// 从 buffer 抽样并（若开启 reanalyze / ROSMO）刷新 unroll 窗口内 policy/value 标签。
///
/// # 流程（reanalyze 开启时）
/// 1. `sample_indexed` clone 出 owned 副本（不对 buffer 内引用原地改）
/// 2. `reanalyze_unroll_window` 刷新 `[start, start+K]` 内各步
/// 3. 交给 `train_batch` 消费同一副本
/// 4. train 后 `writeback_reanalyzed_samples` move 写回 buffer
///
/// # 流程（ROSMO 开启时）
/// 零克隆采样（RNG 消耗与 Borrowed 路径逐 bit 一致）+ 逐样本一步 look-ahead
/// 现算 target（[`rosmo_refresh_window_with_actions`]），**不写回**——见 [`super::rosmo`] 模块文档。
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_train_batch(
    buffer: &ReplayBuffer<SelfPlayGame>,
    train_batch_size: usize,
    k_unroll: usize,
    td_steps: usize,
    reanalyze: bool,
    components: &Components,
    model: &MyZeroModel,
    adapter: &ActionAdapter,
    gamma: f32,
    num_simulations: u32,
    cq: Option<(f32, f32)>,
    image_stack: Option<usize>,
    rng: &mut StdRng,
) -> PreparedBatch {
    if reanalyze {
        let re_cfg = my_zero_mcts_config(
            num_simulations,
            SELF_PLAY_TEMPERATURE,
            gamma,
            components,
            adapter.action_dim(),
            0.0,
        );
        let re_policy = MyZeroSearchPolicy::from_components(components);
        let dyn_model = DynamicsModel::new(model, adapter.candidates().to_vec(), gamma);
        let mut out = Vec::with_capacity(train_batch_size);
        for (idx, mut game) in buffer.sample_indexed(train_batch_size, rng) {
            if game.steps.len() < 2 {
                continue;
            }
            let start = rng.gen_range(0..game.steps.len());
            let actual_k = unroll_len_at(&game.steps, start, k_unroll);
            reanalyze_unroll_window(
                &dyn_model,
                &re_policy,
                &mut game.steps,
                start,
                actual_k,
                &re_cfg,
                cq,
                adapter.action_dim(),
                rng,
            );
            out.push(TrainBatchItem {
                buffer_idx: Some(idx),
                game,
                start,
            });
        }
        PreparedBatch::Owned(out)
    } else {
        // 零克隆路径：只采下标 + 起点，训练时从 buffer 借引用
        //（RNG 消耗顺序与旧 clone 路径逐 bit 一致，哨兵可复现）
        let mut out = Vec::with_capacity(train_batch_size);
        for idx in buffer.sample_indices(train_batch_size, rng) {
            let len = buffer
                .get_ref(idx)
                .map(|g| g.steps.len())
                .unwrap_or_default();
            if len < 2 {
                continue;
            }
            out.push((idx, rng.gen_range(0..len)));
        }
        if !components.rosmo {
            return PreparedBatch::Borrowed(out);
        }
        // ROSMO：对采出的每个 unroll 窗口现算 target（模型前向不消耗 rng，确定性）
        let targets: Vec<RosmoTargets> = out
            .iter()
            .map(|&(idx, start)| {
                let game = buffer.get_ref(idx).expect("buffer 下标应有效");
                let steps = &game.steps;
                let actual_k = unroll_len_at(steps, start, k_unroll);
                let obs_at = |pos: usize| -> Vec<f32> {
                    let src = match image_stack {
                        None => ObsSource::Single(&steps[pos].obs),
                        Some(k) => ObsSource::Stacked {
                            steps,
                            t: pos,
                            stack: k,
                        },
                    };
                    let mut v = Vec::with_capacity(src.dim());
                    src.append_into(&mut v);
                    v
                };
                rosmo_refresh_window_with_actions(
                    &model,
                    steps,
                    start,
                    actual_k,
                    td_steps,
                    gamma,
                    adapter.candidates(),
                    &obs_at,
                )
            })
            .collect();
        PreparedBatch::Refreshed(out, targets)
    }
}

/// reanalyze 路径：train 结束后把刷新过的整局写回 buffer（同 batch 重复 idx 后者覆盖）。
/// 仅 `buffer_idx = Some(_)` 的样本写回；CartPole 现状见 `.issue/items/my_zero_reanalyze_cartpole_regression.md`。
pub(crate) fn writeback_reanalyzed_samples(
    buffer: &mut ReplayBuffer<SelfPlayGame>,
    batch: Vec<TrainBatchItem>,
) {
    let mut by_idx: HashMap<usize, SelfPlayGame> = HashMap::new();
    for item in batch {
        if let Some(idx) = item.buffer_idx {
            by_idx.insert(idx, item.game);
        }
    }
    for (idx, game) in by_idx {
        buffer.update_at(idx, game);
    }
}

/// batch-native 训练：一次 zero_grad + 按 unroll 结构分组做 batch forward/backward + 一次 step。
///
/// # 与逐样本的关系（数学等价，浮点顺序不同）
/// 旧实现对 batch 内每个 position 单独 `train_unroll`+`backward`（各乘 `1/batch_size` 后累积）。
/// 现改为把 position 按 `(actual_k, next_obs 步数)` 分组——**同组结构逐样本一致**——每组一次
/// `[G, X]` 前向 + backward，组 loss（CE/MSE/consistency 均 batch 均值）再乘 `G/batch_size`。
/// 实数域下总梯度 = `(1/batch_size) Σ_g ∇L_g`，与逐样本累积相同；仅浮点归约顺序不同，
/// 故 env-steps 不再逐 bit 复现（改用 solved / 多 seed 统计口径验收）。
/// 分组用 [`BTreeMap`] 保证跨运行**确定性**（同 seed 可复现）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn train_batch(
    model: &MyZeroModel,
    optimizer: &mut Adam,
    samples: &[(&SelfPlayGame, usize)],
    refreshed: Option<&[RosmoTargets]>,
    k_unroll: usize,
    td_steps: usize,
    gamma: f32,
    components: &Components,
    image_stack: Option<usize>,
    burn_in_steps: usize,
    target_model: Option<&MyZeroModel>,
) -> Result<f32, GraphError> {
    if samples.is_empty() {
        return Ok(0.0);
    }

    let batch_size = samples.len() as f32;
    let consistency_coef = if components.consistency {
        components.consistency_coef
    } else {
        0.0
    };
    let reconstruction_coef = if components.reconstruction {
        components.reconstruction_coef
    } else {
        0.0
    };
    // ROSMO 行为正则系数（仅现算 target 路径生效）
    let bc_coef = if refreshed.is_some() {
        components.rosmo_alpha
    } else {
        0.0
    };
    let want_next_obs = components.consistency || components.reconstruction;

    // 1) 逐样本展开成 UnrollItem，并按 (actual_k, next_obs 步数) 分组（结构对齐才能同批堆叠）
    let mut groups: BTreeMap<(usize, usize), Vec<UnrollItem<'_>>> = BTreeMap::new();
    for (i, (game, start)) in samples.iter().enumerate() {
        let steps = &game.steps;
        let len = steps.len();
        let t = *start;
        let actual_k = unroll_len_at(steps, t, k_unroll);
        // ROSMO 现算 target（与存量 target 同槽位结构；prepare 期用同一 unroll_len_at 计算）
        let rosmo_t = refreshed.map(|rts| &rts[i]);
        if let Some(rt) = rosmo_t {
            debug_assert_eq!(rt.policies.len(), actual_k + 1, "ROSMO 槽位结构应对齐");
        }

        // 模型入口 obs：只记录来源（borrow buffer），batch 组装时融合反量化/堆叠
        // 一次性写入最终 flat（Flat 直通；图像模式老 → 新堆叠，语义同 assemble_stacked_obs）
        let obs_at = |pos: usize| -> ObsSource<'_> {
            match image_stack {
                None => ObsSource::Single(&steps[pos].obs),
                Some(k) => ObsSource::Stacked {
                    steps,
                    t: pos,
                    stack: k,
                },
            }
        };

        let uniform_policy = vec![1.0 / model.action_dim as f32; model.action_dim];

        // policy / value target：ROSMO 路径用现算值（不读 buffer 存量），否则读存量
        let target_policies: Vec<Vec<f32>> = match rosmo_t {
            Some(rt) => rt.policies.clone(),
            None => (0..=actual_k)
                .map(|i| {
                    if t + i < len {
                        steps[t + i].policy_target.clone()
                    } else {
                        uniform_policy.clone()
                    }
                })
                .collect(),
        };

        let target_values: Vec<f32> = if let Some(rt) = rosmo_t {
            rt.values.clone()
        } else if let Some(tm) = target_model {
            let bootstrap = |idx: usize| eval_target_bootstrap(tm, steps, idx, image_stack);
            n_step_targets_for_unroll(steps, t, actual_k, td_steps, gamma, Some(&bootstrap))
        } else {
            n_step_targets_for_unroll(steps, t, actual_k, td_steps, gamma, None)
        };

        let target_rewards: Vec<f32> = (0..actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].reward
                } else {
                    0.0
                }
            })
            .collect();
        let target_continuations: Vec<f32> = (0..actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].continuation
                } else {
                    0.0
                }
            })
            .collect();

        let actions: Vec<usize> = (0..actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].action[0] as usize
                } else {
                    0
                }
            })
            .collect();

        // value prefix：把单步 reward 转为累计前缀目标
        let final_rewards = if components.value_prefix {
            reward_prefix_targets(&target_rewards)
        } else {
            target_rewards
        };

        // consistency / reconstruction：收集 unroll 每步对应的真实 next_obs（否则留空）
        let next_obs: Vec<ObsSource<'_>> = if want_next_obs {
            (0..actual_k)
                .take_while(|&i| t + i + 1 < len)
                .map(|i| obs_at(t + i + 1))
                .collect()
        } else {
            Vec::new()
        };

        // POMDP-lite burn-in 上下文
        let (bi_obs, bi_actions, bi_leading, train_prev) = if components.recurrent_posterior {
            let burn_start = t.saturating_sub(burn_in_steps);
            let bi_obs: Vec<ObsSource<'_>> = (burn_start..t).map(&obs_at).collect();
            let bi_actions: Vec<usize> = (burn_start..t)
                .map(|p| steps[p].action[0] as usize)
                .collect();
            let leading = if burn_start > 0 {
                Some(steps[burn_start - 1].action[0] as usize)
            } else {
                None
            };
            let train_prev = if t > 0 {
                Some(steps[t - 1].action[0] as usize)
            } else {
                None
            };
            (bi_obs, bi_actions, leading, train_prev)
        } else {
            (vec![], vec![], None, None)
        };

        let key = (actual_k, next_obs.len());
        groups.entry(key).or_default().push(UnrollItem {
            obs_t: obs_at(t),
            actions,
            target_policies,
            target_values,
            target_rewards: final_rewards,
            target_continuations,
            next_obs,
            bc_weights: rosmo_t.map(|rt| rt.bc_weights.clone()).unwrap_or_default(),
            burn_in_obs: bi_obs,
            burn_in_actions: bi_actions,
            burn_in_leading_action: bi_leading,
            train_prev_action: train_prev,
        });
    }

    // 2) 每组一次 batch 前向 + backward（组 loss × G/batch_size 累积梯度），最后一次 step
    optimizer.zero_grad()?;
    let mut total_loss_val = 0.0;
    for items in groups.into_values() {
        let scale = items.len() as f32 / batch_size;
        let loss = model.train_unroll_batch(
            &items,
            consistency_coef,
            reconstruction_coef,
            components.continuation_coef,
            components.value_prefix,
            bc_coef,
        )?;
        total_loss_val += (&loss * scale).backward()?;
    }

    optimizer.step()?;
    Ok(total_loss_val)
}

/// 贪心 rollout 单局（原始未缩放 return + 步数）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn greedy_one_episode(
    env: &GymEnv,
    model: &MyZeroModel,
    adapter: &ActionAdapter,
    obs_adapter: &mut ObsAdapter,
    components: &Components,
    gamma: f32,
    num_simulations: u32,
    reset_seed: u64,
) -> (f32, usize) {
    use super::network::PrecomputedRootDynamics;

    let eval_cfg = my_zero_mcts_config(
        num_simulations,
        EVAL_TEMPERATURE,
        gamma,
        components,
        adapter.action_dim(),
        0.0,
    );
    let mut rng = StdRng::seed_from_u64(reset_seed);
    let (mut obs, _) = obs_adapter.reset(env, Some(reset_seed));
    let use_posterior = components.recurrent_posterior && model.posterior.is_some();
    let hidden_size = model.posterior.as_ref().map(|p| p.hidden_size).unwrap_or(0);
    let mut posterior_hidden = vec![0.0f32; hidden_size];
    let mut prev_action_idx: Option<usize> = None;
    let mut total_reward = 0.0f32;
    let mut length = 0usize;
    loop {
        let action_idx = if eval_cfg.num_simulations <= 1 {
            let policy = if use_posterior {
                let repr_latent = model.repr_inference(&obs);
                let mut prev_oh = vec![0.0f32; model.action_dim];
                if let Some(a) = prev_action_idx
                    && a < model.action_dim
                {
                    prev_oh[a] = 1.0;
                }
                let (posterior_latent, new_hidden) =
                    model.posterior_step_inference(&repr_latent, &prev_oh, &posterior_hidden);
                posterior_hidden = new_hidden;
                model.pred_inference(&posterior_latent).0
            } else {
                Dynamics::initial_state(&model, &obs).1
            };
            sample_from_probs(&policy, eval_cfg.temperature, &mut rng)
        } else {
            let candidates = adapter.candidates().to_vec();
            let policy = MyZeroSearchPolicy::from_components(components);

            let result = if use_posterior {
                let repr_latent = model.repr_inference(&obs);
                let mut prev_oh = vec![0.0f32; model.action_dim];
                if let Some(a) = prev_action_idx
                    && a < model.action_dim
                {
                    prev_oh[a] = 1.0;
                }
                let (posterior_latent, new_hidden) =
                    model.posterior_step_inference(&repr_latent, &prev_oh, &posterior_hidden);
                posterior_hidden = new_hidden;
                let (root_policy, root_value) = model.pred_inference(&posterior_latent);
                let root_dyn =
                    PrecomputedRootDynamics::new(model, posterior_latent, root_policy, root_value);
                let dyn_model = DynamicsModel::new(root_dyn, candidates, gamma);
                mcts_search(&dyn_model, &policy, &obs, &eval_cfg, &mut rng)
            } else {
                let dyn_model = DynamicsModel::new(model, candidates, gamma);
                mcts_search(&dyn_model, &policy, &obs, &eval_cfg, &mut rng)
            };
            result.recommended_id.index()
        };
        prev_action_idx = Some(action_idx);
        let env_action = adapter.to_env(action_idx);
        let (next_obs, _, reward, terminated, truncated) = obs_adapter.step(env, &env_action);
        total_reward += reward;
        length += 1;
        if terminated || truncated {
            break;
        }
        obs = next_obs;
    }
    (total_reward, length)
}

/// 贪心 eval：跑 `n_episodes` 局，返回均值与各局 return（**原始未缩放**）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn greedy_eval_episodes(
    env: &GymEnv,
    model: &MyZeroModel,
    adapter: &ActionAdapter,
    obs_adapter: &mut ObsAdapter,
    components: &Components,
    gamma: f32,
    n_episodes: usize,
    num_simulations: u32,
    eval_seed: u64,
) -> (f32, Vec<f32>) {
    let mut returns = Vec::with_capacity(n_episodes);
    for i in 0..n_episodes {
        let (r, _) = greedy_one_episode(
            env,
            model,
            adapter,
            obs_adapter,
            components,
            gamma,
            num_simulations,
            greedy_episode_seed(eval_seed, i as u64),
        );
        returns.push(r);
    }
    let mean = if n_episodes == 0 {
        0.0
    } else {
        returns.iter().sum::<f32>() / n_episodes as f32
    };
    (mean, returns)
}

/// 贪心 eval：temperature=0 跑若干局取均值（返回**原始未缩放** return）。
#[allow(dead_code, clippy::too_many_arguments)]
fn eval_episodes(
    env: &GymEnv,
    model: &MyZeroModel,
    adapter: &ActionAdapter,
    obs_adapter: &mut ObsAdapter,
    components: &Components,
    gamma: f32,
    n_episodes: usize,
    num_simulations: u32,
    eval_seed: u64,
) -> f32 {
    greedy_eval_episodes(
        env,
        model,
        adapter,
        obs_adapter,
        components,
        gamma,
        n_episodes,
        num_simulations,
        eval_seed,
    )
    .0
}

/// 单次训练运行的 benchmark 结果（多 seed 汇总用）
struct SeedSummary {
    seed: u64,
    wall_secs: f32,
    greedy_eval: f32,
    /// 首次 greedy eval ≥ solved 时的累计 env-step（None = 未达标）
    solved_at_steps: Option<u64>,
}

/// 取中位数（多 seed 汇总口径）
fn median_f32(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// 打印多 seed 汇总（中位数 = 稳定基线口径，排除单 seed 偶发 spike）
fn print_multiseed_summary(results: &[SeedSummary], solved: f32) {
    let n = results.len();
    let greedy: Vec<f32> = results.iter().map(|r| r.greedy_eval).collect();
    let walls: Vec<f32> = results.iter().map(|r| r.wall_secs).collect();
    let solved_steps: Vec<f32> = results
        .iter()
        .filter_map(|r| r.solved_at_steps.map(|s| s as f32))
        .collect();
    let n_solved = solved_steps.len();

    println!("\n--- 多 seed 汇总 ({n} seeds, 门槛 {solved}) ---");
    for r in results {
        let eff = r
            .solved_at_steps
            .map(|s| s.to_string())
            .unwrap_or_else(|| "未达标".to_string());
        println!(
            "  seed={:<3} greedy={:8.1} total_env_steps={:>8} wall={:.1}s",
            r.seed, r.greedy_eval, eff, r.wall_secs
        );
    }
    let steps_med = if n_solved > 0 {
        format!("{:.0}", median_f32(solved_steps))
    } else {
        "n/a".to_string()
    };
    println!(
        "  中位数 greedy={:.1} | total_env_steps={} ({}/{} 达标) | wall={:.1}s",
        median_f32(greedy),
        steps_med,
        n_solved,
        n,
        median_f32(walls),
    );
}

/// 物化空权重实例（冷启动推理前内部使用）。
pub(crate) fn materialize(
    py: Python<'_>,
    cfg: &MyZeroConfig,
    seed: u64,
) -> Result<MyZero, GraphError> {
    let env = GymEnv::new(py, cfg.env.env_id);
    let adapter = ActionAdapter::resolve(&env, cfg.env.action);
    let obs_spec = ObsAdapter::resolve(&env, cfg.env.observation, cfg.env.obs_mask.clone())
        .model_obs_spec(&env);
    let obs_symlog = obs_spec.uses_obs_symlog();
    let graph = Graph::new_with_seed(seed);
    let model = MyZeroModel::new_with_schemas(
        &graph,
        obs_spec,
        adapter.schema().clone(),
        cfg.model.latent_dim,
    )?
    .with_value_encoding(cfg.components.hl_gauss)
    .with_obs_symlog(obs_symlog)
    .with_recurrent_posterior(cfg.components.recurrent_posterior)?
    .with_stochastic(cfg.components.num_chance_outcomes)?;
    env.close();
    Ok(MyZero::from_parts(cfg.clone(), model, adapter))
}

/// 跑一次完整训练（固定 seed），返回训练后的 [`MyZero`] 与报告。
fn train_one_seed(
    py: Python<'_>,
    seed: u64,
    cfg: &MyZeroConfig,
    base_seed: u64,
) -> Result<(MyZero, TrainReport), GraphError> {
    let wall_t0 = std::time::Instant::now();

    if cfg.components.recurrent_posterior {
        cfg.train.validate_sequence_config();
    }

    let t = &cfg.train;
    let gamma = t.gamma;
    let smoke = cfg.eval.smoke;
    let solved = cfg.eval.solved;
    let profile = std::env::var("PROFILE").is_ok();
    crate::rl::profiling::reset();
    let mut prof_self_play = 0.0f32;
    let mut prof_batch_prepare = 0.0f32;
    let mut prof_train_step = 0.0f32;
    let mut prof_writeback = 0.0f32;
    let mut prof_eval = 0.0f32;

    // completedQ 改进策略目标的超参（开启 completed_q_target 时生效）
    let cq: Option<(f32, f32)> = if cfg.components.completed_q_target {
        Some((cfg.components.cq_c_visit, cfg.components.cq_c_scale))
    } else {
        None
    };

    let env = GymEnv::new(py, cfg.env.env_id);
    let adapter = ActionAdapter::resolve(&env, cfg.env.action);
    let mut obs_adapter = ObsAdapter::resolve(&env, cfg.env.observation, cfg.env.obs_mask.clone());
    let obs_spec = obs_adapter.model_obs_spec(&env);
    let obs_symlog = obs_spec.uses_obs_symlog();
    let obs_dim = obs_spec.dim();
    let action_dim = adapter.action_dim();
    let latent_dim = cfg.model.latent_dim;

    // 图像模式暂不支持 reanalyze（重搜需堆叠组装，recipe 默认关；接回时一并支持）
    if obs_adapter.image_stack().is_some() && cfg.components.reanalyze {
        return Err(GraphError::InvalidOperation(
            "MyZero: 图像 obs 暂不支持 reanalyze（待堆叠组装接回）".into(),
        ));
    }

    if seed == base_seed {
        let obs_desc = match obs_adapter.image_stack() {
            Some(k) => format!("obs=image→{obs_dim}(灰度84²×{k}帧)"),
            None => format!("obs={obs_dim}"),
        };
        println!(
            "[MyZero {}] {obs_desc} {} γ={gamma} acting_sims={} think_sims={}",
            cfg.env.env_id,
            adapter.describe(),
            t.acting_simulations,
            t.num_simulations,
        );
        if !cfg.components.gumbel
            && (cfg.components.sampled || sampled_would_subsample(action_dim, t.num_simulations))
        {
            let p = resolve_sampled_params(cfg.env.action, action_dim, t.num_simulations);
            println!("{}", format_sampled_log(&p, t.num_simulations));
        }
    }

    let graph = Graph::new_with_seed(seed);
    let model =
        MyZeroModel::new_with_schemas(&graph, obs_spec, adapter.schema().clone(), latent_dim)?
            .with_value_encoding(cfg.components.hl_gauss)
            .with_obs_symlog(obs_symlog)
            .with_recurrent_posterior(cfg.components.recurrent_posterior)?
            .with_stochastic(cfg.components.num_chance_outcomes)?;
    // 必须用独立 Graph：同一张图上参数名（如 Repr/fc1_W）不可重复。
    let target_model = if cfg.components.target_net {
        let target_graph = Graph::new_with_seed(seed);
        let tm = MyZeroModel::new_with_schemas(
            &target_graph,
            obs_spec,
            adapter.schema().clone(),
            latent_dim,
        )?
        .with_value_encoding(cfg.components.hl_gauss)
        .with_obs_symlog(obs_symlog)
        .with_recurrent_posterior(cfg.components.recurrent_posterior)?
        .with_stochastic(cfg.components.num_chance_outcomes)?;
        hard_update(&model.parameters(), &tm.parameters());
        Some(tm)
    } else {
        None
    };
    let target_cfg = TargetConfig::efficientzero_hard();
    let mut train_steps: u32 = 0;
    let mut optimizer = Adam::new(&graph, &model.parameters(), t.lr);
    let mut buffer: ReplayBuffer<SelfPlayGame> = ReplayBuffer::new(t.buffer_capacity);
    let mut rng = StdRng::seed_from_u64(seed);

    let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);
    let mut total_steps: u64 = 0;
    let mut hit_solved: Option<(usize, u64)> = None;
    let mut ckpt = BestTracker::new(
        &cfg.eval.checkpoint,
        seed,
        cfg.eval.seed_runs.max(1),
        obs_dim,
        smoke,
    );

    let max_episodes = if smoke { 3 } else { cfg.eval.max_episodes };
    let mut last_ep = 0usize;
    // KL 自适应 lr 乘子（跨局持久；默认关时恒 1.0、零额外前向/RNG，路径逐 bit 不变）
    let mut lr_mult = 1.0f32;
    // 训练期 peak greedy（含未落盘路径；供预算 / 平直判停）
    let mut peak_greedy = f32::NEG_INFINITY;
    let mut flat_stop_armed = cfg.eval.flat_stop_env_steps.is_some();

    for ep in 0..max_episodes {
        last_ep = ep + 1;
        let t0 = std::time::Instant::now();

        let temperature = SELF_PLAY_TEMPERATURE;

        let mcts_cfg = my_zero_mcts_config(
            t.acting_simulations,
            temperature,
            gamma,
            &cfg.components,
            action_dim,
            MctsConfig::default().root_exploration_fraction,
        );

        let sp_t0 = std::time::Instant::now();
        let steps = self_play_one_episode(
            &env,
            &model,
            &adapter,
            &mut obs_adapter,
            &mcts_cfg,
            &cfg.components,
            gamma,
            cq,
            // per-seed 派生：多 seed 时各 seed 拥有独立 env 初始状态流（统计独立性）
            self_play_episode_seed(seed, ep),
            &mut rng,
        );
        prof_self_play += sp_t0.elapsed().as_secs_f32();
        let ep_reward: f32 = steps.iter().map(|s| s.reward).sum();
        let ep_len = steps.len();
        total_steps += ep_len as u64;

        // KL 自适应 lr 探针：训练块前记当局旧 policy（默认关 = 空探针，零开销）
        let probe: Vec<Vec<f32>> = if t.kl_adaptive_lr {
            probe_obs_of(&steps, 8, obs_adapter.image_stack())
        } else {
            Vec::new()
        };
        let probe_old: Vec<Vec<f32>> = probe
            .iter()
            .map(|obs| Dynamics::initial_state(&&model, obs).1)
            .collect();

        buffer.push(SelfPlayGame {
            steps,
            outcome: GameOutcome::InProgress,
        });

        let mut avg_loss = 0.0;
        if buffer.len() >= t.start_training_after {
            let mut loss_sum = 0.0;
            let n_trains = if smoke { 1 } else { t.trains_per_episode };
            for _ in 0..n_trains {
                let prep_t0 = std::time::Instant::now();
                let batch = prepare_train_batch(
                    &buffer,
                    t.train_batch_size,
                    t.k_unroll,
                    t.td_steps,
                    cfg.components.reanalyze,
                    &cfg.components,
                    &model,
                    &adapter,
                    gamma,
                    t.num_simulations,
                    cq,
                    obs_adapter.image_stack(),
                    &mut rng,
                );
                prof_batch_prepare += prep_t0.elapsed().as_secs_f32();
                let train_view: Vec<(&SelfPlayGame, usize)> = match &batch {
                    PreparedBatch::Borrowed(items) | PreparedBatch::Refreshed(items, _) => items
                        .iter()
                        .map(|&(idx, start)| {
                            (buffer.get_ref(idx).expect("buffer 下标应有效"), start)
                        })
                        .collect(),
                    PreparedBatch::Owned(items) => {
                        items.iter().map(|item| (&item.game, item.start)).collect()
                    }
                };
                let refreshed = match &batch {
                    PreparedBatch::Refreshed(_, targets) => Some(targets.as_slice()),
                    _ => None,
                };
                let train_t0 = std::time::Instant::now();
                let l = train_batch(
                    &model,
                    &mut optimizer,
                    &train_view,
                    refreshed,
                    t.k_unroll,
                    t.td_steps,
                    gamma,
                    &cfg.components,
                    obs_adapter.image_stack(),
                    t.burn_in_steps,
                    target_model.as_ref(),
                )?;
                if let Some(tm) = target_model.as_ref() {
                    train_steps = train_steps.saturating_add(1);
                    sync_target(
                        &model.parameters(),
                        &tm.parameters(),
                        &target_cfg,
                        train_steps,
                    );
                }
                prof_train_step += train_t0.elapsed().as_secs_f32();
                drop(train_view);
                if let PreparedBatch::Owned(items) = batch {
                    let write_t0 = std::time::Instant::now();
                    writeback_reanalyzed_samples(&mut buffer, items);
                    prof_writeback += write_t0.elapsed().as_secs_f32();
                }
                loss_sum += l;
            }
            avg_loss = loss_sum / n_trains as f32;

            // KL 自适应 lr：训练块后测 policy 位移，调乘子并落回 optimizer
            if t.kl_adaptive_lr && !probe.is_empty() {
                let kl = probe_policy_kl(&model, &probe, &probe_old);
                lr_mult = kl_lr_multiplier(kl, KL_TARGET, lr_mult);
                optimizer.set_learning_rate(t.lr * lr_mult);
                // 诊断轨迹（组件默认关，仅消融/闸门可见；spike 排查 lr 死亡螺旋用）
                println!("  [kl-lr] ep={} kl={kl:.5} lr_mult={lr_mult:.3}", ep + 1);
            }

            if smoke {
                assert!(avg_loss.is_finite(), "SMOKE: loss={avg_loss} 非有限");
            }
        }

        ep_rewards.push_back(ep_reward);
        if ep_rewards.len() > 100 {
            ep_rewards.pop_front();
        }
        let avg_r = ep_rewards.iter().sum::<f32>() / ep_rewards.len() as f32;

        println!(
            "Ep {:4}: R={:8.1} len={:3} avg_R={:8.1} loss={:.4} temp={:.2} total_env_steps={} t={:.2}s",
            ep + 1,
            ep_reward,
            ep_len,
            avg_r,
            avg_loss,
            temperature,
            total_steps,
            t0.elapsed().as_secs_f32()
        );

        if !smoke && ep_rewards.len() >= 20 && (ep + 1) % cfg.eval.eval_every == 0 {
            let eval_t0 = std::time::Instant::now();
            let eval_r = eval_episodes(
                &env,
                &model,
                &adapter,
                &mut obs_adapter,
                &cfg.components,
                gamma,
                cfg.eval.eval_episodes,
                t.num_simulations,
                seed, // per-seed 派生（同 self-play）
            );
            prof_eval += eval_t0.elapsed().as_secs_f32();
            let recent: f32 = ep_rewards.iter().rev().take(20).sum::<f32>() / 20.0;
            println!(
                "  greedy eval {eval_r:.1} / {solved}（近20局 self-play {recent:.1}，total_env_steps={total_steps}）",
            );
            peak_greedy = peak_greedy.max(eval_r);
            ckpt.maybe_update(&model, cfg, eval_r, ep + 1, total_steps)?;
            if eval_r >= solved {
                hit_solved = Some((ep + 1, total_steps));
                println!("  ✅ 达标 ep={} total_env_steps={}", ep + 1, total_steps);
                break;
            }
        }

        // 预算：env-steps / wall-clock（与 max_episodes 取先到）
        if let Some(cap) = cfg.eval.max_env_steps
            && total_steps >= cap
        {
            println!(
                "  [budget] 达到 max_env_steps={cap}（ep={} total_env_steps={total_steps}）",
                ep + 1
            );
            break;
        }
        if let Some(cap) = cfg.eval.max_wall_secs {
            let elapsed = wall_t0.elapsed().as_secs_f32();
            if elapsed >= cap {
                println!(
                    "  [budget] 达到 max_wall_secs={cap}（ep={} wall={elapsed:.1}s total_env_steps={total_steps}）",
                    ep + 1
                );
                break;
            }
        }
        // 中间平直判停（预注册协议：过线 steps 后 peak 仍低于阈值）
        if flat_stop_armed
            && let (Some(cap), Some(min_g)) =
                (cfg.eval.flat_stop_env_steps, cfg.eval.flat_stop_min_greedy)
            && total_steps >= cap
        {
            let peak = if peak_greedy.is_finite() {
                peak_greedy
            } else {
                f32::NEG_INFINITY
            };
            if peak < min_g {
                println!(
                    "  [flat-stop] total_env_steps={total_steps}≥{cap} 且 peak_greedy={peak:.1}<{min_g}，提前结束"
                );
                break;
            }
            flat_stop_armed = false;
        }
    }

    let wall_secs = wall_t0.elapsed().as_secs_f32();
    let final_greedy = if smoke {
        0.0
    } else {
        let eval_t0 = std::time::Instant::now();
        let r = eval_episodes(
            &env,
            &model,
            &adapter,
            &mut obs_adapter,
            &cfg.components,
            gamma,
            cfg.eval.eval_episodes,
            t.num_simulations,
            seed, // per-seed 派生（同 self-play）
        );
        prof_eval += eval_t0.elapsed().as_secs_f32();
        r
    };

    if profile {
        let measured =
            prof_self_play + prof_batch_prepare + prof_train_step + prof_writeback + prof_eval;
        println!(
            "[PROFILE] self_play={:.2}s batch_prepare={:.2}s train_step={:.2}s writeback={:.2}s eval={:.2}s measured={:.2}s wall={:.2}s",
            prof_self_play,
            prof_batch_prepare,
            prof_train_step,
            prof_writeback,
            prof_eval,
            measured,
            wall_secs,
        );
        crate::rl::profiling::print_report();
    }

    if !smoke {
        ckpt.save_last(&model, cfg, last_ep, total_steps, final_greedy)?;
    }

    if !smoke {
        let eff = hit_solved
            .map(|(e, s)| format!("ep{e} total_env_steps={s}"))
            .unwrap_or_else(|| "未达标".to_string());
        println!(
            "📈 {} greedy={final_greedy:.1} | 门槛 {solved}: {eff} | {wall_secs:.1}s",
            cfg.env.env_id,
        );
    }

    env.close();

    let report = TrainReport {
        seed,
        wall_secs,
        final_greedy,
        solved_at_steps: hit_solved.map(|(_, s)| s),
        solved_threshold: solved,
        best_greedy: if ckpt.best_score().is_finite() {
            ckpt.best_score()
        } else {
            final_greedy
        },
        best_at_episode: ckpt.best_episode(),
        model_path: ckpt.model_path(),
    };
    let mz = MyZero::from_parts(cfg.clone(), model, adapter).with_train_report(report.clone());
    Ok((mz, report))
}

/// 多 seed 训练；返回**最后一 seed** 的权重实例。
pub(crate) fn train_all_seeds(cfg: MyZeroConfig) -> Result<MyZero, GraphError> {
    cfg.components.ensure_supported()?;
    print_components(&cfg.components);
    let smoke = cfg.eval.smoke;
    let n_runs = cfg.eval.seed_runs.max(1);
    let base_seed = cfg.eval.seed;

    Python::attach(|py| {
        let mut summaries = Vec::new();
        let mut last: Option<MyZero> = None;
        for i in 0..n_runs {
            let seed = base_seed.wrapping_add(i);
            if n_runs > 1 {
                println!("\n--- seed {seed} ({}/{n_runs}) ---", i + 1);
            }
            let (mz, report) = train_one_seed(py, seed, &cfg, base_seed)?;
            summaries.push(SeedSummary {
                seed: report.seed,
                wall_secs: report.wall_secs,
                greedy_eval: report.final_greedy,
                solved_at_steps: report.solved_at_steps,
            });
            last = Some(mz);
        }

        if n_runs > 1 && !smoke {
            print_multiseed_summary(&summaries, cfg.eval.solved);
        }

        if smoke {
            println!("[SMOKE] {} 通过", cfg.env.env_id);
        }
        last.ok_or_else(|| GraphError::InvalidOperation("MyZero: 无训练 seed".into()))
    })
}
