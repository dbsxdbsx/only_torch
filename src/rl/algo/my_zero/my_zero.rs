//! MyZero 运行体：Graph + 模型 + 动作适配 + 各尾缀报告。

use super::action::ActionAdapter;
use super::config::{MyZeroConfig, greedy_episode_seed};
use super::model_io::load_weights_into;
use super::network::MyZeroModel;
use super::report::{EvalReport, RunReport, TrainReport};
use super::runner::{greedy_eval_episodes, greedy_one_episode, materialize};
use crate::nn::{Graph, GraphError};
use pyo3::Python;
use std::path::Path;

/// 训练 / 评测 / 部署链式运行体（权重在 [`MyZeroModel::graph`] 内）。
pub struct MyZero {
    pub(crate) cfg: MyZeroConfig,
    pub(crate) model: MyZeroModel,
    pub(crate) adapter: ActionAdapter,
    train_report: Option<TrainReport>,
    eval_report: Option<EvalReport>,
    run_report: Option<RunReport>,
}

impl MyZero {
    /// 唯一入口：声明 Gymnasium 环境 ID（如 `"CartPole-v1"`）。
    pub fn new(env_id: &'static str) -> super::builder::MyZeroBuilder {
        assert!(!env_id.is_empty(), "MyZero: env_id 不能为空");
        let mut cfg = MyZeroConfig::default();
        cfg.env.env_id = env_id;
        cfg.components = super::recipe::components_for(env_id);
        super::builder::MyZeroBuilder {
            cfg,
            solved_set: false,
            max_episodes_set: false,
        }
    }

    pub(crate) fn from_parts(
        cfg: MyZeroConfig,
        model: MyZeroModel,
        adapter: ActionAdapter,
    ) -> Self {
        Self {
            cfg,
            model,
            adapter,
            train_report: None,
            eval_report: None,
            run_report: None,
        }
    }

    pub(crate) fn with_train_report(mut self, report: TrainReport) -> Self {
        self.train_report = Some(report);
        self
    }

    fn graph(&self) -> &Graph {
        &self.model.graph
    }

    pub fn train_report(&self) -> Option<&TrainReport> {
        self.train_report.as_ref()
    }

    pub fn eval_report(&self) -> Option<&EvalReport> {
        self.eval_report.as_ref()
    }

    pub fn run_report(&self) -> Option<&RunReport> {
        self.run_report.as_ref()
    }

    /// 若 `path.otm` 存在则加载权重；否则保持当前权重不变。
    ///
    /// `path` 为基名（不含 `.otm` 后缀），**不能为空**。
    pub fn load_model_if_exists(self, path: impl AsRef<Path>) -> Result<Self, GraphError> {
        let base = path.as_ref();
        if base.as_os_str().is_empty() {
            return Err(GraphError::InvalidOperation(
                "MyZero::load_model_if_exists: 路径不能为空".into(),
            ));
        }
        let otm = base.with_extension("otm");
        if otm.is_file() {
            load_weights_into(self.graph(), &self.cfg, base)?;
            println!("[MyZero] 已加载模型 {}", otm.display());
        }
        Ok(self)
    }

    /// 独立 `n` 局 greedy 评测（原始未缩放 return）。
    pub fn eval(mut self, n: usize) -> Result<Self, GraphError> {
        let gamma = self.cfg.train.gamma;
        let num_simulations = self.cfg.train.num_simulations;
        let base_seed = self.cfg.eval.seed;
        let env_id = self.cfg.env.env_id;
        Python::attach(|py| {
            let env = crate::rl::GymEnv::new(py, env_id);
            self.graph().inference();
            let (mean, returns) = greedy_eval_episodes(
                &env,
                &self.model,
                &self.adapter,
                gamma,
                n,
                num_simulations,
                base_seed,
            );
            env.close();
            println!("[MyZero] eval {env_id} ×{n}: mean={mean:.1}");
            self.eval_report = Some(EvalReport {
                n_episodes: n,
                mean_return: mean,
                episode_returns: returns,
            });
            Ok(self)
        })
    }

    /// 在已配置 env 里贪心 rollout（部署 / 演示）。
    ///
    /// - `Some(n)`：玩 `n` 局后返回（`n=0` 报错）
    /// - `None`：无限循环，每局结束 `reset` 再开；须 Ctrl+C 中断
    pub fn run(mut self, episodes: Option<usize>) -> Result<Self, GraphError> {
        if matches!(episodes, Some(0)) {
            return Err(GraphError::InvalidOperation(
                "MyZero::run: 局数须为 Some(n>0) 或 None（无限）".into(),
            ));
        }
        let gamma = self.cfg.train.gamma;
        let num_simulations = self.cfg.train.num_simulations;
        let base_seed = self.cfg.eval.seed;
        let env_id = self.cfg.env.env_id;
        Python::attach(|py| {
            let env = crate::rl::GymEnv::new(py, env_id);
            self.graph().inference();
            let mut returns = Vec::new();
            let mut lengths = Vec::new();
            let mut i = 0u64;
            loop {
                if let Some(n) = episodes
                    && i as usize >= n
                {
                    break;
                }
                let (ret, len) = greedy_one_episode(
                    &env,
                    &self.model,
                    &self.adapter,
                    gamma,
                    num_simulations,
                    greedy_episode_seed(base_seed, i),
                );
                let ep_no = i as usize + 1;
                if episodes.is_none() {
                    println!("[MyZero] run {env_id} #{ep_no}: return={ret:.1} len={len}");
                }
                returns.push(ret);
                lengths.push(len);
                i += 1;
            }
            env.close();
            let completed = returns.len();
            let mean = if completed == 0 {
                0.0
            } else {
                returns.iter().sum::<f32>() / completed as f32
            };
            if let Some(n) = episodes {
                println!("[MyZero] run {env_id} ×{n}: mean={mean:.1} returns={returns:?}",);
            }
            self.run_report = Some(RunReport {
                episodes_requested: episodes,
                episodes_completed: completed,
                episode_returns: returns,
                episode_lengths: lengths,
                mean_return: mean,
            });
            Ok(self)
        })
    }

    /// 从配置物化空权重实例（冷启动推理前内部使用）。
    pub(crate) fn materialize_from_cfg(cfg: &MyZeroConfig, seed: u64) -> Result<Self, GraphError> {
        Python::attach(|py| materialize(py, cfg, seed))
    }
}
