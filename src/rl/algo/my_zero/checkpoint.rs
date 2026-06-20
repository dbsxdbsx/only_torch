//! 训练期 best 模型落盘（挂在 periodic greedy eval 上，写 `.otm`）。

use super::config::{CheckpointSettings, MyZeroConfig};
use super::model_io::save_myzero_model;
use super::network::MyZeroModel;
use crate::nn::GraphError;
use std::fs;
use std::path::PathBuf;

/// 训练期追踪 greedy eval 最高分并按需写入 best `.otm`。
pub(crate) struct BestTracker {
    enabled: bool,
    min_delta: f32,
    save_last: bool,
    obs_dim: usize,
    seed_dir: PathBuf,
    best_base: PathBuf,
    best_score: f32,
    best_episode: usize,
    best_steps: u64,
}

impl BestTracker {
    pub fn new(
        checkpoint: &CheckpointSettings,
        env_id: &str,
        seed: u64,
        obs_dim: usize,
        smoke: bool,
    ) -> Self {
        let dir = checkpoint.dir.clone().or_else(|| {
            if smoke {
                None
            } else {
                Some(super::model_io::default_model_dir(env_id))
            }
        });
        let enabled = dir.is_some() && !smoke;
        let seed_dir = dir
            .as_ref()
            .map(|d| d.join(format!("seed_{seed}")))
            .unwrap_or_default();
        let best_base = seed_dir.join("best");
        Self {
            enabled,
            min_delta: checkpoint.min_delta,
            save_last: checkpoint.save_last,
            obs_dim,
            seed_dir,
            best_base,
            best_score: f32::NEG_INFINITY,
            best_episode: 0,
            best_steps: 0,
        }
    }

    pub fn best_score(&self) -> f32 {
        self.best_score
    }

    pub fn best_episode(&self) -> Option<usize> {
        if self.best_score.is_finite() {
            Some(self.best_episode)
        } else {
            None
        }
    }

    pub fn model_path(&self) -> Option<PathBuf> {
        if self.enabled && self.best_score.is_finite() {
            Some(self.best_base.clone())
        } else {
            None
        }
    }

    /// 是否相对历史 best 有提升（`score >= best + min_delta`；首次 eval 恒为真）。
    pub fn should_update(&self, score: f32) -> bool {
        if !self.enabled {
            return false;
        }
        if !self.best_score.is_finite() {
            return true;
        }
        score >= self.best_score + self.min_delta
    }

    /// periodic greedy eval 后调用；创新高则覆盖写入 `best.otm`。
    pub fn maybe_update(
        &mut self,
        model: &MyZeroModel,
        cfg: &MyZeroConfig,
        score: f32,
        episode: usize,
        env_steps: u64,
    ) -> Result<bool, GraphError> {
        if !self.should_update(score) {
            return Ok(false);
        }
        if let Some(parent) = self.best_base.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| GraphError::ComputationError(format!("创建模型目录失败: {e}")))?;
        }
        save_myzero_model(model, cfg, self.obs_dim, &self.best_base)?;
        println!(
            "[MyZero] best 模型 greedy={score:.1} ep={episode} steps={env_steps} -> {}.otm",
            self.best_base.display()
        );
        self.best_score = score;
        self.best_episode = episode;
        self.best_steps = env_steps;
        Ok(true)
    }

    /// 可选：额外写入当前权重到 `last.otm`（不参与 best 比较）。
    pub fn save_last(
        &self,
        model: &MyZeroModel,
        cfg: &MyZeroConfig,
        episode: usize,
        _env_steps: u64,
        score: f32,
    ) -> Result<(), GraphError> {
        if !self.enabled || !self.save_last {
            return Ok(());
        }
        fs::create_dir_all(&self.seed_dir)
            .map_err(|e| GraphError::ComputationError(format!("创建模型目录失败: {e}")))?;
        let last_base = self.seed_dir.join("last");
        save_myzero_model(model, cfg, self.obs_dim, &last_base)?;
        println!(
            "[MyZero] last 模型 greedy={score:.1} ep={episode} -> {}.otm",
            last_base.display()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::CheckpointSettings;
    use super::*;

    #[test]
    fn should_update_first_eval_always() {
        let t = BestTracker::new(
            &CheckpointSettings {
                dir: Some(PathBuf::from("/tmp/x")),
                min_delta: 0.0,
                save_last: false,
            },
            "CartPole-v1",
            42,
            4,
            false,
        );
        assert!(t.should_update(9.4));
    }

    #[test]
    fn min_delta_zero_requires_strict_improvement_or_equal() {
        let mut t = BestTracker::new(
            &CheckpointSettings {
                dir: Some(PathBuf::from("/tmp/x")),
                min_delta: 0.0,
                save_last: false,
            },
            "CartPole-v1",
            42,
            4,
            false,
        );
        t.best_score = 100.0;
        assert!(t.should_update(100.0));
        assert!(!t.should_update(99.9));
        assert!(t.should_update(100.1));
    }

    #[test]
    fn min_delta_one_uses_gte_threshold() {
        let mut t = BestTracker::new(
            &CheckpointSettings {
                dir: Some(PathBuf::from("/tmp/x")),
                min_delta: 1.0,
                save_last: false,
            },
            "CartPole-v1",
            42,
            4,
            false,
        );
        t.best_score = 499.0;
        assert!(t.should_update(500.0));
        assert!(!t.should_update(499.5));
    }

    #[test]
    fn disabled_when_smoke() {
        let t = BestTracker::new(
            &CheckpointSettings {
                dir: Some(PathBuf::from("/tmp/x")),
                min_delta: 0.0,
                save_last: false,
            },
            "CartPole-v1",
            42,
            4,
            true,
        );
        assert!(!t.should_update(500.0));
    }

    #[test]
    fn default_dir_when_not_smoke_and_dir_none() {
        let t = BestTracker::new(&CheckpointSettings::default(), "CartPole-v1", 7, 4, false);
        assert!(t.should_update(500.0));
        assert_eq!(
            t.best_base,
            PathBuf::from("models/my_zero/CartPole-v1/seed_7/best")
        );
    }
}
