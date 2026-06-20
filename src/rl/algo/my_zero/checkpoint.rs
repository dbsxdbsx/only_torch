//! 训练期 best 模型落盘（仅在 periodic greedy eval 创新高时写入 `.otm`）。

use super::config::{CheckpointSettings, MyZeroConfig};
use super::model_io::save_myzero_model;
use super::network::MyZeroModel;
use crate::nn::GraphError;
use std::fs;
use std::path::{Path, PathBuf};

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

/// 多 seed 时在用户给定路径的父目录下插入 `seed_{seed}/`，单 seed 则原样使用。
fn resolve_best_base(base: &Path, seed: u64, seed_runs: u64) -> PathBuf {
    if seed_runs <= 1 {
        return base.to_path_buf();
    }
    let parent = base.parent().unwrap_or_else(|| Path::new("."));
    let name = base
        .file_name()
        .map(|s| s.to_owned())
        .unwrap_or_else(|| std::ffi::OsString::from("best"));
    parent.join(format!("seed_{seed}")).join(name)
}

impl BestTracker {
    pub fn new(
        checkpoint: &CheckpointSettings,
        seed: u64,
        seed_runs: u64,
        obs_dim: usize,
        smoke: bool,
    ) -> Self {
        let enabled = checkpoint.enabled && checkpoint.best_base.is_some() && !smoke;
        let best_base = checkpoint
            .best_base
            .as_ref()
            .map(|b| resolve_best_base(b, seed, seed_runs))
            .unwrap_or_default();
        let seed_dir = best_base
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
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
            "[MyZero] best 模型 greedy={score:.1} ep={episode} total_env_steps={env_steps} -> {}.otm",
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

    fn enabled_checkpoint(base: PathBuf) -> CheckpointSettings {
        CheckpointSettings {
            enabled: true,
            best_base: Some(base),
            min_delta: 0.0,
            save_last: false,
        }
    }

    #[test]
    fn should_update_first_eval_always() {
        let t = BestTracker::new(
            &enabled_checkpoint(PathBuf::from("/tmp/x/best")),
            42,
            1,
            4,
            false,
        );
        assert!(t.should_update(9.4));
    }

    #[test]
    fn min_delta_zero_requires_strict_improvement_or_equal() {
        let mut t = BestTracker::new(
            &enabled_checkpoint(PathBuf::from("/tmp/x/best")),
            42,
            1,
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
                enabled: true,
                best_base: Some(PathBuf::from("/tmp/x/best")),
                min_delta: 1.0,
                save_last: false,
            },
            42,
            1,
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
            &enabled_checkpoint(PathBuf::from("/tmp/x/best")),
            42,
            1,
            4,
            true,
        );
        assert!(!t.should_update(500.0));
    }

    #[test]
    fn disabled_by_default() {
        let t = BestTracker::new(&CheckpointSettings::default(), 7, 1, 4, false);
        assert!(!t.should_update(500.0));
        assert!(t.model_path().is_none());
    }

    #[test]
    fn uses_explicit_path_single_seed() {
        let base = PathBuf::from("/tmp/my_cartpole_best");
        let t = BestTracker::new(&enabled_checkpoint(base.clone()), 42, 1, 4, false);
        assert!(t.should_update(500.0));
        assert_eq!(t.best_base, base);
    }

    #[test]
    fn multi_seed_inserts_seed_subdir() {
        let t = BestTracker::new(
            &enabled_checkpoint(PathBuf::from("models/foo/best")),
            43,
            3,
            4,
            false,
        );
        assert_eq!(t.best_base, PathBuf::from("models/foo/seed_43/best"));
    }
}
