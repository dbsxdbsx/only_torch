//! checkpoint 旁路 manifest（与 `graph.save_weights` 的 `.bin` 配对）。

use super::config::{ActionPlan, MyZeroConfig};
use crate::nn::GraphError;
use std::fs;
use std::path::{Path, PathBuf};

const MANIFEST_SUFFIX: &str = ".myzero.json";
const META_SUFFIX: &str = ".meta.json";

fn manifest_path(weights_base: &Path) -> PathBuf {
    PathBuf::from(format!("{}{MANIFEST_SUFFIX}", weights_base.display()))
}

/// 写入 manifest（`path` 为不含 `.bin` 的权重基名，与 [`Graph::save_weights`] 一致）。
pub fn write_manifest(weights_base: &Path, cfg: &MyZeroConfig) -> Result<(), GraphError> {
    let action = match cfg.env.action {
        ActionPlan::Auto => "auto".to_string(),
        ActionPlan::Discretize { buckets } => format!("discretize:{buckets}"),
    };
    let body = format!(
        "env_id={}\naction={}\nreward_scale={}\nlatent_dim={}\n",
        cfg.env.env_id, action, cfg.env.reward_scale, cfg.model.latent_dim,
    );
    fs::write(manifest_path(weights_base), body)
        .map_err(|e| GraphError::ComputationError(format!("写入 manifest 失败: {e}")))
}

/// 读取并校验 manifest 与当前 [`MyZeroConfig`] 契约一致。
pub fn verify_manifest(weights_base: &Path, cfg: &MyZeroConfig) -> Result<(), GraphError> {
    let path = manifest_path(weights_base);
    let text = fs::read_to_string(&path).map_err(|e| {
        GraphError::ComputationError(format!("读取 manifest 失败（{}）: {e}", path.display()))
    })?;
    let mut env_id = None;
    let mut action = None;
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k.trim() {
            "env_id" => env_id = Some(v.trim()),
            "action" => action = Some(v.trim()),
            _ => {}
        }
    }
    let file_env =
        env_id.ok_or_else(|| GraphError::InvalidOperation("manifest 缺少 env_id".into()))?;
    if file_env != cfg.env.env_id {
        return Err(GraphError::InvalidOperation(format!(
            "manifest env_id={file_env} 与配置 {} 不一致",
            cfg.env.env_id
        )));
    }
    let file_action = action.unwrap_or("auto");
    let cfg_action = match cfg.env.action {
        ActionPlan::Auto => "auto",
        ActionPlan::Discretize { buckets } => {
            let expected = format!("discretize:{buckets}");
            if file_action != expected {
                return Err(GraphError::InvalidOperation(format!(
                    "manifest action={file_action} 与配置 {expected} 不一致"
                )));
            }
            return Ok(());
        }
    };
    if file_action != cfg_action {
        return Err(GraphError::InvalidOperation(format!(
            "manifest action={file_action} 与配置 {cfg_action} 不一致"
        )));
    }
    Ok(())
}

fn meta_path(weights_base: &Path) -> PathBuf {
    PathBuf::from(format!("{}{META_SUFFIX}", weights_base.display()))
}

/// 写入 checkpoint 性能元数据（与权重基名配对，如 `best.meta.json`）。
pub fn write_checkpoint_meta(
    weights_base: &Path,
    episode: usize,
    env_steps: u64,
    greedy_mean: f32,
    eval_episodes: usize,
    seed: u64,
) -> Result<(), GraphError> {
    let body = format!(
        "episode={episode}\nenv_steps={env_steps}\ngreedy_mean={greedy_mean}\neval_episodes={eval_episodes}\nseed={seed}\n",
    );
    fs::write(meta_path(weights_base), body)
        .map_err(|e| GraphError::ComputationError(format!("写入 checkpoint meta 失败: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn write_and_verify_manifest_roundtrip() {
        let base = temp_dir().join("myzero_manifest_test");
        let _ = fs::remove_file(manifest_path(&base));
        let cfg = MyZeroConfig::default();
        write_manifest(&base, &cfg).unwrap();
        verify_manifest(&base, &cfg).unwrap();
        let _ = fs::remove_file(manifest_path(&base));
    }
}
