//! 棋盘 MyZero 的 OTM 契约适配器。
//!
//! 文件格式、拓扑与权重 I/O 复用 [`super::model_io`] 通用内核；本模块只负责把
//! [`BoardTrainConfig`](super::board::BoardTrainConfig) 映射成可校验的棋盘契约。

use super::board::BoardTrainConfig;
use super::model_io::{load_model_with_contract, save_model_with_contract};
use super::network::MyZeroModel;
use crate::nn::GraphError;
use serde::{Deserialize, Serialize};
use std::path::Path;

const BOARD_OTM_SCHEMA_VERSION: u32 = 1;
const BOARD_MODEL_NAME: &str = "myzero-board";

/// 棋盘 MyZero 交付模型契约；既锁架构，也保留产生该权重的关键 recipe。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct BoardOtmContract {
    schema_version: u32,
    env_id: String,
    obs_dim: usize,
    action_dim: usize,
    latent_dim: usize,
    cnn_repr: bool,
    training_seed: u64,
    max_episodes: usize,
    num_simulations: u32,
    augment: bool,
    true_rules_tree: bool,
    reconstruction_coef: Option<f32>,
    consistency_coef: Option<f32>,
    per: bool,
}

fn contract_from_cfg(
    cfg: &BoardTrainConfig,
    obs_dim: usize,
    action_dim: usize,
) -> BoardOtmContract {
    BoardOtmContract {
        schema_version: BOARD_OTM_SCHEMA_VERSION,
        env_id: cfg.env_id.to_string(),
        obs_dim,
        action_dim,
        latent_dim: cfg.latent_dim,
        cnn_repr: cfg.cnn_repr,
        training_seed: cfg.seed,
        max_episodes: cfg.max_episodes,
        num_simulations: cfg.num_simulations,
        augment: cfg.augment,
        true_rules_tree: cfg.true_rules_tree,
        reconstruction_coef: cfg
            .components
            .reconstruction
            .then_some(cfg.components.reconstruction_coef),
        consistency_coef: cfg
            .components
            .consistency
            .then_some(cfg.components.consistency_coef),
        per: cfg.per,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn verify_contract(
    file: &BoardOtmContract,
    cfg: &BoardTrainConfig,
    obs_dim: usize,
    action_dim: usize,
) -> Result<(), GraphError> {
    let expected = contract_from_cfg(cfg, obs_dim, action_dim);
    if file != &expected {
        return Err(GraphError::InvalidOperation(format!(
            "棋盘 MyZero 模型契约不匹配：文件 {file:?}，期望 {expected:?}"
        )));
    }
    Ok(())
}

/// 保存棋盘 MyZero final 权重与可校验 recipe 契约。
pub(crate) fn save_board_model(
    model: &MyZeroModel,
    cfg: &BoardTrainConfig,
    obs_dim: usize,
    action_dim: usize,
    path: &Path,
) -> Result<(), GraphError> {
    let contract = contract_from_cfg(cfg, obs_dim, action_dim);
    save_model_with_contract(model, obs_dim, path, BOARD_MODEL_NAME, &contract)
}

/// 将棋盘 `.otm` 权重加载进已按同一配置物化的模型，并先校验完整契约。
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn load_board_weights_into(
    model: &MyZeroModel,
    cfg: &BoardTrainConfig,
    obs_dim: usize,
    action_dim: usize,
    path: &Path,
) -> Result<(), GraphError> {
    load_model_with_contract::<BoardOtmContract, _>(
        &model.graph,
        path,
        BOARD_MODEL_NAME,
        |contract| verify_contract(contract, cfg, obs_dim, action_dim),
    )
}
