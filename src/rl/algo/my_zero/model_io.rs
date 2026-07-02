//! MyZero `.otm` 持久化（契约写入 metadata；用户见 `save_model_when_eval` / `load_model_if_exists`）。

use super::config::{ActionPlan, MyZeroConfig};
use super::network::MyZeroModel;
use crate::nn::{
    Graph, GraphError, OTM_FORMAT_VERSION, OtmMetadata, Var, apply_params_to_graph, read_otm_file,
    write_otm_file,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// `.otm` 内嵌的 MyZero 运行契约（`OtmMetadata.myzero` 字段）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct MyZeroOtmContract {
    pub schema_version: u32,
    pub env_id: String,
    pub action: String,
    pub reward_scale: f32,
    pub latent_dim: usize,
}

const SCHEMA_VERSION: u32 = 1;

fn action_to_str(action: ActionPlan) -> String {
    match action {
        ActionPlan::Auto => "auto".to_string(),
        ActionPlan::Discretize { buckets } => format!("discretize:{buckets}"),
    }
}

pub(crate) fn contract_from_cfg(cfg: &MyZeroConfig) -> MyZeroOtmContract {
    MyZeroOtmContract {
        schema_version: SCHEMA_VERSION,
        env_id: cfg.env.env_id.to_string(),
        action: action_to_str(cfg.env.action),
        reward_scale: cfg.env.reward_scale,
        latent_dim: cfg.model.latent_dim,
    }
}

fn parse_contract(metadata: &OtmMetadata) -> Result<MyZeroOtmContract, GraphError> {
    let value = metadata.myzero.as_ref().ok_or_else(|| {
        GraphError::InvalidOperation("该 .otm 不是 MyZero 模型（缺少 myzero 契约字段）".into())
    })?;
    serde_json::from_value(value.clone())
        .map_err(|e| GraphError::ComputationError(format!("解析 MyZero 契约失败: {e}")))
}

pub(crate) fn verify_contract(
    file: &MyZeroOtmContract,
    cfg: &MyZeroConfig,
) -> Result<(), GraphError> {
    if file.schema_version != SCHEMA_VERSION {
        return Err(GraphError::InvalidOperation(format!(
            "不支持的 MyZero 契约版本: {}",
            file.schema_version
        )));
    }
    if file.env_id != cfg.env.env_id {
        return Err(GraphError::InvalidOperation(format!(
            "模型环境 {} 与声明 {} 不一致",
            file.env_id, cfg.env.env_id
        )));
    }
    let expected_action = action_to_str(cfg.env.action);
    if file.action != expected_action {
        return Err(GraphError::InvalidOperation(format!(
            "模型动作方案 {} 与声明 {expected_action} 不一致",
            file.action
        )));
    }
    if (file.reward_scale - cfg.env.reward_scale).abs() > 1e-5 {
        return Err(GraphError::InvalidOperation(format!(
            "模型 reward_scale={} 与声明 {} 不一致",
            file.reward_scale, cfg.env.reward_scale
        )));
    }
    if file.latent_dim != cfg.model.latent_dim {
        return Err(GraphError::InvalidOperation(format!(
            "模型 latent_dim={} 与声明 {} 不一致",
            file.latent_dim, cfg.model.latent_dim
        )));
    }
    Ok(())
}

/// 将 MyZero 模型写入 `.otm`（`path` 不含后缀，库自动加 `.otm`）。
pub(crate) fn save_myzero_model(
    model: &MyZeroModel,
    cfg: &MyZeroConfig,
    obs_dim: usize,
    path: &Path,
) -> Result<(), GraphError> {
    let outputs = model.otm_output_vars(obs_dim)?;
    let output_refs: Vec<&Var> = outputs.iter().collect();
    let desc = Var::vars_to_graph_descriptor(&output_refs, "myzero");
    let contract = contract_from_cfg(cfg);
    let myzero_json = serde_json::to_value(&contract)
        .map_err(|e| GraphError::ComputationError(format!("序列化 MyZero 契约失败: {e}")))?;
    let metadata = OtmMetadata {
        format_version: OTM_FORMAT_VERSION,
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        model_name: "myzero".to_string(),
        graph: desc,
        evolution: None,
        myzero: Some(myzero_json),
    };
    let params = model.graph.inner().get_all_parameters();
    write_otm_file(path, &metadata, &params)
}

/// 从 `.otm` 加载权重到已物化的图，并校验与 `cfg` 的契约一致。
pub(crate) fn load_weights_into(
    graph: &Graph,
    cfg: &MyZeroConfig,
    path: &Path,
) -> Result<(), GraphError> {
    let (metadata, params) = read_otm_file(path)?;
    let contract = parse_contract(&metadata)?;
    verify_contract(&contract, cfg)?;
    apply_params_to_graph(graph, &params)?;
    Ok(())
}
