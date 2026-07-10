//! MyZero `.otm` 持久化（契约写入 metadata；用户见 `save_model_when_eval` / `load_model_if_exists`）。

use super::config::{ActionPlan, MyZeroConfig};
use super::network::MyZeroModel;
use crate::nn::{
    Graph, GraphError, OTM_FORMAT_VERSION, OtmMetadata, Var, apply_params_to_graph, read_otm_file,
    write_otm_file,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
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

/// MyZero 各载体共用的 OTM 写入内核。
///
/// 拓扑与权重来自同一个 [`MyZeroModel`]；环境/recipe 差异由调用方提供的契约承载。
/// 单智能体与棋盘只应实现各自契约适配器，不重复文件格式 I/O。
pub(crate) fn save_model_with_contract<T: Serialize>(
    model: &MyZeroModel,
    obs_dim: usize,
    path: &Path,
    model_name: &str,
    contract: &T,
) -> Result<(), GraphError> {
    let outputs = model.otm_output_vars(obs_dim)?;
    let output_refs: Vec<&Var> = outputs.iter().collect();
    let desc = Var::vars_to_graph_descriptor(&output_refs, model_name);
    let contract_json = serde_json::to_value(contract)
        .map_err(|e| GraphError::ComputationError(format!("序列化 MyZero 契约失败: {e}")))?;
    let metadata = OtmMetadata {
        format_version: OTM_FORMAT_VERSION,
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        model_name: model_name.to_string(),
        graph: desc,
        evolution: None,
        myzero: Some(contract_json),
    };
    let params = model.graph.inner().get_all_parameters();
    write_otm_file(path, &metadata, &params)
}

/// MyZero 各载体共用的 OTM 读取内核：先解析并校验载体契约，再应用权重。
pub(crate) fn load_model_with_contract<T, F>(
    graph: &Graph,
    path: &Path,
    expected_model_name: &str,
    verify: F,
) -> Result<(), GraphError>
where
    T: DeserializeOwned,
    F: FnOnce(&T) -> Result<(), GraphError>,
{
    let (metadata, params) = read_otm_file(path)?;
    if metadata.model_name != expected_model_name {
        return Err(GraphError::InvalidOperation(format!(
            "MyZero 模型类型不匹配：文件 {}，期望 {expected_model_name}",
            metadata.model_name
        )));
    }
    let value = metadata
        .myzero
        .as_ref()
        .ok_or_else(|| GraphError::InvalidOperation("该 .otm 缺少 MyZero 契约字段".into()))?;
    let contract: T = serde_json::from_value(value.clone())
        .map_err(|e| GraphError::ComputationError(format!("解析 MyZero 契约失败: {e}")))?;
    verify(&contract)?;
    apply_params_to_graph(graph, &params)?;
    Ok(())
}

/// 将 MyZero 模型写入 `.otm`（`path` 不含后缀，库自动加 `.otm`）。
pub(crate) fn save_myzero_model(
    model: &MyZeroModel,
    cfg: &MyZeroConfig,
    obs_dim: usize,
    path: &Path,
) -> Result<(), GraphError> {
    let contract = contract_from_cfg(cfg);
    save_model_with_contract(model, obs_dim, path, "myzero", &contract)
}

/// 从 `.otm` 加载权重到已物化的图，并校验与 `cfg` 的契约一致。
pub(crate) fn load_weights_into(
    graph: &Graph,
    cfg: &MyZeroConfig,
    path: &Path,
) -> Result<(), GraphError> {
    load_model_with_contract::<MyZeroOtmContract, _>(graph, path, "myzero", |contract| {
        verify_contract(contract, cfg)
    })
}
