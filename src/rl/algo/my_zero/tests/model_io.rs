//! `model_io.rs` `.otm` 持久化测试：save/load roundtrip、契约校验

use super::super::config::MyZeroConfig;
use super::super::model_io::{
    MyZeroOtmContract, contract_from_cfg, load_weights_into, save_myzero_model, verify_contract,
};
use super::super::network::MyZeroModel;
use crate::nn::{Graph, GraphError, read_otm_file};
use crate::rl::mcts::Dynamics;
use std::env::temp_dir;

fn dummy_model(obs_dim: usize, action_dim: usize) -> MyZeroModel {
    dummy_model_seed(obs_dim, action_dim, 0)
}

fn dummy_model_seed(obs_dim: usize, action_dim: usize, seed: u64) -> MyZeroModel {
    let graph = Graph::new_with_seed(seed);
    MyZeroModel::new(&graph, obs_dim, action_dim, 64).unwrap()
}

#[test]
fn save_and_load_roundtrip() {
    let base = temp_dir().join("myzero_otm_roundtrip");
    let _ = std::fs::remove_file(base.with_extension("otm"));
    let cfg = MyZeroConfig::default();
    let model = dummy_model(4, 2);
    save_myzero_model(&model, &cfg, 4, &base).unwrap();
    let graph2 = Graph::new_with_seed(1);
    let model2 = MyZeroModel::new(&graph2, 4, 2, 64).unwrap();
    load_weights_into(&model2.graph, &cfg, &base).unwrap();
    let _ = std::fs::remove_file(base.with_extension("otm"));
}

#[test]
fn env_mismatch_is_error() {
    let base = temp_dir().join("myzero_otm_mismatch");
    let _ = std::fs::remove_file(base.with_extension("otm"));
    let cfg = MyZeroConfig::default();
    let model = dummy_model(4, 2);
    save_myzero_model(&model, &cfg, 4, &base).unwrap();

    let mut other = cfg.clone();
    other.env.env_id = "Pendulum-v1";
    let graph2 = Graph::new_with_seed(1);
    let model2 = MyZeroModel::new(&graph2, 4, 2, 64).unwrap();
    let err = load_weights_into(&model2.graph, &other, &base).unwrap_err();
    assert!(
        matches!(err, GraphError::InvalidOperation(_)),
        "expected InvalidOperation, got {err:?}"
    );
    let _ = std::fs::remove_file(base.with_extension("otm"));
}

#[test]
fn contract_roundtrip() {
    let cfg = MyZeroConfig::default();
    let c = contract_from_cfg(&cfg);
    verify_contract(&c, &cfg).unwrap();
}

#[test]
fn legacy_contract_without_observation_field_remains_readable() {
    let legacy = serde_json::json!({
        "schema_version": 1,
        "env_id": "CartPole-v1",
        "action": "auto",
        "reward_scale": 1.0,
        "latent_dim": 64
    });
    let contract: MyZeroOtmContract = serde_json::from_value(legacy).unwrap();
    assert_eq!(contract.observation, None);
    verify_contract(&contract, &MyZeroConfig::default()).unwrap();
}

#[test]
fn legacy_otm_without_observation_field_loads_all_weights() {
    let current = temp_dir().join("myzero_otm_current_contract");
    let legacy = temp_dir().join("myzero_otm_legacy_contract");
    let _ = std::fs::remove_file(current.with_extension("otm"));
    let _ = std::fs::remove_file(legacy.with_extension("otm"));

    let cfg = MyZeroConfig::default();
    let source = dummy_model(4, 2);
    save_myzero_model(&source, &cfg, 4, &current).unwrap();
    let (mut metadata, _) = read_otm_file(&current).unwrap();
    metadata
        .myzero
        .as_mut()
        .and_then(serde_json::Value::as_object_mut)
        .expect("MyZero contract 应为 JSON object")
        .remove("observation");
    let current_file = current.with_extension("otm");
    let bytes = std::fs::read(&current_file).unwrap();
    let old_json_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let weight_tail = &bytes[12 + old_json_len..];
    let json = serde_json::to_vec_pretty(&metadata).unwrap();
    let mut legacy_bytes = Vec::with_capacity(12 + json.len() + weight_tail.len());
    legacy_bytes.extend_from_slice(&bytes[..8]);
    legacy_bytes.extend_from_slice(&(json.len() as u32).to_le_bytes());
    legacy_bytes.extend_from_slice(&json);
    legacy_bytes.extend_from_slice(weight_tail);
    std::fs::write(legacy.with_extension("otm"), legacy_bytes).unwrap();

    let target = dummy_model_seed(4, 2, 1);
    load_weights_into(&target.graph, &cfg, &legacy).unwrap();
    let obs = [0.1, -0.2, 0.3, 0.4];
    let expected = Dynamics::initial_state(&&source, &obs);
    let actual = Dynamics::initial_state(&&target, &obs);
    assert_eq!(actual.0, expected.0, "legacy OTM 应完整恢复 repr 权重");
    assert_eq!(actual.1, expected.1, "legacy OTM 应完整恢复 policy 权重");
    assert_eq!(actual.2, expected.2, "legacy OTM 应完整恢复 value 权重");

    let _ = std::fs::remove_file(current.with_extension("otm"));
    let _ = std::fs::remove_file(legacy.with_extension("otm"));
}

#[test]
#[ignore = "manual: 读取本机历史 CartPole best.otm（制品被 gitignore，不作为 CI fixture）"]
fn real_legacy_cartpole_artifact_loads_and_changes_predictions() {
    let base = std::path::Path::new("models/my_zero/CartPole-v1/best");
    assert!(
        base.with_extension("otm").exists(),
        "缺少历史制品 models/my_zero/CartPole-v1/best.otm"
    );
    let cfg = MyZeroConfig::default();
    let target = dummy_model_seed(4, 2, 0xC0FFEE);
    let obs = [0.1, -0.2, 0.3, 0.4];
    let before = Dynamics::initial_state(&&target, &obs);
    load_weights_into(&target.graph, &cfg, base).unwrap();
    let after = Dynamics::initial_state(&&target, &obs);
    assert_ne!(after.0, before.0, "加载历史制品后 repr 输出应改变");
    assert_ne!(after.1, before.1, "加载历史制品后 policy 输出应改变");
    assert!(after.2.is_finite());
}
