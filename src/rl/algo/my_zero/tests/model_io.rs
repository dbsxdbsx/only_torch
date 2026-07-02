//! `model_io.rs` `.otm` 持久化测试：save/load roundtrip、契约校验

use super::super::config::MyZeroConfig;
use super::super::model_io::{
    contract_from_cfg, load_weights_into, save_myzero_model, verify_contract,
};
use super::super::network::MyZeroModel;
use crate::nn::{Graph, GraphError};
use std::env::temp_dir;

fn dummy_model(obs_dim: usize, action_dim: usize) -> MyZeroModel {
    let graph = Graph::new_with_seed(0);
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
