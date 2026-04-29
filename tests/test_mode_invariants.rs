//! Mode（Train / Inference）契约测试
//!
//! 覆盖：
//! - `Graph::train()` / `Graph::inference()` / `Graph::mode()` / `Graph::is_training()` 的语义；
//! - `Graph::inference_scope` 闭包式临时切换，必须在闭包退出时回滚到进入前的模式；
//! - `Inference` 模式下 `backward()` 必须返回硬错误，不再降级为警告；
//! - `Graph::load_model()` 默认进入 `Inference` 模式。
//!
//! 设计文档：`.doc/design/mode_design.md`

use only_torch::nn::{Graph, Linear, Mode};
use only_torch::tensor::Tensor;

#[test]
fn default_mode_is_train() {
    let graph = Graph::new();
    assert_eq!(graph.mode(), Mode::Train);
    assert!(graph.is_training());
}

#[test]
fn train_inference_switch_changes_mode_only() {
    let graph = Graph::new();

    graph.inference();
    assert_eq!(graph.mode(), Mode::Inference);
    assert!(!graph.is_training());

    graph.train();
    assert_eq!(graph.mode(), Mode::Train);
    assert!(graph.is_training());
}

#[test]
fn set_mode_explicit_value_round_trips() {
    let graph = Graph::new();

    graph.set_mode(Mode::Inference);
    assert_eq!(graph.mode(), Mode::Inference);

    graph.set_mode(Mode::Train);
    assert_eq!(graph.mode(), Mode::Train);
}

#[test]
fn inference_scope_temporarily_switches_and_restores() {
    let graph = Graph::new();
    assert_eq!(graph.mode(), Mode::Train);

    let result = graph.inference_scope(|g| {
        assert_eq!(g.mode(), Mode::Inference);
        assert!(!g.is_training());
        42
    });
    assert_eq!(result, 42);

    // 闭包退出后必须回滚到进入前的模式
    assert_eq!(graph.mode(), Mode::Train);
    assert!(graph.is_training());
}

#[test]
fn inference_scope_restores_even_from_inference_entry() {
    let graph = Graph::new();
    graph.inference();

    graph.inference_scope(|g| {
        assert_eq!(g.mode(), Mode::Inference);
    });

    // 进入前是 Inference，退出后仍应是 Inference
    assert_eq!(graph.mode(), Mode::Inference);
}

#[test]
fn inference_scope_restores_when_closure_panics() {
    let graph = Graph::new();
    assert_eq!(graph.mode(), Mode::Train);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        graph.inference_scope(|g| {
            assert_eq!(g.mode(), Mode::Inference);
            panic!("intentional panic inside inference_scope");
        });
    }));

    assert!(result.is_err());
    assert_eq!(graph.mode(), Mode::Train);
}

#[test]
fn mode_helpers_describe_caching_and_backward_capability() {
    assert!(Mode::Train.is_training());
    assert!(Mode::Train.caches_for_backward());
    assert!(Mode::Train.allows_backward());

    assert!(!Mode::Inference.is_training());
    assert!(!Mode::Inference.caches_for_backward());
    assert!(!Mode::Inference.allows_backward());
}

#[test]
fn inference_mode_rejects_backward_with_hard_error() {
    let graph = Graph::new();
    let x = graph.input(&Tensor::ones(&[1, 2])).unwrap();
    let linear = Linear::new(&graph, 2, 1, true, "fc").unwrap();
    let loss = linear.forward(&x);

    graph.forward(&loss).unwrap();

    graph.inference();
    let err = graph
        .backward(&loss)
        .expect_err("inference 模式下 backward 必须直接报错");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("inference"),
        "错误信息应明确指出 inference 模式: {msg}",
    );
}

#[test]
fn load_model_defaults_to_inference_mode() {
    let dir = std::path::PathBuf::from("target/test_mode_invariants");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("linear_{}", std::process::id()));
    let otm_path = path.with_extension("otm");
    let _ = std::fs::remove_file(&otm_path);

    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[1, 2])).unwrap();
    let linear = Linear::new(&graph, 2, 1, true, "fc").unwrap();
    let out = linear.forward(&x);
    graph.forward(&out).unwrap();
    graph.save_model(&path, &[&out]).unwrap();

    let loaded = Graph::load_model(&path).unwrap();
    assert_eq!(loaded.graph.mode(), Mode::Inference);
    assert!(!loaded.graph.is_training());

    let _ = std::fs::remove_file(&otm_path);
}
