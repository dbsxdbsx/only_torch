use only_torch::nn::{ExecutionContext, Graph, Linear};
use only_torch::tensor::Tensor;

#[test]
fn train_eval_do_not_toggle_grad_enabled() {
    let graph = Graph::new();

    assert_eq!(graph.execution_ctx(), ExecutionContext::training());

    graph.eval();
    assert!(!graph.training());
    assert!(graph.is_grad_enabled(), "eval 只改变层行为，不关闭 grad");

    graph.train();
    assert!(graph.training());
    assert!(graph.is_grad_enabled(), "train 也不负责重新打开 grad");
}

#[test]
fn no_grad_scope_only_temporarily_disables_grad() {
    let graph = Graph::new();

    graph.no_grad_scope(|g| {
        assert!(g.training());
        assert!(!g.is_grad_enabled());
    });
    assert!(graph.training());
    assert!(graph.is_grad_enabled());

    graph.eval();
    graph.no_grad_scope(|g| {
        assert!(!g.training());
        assert!(!g.is_grad_enabled());
    });
    assert!(!graph.training());
    assert!(graph.is_grad_enabled());
}

#[test]
fn mode_switch_inside_no_grad_preserves_training_change_but_restores_grad() {
    let graph = Graph::new();

    graph.no_grad_scope(|g| {
        assert!(g.training());
        assert!(!g.is_grad_enabled());

        g.eval();
        assert!(!g.training());
        assert!(!g.is_grad_enabled());
    });

    assert!(
        !graph.training(),
        "no_grad 不回滚闭包内显式 train/eval 切换"
    );
    assert!(graph.is_grad_enabled(), "no_grad 只恢复进入前的 grad 开关");
}

#[test]
fn load_model_defaults_to_inference_context() {
    let dir = std::path::PathBuf::from("target/test_execution_context_invariants");
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
    assert_eq!(loaded.graph.execution_ctx(), ExecutionContext::inference());

    let _ = std::fs::remove_file(&otm_path);
}
