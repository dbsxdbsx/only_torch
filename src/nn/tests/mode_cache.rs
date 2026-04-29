use crate::nn::{
    Graph, GraphError, GraphInner, Init, Mode, NodeInner, VarActivationOps, VarLossOps,
};
use crate::tensor::Tensor;
use std::rc::Rc;

#[test]
fn inference_backward_rejects_before_ensure_forward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let x = graph.parameter(&[2, 2], Init::Ones, "x")?;
    let target = graph.input(&Tensor::zeros(&[2, 2]))?;
    let loss = x.square().mse_loss(&target)?;

    graph.inference();
    let err = match loss.backward() {
        Ok(_) => panic!("Inference 模式下 backward 必须在 ensure-forward 前报错"),
        Err(err) => err,
    };
    assert!(
        format!("{err:?}").contains("inference 模式不允许 backward"),
        "错误信息应明确拒绝 Inference backward: {err:?}",
    );
    assert!(
        loss.node().value().is_none(),
        "Inference backward 被拒绝后不应先触发 ensure-forward",
    );

    Ok(())
}

fn assert_inference_forward_skips_cache(
    name: &str,
    input: Tensor,
    build: impl FnOnce(&mut GraphInner, Rc<NodeInner>) -> Result<Rc<NodeInner>, GraphError>,
) -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();
    let input_node = inner
        .borrow_mut()
        .create_basic_input_node(input.shape(), Some("x"))?;
    input_node.set_value(Some(&input))?;

    let op = {
        let mut g = inner.borrow_mut();
        build(&mut g, Rc::clone(&input_node))?
    };

    op.forward_recursive(1, Mode::Inference)?;
    let value_shape = op
        .value()
        .expect("Inference forward 仍应正常产出节点值")
        .shape()
        .to_vec();
    let upstream = Tensor::ones(&value_shape);

    let err = match op.calc_grad_to_parent_index(0, &upstream) {
        Ok(_) => panic!("{name} 在 Inference forward 后不应保留 backward 缓存"),
        Err(err) => err,
    };
    match err {
        GraphError::BackwardCacheMissing { node, cache } => {
            assert!(!node.is_empty(), "{name} 应返回缺失缓存所属节点");
            assert!(!cache.is_empty(), "{name} 应返回缺失缓存名称");
        }
        other => panic!("{name} 应返回结构化 BackwardCacheMissing 错误，实际: {other:?}"),
    }

    Ok(())
}

#[test]
fn inference_forward_skips_softmax_cache() -> Result<(), GraphError> {
    assert_inference_forward_skips_cache(
        "Softmax",
        Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]),
        |g, x| g.create_softmax_node(x, Some("softmax")),
    )
}

#[test]
fn inference_forward_skips_log_softmax_cache() -> Result<(), GraphError> {
    assert_inference_forward_skips_cache(
        "LogSoftmax",
        Tensor::new(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]),
        |g, x| g.create_log_softmax_node(x, Some("log_softmax")),
    )
}

#[test]
fn inference_forward_skips_layer_norm_cache() -> Result<(), GraphError> {
    assert_inference_forward_skips_cache(
        "LayerNormOp",
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], &[2, 4]),
        |g, x| g.create_layer_norm_op_node(x, 1, 1e-5, Some("layer_norm")),
    )
}

#[test]
fn inference_forward_skips_rms_norm_cache() -> Result<(), GraphError> {
    assert_inference_forward_skips_cache(
        "RMSNormOp",
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], &[2, 4]),
        |g, x| g.create_rms_norm_op_node(x, 1, 1e-5, Some("rms_norm")),
    )
}

#[test]
fn inference_forward_skips_abs_cache() -> Result<(), GraphError> {
    assert_inference_forward_skips_cache(
        "Abs",
        Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]),
        |g, x| g.create_abs_node(x, Some("abs")),
    )
}

#[test]
fn inference_forward_skips_square_cache() -> Result<(), GraphError> {
    assert_inference_forward_skips_cache(
        "Square",
        Tensor::new(&[0.5, -1.0, 2.0, 3.0], &[2, 2]),
        |g, x| g.create_square_node(x, Some("square")),
    )
}

#[test]
fn inference_forward_skips_pow_cache() -> Result<(), GraphError> {
    assert_inference_forward_skips_cache(
        "Pow",
        Tensor::new(&[0.5, 1.0, 2.0, 3.0], &[2, 2]),
        |g, x| g.create_pow_node(x, 2.5, Some("pow")),
    )
}

#[test]
fn inference_forward_skips_clip_cache() -> Result<(), GraphError> {
    assert_inference_forward_skips_cache(
        "Clip",
        Tensor::new(&[-1.0, 0.5, 2.0, 3.0], &[2, 2]),
        |g, x| g.create_clip_node(x, 0.0, 2.0, Some("clip")),
    )
}

#[test]
fn inference_forward_skips_reciprocal_cache() -> Result<(), GraphError> {
    assert_inference_forward_skips_cache(
        "Reciprocal",
        Tensor::new(&[0.5, 1.0, 2.0, 3.0], &[2, 2]),
        |g, x| g.create_reciprocal_node(x, Some("reciprocal")),
    )
}

#[test]
fn inference_forward_skips_ln_cache() -> Result<(), GraphError> {
    assert_inference_forward_skips_cache(
        "Ln",
        Tensor::new(&[0.5, 1.0, 2.0, 3.0], &[2, 2]),
        |g, x| g.create_ln_node(x, Some("ln")),
    )
}

#[test]
fn inference_forward_skips_log2_cache() -> Result<(), GraphError> {
    assert_inference_forward_skips_cache(
        "Log2",
        Tensor::new(&[0.5, 1.0, 2.0, 3.0], &[2, 2]),
        |g, x| g.create_log2_node(x, Some("log2")),
    )
}

#[test]
fn inference_forward_skips_log10_cache() -> Result<(), GraphError> {
    assert_inference_forward_skips_cache(
        "Log10",
        Tensor::new(&[0.5, 1.0, 2.0, 3.0], &[2, 2]),
        |g, x| g.create_log10_node(x, Some("log10")),
    )
}
