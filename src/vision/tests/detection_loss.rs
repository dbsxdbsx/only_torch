use approx::assert_abs_diff_eq;

use crate::nn::Graph;
use crate::tensor::Tensor;
use crate::vision::detection::{DetectionLossComponents, DetectionLossWeights};

#[test]
fn test_weighted_total_combines_existing_components() {
    let graph = Graph::new();
    let bbox = graph.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();
    let objectness = graph.input(&Tensor::new(&[3.0], &[1, 1])).unwrap();
    let class = graph.input(&Tensor::new(&[5.0], &[1, 1])).unwrap();

    let total = DetectionLossComponents::from_required(bbox, objectness, class)
        .weighted_total(DetectionLossWeights::new(1.0, 2.0, 0.5))
        .unwrap();
    total.forward().unwrap();

    assert_abs_diff_eq!(total.item().unwrap(), 10.5, epsilon = 1e-6);
}

#[test]
fn test_weighted_total_skips_missing_and_zero_weight_components() {
    let graph = Graph::new();
    let bbox = graph.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();
    let class = graph.input(&Tensor::new(&[5.0], &[1, 1])).unwrap();

    let total = DetectionLossComponents::new(Some(bbox), None, Some(class))
        .weighted_total(DetectionLossWeights::new(0.0, 10.0, 0.5))
        .unwrap();
    total.forward().unwrap();

    assert_abs_diff_eq!(total.item().unwrap(), 2.5, epsilon = 1e-6);
}

#[test]
fn test_weighted_total_rejects_all_zero_weights() {
    let graph = Graph::new();
    let bbox = graph.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();

    let err = match DetectionLossComponents::new(Some(bbox), None, None)
        .weighted_total(DetectionLossWeights::new(0.0, 0.0, 0.0))
    {
        Ok(_) => panic!("全 0 权重 detection loss 组合应返回错误"),
        Err(err) => err,
    };

    assert!(
        format!("{err}").contains("权重非零"),
        "错误信息应说明缺少非零权重组件，实际: {err}"
    );
}

#[test]
fn test_weighted_total_rejects_empty_components() {
    let err = match DetectionLossComponents::new(None, None, None)
        .weighted_total(DetectionLossWeights::default())
    {
        Ok(_) => panic!("空 detection loss 组合应返回错误"),
        Err(err) => err,
    };

    assert!(
        format!("{err}").contains("至少需要一个"),
        "错误信息应说明缺少有效 loss 组件，实际: {err}"
    );
}
