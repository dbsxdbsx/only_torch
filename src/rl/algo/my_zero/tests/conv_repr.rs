//! CNN representation 编码器单测（v0.26 Phase 1）：shape / 归一化 / batch 等价 / 反传 / 序列化往返。

use crate::nn::{Adam, Graph, Module, Optimizer, VarLossOps};
use crate::rl::algo::my_zero::network::{ConvRepresentationNet, MyZeroModel, ObsSpec};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

const SIDE: usize = 42; // 测试用 42（栈 3 层，比 84 快）
const CH: usize = 4;
const LATENT: usize = 32;

fn fake_obs(len: usize, seed: u64) -> Vec<f32> {
    // 简单 LCG 造确定性假图像（[0,1]）
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

/// 前向输出 [B, latent] 且经 min-max 归一化（每行 min=0、max=1）
#[test]
fn conv_repr_forward_shape_and_normalization() {
    let graph = Graph::new_with_seed(42);
    let net = ConvRepresentationNet::new(&graph, CH, SIDE, LATENT).unwrap();

    let obs = fake_obs(CH * SIDE * SIDE, 7);
    let latent = net
        .forward(Tensor::new(&obs, &[1, CH * SIDE * SIDE]))
        .unwrap();
    latent.forward().unwrap();
    let v = latent.value().unwrap().unwrap();
    assert_eq!(v.shape(), &[1, LATENT]);

    let data = v.data_as_slice();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert_abs_diff_eq!(min, 0.0, epsilon = 1e-4);
    assert_abs_diff_eq!(max, 1.0, epsilon = 1e-3);
}

/// batch=2 前向与两次 batch=1 前向逐元素一致（conv 路径 batch 语义正确）
#[test]
fn conv_repr_batch_matches_single() {
    let graph = Graph::new_with_seed(42);
    let net = ConvRepresentationNet::new(&graph, CH, SIDE, LATENT).unwrap();
    let dim = CH * SIDE * SIDE;

    let o1 = fake_obs(dim, 1);
    let o2 = fake_obs(dim, 2);

    let single = |obs: &[f32]| -> Vec<f32> {
        let latent = net.forward(Tensor::new(obs, &[1, dim])).unwrap();
        latent.forward().unwrap();
        latent.value().unwrap().unwrap().data_as_slice().to_vec()
    };
    let v1 = single(&o1);
    let v2 = single(&o2);

    let mut both = o1.clone();
    both.extend_from_slice(&o2);
    let latent_b = net.forward(Tensor::new(&both, &[2, dim])).unwrap();
    latent_b.forward().unwrap();
    let vb = latent_b.value().unwrap().unwrap();
    assert_eq!(vb.shape(), &[2, LATENT]);
    let vb = vb.data_as_slice();

    for i in 0..LATENT {
        assert_abs_diff_eq!(vb[i], v1[i], epsilon = 1e-5);
        assert_abs_diff_eq!(vb[LATENT + i], v2[i], epsilon = 1e-5);
    }
}

/// 反传后 conv 参数确实被更新（梯度流通到卷积核）
#[test]
fn conv_repr_backward_updates_conv_params() {
    let graph = Graph::new_with_seed(42);
    let net = ConvRepresentationNet::new(&graph, CH, SIDE, LATENT).unwrap();
    let params = net.parameters();
    let before: Vec<Tensor> = params
        .iter()
        .map(|p| p.value().unwrap().unwrap().clone())
        .collect();

    let mut opt = Adam::new(&graph, &params, 0.01);
    let obs = fake_obs(CH * SIDE * SIDE, 3);
    let latent = net
        .forward(Tensor::new(&obs, &[1, CH * SIDE * SIDE]))
        .unwrap();
    let target = Tensor::new(vec![0.5f32; LATENT], &[1, LATENT]);
    let loss = latent.mse_loss(&target).unwrap();
    opt.zero_grad().unwrap();
    let lv = loss.backward().unwrap();
    assert!(lv.is_finite());
    opt.step().unwrap();

    let mut changed = 0;
    for (p, b) in params.iter().zip(&before) {
        let after = p.value().unwrap().unwrap();
        if after
            .data_as_slice()
            .iter()
            .zip(b.data_as_slice())
            .any(|(x, y)| (x - y).abs() > 1e-9)
        {
            changed += 1;
        }
    }
    assert_eq!(
        changed,
        params.len(),
        "全部 conv/fc 参数都应被更新，仅 {changed}/{} 变化",
        params.len()
    );
}

/// MyZeroModel Image spec：train_unroll_batch 一次完整前向反向可跑通
#[test]
fn my_zero_model_image_spec_trains() {
    use crate::rl::algo::my_zero::network::UnrollItem;

    let graph = Graph::new_with_seed(42);
    let spec = ObsSpec::Image {
        channels: CH,
        side: SIDE,
    };
    let action_dim = 3;
    let model = MyZeroModel::new_with_spec(&graph, spec, action_dim, LATENT).unwrap();
    let mut opt = Adam::new(&graph, &model.parameters(), 0.01);

    let dim = spec.dim();
    let item = UnrollItem {
        obs_t: fake_obs(dim, 11).into(),
        actions: vec![1, 2],
        target_policies: vec![vec![0.5, 0.3, 0.2]; 3],
        target_values: vec![1.0, 0.8, 0.5],
        target_rewards: vec![1.0, 0.0],
        target_continuations: vec![1.0, 1.0],
        next_obs: vec![fake_obs(dim, 12).into(), fake_obs(dim, 13).into()],
        bc_weights: Vec::new(),
    };
    let loss = model
        .train_unroll_batch(&[item], 2.0, 0.0, 1.0, false, 0.0)
        .unwrap();
    opt.zero_grad().unwrap();
    let lv = loss.backward().unwrap();
    assert!(lv.is_finite(), "图像模型 loss 应有限，got {lv}");
    opt.step().unwrap();
}

/// Image spec 的搜索期推理（initial_state + recurrent）产出合法输出
#[test]
fn my_zero_model_image_spec_inference() {
    use crate::rl::mcts::{ActionId, ActionPayload, Dynamics};

    let graph = Graph::new_with_seed(42);
    let spec = ObsSpec::Image {
        channels: CH,
        side: SIDE,
    };
    let model = MyZeroModel::new_with_spec(&graph, spec, 3, LATENT).unwrap();

    let obs = fake_obs(spec.dim(), 21);
    let (latent, policy, value) = Dynamics::initial_state(&&model, &obs);
    assert_eq!(latent.len(), LATENT);
    assert_eq!(policy.len(), 3);
    assert!(value.is_finite());
    assert_abs_diff_eq!(policy.iter().sum::<f32>(), 1.0, epsilon = 1e-4);

    let out =
        Dynamics::recurrent_with_id(&&model, &latent, ActionId(1), &ActionPayload::Discrete(1));
    assert_eq!(out.next_state.len(), LATENT);
    assert!(out.reward.is_finite() && out.value.is_finite());
}
