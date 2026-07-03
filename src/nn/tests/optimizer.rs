/*
 * @Description  : Optimizer 测试
 *
 * 测试 PyTorch 风格的 Optimizer API
 *
 * 测试覆盖：
 * - SGD/Adam 基本训练流程
 * - minimize() 一步完成
 * - learning_rate() / set_learning_rate() / reset()
 * - params() 参数列表查询
 * - Adam 状态查询：get_momentum() / get_velocity() / timestep()
 */

use crate::nn::graph::Graph;
use crate::nn::layer::Linear;
use crate::nn::{Adam, Module, Optimizer, SGD, VarActivationOps, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;

// ==================== 融合原地更新 vs 参考实现（逐 bit 金测试）====================
//
// SGD/Adam 的 step 已改为单趟 Zip 融合原地更新（消除 grad/value clone 与临时张量链）。
// 融合不允许改变任何逐元素浮点运算顺序——这里用旧张量表达式链作为参考实现，
// 在随机数据上多步对比，必须逐 bit 相等。

/// 旧 SGD 更新的参考实现：`new = current - lr * grad`
fn sgd_reference(current: &Tensor, grad: &Tensor, lr: f32) -> Tensor {
    current - lr * grad
}

/// 旧 Adam 更新的参考实现（与优化历史版本的张量表达式链完全一致）
#[allow(clippy::too_many_arguments)]
fn adam_reference(
    current: &Tensor,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step_size: f32,
    bc2: f32,
) -> Tensor {
    *m *= beta1;
    *m += &(grad * (1.0 - beta1));

    let mut grad_sq = grad * grad;
    grad_sq *= 1.0 - beta2;
    *v *= beta2;
    *v += &grad_sq;

    let mut denom = (&*v / bc2).sqrt();
    denom += epsilon;
    let update = &*m / &denom;
    current - step_size * &update
}

/// SGD 融合原地更新必须与参考实现逐 bit 一致（多步）
#[test]
fn test_sgd_step_inplace_bitwise_matches_reference() {
    let lr = 0.037f32;
    let mut fused = Tensor::normal_seeded(0.0, 1.0, &[4, 7], 11);
    let mut reference = fused.clone();

    for step in 0..5 {
        let grad = Tensor::normal_seeded(0.0, 0.5, &[4, 7], 100 + step);

        fused.sgd_step_inplace(&grad, lr);
        reference = sgd_reference(&reference, &grad, lr);

        let f = fused.data_as_slice();
        let r = reference.data_as_slice();
        for i in 0..f.len() {
            assert_eq!(
                f[i].to_bits(),
                r[i].to_bits(),
                "SGD 融合与参考实现不逐 bit 一致：step={step} i={i} fused={} ref={}",
                f[i],
                r[i]
            );
        }
    }
}

/// Adam 融合原地更新必须与参考实现逐 bit 一致（多步、含 m/v 状态演化）
#[test]
fn test_adam_step_inplace_bitwise_matches_reference() {
    let (beta1, beta2, epsilon, lr) = (0.9f32, 0.999f32, 1e-8f32, 0.003f32);

    let mut fused_param = Tensor::normal_seeded(0.0, 1.0, &[3, 5], 21);
    let mut ref_param = fused_param.clone();
    let mut fused_m = Tensor::zeros(&[3, 5]);
    let mut fused_v = Tensor::zeros(&[3, 5]);
    let mut ref_m = Tensor::zeros(&[3, 5]);
    let mut ref_v = Tensor::zeros(&[3, 5]);

    for t in 1..=8 {
        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);
        let step_size = lr / bc1;
        let grad = Tensor::normal_seeded(0.0, 0.7, &[3, 5], 200 + t as u64);

        fused_param.adam_step_inplace(
            &grad,
            &mut fused_m,
            &mut fused_v,
            beta1,
            beta2,
            epsilon,
            step_size,
            bc2,
        );
        ref_param = adam_reference(
            &ref_param, &grad, &mut ref_m, &mut ref_v, beta1, beta2, epsilon, step_size, bc2,
        );

        let f = fused_param.data_as_slice();
        let r = ref_param.data_as_slice();
        for i in 0..f.len() {
            assert_eq!(
                f[i].to_bits(),
                r[i].to_bits(),
                "Adam 融合与参考实现不逐 bit 一致：t={t} i={i} fused={} ref={}",
                f[i],
                r[i]
            );
        }
        // m/v 状态本身也应逐 bit 一致
        let (fm, rm) = (fused_m.data_as_slice(), ref_m.data_as_slice());
        let (fv, rv) = (fused_v.data_as_slice(), ref_v.data_as_slice());
        for i in 0..fm.len() {
            assert_eq!(fm[i].to_bits(), rm[i].to_bits(), "m 状态漂移：t={t} i={i}");
            assert_eq!(fv[i].to_bits(), rv[i].to_bits(), "v 状态漂移：t={t} i={i}");
        }
    }
}

/// Adam 融合更新对非连续梯度（permute 视图产物）也必须按逻辑序正确
///
/// 优化器融合走 ndarray `Zip`（stride 感知）；本测试用转置视图梯度对比
/// 「先物化连续再更新」的结果，确保非连续布局不会静默算错。
#[test]
fn test_adam_step_inplace_noncontiguous_grad() {
    let (beta1, beta2, epsilon, step_size, bc2) = (0.9f32, 0.999f32, 1e-8f32, 0.01f32, 0.5f32);

    // 非连续梯度：transpose 产生的置换视图（逻辑形状 [2,3]）
    let base = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let grad_noncontig = base.transpose(); // [2,3]，非连续
    assert!(
        !grad_noncontig.is_contiguous(),
        "前置条件：梯度应为非连续布局"
    );
    let grad_contig = Tensor::new(grad_noncontig.to_vec(), &[2, 3]); // 逻辑序物化

    let init = Tensor::new(&[0.5, -0.5, 1.0, -1.0, 2.0, -2.0], &[2, 3]);

    let mut p1 = init.clone();
    let (mut m1, mut v1) = (Tensor::zeros(&[2, 3]), Tensor::zeros(&[2, 3]));
    p1.adam_step_inplace(
        &grad_noncontig,
        &mut m1,
        &mut v1,
        beta1,
        beta2,
        epsilon,
        step_size,
        bc2,
    );

    let mut p2 = init.clone();
    let (mut m2, mut v2) = (Tensor::zeros(&[2, 3]), Tensor::zeros(&[2, 3]));
    p2.adam_step_inplace(
        &grad_contig,
        &mut m2,
        &mut v2,
        beta1,
        beta2,
        epsilon,
        step_size,
        bc2,
    );

    let (a, b) = (p1.to_vec(), p2.to_vec());
    for i in 0..a.len() {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "非连续梯度与连续物化结果不一致：i={i}"
        );
    }
}

#[test]
fn test_sgd_basic() {
    let graph = Graph::new_with_seed(42);

    // 简单线性模型：y = x * w
    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let w = graph
        .parameter(&[2, 1], crate::nn::Init::Constant(0.5), "w")
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    let y = x.matmul(&w).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    // 初始 loss
    loss.forward().unwrap();
    let initial_loss = loss.item().unwrap();

    // 创建 SGD 优化器
    let mut optimizer = SGD::new(&graph, std::slice::from_ref(&w), 0.1);

    // 训练一步
    optimizer.zero_grad().unwrap();
    loss.backward().unwrap();
    optimizer.step().unwrap();

    // 重新计算 loss
    // 需要重新构建计算图，因为 w 已更新
    let y2 = x.matmul(&w).unwrap();
    let loss2 = y2.mse_loss(&target).unwrap();
    loss2.forward().unwrap();
    let new_loss = loss2.item().unwrap();

    // loss 应该下降
    assert!(
        new_loss < initial_loss,
        "Loss 应该下降: {} -> {}",
        initial_loss,
        new_loss
    );
}

#[test]
fn test_sgd_minimize() {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let w = graph
        .parameter(&[2, 1], crate::nn::Init::Constant(0.5), "w")
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    let y = x.matmul(&w).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    let mut optimizer = SGD::new(&graph, std::slice::from_ref(&w), 0.1);

    // 使用 minimize
    let loss_val = optimizer.minimize(&loss).unwrap();
    assert!(loss_val > 0.0, "Loss 应该是正数");
}

#[test]
fn test_adam_basic() {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let w = graph
        .parameter(&[2, 1], crate::nn::Init::Constant(0.5), "w")
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    let y = x.matmul(&w).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    // 初始 loss
    loss.forward().unwrap();
    let initial_loss = loss.item().unwrap();

    // 创建 Adam 优化器
    let mut optimizer = Adam::new(&graph, std::slice::from_ref(&w), 0.1);

    // 训练一步
    optimizer.zero_grad().unwrap();
    loss.backward().unwrap();
    optimizer.step().unwrap();

    // 重新计算 loss
    let y2 = x.matmul(&w).unwrap();
    let loss2 = y2.mse_loss(&target).unwrap();
    loss2.forward().unwrap();
    let new_loss = loss2.item().unwrap();

    // loss 应该下降
    assert!(
        new_loss < initial_loss,
        "Loss 应该下降: {} -> {}",
        initial_loss,
        new_loss
    );
}

#[test]
fn test_adam_minimize() {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let w = graph
        .parameter(&[2, 1], crate::nn::Init::Constant(0.5), "w")
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    let y = x.matmul(&w).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    let mut optimizer = Adam::new(&graph, std::slice::from_ref(&w), 0.1);

    // 使用 minimize
    let loss_val = optimizer.minimize(&loss).unwrap();
    assert!(loss_val > 0.0, "Loss 应该是正数");
}

#[test]
fn test_adam_reset() {
    let graph = Graph::new_with_seed(42);

    let w = graph
        .parameter(&[2, 1], crate::nn::Init::Constant(0.5), "w")
        .unwrap();

    let mut optimizer = Adam::new(&graph, &[w], 0.001);

    // 模拟一些更新
    // t 初始为 0，reset 后也应该是 0
    assert_eq!(optimizer.learning_rate(), 0.001);

    optimizer.set_learning_rate(0.01);
    assert_eq!(optimizer.learning_rate(), 0.01);

    optimizer.reset();
    // reset 不改变学习率，只清除动量
    assert_eq!(optimizer.learning_rate(), 0.01);
}

#[test]
fn test_sgd_with_linear_layer() {
    let graph = Graph::new_with_seed(42);

    // 使用 Linear 层
    let fc = Linear::new(&graph, 3, 2, true, "fc").unwrap();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0, 0.0], &[1, 2])).unwrap();

    let y = fc.forward(&x);
    let loss = y.mse_loss(&target).unwrap();

    // 创建优化器，使用 Module 的 parameters()
    let params = fc.parameters();
    let mut optimizer = SGD::new(&graph, &params, 0.01);

    // 训练多步
    for _ in 0..10 {
        optimizer.zero_grad().unwrap();
        loss.backward().unwrap();
        optimizer.step().unwrap();
    }

    // 参数应该有梯度
    for p in &params {
        assert!(p.grad().unwrap().is_some());
    }
}

#[test]
fn test_adam_with_mlp() {
    let graph = Graph::new_with_seed(42);

    // 两层 MLP
    let fc1 = Linear::new(&graph, 3, 4, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 4, 2, true, "fc2").unwrap();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0, 0.0], &[1, 2])).unwrap();

    let h = fc1.forward(&x).relu();
    let y = fc2.forward(&h);
    let loss = y.mse_loss(&target).unwrap();

    // 收集所有参数
    let mut params = fc1.parameters();
    params.extend(fc2.parameters());

    let mut optimizer = Adam::new(&graph, &params, 0.01);

    // 训练
    let initial_loss = loss.backward().unwrap();
    optimizer.step().unwrap();

    // 训练更多步
    for _ in 0..5 {
        let _ = optimizer.minimize(&loss).unwrap();
    }

    // 最终 loss 应该下降
    loss.forward().unwrap();
    let final_loss = loss.item().unwrap();
    assert!(
        final_loss < initial_loss,
        "Loss 应该下降: {} -> {}",
        initial_loss,
        final_loss
    );
}

#[test]
fn test_optimizer_params_accessor() {
    let graph = Graph::new_with_seed(42);

    let w1 = graph
        .parameter(&[2, 3], crate::nn::Init::Constant(0.1), "w1")
        .unwrap();
    let w2 = graph
        .parameter(&[3, 1], crate::nn::Init::Constant(0.2), "w2")
        .unwrap();

    // SGD params() 测试
    let sgd = SGD::new(&graph, &[w1.clone(), w2.clone()], 0.01);
    let sgd_params = sgd.params();
    assert_eq!(sgd_params.len(), 2);
    assert_eq!(sgd_params[0].node_id(), w1.node_id());
    assert_eq!(sgd_params[1].node_id(), w2.node_id());

    // Adam params() 测试
    let adam = Adam::new(&graph, &[w1.clone(), w2.clone()], 0.001);
    let adam_params = adam.params();
    assert_eq!(adam_params.len(), 2);
    assert_eq!(adam_params[0].node_id(), w1.node_id());
    assert_eq!(adam_params[1].node_id(), w2.node_id());
}

#[test]
fn test_adam_state_accessors() {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let w = graph
        .parameter(&[2, 1], crate::nn::Init::Constant(0.5), "w")
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    let y = x.matmul(&w).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    let mut optimizer = Adam::new(&graph, std::slice::from_ref(&w), 0.1);

    // 初始状态：timestep = 0，无动量/速度
    assert_eq!(optimizer.timestep(), 0);
    assert!(optimizer.get_momentum(&w).is_none());
    assert!(optimizer.get_velocity(&w).is_none());

    // 执行一步优化
    optimizer.zero_grad().unwrap();
    loss.backward().unwrap();
    optimizer.step().unwrap();

    // 优化后：timestep = 1，有动量/速度
    assert_eq!(optimizer.timestep(), 1);
    assert!(optimizer.get_momentum(&w).is_some());
    assert!(optimizer.get_velocity(&w).is_some());

    // 验证动量/速度的形状与参数一致
    let momentum = optimizer.get_momentum(&w).unwrap();
    let velocity = optimizer.get_velocity(&w).unwrap();
    assert_eq!(momentum.shape(), &[2, 1]);
    assert_eq!(velocity.shape(), &[2, 1]);

    // 再执行一步
    optimizer.minimize(&loss).unwrap();
    assert_eq!(optimizer.timestep(), 2);

    // reset 后状态清零
    optimizer.reset();
    assert_eq!(optimizer.timestep(), 0);
    assert!(optimizer.get_momentum(&w).is_none());
    assert!(optimizer.get_velocity(&w).is_none());
}
