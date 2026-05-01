/*
 * @Author       : 老董
 * @Description  : CSE 节点去重集成测试
 *
 * 模拟 Hybrid SAC 双 Critic 前向场景：
 * 两个 Critic 共享同一 obs+action 拼接输入，CSE 应去重 Concat 节点。
 * 验证去重后计算图正确性（forward + backward 全流程）。
 */

use only_torch::nn::layer::Linear;
use only_torch::nn::{Graph, GraphError, Module, Var, VarActivationOps, VarLossOps};
use only_torch::tensor::Tensor;

/// 模拟双 Critic 共享 obs+action 拼接输入的场景
///
/// 构造：
///   obs [batch, 4] + action [batch, 2] → concat → [batch, 6]
///   Critic1: concat → fc1_1 → relu → fc2_1 → q1 [batch, 1]
///   Critic2: concat → fc1_2 → relu → fc2_2 → q2 [batch, 1]
///   loss = mse(q1, target) + mse(q2, target)
///
/// 预期：
///   - Var::concat 对同一对 (obs, action) 只创建一个 Concat 节点
///   - forward + backward 全流程梯度正确
#[test]
fn test_cse_dual_critic_shared_concat() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let obs_dim = 4;
    let act_dim = 2;
    let hidden = 8;
    let batch = 2;

    // 创建 obs 和 action 输入
    let obs = graph.input(&Tensor::new(
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        &[batch, obs_dim],
    ))?;
    let action = graph.input(&Tensor::new(&[0.9, 1.0, 1.1, 1.2], &[batch, act_dim]))?;

    // 两个 Critic 各自调 Var::concat，但输入完全相同
    // CSE 应让它们共享同一个 Concat 节点
    let cat_for_critic1 = Var::concat(&[&obs, &action], 1)?; // [batch, 6]
    let cat_for_critic2 = Var::concat(&[&obs, &action], 1)?; // 应与上面同一个节点

    // 验证 CSE 去重
    assert_eq!(
        cat_for_critic1.node_id(),
        cat_for_critic2.node_id(),
        "双 Critic 共享的 concat 应去重为同一个节点"
    );

    // Critic1 网络
    let c1_fc1 = Linear::new(&graph, obs_dim + act_dim, hidden, true, "c1_fc1")?;
    let c1_fc2 = Linear::new(&graph, hidden, 1, true, "c1_fc2")?;
    let q1 = c1_fc2.forward(c1_fc1.forward(&cat_for_critic1).relu());

    // Critic2 网络（不同权重）
    let c2_fc1 = Linear::new(&graph, obs_dim + act_dim, hidden, true, "c2_fc1")?;
    let c2_fc2 = Linear::new(&graph, hidden, 1, true, "c2_fc2")?;
    let q2 = c2_fc2.forward(c2_fc1.forward(&cat_for_critic2).relu());

    // loss = mse(q1, 0) + mse(q2, 0)
    let target = graph.input(&Tensor::zeros(&[batch, 1]))?;
    let loss1 = q1.mse_loss(&target)?;
    let loss2 = q2.mse_loss(&target)?;
    let total_loss = &loss1 + &loss2;

    // Forward
    total_loss.forward()?;
    let loss_val = total_loss.value()?.unwrap();
    assert!(
        loss_val[[0, 0]].is_finite(),
        "loss 值应有限: {:?}",
        loss_val
    );

    // Backward
    graph.zero_grad()?;
    total_loss.backward()?;

    // 验证所有参数都收到了梯度
    for param in c1_fc1
        .parameters()
        .iter()
        .chain(c1_fc2.parameters().iter())
        .chain(c2_fc1.parameters().iter())
        .chain(c2_fc2.parameters().iter())
    {
        let grad = param.grad()?.expect("所有 Critic 参数应有梯度");
        assert!(
            grad.sum().get_data_number().unwrap().abs() > 1e-10,
            "梯度不应全为零"
        );
    }

    Ok(())
}
