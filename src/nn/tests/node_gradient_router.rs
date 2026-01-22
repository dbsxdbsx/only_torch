/*
 * GradientRouter 节点单元测试
 *
 * GradientRouter 是 ModelState 内部使用的特殊节点，用于实现：
 * - 统一缓存（所有输入类型使用相同的计算路径）
 * - 动态 detached 状态
 * - 梯度路由（将梯度传回源 Var）
 */

use crate::nn::var_ops::{VarLossOps, VarMatrixOps};
use crate::nn::{Graph, Init};
use crate::tensor::Tensor;

/// 测试: GradientRouter 基本创建和值设置
#[test]
fn test_gradient_router_basic() {
    let graph = Graph::new();

    // 创建 GradientRouter 节点
    let router_id = graph
        .inner_mut()
        .new_gradient_router_node(&[2, 3], Some("test_router"))
        .unwrap();

    // 验证节点存在
    assert!(graph.inner().nodes_count() == 1);

    // 设置值
    let value = Tensor::ones(&[2, 3]);
    graph
        .inner_mut()
        .set_node_value(router_id, Some(&value))
        .unwrap();

    // 读取值
    let inner = graph.inner();
    let retrieved = inner.get_node_value(router_id).unwrap().unwrap();
    assert_eq!(retrieved.shape(), &[2, 3]);
}

/// 测试: GradientRouter 动态 detached 状态
#[test]
fn test_gradient_router_dynamic_detached() {
    let graph = Graph::new();

    let router_id = graph
        .inner_mut()
        .new_gradient_router_node(&[2, 2], None)
        .unwrap();

    // 默认非 detached
    assert!(!graph.inner().is_node_detached(router_id).unwrap());

    // 设置为 detached
    graph
        .inner_mut()
        .set_router_detached(router_id, true)
        .unwrap();
    assert!(graph.inner().is_node_detached(router_id).unwrap());

    // 切换回非 detached
    graph
        .inner_mut()
        .set_router_detached(router_id, false)
        .unwrap();
    assert!(!graph.inner().is_node_detached(router_id).unwrap());
}

/// 测试: GradientRouter 梯度路由设置
#[test]
fn test_gradient_router_gradient_target() {
    let graph = Graph::new();

    // 创建一个普通节点作为路由目标
    let target = graph.parameter(&[2, 2], Init::Ones, "target").unwrap();
    let target_id = target.node_id();

    // 创建 GradientRouter
    let router_id = graph
        .inner_mut()
        .new_gradient_router_node(&[2, 2], None)
        .unwrap();

    // 默认无路由目标
    assert!(
        graph
            .inner()
            .get_gradient_target(router_id)
            .unwrap()
            .is_none()
    );

    // 设置路由目标
    graph
        .inner_mut()
        .set_gradient_target(router_id, Some(target_id))
        .unwrap();
    assert_eq!(
        graph.inner().get_gradient_target(router_id).unwrap(),
        Some(target_id)
    );

    // 清除路由目标
    graph
        .inner_mut()
        .set_gradient_target(router_id, None)
        .unwrap();
    assert!(
        graph
            .inner()
            .get_gradient_target(router_id)
            .unwrap()
            .is_none()
    );
}

/// 测试: GradientRouter 梯度路由功能（核心测试）
///
/// 使用参数节点作为梯度路由目标，验证梯度正确路由。
#[test]
fn test_gradient_router_gradient_routing() {
    let graph = Graph::new_with_seed(42);

    // D 的参数
    let d_w = graph.parameter(&[2, 1], Init::Ones, "d_w").unwrap();

    // G 的参数（作为梯度路由目标）
    let g_w = graph.parameter(&[1, 2], Init::Ones, "g_w").unwrap();

    // 模拟 G 的输出：z @ g_w
    let z = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let fake = z.matmul(&g_w).unwrap(); // [1,1] @ [1,2] -> [1,2]

    // 创建 GradientRouter（模拟 ModelState 的入口）
    let router_id = graph
        .inner_mut()
        .new_gradient_router_node(&[1, 2], Some("router"))
        .unwrap();
    let router = crate::nn::Var::new(router_id, graph.inner_rc());

    // 设置 GradientRouter 的值（从 fake 复制）
    fake.forward().unwrap();
    router.set_value(&fake.value().unwrap().unwrap()).unwrap();

    // 设置梯度路由目标为 fake（模拟训练 G 时的行为）
    graph
        .inner_mut()
        .set_gradient_target(router_id, Some(fake.node_id()))
        .unwrap();

    // 构建 D 的计算：router @ d_w
    let d_out = router.matmul(&d_w).unwrap();

    // 创建 target 和 loss
    let target = graph.input(&Tensor::ones(&[1, 1])).unwrap();
    let loss = d_out.mse_loss(&target).unwrap();

    // 反向传播
    loss.backward().unwrap();

    // 验证：d_w 应该有梯度（D 的参数）
    assert!(d_w.grad().unwrap().is_some(), "d_w 应该有梯度");

    // 验证：g_w 应该通过梯度路由收到梯度（G 的参数）
    // 梯度路径：loss → d_out → router → (路由到) fake → g_w
    assert!(
        g_w.grad().unwrap().is_some(),
        "g_w 应该通过梯度路由收到梯度"
    );
}

/// 测试: GradientRouter detached 时不路由梯度
#[test]
fn test_gradient_router_detached_no_routing() {
    let graph = Graph::new_with_seed(42);

    // D 的参数
    let d_w = graph.parameter(&[2, 1], Init::Ones, "d_w").unwrap();

    // G 的参数
    let g_w = graph.parameter(&[1, 2], Init::Ones, "g_w").unwrap();

    // 模拟 G 的输出
    let z = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let fake = z.matmul(&g_w).unwrap();

    // 创建 GradientRouter 并设置为 detached（模拟训练 D）
    let router_id = graph
        .inner_mut()
        .new_gradient_router_node(&[1, 2], None)
        .unwrap();
    let router = crate::nn::Var::new(router_id, graph.inner_rc());

    fake.forward().unwrap();
    router.set_value(&fake.value().unwrap().unwrap()).unwrap();

    // 设置路由目标 AND detached
    // 即使设置了路由目标，detached 时也不应该路由梯度
    graph
        .inner_mut()
        .set_gradient_target(router_id, Some(fake.node_id()))
        .unwrap();
    graph
        .inner_mut()
        .set_router_detached(router_id, true)
        .unwrap();

    // 构建 D 的计算
    let d_out = router.matmul(&d_w).unwrap();
    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = d_out.mse_loss(&target).unwrap();

    // 反向传播
    loss.backward().unwrap();

    // d_w 应该有梯度（D 的参数）
    assert!(d_w.grad().unwrap().is_some(), "d_w 应该有梯度");

    // g_w 不应该有梯度（因为 GradientRouter 是 detached）
    assert!(
        g_w.grad().unwrap().is_none(),
        "g_w 不应有梯度（GradientRouter 是 detached）"
    );
}

/// 测试: GradientRouter 可视化样式
#[test]
fn test_gradient_router_visualization() {
    let graph = Graph::new();

    // 创建 GradientRouter
    let _router_id = graph
        .inner_mut()
        .new_gradient_router_node(&[2, 2], Some("test_router"))
        .unwrap();

    // 生成 DOT 字符串
    let dot = graph.inner().to_dot();

    // 验证 GradientRouter 使用特殊样式（虚线边框）
    assert!(dot.contains("dashed"), "GradientRouter 应使用虚线样式");
    assert!(
        dot.contains("GradientRouter"),
        "DOT 应包含 GradientRouter 类型"
    );
}

/// 测试: GradientRouter 支持动态 batch（类似 Keras）
///
/// GradientRouter 只验证特征维度匹配，允许不同 batch_size 的值。
#[test]
fn test_gradient_router_dynamic_batch() {
    let graph = Graph::new();

    // 创建 GradientRouter（首次使用 batch=4）
    let router_id = graph
        .inner_mut()
        .new_gradient_router_node(&[4, 3], Some("router"))
        .unwrap();

    // 设置初始值 [4, 3]
    let value1 = Tensor::ones(&[4, 3]);
    graph
        .inner_mut()
        .set_node_value(router_id, Some(&value1))
        .unwrap();

    // 设置不同 batch 的值 [2, 3]（应该成功）
    let value2 = Tensor::ones(&[2, 3]);
    graph
        .inner_mut()
        .set_node_value(router_id, Some(&value2))
        .unwrap();

    // 设置 batch=1 的值 [1, 3]（应该成功）
    let value3 = Tensor::ones(&[1, 3]);
    graph
        .inner_mut()
        .set_node_value(router_id, Some(&value3))
        .unwrap();

    // 验证值已更新
    let inner = graph.inner();
    let retrieved = inner.get_node_value(router_id).unwrap().unwrap();
    assert_eq!(retrieved.shape(), &[1, 3]);
}

/// 测试: GradientRouter 仍然验证特征维度
///
/// 虽然 batch 维度可变，但特征维度必须匹配。
#[test]
fn test_gradient_router_feature_shape_must_match() {
    let graph = Graph::new();

    // 创建 GradientRouter（特征维度=3）
    let router_id = graph
        .inner_mut()
        .new_gradient_router_node(&[4, 3], Some("router"))
        .unwrap();

    // 设置初始值 [4, 3]
    let value1 = Tensor::ones(&[4, 3]);
    graph
        .inner_mut()
        .set_node_value(router_id, Some(&value1))
        .unwrap();

    // 尝试设置不同特征维度的值 [4, 5]（应该失败）
    let value2 = Tensor::ones(&[4, 5]);
    let result = graph.inner_mut().set_node_value(router_id, Some(&value2));

    assert!(result.is_err(), "特征维度不匹配应该报错");
    if let Err(crate::nn::GraphError::ShapeMismatch { message, .. }) = result {
        assert!(
            message.contains("动态形状") && message.contains("不兼容"),
            "错误信息应提及动态形状不兼容: {}",
            message
        );
    }
}
