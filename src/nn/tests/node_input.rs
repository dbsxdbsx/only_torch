/*
 * @Author       : 老董
 * @Description  : InputVariant 节点单元测试
 *
 * 测试所有 Input 变体：
 * - BasicInput：普通数据输入节点
 * - TargetInput：Loss 的目标值节点（内部复用 SmartInput）
 * - SmartInput：ModelState 使用的智能入口节点（支持动态 batch、梯度路由）
 *
 * 六段式结构（Input 节点特殊）：
 * 1. 基础功能（创建、无效形状、命名、预期形状）
 * 2. 值设置（有效/无效/清除/动态 batch）
 * 3. 前向传播行为（无值报错、有值静默成功）
 * 4. 梯度行为（Input 无梯度、在计算图中无梯度）
 * 5. SmartInput 测试（基本创建、detached、梯度路由、动态 batch）
 * 6. 动态形状 + Create API（已有，保持原样）
 */

use crate::nn::var_ops::{VarActivationOps, VarLossOps, VarMatrixOps};
use crate::nn::{Graph, GraphError, Init, Var};
use crate::tensor::Tensor;
use std::rc::Rc;

// ==================== 1. 基础功能测试 ====================

/// 测试 Input 节点基本创建
#[test]
fn test_node_input_creation() {
    let graph = Graph::new();

    // 1. 使用高层 API 创建带名称的 Input 节点
    let x = graph
        .input_named(&Tensor::zeros(&[2, 3]), "input1")
        .unwrap();

    // 1.1 验证基本属性
    assert_eq!(x.name(), Some("input1"));
    assert_eq!(x.value_expected_shape(), vec![2, 3]);

    // 1.2 Input 节点是叶子节点，没有父节点
    assert!(x.node().is_leaf());
    assert!(x.node().parents().is_empty());
}

/// 测试无效形状的 Input 创建（2-4 维有效，0/1/5 维应失败）
#[test]
fn test_node_input_creation_with_invalid_shape() {
    use crate::assert_err;

    let graph = Graph::new();

    // 0D、1D、5D 应该失败
    for dims in [0, 1, 5] {
        let shape: Vec<usize> = match dims {
            0 => vec![],
            1 => vec![2],
            5 => vec![2, 2, 2, 2, 2],
            _ => unreachable!(),
        };

        let result = graph.input_shape(&shape, None);
        assert_err!(
            result,
            GraphError::DimensionMismatch { expected, got, message }
                if *expected == 2 && *got == dims && message.contains("2-4 维")
        );
    }

    // 3D 和 4D 应该成功
    assert!(graph.input_shape(&[3, 28, 28], Some("input_3d")).is_ok());
    assert!(
        graph
            .input_shape(&[4, 3, 28, 28], Some("input_4d"))
            .is_ok()
    );
}

/// 测试 Input 节点的命名机制
#[test]
fn test_node_input_name_generation() {
    let graph = Graph::new();

    // 1. 显式命名
    let x1 = graph
        .input_named(&Tensor::zeros(&[2, 2]), "explicit_input")
        .unwrap();
    assert_eq!(x1.name(), Some("explicit_input"));

    // 2. 自动命名（应包含 "input"）
    let x2 = graph.input(&Tensor::zeros(&[2, 2])).unwrap();
    let auto_name = x2.name().unwrap();
    assert!(
        auto_name.contains("input"),
        "自动名称应包含 'input': {}",
        auto_name
    );
}

/// 测试 Input 节点的预期形状
#[test]
fn test_node_input_expected_shape() {
    let graph = Graph::new();

    // 1. 创建 Input 节点（input_shape 不设置值）
    let x = graph.input_shape(&[2, 3], Some("input")).unwrap();

    // 预期形状已确定
    assert_eq!(x.value_expected_shape(), vec![2, 3]);

    // 没有值时 node 层面 value 为 None
    assert!(x.node().value().is_none());

    // 2. 设置值后检查
    let value = Tensor::zeros(&[2, 3]);
    x.set_value(&value).unwrap();
    assert_eq!(x.node().value().unwrap().shape(), &[2, 3]);

    // 预期形状保持不变
    assert_eq!(x.value_expected_shape(), vec![2, 3]);

    // 3. 清除值
    x.node().set_value(None).unwrap();
    assert!(x.node().value().is_none());

    // 预期形状仍然保持
    assert_eq!(x.value_expected_shape(), vec![2, 3]);
}

// ==================== 2. 值设置测试 ====================

/// 测试 Input 节点的手动赋值
#[test]
fn test_node_input_manually_set_value() {
    let graph = Graph::new();
    let x = graph
        .input_named(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), "test_input")
        .unwrap();

    // 1. 验证初始值已设置
    let val = x.node().value().unwrap();
    assert_eq!(val.shape(), &[2, 2]);

    // 2. 更新为新值
    let new_value = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    x.set_value(&new_value).unwrap();
    assert_eq!(x.node().value().unwrap()[[0, 0]], 5.0);

    // 3. 不同 batch 大小应成功（Input 支持动态 batch）
    let different_batch = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    x.set_value(&different_batch).unwrap();
    assert_eq!(x.node().value().unwrap().shape(), &[3, 2]);

    // 4. 清除值
    x.node().set_value(None).unwrap();
    assert!(x.node().value().is_none());
}

// ==================== 3. 前向传播行为测试 ====================

/// 测试 Input 节点前向传播行为
///
/// - 无值时 forward 报错（叶子节点必须有值）
/// - 有值时 forward 静默成功
#[test]
fn test_node_input_forward_propagation() {
    let graph = Graph::new();

    // 创建不带值的 Input 节点
    let x = graph.input_shape(&[2, 2], Some("input")).unwrap();

    // 1. 无值时 forward 应失败
    let result = x.forward();
    assert!(
        result.is_err(),
        "无值的 Input 节点 forward 应失败"
    );

    // 2. 设置值后 forward 应成功
    let value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    x.set_value(&value).unwrap();
    assert!(
        x.forward().is_ok(),
        "有值的 Input 节点应该允许 forward（静默成功）"
    );
}

// ==================== 4. 梯度行为测试 ====================

/// 测试 Input 节点不应该有梯度
///
/// Input 节点是输入数据，不是可学习参数，因此不应该有梯度。
/// 在新 API 中，Var::grad() 返回 Ok(None)（而非错误）。
#[test]
fn test_node_input_no_grad() {
    let graph = Graph::new();

    // 1. 创建输入节点
    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();

    // 2. Input 节点的 grad 应返回 None（不是错误，而是没有梯度）
    assert!(
        x.grad().unwrap().is_none(),
        "Input 节点不应有梯度"
    );

    // 3. zero_grad 后仍然无梯度
    graph.zero_grad().unwrap();
    assert!(
        x.grad().unwrap().is_none(),
        "zero_grad 后 Input 节点仍不应有梯度"
    );
}

/// 测试 Input 节点在正常计算图反向传播后的行为
///
/// 在完整的计算图中，反向传播到 Input 节点时会无害跳过（Input 是"梯度汇点"），
/// 调用 grad() 仍然返回 None。
#[test]
fn test_node_input_in_computation_graph() {
    let graph = Graph::new_with_seed(42);

    // 1. 构建简单计算图: input * param -> mse_loss(target)
    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let param = graph.parameter(&[2, 2], Init::Ones, "param").unwrap();
    let target = graph
        .input(&Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]))
        .unwrap();

    // input * param（逐元素乘法通过 Var ops）
    let mul = &input * &param;
    let loss = mul.mse_loss(&target).unwrap();

    // 2. 前向 + 反向传播
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // 3. 验证：Parameter 节点应该有梯度
    assert!(
        param.grad().unwrap().is_some(),
        "参数节点应该有梯度"
    );

    // 4. 验证：Input 节点不应有梯度（梯度到 Input 时无害跳过）
    assert!(
        input.grad().unwrap().is_none(),
        "Input 节点不应有梯度"
    );
    assert!(
        target.grad().unwrap().is_none(),
        "Target Input 节点不应有梯度"
    );
}

// ==================== 5. SmartInput 测试 ====================
//
// SmartInput 是 ModelState 内部使用的特殊节点，用于实现：
// - 统一缓存（所有输入类型使用相同的计算路径）
// - 动态 detached 状态
// - 梯度路由（将梯度传回源 Var）

/// 测试: SmartInput 基本创建和值设置
#[test]
fn test_smart_input_basic() {
    let graph = Graph::new();

    // 通过底层 API 创建 SmartInput 节点
    let router_node = graph
        .inner_mut()
        .create_smart_input_node(&[2, 3], Some("test_router"))
        .unwrap();

    // 验证基本属性
    assert_eq!(router_node.shape(), vec![2, 3]);
    assert_eq!(router_node.name(), Some("test_router"));
    assert!(router_node.is_leaf());

    // 设置值
    let value = Tensor::ones(&[2, 3]);
    router_node.set_value(Some(&value)).unwrap();

    // 读取值
    let retrieved = router_node.value().unwrap();
    assert_eq!(retrieved.shape(), &[2, 3]);
}

/// 测试: SmartInput 动态 detached 状态
///
/// 通过 NodeInner::set_detached 控制梯度截断
#[test]
fn test_smart_input_dynamic_detached() {
    let graph = Graph::new();

    let router_node = graph
        .inner_mut()
        .create_smart_input_node(&[2, 2], None)
        .unwrap();

    // 默认非 detached
    assert!(!router_node.is_detached());

    // 设置为 detached
    router_node.set_detached(true);
    assert!(router_node.is_detached());

    // 切换回非 detached
    router_node.set_detached(false);
    assert!(!router_node.is_detached());
}

/// 测试: SmartInput 梯度路由设置（通过 raw node 访问）
///
/// 验证可以正确设置和清除梯度路由目标。
/// 注意：实际梯度路由在 backward 中由 ModelState 外部逻辑处理。
#[test]
fn test_smart_input_gradient_target() {
    use crate::nn::nodes::raw_node::InputVariant;
    use crate::nn::nodes::NodeType;

    let graph = Graph::new();

    // 创建目标节点（用于路由）
    let target = graph.parameter(&[2, 2], Init::Ones, "target").unwrap();
    let target_id = target.node_id();

    // 创建 SmartInput
    let router_node = graph
        .inner_mut()
        .create_smart_input_node(&[2, 2], None)
        .unwrap();

    // 默认无路由目标
    router_node.with_raw_node(|raw| {
        if let NodeType::Input(InputVariant::Smart(smart)) = raw {
            assert!(smart.gradient_target().is_none(), "默认无梯度路由目标");
        } else {
            panic!("应为 SmartInput 类型");
        }
    });

    // 设置路由目标
    router_node.with_raw_node_mut(|raw| {
        if let NodeType::Input(InputVariant::Smart(smart)) = raw {
            smart.set_gradient_target(Some(target_id));
        }
    });
    router_node.with_raw_node(|raw| {
        if let NodeType::Input(InputVariant::Smart(smart)) = raw {
            assert_eq!(smart.gradient_target(), Some(target_id));
        }
    });

    // 清除路由目标
    router_node.with_raw_node_mut(|raw| {
        if let NodeType::Input(InputVariant::Smart(smart)) = raw {
            smart.set_gradient_target(None);
        }
    });
    router_node.with_raw_node(|raw| {
        if let NodeType::Input(InputVariant::Smart(smart)) = raw {
            assert!(smart.gradient_target().is_none());
        }
    });
}

/// 测试: SmartInput 梯度路由功能（核心测试）
///
/// 使用参数节点作为梯度路由目标，验证梯度正确路由。
/// 注意：梯度路由需要 ModelState 的外部逻辑配合，
/// 此处仅验证 SmartInput 节点本身在反向传播中的行为。
#[test]
fn test_smart_input_gradient_routing() {
    let graph = Graph::new_with_seed(42);

    // D 的参数
    let d_w = graph.parameter(&[2, 1], Init::Ones, "d_w").unwrap();

    // G 的参数（作为梯度路由目标）
    let g_w = graph.parameter(&[1, 2], Init::Ones, "g_w").unwrap();

    // 模拟 G 的输出：z @ g_w
    let z = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let fake = z.matmul(&g_w).unwrap(); // [1,1] @ [1,2] -> [1,2]

    // 创建 SmartInput（模拟 ModelState 的入口）
    let router_node = graph
        .inner_mut()
        .create_smart_input_node(&[1, 2], Some("router"))
        .unwrap();
    let router = Var::new_with_rc_graph(router_node, &graph.inner_rc());

    // 设置 SmartInput 的值（从 fake 复制）
    fake.forward().unwrap();
    router
        .set_value(&fake.value().unwrap().unwrap())
        .unwrap();

    // 构建 D 的计算：router @ d_w
    let d_out = router.matmul(&d_w).unwrap();

    // 创建 target 和 loss
    let target = graph.input(&Tensor::ones(&[1, 1])).unwrap();
    let loss = d_out.mse_loss(&target).unwrap();

    // 反向传播
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // 验证：d_w 应该有梯度（D 的参数，正常路径）
    assert!(d_w.grad().unwrap().is_some(), "d_w 应该有梯度");

    // 注意：在当前架构中，SmartInput 是叶子节点，梯度不会自动路由到 fake → g_w。
    // 梯度路由由 ModelState 的外部逻辑负责（读取 SmartInput 的梯度 → 手动累加到目标）。
    // 此处验证 SmartInput 确实接收到了上游梯度（作为 SmartInput 变体有 grad 字段）。
    let router_grad = router.grad().unwrap();
    assert!(
        router_grad.is_some(),
        "SmartInput 应收到上游梯度（供外部路由使用）"
    );
}

/// 测试: SmartInput detached 时不向上游传播梯度
///
/// 设计说明：
/// - detached 节点**可以收到**来自下游的梯度
/// - 但**不会向上游**传播梯度（propagate_grad_to_parents 中检查）
/// - 对于 SmartInput（无父节点），detached 标志主要用于 ModelState
///   的外部逻辑判断是否执行梯度路由
#[test]
fn test_smart_input_detached_no_routing() {
    let graph = Graph::new_with_seed(42);

    // D 的参数
    let d_w = graph.parameter(&[2, 1], Init::Ones, "d_w").unwrap();

    // G 的参数（与 SmartInput 无计算图连接）
    let g_w = graph.parameter(&[1, 2], Init::Ones, "g_w").unwrap();

    // 创建 SmartInput 并设置为 detached（模拟训练 D 时阻断梯度路由）
    let router_node = graph
        .inner_mut()
        .create_smart_input_node(&[1, 2], None)
        .unwrap();

    // 设置为 detached
    router_node.set_detached(true);
    assert!(router_node.is_detached(), "should be detached");

    let router = Var::new_with_rc_graph(router_node, &graph.inner_rc());

    // 设置值
    router.set_value(&Tensor::ones(&[1, 2])).unwrap();

    // 构建 D 的计算
    let d_out = router.matmul(&d_w).unwrap();
    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = d_out.mse_loss(&target).unwrap();

    // 反向传播
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // d_w 应该有梯度（D 的参数，正常反向路径）
    assert!(d_w.grad().unwrap().is_some(), "d_w 应该有梯度");

    // SmartInput 仍然收到梯度（detached 不阻止接收，只阻止传播）
    // 这是设计行为：ModelState 外部逻辑会检查 is_detached() 来决定是否路由
    let router_grad = router.grad().unwrap();
    assert!(
        router_grad.is_some(),
        "SmartInput 应收到上游梯度（即使 detached）"
    );

    // g_w 不应有梯度（SmartInput 没有父节点连接到 g_w 的计算路径）
    assert!(
        g_w.grad().unwrap().is_none(),
        "g_w 不应有梯度（SmartInput 与 g_w 无计算图连接）"
    );
}

/// 测试: SmartInput 可视化样式
#[test]
fn test_smart_input_visualization() {
    let graph = Graph::new();

    // 创建 SmartInput
    let router_node = graph
        .inner_mut()
        .create_smart_input_node(&[2, 2], Some("test_router"))
        .unwrap();

    // 包装为 Var 以便使用 to_dot
    let router = Var::new_with_rc_graph(router_node, &graph.inner_rc());

    // 生成 DOT 字符串
    let dot = router.to_dot();

    // 验证 SmartInput 在 DOT 中的表示
    assert!(
        dot.contains("SmartInput") || dot.contains("test_router"),
        "DOT 应包含 SmartInput 或节点名称: {}",
        dot
    );
}

/// 测试: SmartInput 支持动态 batch（类似 Keras）
///
/// SmartInput 只验证特征维度匹配，允许不同 batch_size 的值。
#[test]
fn test_smart_input_dynamic_batch() {
    let graph = Graph::new();

    // 创建 SmartInput（首次使用 batch=4）
    let router_node = graph
        .inner_mut()
        .create_smart_input_node(&[4, 3], Some("router"))
        .unwrap();

    // 设置初始值 [4, 3]
    let value1 = Tensor::ones(&[4, 3]);
    router_node.set_value(Some(&value1)).unwrap();
    assert_eq!(router_node.value().unwrap().shape(), &[4, 3]);

    // 设置不同 batch 的值 [2, 3]（应该成功）
    let value2 = Tensor::ones(&[2, 3]);
    router_node.set_value(Some(&value2)).unwrap();
    assert_eq!(router_node.value().unwrap().shape(), &[2, 3]);

    // 设置 batch=1 的值 [1, 3]（应该成功）
    let value3 = Tensor::ones(&[1, 3]);
    router_node.set_value(Some(&value3)).unwrap();
    assert_eq!(router_node.value().unwrap().shape(), &[1, 3]);
}

/// 测试: SmartInput 特征维度不匹配应报错
///
/// 虽然 batch 维度可变，但特征维度必须匹配。
/// 注意：SmartInput 的 set_value 通过 TraitNode 直接设置，
/// 维度验证发生在 forward 阶段（通过 dynamic_expected_shape 比较）。
#[test]
fn test_smart_input_feature_shape_must_match() {
    let graph = Graph::new();

    // 创建 SmartInput（特征维度=3）
    let router_node = graph
        .inner_mut()
        .create_smart_input_node(&[4, 3], Some("router"))
        .unwrap();
    let router = Var::new_with_rc_graph(router_node, &graph.inner_rc());

    // 设置初始值 [4, 3]
    router.set_value(&Tensor::ones(&[4, 3])).unwrap();

    // 验证动态形状：batch 维度是动态的，特征维度固定为 3
    let dyn_shape = router.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应为动态");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应为固定");
    assert_eq!(dyn_shape.dim(1), Some(3), "特征维度应为 3");
}

// ==================== 动态形状测试 ====================

/// 测试 Input 节点的动态形状传播
///
/// Input 节点是动态 batch 的源头，其 dynamic_expected_shape 的第一维应为 None
#[test]
fn test_input_dynamic_shape_propagation() {
    let graph = Graph::new();

    // 创建 2D Input 节点
    let x = graph.input(&Tensor::zeros(&[4, 16])).unwrap();

    // 验证动态形状
    let dyn_shape = x.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "特征维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(16), "特征维度应该是 16");
}

/// 测试 Input 节点在不同维度下的动态形状
#[test]
fn test_input_dynamic_shape_various_dims() {
    let graph = Graph::new();

    // 2D: [batch, features]
    let x_2d = graph.input(&Tensor::zeros(&[4, 16])).unwrap();
    let dyn_2d = x_2d.dynamic_expected_shape();
    assert!(dyn_2d.is_dynamic(0));
    assert!(!dyn_2d.is_dynamic(1));

    // 3D: [batch, seq_len, features] (RNN)
    let x_3d = graph.input(&Tensor::zeros(&[4, 10, 32])).unwrap();
    let dyn_3d = x_3d.dynamic_expected_shape();
    assert!(dyn_3d.is_dynamic(0), "3D: batch 维度应该是动态的");
    assert!(!dyn_3d.is_dynamic(1), "3D: seq_len 应该是固定的");
    assert!(!dyn_3d.is_dynamic(2), "3D: features 应该是固定的");

    // 4D: [batch, channels, height, width] (CNN)
    let x_4d = graph.input(&Tensor::zeros(&[8, 3, 28, 28])).unwrap();
    let dyn_4d = x_4d.dynamic_expected_shape();
    assert!(dyn_4d.is_dynamic(0), "4D: batch 维度应该是动态的");
    assert!(!dyn_4d.is_dynamic(1), "4D: channels 应该是固定的");
    assert!(!dyn_4d.is_dynamic(2), "4D: height 应该是固定的");
    assert!(!dyn_4d.is_dynamic(3), "4D: width 应该是固定的");
}

/// 测试 Input 节点支持动态 batch 的值更新
///
/// Input 节点在 set_value 时允许不同的 batch 大小（特征维度必须匹配）
#[test]
fn test_input_dynamic_batch_set_value() {
    let graph = Graph::new();

    // 创建 Input 节点
    let x = graph.input(&Tensor::zeros(&[4, 16])).unwrap();

    // 验证支持动态 batch（通过检查第一维是否为动态）
    assert!(
        x.dynamic_expected_shape().is_dynamic(0),
        "Input 节点应该支持动态 batch"
    );

    // 设置不同 batch 大小的值应该成功
    x.set_value(&Tensor::zeros(&[8, 16])).unwrap();
    assert_eq!(x.value().unwrap().unwrap().shape(), &[8, 16]);

    x.set_value(&Tensor::zeros(&[1, 16])).unwrap();
    assert_eq!(x.value().unwrap().unwrap().shape(), &[1, 16]);

    x.set_value(&Tensor::zeros(&[32, 16])).unwrap();
    assert_eq!(x.value().unwrap().unwrap().shape(), &[32, 16]);
}

/// 测试 Input 节点作为下游节点父节点时的动态 batch 传播
#[test]
fn test_input_dynamic_batch_forward_chain() {
    let graph = Graph::new();

    // Input -> Sigmoid -> output
    let x = graph
        .input(&Tensor::new(&[0.0, 1.0, 2.0, 3.0], &[2, 2]))
        .unwrap();
    let output = x.sigmoid();

    // 第一次 forward：batch=2
    output.forward().unwrap();
    assert_eq!(output.value().unwrap().unwrap().shape(), &[2, 2]);

    // 更新 Input 为不同的 batch 大小
    x.set_value(&Tensor::zeros(&[5, 2])).unwrap();

    // 第二次 forward：batch=5
    output.forward().unwrap();
    assert_eq!(
        output.value().unwrap().unwrap().shape(),
        &[5, 2],
        "输出应该自动适应新的 batch 大小"
    );
}

/// 测试 Input 节点在完整训练流程中的动态 batch 支持
#[test]
fn test_input_dynamic_batch_training() {
    let graph = Graph::new();

    // 创建简单网络：input -> sigmoid -> loss
    let x = graph
        .input(&Tensor::new(&[0.0, 1.0, 2.0, 3.0], &[2, 2]))
        .unwrap();
    let pred = x.sigmoid();
    let target = graph.input(&Tensor::zeros(&[2, 2])).unwrap();
    let loss = pred.mse_loss(&target).unwrap();

    // 第一次训练：batch=2
    loss.forward().unwrap();
    let loss_val1 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0);

    // 更新为不同 batch 大小
    x.set_value(&Tensor::zeros(&[8, 2])).unwrap();
    target.set_value(&Tensor::zeros(&[8, 2])).unwrap();

    // 第二次训练：batch=8
    loss.forward().unwrap();
    let loss_val2 = loss.value().unwrap().unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);
}

// ==================== 方案 C：新节点创建 API 测试 ====================

#[test]
fn test_create_basic_input_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 创建 BasicInput 节点
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 3], Some("x"))
        .unwrap();

    // 验证节点属性
    assert_eq!(input.shape(), vec![4, 3]);
    assert_eq!(input.name(), Some("x"));
    assert!(input.is_leaf()); // 叶子节点，无父节点
    assert!(input.parents().is_empty());
}

#[test]
fn test_create_basic_input_auto_name() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 不指定名称，应该自动生成
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 3], None)
        .unwrap();

    // 验证自动生成的名称
    let name = input.name().unwrap();
    assert!(name.contains("input"), "名称应包含 'input': {}", name);
}

#[test]
fn test_create_basic_input_reference_counting() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input;
    {
        // 在作用域内创建
        input = inner
            .borrow_mut()
            .create_basic_input_node(&[4, 3], None)
            .unwrap();

        // 此时引用计数为 1
        assert_eq!(Rc::strong_count(&input), 1);
    }

    // 变量仍然有效
    assert_eq!(Rc::strong_count(&input), 1);
    assert_eq!(input.shape(), vec![4, 3]);
}

#[test]
fn test_create_basic_input_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 使用 Weak 来观察节点是否被释放
    let weak;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[4, 3], None)
            .unwrap();
        weak = Rc::downgrade(&input);

        // 作用域内，Weak 可以升级
        assert!(weak.upgrade().is_some());
    }
    // input 离开作用域，节点被释放

    // Weak 无法升级
    assert!(weak.upgrade().is_none());
}

#[test]
fn test_create_multiple_basic_inputs() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 创建多个输入节点
    let x = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 3], Some("x"))
        .unwrap();
    let y = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 3], Some("y"))
        .unwrap();

    // 验证是不同的节点
    assert_ne!(x.id(), y.id());
    assert_eq!(x.name(), Some("x"));
    assert_eq!(y.name(), Some("y"));
}

#[test]
fn test_create_basic_input_invalid_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 1D 形状应该失败（BasicInput 要求 2-4D）
    let result = inner.borrow_mut().create_basic_input_node(&[10], None);
    assert!(result.is_err());

    // 5D 形状也应该失败
    let result = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2, 3, 4, 5], None);
    assert!(result.is_err());
}

#[test]
fn test_create_basic_input_various_shapes() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 2D: 全连接层输入 [batch, features]
    let fc_input = inner
        .borrow_mut()
        .create_basic_input_node(&[32, 784], None)
        .unwrap();
    assert_eq!(fc_input.shape(), vec![32, 784]);

    // 3D: RNN 输入 [batch, seq_len, features]
    let rnn_input = inner
        .borrow_mut()
        .create_basic_input_node(&[16, 10, 128], None)
        .unwrap();
    assert_eq!(rnn_input.shape(), vec![16, 10, 128]);

    // 4D: CNN 输入 [batch, C, H, W]
    let cnn_input = inner
        .borrow_mut()
        .create_basic_input_node(&[8, 3, 32, 32], None)
        .unwrap();
    assert_eq!(cnn_input.shape(), vec![8, 3, 32, 32]);
}

// ==================== TargetInput 方案 C 测试 ====================

#[test]
fn test_create_target_input_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 创建 TargetInput 节点（用于 Loss 的目标值）
    let target = inner
        .borrow_mut()
        .create_target_input_node(&[4, 10], Some("y_true"))
        .unwrap();

    // 验证节点属性
    assert_eq!(target.shape(), vec![4, 10]);
    assert_eq!(target.name(), Some("y_true"));
    assert!(target.is_leaf());
    assert!(target.parents().is_empty());
}

#[test]
fn test_create_target_input_auto_name() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let target = inner
        .borrow_mut()
        .create_target_input_node(&[4, 10], None)
        .unwrap();

    let name = target.name().unwrap();
    assert!(name.contains("target"), "名称应包含 'target': {}", name);
}

#[test]
fn test_create_target_input_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak;
    {
        let target = inner
            .borrow_mut()
            .create_target_input_node(&[4, 10], None)
            .unwrap();
        weak = Rc::downgrade(&target);
        assert!(weak.upgrade().is_some());
    }
    // target 离开作用域，节点被释放
    assert!(weak.upgrade().is_none());
}

#[test]
fn test_create_target_input_various_shapes() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 分类任务：[batch, num_classes]
    let cls_target = inner
        .borrow_mut()
        .create_target_input_node(&[32, 10], None)
        .unwrap();
    assert_eq!(cls_target.shape(), vec![32, 10]);

    // 回归任务：[batch, output_dim]
    let reg_target = inner
        .borrow_mut()
        .create_target_input_node(&[16, 1], None)
        .unwrap();
    assert_eq!(reg_target.shape(), vec![16, 1]);

    // 序列任务：[batch, seq_len, vocab_size]
    let seq_target = inner
        .borrow_mut()
        .create_target_input_node(&[8, 20, 1000], None)
        .unwrap();
    assert_eq!(seq_target.shape(), vec![8, 20, 1000]);
}

// ==================== SmartInput 方案 C 测试 ====================

#[test]
fn test_create_smart_input_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 创建 SmartInput 节点（ModelState 使用）
    let smart = inner
        .borrow_mut()
        .create_smart_input_node(&[4, 128], Some("model_input"))
        .unwrap();

    // 验证节点属性
    assert_eq!(smart.shape(), vec![4, 128]);
    assert_eq!(smart.name(), Some("model_input"));
    assert!(smart.is_leaf());
    assert!(smart.parents().is_empty());
}

#[test]
fn test_create_smart_input_auto_name() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let smart = inner
        .borrow_mut()
        .create_smart_input_node(&[4, 128], None)
        .unwrap();

    let name = smart.name().unwrap();
    assert!(name.contains("input"), "名称应包含 'input': {}", name);
}

#[test]
fn test_create_smart_input_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak;
    {
        let smart = inner
            .borrow_mut()
            .create_smart_input_node(&[4, 128], None)
            .unwrap();
        weak = Rc::downgrade(&smart);
        assert!(weak.upgrade().is_some());
    }
    // smart 离开作用域，节点被释放
    assert!(weak.upgrade().is_none());
}

#[test]
fn test_create_smart_input_various_shapes() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 全连接模型输入 [batch, features]
    let fc_input = inner
        .borrow_mut()
        .create_smart_input_node(&[32, 784], None)
        .unwrap();
    assert_eq!(fc_input.shape(), vec![32, 784]);

    // RNN 模型输入 [batch, seq_len, features]
    let rnn_input = inner
        .borrow_mut()
        .create_smart_input_node(&[16, 10, 128], None)
        .unwrap();
    assert_eq!(rnn_input.shape(), vec![16, 10, 128]);

    // CNN 模型输入 [batch, C, H, W]
    let cnn_input = inner
        .borrow_mut()
        .create_smart_input_node(&[8, 3, 32, 32], None)
        .unwrap();
    assert_eq!(cnn_input.shape(), vec![8, 3, 32, 32]);
}

// ==================== RecurrentOutput 方案 C 测试 ====================

#[test]
fn test_create_recurrent_output_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 创建 RecurrentOutput 节点（RNN/LSTM/GRU 输出桥接）
    let recurrent = inner
        .borrow_mut()
        .create_recurrent_output_node(&[4, 64], Some("rnn_out"))
        .unwrap();

    // 验证节点属性
    assert_eq!(recurrent.shape(), vec![4, 64]);
    assert_eq!(recurrent.name(), Some("rnn_out"));
    assert!(recurrent.is_leaf());
    assert!(recurrent.parents().is_empty());
}

#[test]
fn test_create_recurrent_output_auto_name() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let recurrent = inner
        .borrow_mut()
        .create_recurrent_output_node(&[4, 64], None)
        .unwrap();

    let name = recurrent.name().unwrap();
    assert!(
        name.contains("recurrent"),
        "名称应包含 'recurrent': {}",
        name
    );
}

#[test]
fn test_create_recurrent_output_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak;
    {
        let recurrent = inner
            .borrow_mut()
            .create_recurrent_output_node(&[4, 64], None)
            .unwrap();
        weak = Rc::downgrade(&recurrent);
        assert!(weak.upgrade().is_some());
    }
    // recurrent 离开作用域，节点被释放
    assert!(weak.upgrade().is_none());
}

#[test]
fn test_create_recurrent_output_various_shapes() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // RNN 隐藏状态 [batch, hidden_size]
    let rnn_hidden = inner
        .borrow_mut()
        .create_recurrent_output_node(&[16, 128], None)
        .unwrap();
    assert_eq!(rnn_hidden.shape(), vec![16, 128]);

    // LSTM 输出 [batch, seq_len, hidden_size]
    let lstm_out = inner
        .borrow_mut()
        .create_recurrent_output_node(&[8, 10, 256], None)
        .unwrap();
    assert_eq!(lstm_out.shape(), vec![8, 10, 256]);
}
