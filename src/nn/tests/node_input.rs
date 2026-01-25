/*
 * @Author       : 老董
 * @Description  : InputVariant 节点单元测试
 *
 * 测试所有 Input 变体：
 * - BasicInput：普通数据输入节点
 * - TargetInput：Loss 的目标值节点（内部复用 BasicInput）
 * - SmartInput：ModelState 使用的智能入口节点（支持动态 batch、梯度路由）
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）
 * 2. 值设置测试
 * 3. 前向传播行为测试
 * 4. 梯度行为测试（Input 是梯度汇点，不应有梯度）
 * 5. 动态形状测试
 * 6. SmartInput 特有功能（梯度路由、detached 状态）
 */

use crate::assert_err;
use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;

#[test]
fn test_node_input_creation() {
    let mut graph = GraphInner::new();

    // 1. 测试基本创建
    let input = graph.new_basic_input_node(&[2, 3], Some("input1")).unwrap();

    // 1.1 验证基本属性
    assert_eq!(graph.get_node_name(input).unwrap(), "input1");
    assert_eq!(graph.get_node_parents(input).unwrap().len(), 0);
    assert_eq!(graph.get_node_children(input).unwrap().len(), 0);
    assert!(!graph.is_node_inited(input).unwrap()); // Input节点创建时未初始化
}

#[test]
fn test_node_input_creation_with_invalid_shape() {
    let mut graph = GraphInner::new();

    // 测试不同维度的形状（支持 2-4 维，0/1/5 维应该失败）
    for dims in [0, 1, 5] {
        let shape = match dims {
            0 => vec![],
            1 => vec![2],
            5 => vec![2, 2, 2, 2, 2],
            _ => unreachable!(),
        };

        let result = graph.new_basic_input_node(&shape, None);
        assert_err!(
            result,
            GraphError::DimensionMismatch { expected, got, message }
                if *expected == 2 && *got == dims && message == &format!(
                    "节点张量必须是 2-4 维（支持 FC、RNN 和 CNN），但收到的维度是 {} 维。",
                    dims
                )
        );
    }

    // 3D 和 4D 现在应该成功（CNN 支持）
    assert!(graph.new_basic_input_node(&[3, 28, 28], Some("input_3d")).is_ok());
    assert!(
        graph
            .new_basic_input_node(&[4, 3, 28, 28], Some("input_4d"))
            .is_ok()
    );
}

#[test]
fn test_node_input_name_generation() {
    let mut graph = GraphInner::new();

    // 1. 测试节点显式命名
    let input1 = graph
        .new_basic_input_node(&[2, 2], Some("explicit_input"))
        .unwrap();
    assert_eq!(graph.get_node_name(input1).unwrap(), "explicit_input");

    // 2. 测试节点自动命名
    let input2 = graph.new_basic_input_node(&[2, 2], None).unwrap();
    assert_eq!(graph.get_node_name(input2).unwrap(), "input_1");

    // 3. 测试节点名称重复
    let result = graph.new_basic_input_node(&[2, 2], Some("explicit_input"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点explicit_input在图default_graph中重复")
    );
}

#[test]
fn test_node_input_manually_set_value() {
    let mut graph = GraphInner::new();
    let input = graph.new_basic_input_node(&[2, 2], Some("test_input")).unwrap();

    // 1. 测试有效赋值
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    {
        let cloned_tensor = test_value.clone();
        graph.set_node_value(input, Some(&cloned_tensor)).unwrap();
    } // cloned_tensor在这里被释放

    // 1.1 验证节点状态
    assert!(graph.is_node_inited(input).unwrap());
    assert_eq!(graph.get_node_value(input).unwrap().unwrap(), &test_value);

    // 2. 测试错误形状的赋值（特征维度不匹配）
    // 由于 Input 节点支持动态 batch，错误消息会提示"动态形状不兼容"
    let invalid_cases = [
        Tensor::new(&[1.0], &[1, 1]),           // 特征维度 [1] != [2]
        Tensor::new(&[1.0, 2.0], &[2, 1]),      // 特征维度 [1] != [2]
        Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]), // 特征维度 [1] != [2]
    ];
    for value in invalid_cases {
        let result = graph.set_node_value(input, Some(&value));
        assert_err!(
            result,
            GraphError::ShapeMismatch { message, .. }
                if message.contains("动态形状") && message.contains("不兼容")
        );
    }

    // 3. 不同的 batch 大小应该成功（动态 batch）
    let different_batch = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    graph.set_node_value(input, Some(&different_batch)).unwrap();

    // 3. 测试设置空值（清除值）
    graph.set_node_value(input, None).unwrap();
    assert!(!graph.is_node_inited(input).unwrap());
    assert!(graph.get_node_value(input).unwrap().is_none());
}

#[test]
fn test_node_input_expected_shape() {
    let mut graph = GraphInner::new();

    // 1. 测试基本的Input节点预期形状
    let input = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();
    assert_eq!(graph.get_node_value_shape(input).unwrap(), None); // 实际值形状为None
    assert_eq!(graph.get_node_value_expected_shape(input).unwrap(), &[2, 3]); // 预期形状已确定

    // 2. 设置值后检查
    let value = Tensor::zeros(&[2, 3]);
    graph.set_node_value(input, Some(&value)).unwrap();
    assert_eq!(graph.get_node_value_shape(input).unwrap().unwrap(), &[2, 3]); // 设置值后实际形状
    assert_eq!(graph.get_node_value_expected_shape(input).unwrap(), &[2, 3]); // 预期形状保持不变

    // 3. 清除值后检查
    graph.set_node_value(input, None).unwrap();
    assert_eq!(graph.get_node_value_shape(input).unwrap(), None); // 清除后实际值形状为None
    assert_eq!(graph.get_node_value_expected_shape(input).unwrap(), &[2, 3]); // 预期形状仍然保持
}

#[test]
fn test_node_input_forward_propagation() {
    let mut graph = GraphInner::new();
    let input = graph.new_basic_input_node(&[2, 2], Some("input")).unwrap();

    // 1. 测试前向传播（应该失败，因为Input节点不支持前向传播）
    assert_err!(
        graph.forward(input),
        GraphError::InvalidOperation(
            "节点[id=1, name=input, type=Input]是输入/参数/状态节点，其值应通过set_value设置，而不是通过父节点前向传播计算"
        )
    );

    // 2. 设置值后仍然不能前向传播
    let value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();
    assert_err!(
        graph.forward(input),
        GraphError::InvalidOperation(
            "节点[id=1, name=input, type=Input]是输入/参数/状态节点，其值应通过set_value设置，而不是通过父节点前向传播计算"
        )
    );
}

/// 测试 Input 节点不应该有梯度（VJP 模式）
///
/// Input 节点是输入数据，不是可学习参数，因此不应该有梯度。
#[test]
fn test_node_input_no_grad() {
    let mut graph = GraphInner::new();

    // 1. 创建输入节点
    let input = graph.new_basic_input_node(&[2, 2], Some("input")).unwrap();
    let value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // 2. 对 Input 节点调用 get_node_grad 应该返回错误
    assert_err!(
        graph.get_node_grad(input),
        GraphError::InvalidOperation(msg) if msg == "输入节点[id=1, name=input, type=Input]不应该有梯度"
    );

    // 3. 设置值后，get_node_grad 仍应返回错误
    graph.set_node_value(input, Some(&value)).unwrap();
    assert_err!(
        graph.get_node_grad(input),
        GraphError::InvalidOperation(msg) if msg == "输入节点[id=1, name=input, type=Input]不应该有梯度"
    );

    // 4. zero_grad 后，get_node_grad 仍应返回错误
    graph.zero_grad().unwrap();
    assert_err!(
        graph.get_node_grad(input),
        GraphError::InvalidOperation(msg) if msg == "输入节点[id=1, name=input, type=Input]不应该有梯度"
    );
}

/// 测试 Input 节点在正常计算图反向传播后的行为
///
/// 在完整的计算图中，反向传播到 Input 节点时会无害跳过（Input 是"梯度汇点"），
/// 调用 get_node_grad(input) 仍然返回错误。
#[test]
fn test_node_input_in_computation_graph() {
    let mut graph = GraphInner::new();

    // 1. 构建简单计算图: input -> param -> mse_loss
    let input = graph.new_basic_input_node(&[2, 2], Some("input")).unwrap();
    let param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();
    let target = graph.new_basic_input_node(&[2, 2], Some("target")).unwrap();

    // input * param
    let mul = graph.new_multiply_node(input, param, None).unwrap();
    // MSE Loss
    let loss = graph.new_mse_loss_node(mul, target, None).unwrap();

    // 2. 设置输入值
    let input_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let target_val = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);
    graph.set_node_value(input, Some(&input_val)).unwrap();
    graph.set_node_value(target, Some(&target_val)).unwrap();

    // 3. 前向传播
    graph.forward(loss).unwrap();

    // 4. 反向传播
    graph.zero_grad().unwrap();
    let _loss_val = graph.backward(loss).unwrap();

    // 5. 验证：Parameter 节点应该有梯度
    assert!(graph.get_node_grad(param).unwrap().is_some());

    // 6. 验证：Input 节点仍然不应该有梯度（调用 get_node_grad 返回错误）
    //    反向传播到 Input 节点时是无害跳过的，不会报错，但 get_node_grad 仍返回错误
    assert_err!(
        graph.get_node_grad(input),
        GraphError::InvalidOperation(msg) if msg == "输入节点[id=1, name=input, type=Input]不应该有梯度"
    );
    assert_err!(
        graph.get_node_grad(target),
        GraphError::InvalidOperation(msg) if msg == "输入节点[id=3, name=target, type=Input]不应该有梯度"
    );
}

// ==================== 动态形状测试 ====================

/// 测试 Input 节点的动态形状传播
///
/// Input 节点是动态 batch 的源头，其 dynamic_expected_shape 的第一维应为 None
#[test]
fn test_input_dynamic_shape_propagation() {
    use crate::nn::Graph;

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
    use crate::nn::Graph;

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
    use crate::nn::Graph;

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
    use crate::nn::Graph;
    use crate::nn::var_ops::VarActivationOps;

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
    use crate::nn::Graph;
    use crate::nn::var_ops::{VarActivationOps, VarLossOps};

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

// ==================== SmartInput 测试 ====================
//
// SmartInput 是 ModelState 内部使用的特殊节点，用于实现：
// - 统一缓存（所有输入类型使用相同的计算路径）
// - 动态 detached 状态
// - 梯度路由（将梯度传回源 Var）

use crate::nn::var_ops::{VarLossOps, VarMatrixOps};
use crate::nn::{Graph, Init};

/// 测试: SmartInput 基本创建和值设置
#[test]
fn test_smart_input_basic() {
    let graph = Graph::new();

    // 创建 SmartInput 节点
    let router_id = graph
        .inner_mut()
        .new_smart_input_node(&[2, 3], Some("test_router"))
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

/// 测试: SmartInput 动态 detached 状态
#[test]
fn test_smart_input_dynamic_detached() {
    let graph = Graph::new();

    let router_id = graph
        .inner_mut()
        .new_smart_input_node(&[2, 2], None)
        .unwrap();

    // 默认非 detached
    assert!(!graph.inner().is_node_detached(router_id).unwrap());

    // 设置为 detached（显式 detach）
    graph
        .inner_mut()
        .set_router_detached(router_id, true, true)
        .unwrap();
    assert!(graph.inner().is_node_detached(router_id).unwrap());

    // 切换回非 detached
    graph
        .inner_mut()
        .set_router_detached(router_id, false, false)
        .unwrap();
    assert!(!graph.inner().is_node_detached(router_id).unwrap());
}

/// 测试: SmartInput 梯度路由设置
#[test]
fn test_smart_input_gradient_target() {
    let graph = Graph::new();

    // 创建一个普通节点作为路由目标
    let target = graph.parameter(&[2, 2], Init::Ones, "target").unwrap();
    let target_id = target.node_id();

    // 创建 SmartInput
    let router_id = graph
        .inner_mut()
        .new_smart_input_node(&[2, 2], None)
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

/// 测试: SmartInput 梯度路由功能（核心测试）
///
/// 使用参数节点作为梯度路由目标，验证梯度正确路由。
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
    let router_id = graph
        .inner_mut()
        .new_smart_input_node(&[1, 2], Some("router"))
        .unwrap();
    let router = crate::nn::Var::new(router_id, graph.inner_rc());

    // 设置 SmartInput 的值（从 fake 复制）
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

/// 测试: SmartInput detached 时不路由梯度
#[test]
fn test_smart_input_detached_no_routing() {
    let graph = Graph::new_with_seed(42);

    // D 的参数
    let d_w = graph.parameter(&[2, 1], Init::Ones, "d_w").unwrap();

    // G 的参数
    let g_w = graph.parameter(&[1, 2], Init::Ones, "g_w").unwrap();

    // 模拟 G 的输出
    let z = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let fake = z.matmul(&g_w).unwrap();

    // 创建 SmartInput 并设置为 detached（模拟训练 D）
    let router_id = graph
        .inner_mut()
        .new_smart_input_node(&[1, 2], None)
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
        .set_router_detached(router_id, true, true)
        .unwrap();

    // 构建 D 的计算
    let d_out = router.matmul(&d_w).unwrap();
    let target = graph.input(&Tensor::zeros(&[1, 1])).unwrap();
    let loss = d_out.mse_loss(&target).unwrap();

    // 反向传播
    loss.backward().unwrap();

    // d_w 应该有梯度（D 的参数）
    assert!(d_w.grad().unwrap().is_some(), "d_w 应该有梯度");

    // g_w 不应该有梯度（因为 SmartInput 是 detached）
    assert!(
        g_w.grad().unwrap().is_none(),
        "g_w 不应有梯度（SmartInput 是 detached）"
    );
}

/// 测试: SmartInput 可视化样式
#[test]
fn test_smart_input_visualization() {
    let graph = Graph::new();

    // 创建 SmartInput
    let _router_id = graph
        .inner_mut()
        .new_smart_input_node(&[2, 2], Some("test_router"))
        .unwrap();

    // 生成 DOT 字符串
    let dot = graph.inner().to_dot();

    // 验证 SmartInput 使用实线样式和灰色填充
    assert!(dot.contains("SmartInput"), "DOT 应包含 SmartInput 类型");
    assert!(dot.contains("#E0E0E0"), "SmartInput 应使用灰色填充");
}

/// 测试: SmartInput 支持动态 batch（类似 Keras）
///
/// SmartInput 只验证特征维度匹配，允许不同 batch_size 的值。
#[test]
fn test_smart_input_dynamic_batch() {
    let graph = Graph::new();

    // 创建 SmartInput（首次使用 batch=4）
    let router_id = graph
        .inner_mut()
        .new_smart_input_node(&[4, 3], Some("router"))
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

/// 测试: SmartInput 仍然验证特征维度
///
/// 虽然 batch 维度可变，但特征维度必须匹配。
#[test]
fn test_smart_input_feature_shape_must_match() {
    let graph = Graph::new();

    // 创建 SmartInput（特征维度=3）
    let router_id = graph
        .inner_mut()
        .new_smart_input_node(&[4, 3], Some("router"))
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
    if let Err(GraphError::ShapeMismatch { message, .. }) = result {
        assert!(
            message.contains("动态形状") && message.contains("不兼容"),
            "错误信息应提及动态形状不兼容: {}",
            message
        );
    }
}
