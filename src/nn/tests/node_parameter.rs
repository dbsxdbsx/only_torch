/*
 * @Author       : 老董
 * @Description  : Parameter 节点单元测试
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）
 * 2. 值设置测试
 * 3. 前向传播行为测试
 * 4. 反向传播测试（Parameter 是可训练参数，应有梯度）
 * 5. 动态形状测试（Parameter 形状固定，不支持动态 batch）
 */

use approx::assert_abs_diff_eq;

use crate::assert_err;
use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;

#[test]
fn test_node_parameter_creation() {
    let mut graph = GraphInner::new();

    // 1. 测试基本创建
    let param = graph.new_parameter_node(&[2, 3], Some("param1")).unwrap();

    // 1.1 验证基本属性
    assert_eq!(graph.get_node_name(param).unwrap(), "param1");
    assert_eq!(graph.get_node_parents(param).unwrap().len(), 0);
    assert_eq!(graph.get_node_children(param).unwrap().len(), 0);
    assert!(graph.is_node_inited(param).unwrap()); // Parameter节点创建时已初始化

    // 1.2 验证初始化值
    let value = graph.get_node_value(param).unwrap().unwrap();
    let mean = value.mean();
    let std_dev = value.std_dev();
    assert_abs_diff_eq!(mean, 0.0, epsilon = 0.1); // 均值应接近0
    assert_abs_diff_eq!(std_dev, 0.001, epsilon = 0.001); // 标准差应接近0.001
}

#[test]
fn test_node_parameter_creation_with_invalid_shape() {
    let mut graph = GraphInner::new();

    // 测试不同维度的形状（支持 2-4 维，0/1/5 维应该失败）
    for dims in [0, 1, 5] {
        let shape = match dims {
            0 => vec![],
            1 => vec![2],
            5 => vec![2, 2, 2, 2, 2],
            _ => unreachable!(),
        };

        let result = graph.new_parameter_node(&shape, None);
        assert_err!(
            result,
            GraphError::DimensionMismatch { expected, got, message }
                if *expected == 2 && *got == dims && message == &format!(
                    "参数张量必须是 2-4 维（支持 FC 权重和 CNN 卷积核），但收到的维度是 {} 维。",
                    dims
                )
        );
    }

    // 3D 和 4D 现在应该成功（CNN 卷积核支持）
    assert!(
        graph
            .new_parameter_node(&[16, 3, 3], Some("param_3d"))
            .is_ok()
    );
    assert!(
        graph
            .new_parameter_node(&[32, 16, 3, 3], Some("conv_kernel"))
            .is_ok()
    );
}

#[test]
fn test_node_parameter_name_generation() {
    let mut graph = GraphInner::new();

    // 1. 测试节点显式命名
    let param1 = graph
        .new_parameter_node(&[2, 2], Some("explicit_param"))
        .unwrap();
    assert_eq!(graph.get_node_name(param1).unwrap(), "explicit_param");

    // 2. 测试节点自动命名
    let param2 = graph.new_parameter_node(&[2, 2], None).unwrap();
    assert_eq!(graph.get_node_name(param2).unwrap(), "parameter_1");

    // 3. 测试节点名称重复
    let result = graph.new_parameter_node(&[2, 2], Some("explicit_param"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点explicit_param在图default_graph中重复")
    );
}

#[test]
fn test_node_parameter_manually_set_value() {
    let mut graph = GraphInner::new();
    let param = graph
        .new_parameter_node(&[2, 2], Some("test_param"))
        .unwrap();

    // 1. 测试有效赋值
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    {
        let cloned_tensor = test_value.clone();
        graph.set_node_value(param, Some(&cloned_tensor)).unwrap();
    } // cloned_tensor在这里被释放

    // 1.1 验证节点状态
    assert!(graph.is_node_inited(param).unwrap());
    assert_eq!(graph.get_node_value(param).unwrap().unwrap(), &test_value);

    // 2. 测试错误形状的赋值
    // 注意：Parameter 不支持动态 batch，所以任何形状不匹配都会报错
    let invalid_cases = [
        Tensor::new(&[1.0], &[1, 1]),
        Tensor::new(&[1.0, 2.0], &[2, 1]),
        Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]),
    ];
    for value in invalid_cases {
        let result = graph.set_node_value(param, Some(&value));
        assert!(result.is_err(), "形状不匹配应该返回错误");
    }

    // 3. 测试设置空值（清除值）
    graph.set_node_value(param, None).unwrap();
    assert!(!graph.is_node_inited(param).unwrap());
    assert!(graph.get_node_value(param).unwrap().is_none());
}

#[test]
fn test_node_parameter_expected_shape() {
    let mut graph = GraphInner::new();

    // 1. 测试基本的Parameter节点预期形状
    let param = graph.new_parameter_node(&[2, 3], Some("param")).unwrap();
    assert_eq!(graph.get_node_value_shape(param).unwrap().unwrap(), &[2, 3]); // 实际值形状（已初始化）
    assert_eq!(graph.get_node_value_expected_shape(param).unwrap(), &[2, 3]); // 预期形状已确定

    // 2. 设置新值后检查
    let value = Tensor::zeros(&[2, 3]);
    graph.set_node_value(param, Some(&value)).unwrap();
    assert_eq!(graph.get_node_value_shape(param).unwrap().unwrap(), &[2, 3]); // 设置值后实际形状
    assert_eq!(graph.get_node_value_expected_shape(param).unwrap(), &[2, 3]); // 预期形状保持不变

    // 3. 清除值后检查
    graph.set_node_value(param, None).unwrap();
    assert_eq!(graph.get_node_value_shape(param).unwrap(), None); // 清除后实际值形状为None
    assert_eq!(graph.get_node_value_expected_shape(param).unwrap(), &[2, 3]); // 预期形状仍然保持
}

/// 测试 Parameter 节点的前向传播行为
///
/// Parameter 节点是外部设置的参数，不从父节点计算。
/// - 没有值时：forward 报错
/// - 有值时：forward 静默成功（支持 RNN 缓存等场景）
#[test]
fn test_node_parameter_forward_propagation() {
    let mut graph = GraphInner::new();
    let param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();

    // Parameter 节点创建时默认有值（Xavier 初始化），所以 forward 应该成功
    assert!(
        graph.forward(param).is_ok(),
        "有值的 Parameter 节点应该允许 forward（静默成功）"
    );

    // 设置新值后，forward 仍然成功
    let value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    graph.set_node_value(param, Some(&value)).unwrap();
    assert!(
        graph.forward(param).is_ok(),
        "有值的 Parameter 节点应该允许 forward（静默成功）"
    );
}

/// 测试 Parameter 节点在完整计算图中的反向传播行为
///
/// Parameter 节点是可学习参数，在反向传播后应该有梯度。
#[test]
fn test_node_parameter_backward_propagation() {
    let mut graph = GraphInner::new();

    // 1. 构建计算图: input * param -> mse_loss
    let input = graph.new_basic_input_node(&[2, 2], Some("input")).unwrap();
    let param = graph.new_parameter_node(&[2, 2], Some("param")).unwrap();
    let target = graph.new_basic_input_node(&[2, 2], Some("target")).unwrap();

    let mul = graph.new_multiply_node(input, param, None).unwrap();
    let loss = graph.new_mse_loss_node(mul, target, None).unwrap();

    // 2. 设置输入值
    let input_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let param_val = Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2]);
    let target_val = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    graph.set_node_value(input, Some(&input_val)).unwrap();
    graph.set_node_value(param, Some(&param_val)).unwrap();
    graph.set_node_value(target, Some(&target_val)).unwrap();

    // 3. 初始时梯度应为空
    assert!(graph.get_node_grad(param).unwrap().is_none());

    // 4. 前向传播 + 反向传播
    graph.forward(loss).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss).unwrap();

    // 5. Parameter 节点应该有梯度
    let grad = graph.get_node_grad(param).unwrap();
    assert!(grad.is_some(), "Parameter 节点应该有梯度");

    // 6. 清除梯度并验证
    graph.zero_grad().unwrap();
    assert!(graph.get_node_grad(param).unwrap().is_none());
}

/// 测试 Parameter 节点的梯度值正确性
#[test]
fn test_node_parameter_gradient_correctness() {
    let mut graph = GraphInner::new();

    // 简单计算图: param -> mse_loss(param, target)
    // loss = mean((param - target)^2)
    // d_loss/d_param = 2 * (param - target) / n
    let param = graph.new_parameter_node(&[1, 2], Some("param")).unwrap();
    let target = graph.new_basic_input_node(&[1, 2], Some("target")).unwrap();
    let loss = graph.new_mse_loss_node(param, target, None).unwrap();

    // 设置值: param = [1.0, 2.0], target = [0.0, 0.0]
    let param_val = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let target_val = Tensor::new(&[0.0, 0.0], &[1, 2]);
    graph.set_node_value(param, Some(&param_val)).unwrap();
    graph.set_node_value(target, Some(&target_val)).unwrap();

    // 前向 + 反向
    graph.forward(loss).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss).unwrap();

    // 验证梯度
    // loss = mean((1-0)^2 + (2-0)^2) = mean(1 + 4) = 2.5
    // d_loss/d_param = 2 * (param - target) / 2 = (param - target)
    // d_loss/d_param = [1.0, 2.0]
    let grad = graph.get_node_grad(param).unwrap().unwrap();
    assert_eq!(grad.shape(), &[1, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 2.0, epsilon = 1e-5);
}

// ==================== 动态形状测试 ====================

/// 测试 Parameter 节点的动态形状（固定形状，不支持动态 batch）
///
/// Parameter 节点是权重参数，其形状不随 batch 变化：
/// - FC 权重：[in_features, out_features]
/// - CNN 卷积核：[C_out, C_in, kH, kW]
#[test]
fn test_parameter_dynamic_shape_fixed() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建 2D Parameter（FC 权重）
    let param = graph.new_parameter_node(&[16, 32], Some("fc_weight"))?;

    // 获取动态形状
    let node = graph.get_node(param)?;
    let dyn_shape = node.dynamic_expected_shape();

    // 所有维度都应该是固定的
    assert!(!dyn_shape.is_dynamic(0), "Parameter 第一维应该是固定的");
    assert!(!dyn_shape.is_dynamic(1), "Parameter 第二维应该是固定的");
    assert_eq!(dyn_shape.dim(0), Some(16));
    assert_eq!(dyn_shape.dim(1), Some(32));

    // 不支持动态 batch
    assert!(
        !node.supports_dynamic_batch(),
        "Parameter 节点不应该支持动态 batch"
    );

    Ok(())
}

/// 测试 Parameter 节点在不同维度下的固定形状
#[test]
fn test_parameter_dynamic_shape_various_dims() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 2D: FC 权重 [in, out]
    let param_2d = graph.new_parameter_node(&[16, 32], Some("fc_weight"))?;
    let node_2d = graph.get_node(param_2d)?;
    let dyn_2d = node_2d.dynamic_expected_shape();
    assert!(!dyn_2d.is_dynamic(0));
    assert!(!dyn_2d.is_dynamic(1));

    // 3D: 某些特殊权重
    let param_3d = graph.new_parameter_node(&[16, 3, 3], Some("weight_3d"))?;
    let node_3d = graph.get_node(param_3d)?;
    let dyn_3d = node_3d.dynamic_expected_shape();
    assert!(!dyn_3d.is_dynamic(0));
    assert!(!dyn_3d.is_dynamic(1));
    assert!(!dyn_3d.is_dynamic(2));

    // 4D: CNN 卷积核 [C_out, C_in, kH, kW]
    let param_4d = graph.new_parameter_node(&[32, 16, 3, 3], Some("conv_kernel"))?;
    let node_4d = graph.get_node(param_4d)?;
    let dyn_4d = node_4d.dynamic_expected_shape();
    assert!(!dyn_4d.is_dynamic(0), "4D: C_out 应该固定");
    assert!(!dyn_4d.is_dynamic(1), "4D: C_in 应该固定");
    assert!(!dyn_4d.is_dynamic(2), "4D: kH 应该固定");
    assert!(!dyn_4d.is_dynamic(3), "4D: kW 应该固定");

    Ok(())
}

/// 测试 Parameter 节点在不同 batch 输入下的行为
///
/// Parameter 形状固定，但可以与不同 batch 的输入进行运算
#[test]
fn test_parameter_with_dynamic_batch_input() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建 Input（支持动态 batch）和 Parameter（固定形状）
    let input = graph.new_basic_input_node(&[4, 16], Some("input"))?;
    let weight = graph.new_parameter_node(&[16, 32], Some("weight"))?;

    // MatMul: [batch, 16] @ [16, 32] -> [batch, 32]
    let output = graph.new_mat_mul_node(input, weight, Some("output"))?;

    // 设置初始值
    graph.set_node_value(input, Some(&Tensor::ones(&[4, 16])))?;
    graph.set_node_value(weight, Some(&Tensor::ones(&[16, 32])))?;

    // 第一次 forward：batch=4
    graph.forward(output)?;
    let value1 = graph.get_node_value(output)?.unwrap();
    assert_eq!(value1.shape(), &[4, 32], "第一次 forward: batch=4");

    // 更新 Input 为不同 batch
    graph.set_node_value(input, Some(&Tensor::ones(&[8, 16])))?;

    // 第二次 forward：batch=8（Parameter 形状不变）
    graph.forward(output)?;
    let value2 = graph.get_node_value(output)?.unwrap();
    assert_eq!(value2.shape(), &[8, 32], "第二次 forward: batch=8");

    // 验证 Parameter 形状仍然固定
    let weight_val = graph.get_node_value(weight)?.unwrap();
    assert_eq!(weight_val.shape(), &[16, 32], "Parameter 形状应保持不变");

    Ok(())
}

/// 测试 Parameter 梯度在不同 batch 下的行为
#[test]
fn test_parameter_gradient_with_dynamic_batch() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    graph.set_train_mode();

    // 创建网络：input @ weight -> output -> loss
    let input = graph.new_basic_input_node(&[2, 4], Some("input"))?;
    let weight = graph.new_parameter_node(&[4, 8], Some("weight"))?;
    let output = graph.new_mat_mul_node(input, weight, Some("output"))?;
    let target = graph.new_basic_input_node(&[2, 8], Some("target"))?;
    let loss = graph.new_mse_loss_node(output, target, Some("loss"))?;

    // 设置初始值
    graph.set_node_value(input, Some(&Tensor::ones(&[2, 4])))?;
    graph.set_node_value(weight, Some(&Tensor::normal_seeded(0.0, 0.1, &[4, 8], 42)))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 8])))?;

    // 第一次训练：batch=2
    graph.forward(loss)?;
    graph.zero_grad()?;
    graph.backward(loss)?;
    let grad1 = graph.get_node_grad(weight)?.unwrap().clone();
    assert_eq!(grad1.shape(), &[4, 8], "梯度形状应与 Parameter 一致");

    // 更新为不同 batch
    graph.set_node_value(input, Some(&Tensor::ones(&[6, 4])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[6, 8])))?;

    // 第二次训练：batch=6
    graph.forward(loss)?;
    graph.zero_grad()?;
    graph.backward(loss)?;
    let grad2 = graph.get_node_grad(weight)?.unwrap();
    assert_eq!(
        grad2.shape(),
        &[4, 8],
        "梯度形状应始终与 Parameter 一致（不随 batch 变化）"
    );

    Ok(())
}
