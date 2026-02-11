/*
 * @Author       : 老董
 * @Description  : Parameter 节点单元测试
 *
 * 测试策略：
 * 1. 基础功能测试（创建+初始化、无效形状、命名）
 * 2. 值设置测试（有效/无效形状/清除、预期形状）
 * 3. 前向传播行为测试（有初始值时 forward 应成功）
 * 4. 反向传播+梯度正确性测试（Parameter 有梯度、梯度值验证）
 * 5. 动态形状测试（Parameter 固定形状、与动态 batch Input 配合）
 * 6. Create API 测试（底层 GraphInner 创建 API）
 */

use approx::assert_abs_diff_eq;

use crate::nn::{Graph, Init, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;
use std::rc::Rc;

// ==================== 1. 基础功能测试 ====================

/// 测试 Parameter 节点创建与不同初始化方式
#[test]
fn test_node_parameter_creation() {
    let graph = Graph::new();

    // 1. Zeros 初始化
    let param_z = graph.parameter(&[2, 3], Init::Zeros, "p_zeros").unwrap();
    let val_z = param_z.value().unwrap().unwrap();
    assert_eq!(val_z.shape(), &[2, 3]);
    for &v in val_z.data_as_slice() {
        assert_abs_diff_eq!(v, 0.0, epsilon = 1e-7);
    }

    // 2. Ones 初始化
    let param_o = graph.parameter(&[3, 2], Init::Ones, "p_ones").unwrap();
    let val_o = param_o.value().unwrap().unwrap();
    assert_eq!(val_o.shape(), &[3, 2]);
    for &v in val_o.data_as_slice() {
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-7);
    }

    // 3. Constant 初始化
    let param_c = graph
        .parameter(&[2, 2], Init::Constant(0.5), "p_const")
        .unwrap();
    let val_c = param_c.value().unwrap().unwrap();
    for &v in val_c.data_as_slice() {
        assert_abs_diff_eq!(v, 0.5, epsilon = 1e-7);
    }

    // 4. Kaiming 初始化（均值接近 0）
    let param_k = graph
        .parameter(&[64, 32], Init::Kaiming, "p_kaiming")
        .unwrap();
    let val_k = param_k.value().unwrap().unwrap();
    assert_eq!(val_k.shape(), &[64, 32]);
    let data = val_k.data_as_slice();
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert_abs_diff_eq!(mean, 0.0, epsilon = 0.05);
}

/// 测试 Parameter 节点创建时无效形状应报错
#[test]
fn test_node_parameter_creation_with_invalid_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 0D / 1D / 5D 均应失败
    assert!(
        inner.borrow_mut().create_parameter_node(&[], None).is_err(),
        "0D 形状应失败"
    );
    assert!(
        inner
            .borrow_mut()
            .create_parameter_node(&[10], None)
            .is_err(),
        "1D 形状应失败"
    );
    assert!(
        inner
            .borrow_mut()
            .create_parameter_node(&[2, 2, 2, 2, 2], None)
            .is_err(),
        "5D 形状应失败"
    );

    // 2D 应成功（FC 权重 [in, out]）
    assert!(graph.parameter(&[4, 8], Init::Zeros, "fc_w").is_ok());

    // 3D 应成功（某些特殊权重）
    assert!(
        inner
            .borrow_mut()
            .create_parameter_node(&[16, 3, 3], Some("p3d"))
            .is_ok()
    );

    // 4D 应成功（CNN 卷积核 [C_out, C_in, kH, kW]）
    assert!(
        inner
            .borrow_mut()
            .create_parameter_node(&[32, 16, 3, 3], Some("conv"))
            .is_ok()
    );
}

/// 测试 Parameter 节点的命名（显式/自动/重复）
#[test]
fn test_node_parameter_name_generation() {
    let graph = Graph::new();

    // 1. 显式命名
    let param1 = graph
        .parameter(&[2, 2], Init::Zeros, "explicit_param")
        .unwrap();
    assert_eq!(param1.name(), Some("explicit_param"));

    // 2. 底层 API 自动命名
    let inner = graph.inner_rc();
    let param_auto = inner
        .borrow_mut()
        .create_parameter_node(&[2, 2], None)
        .unwrap();
    let name = param_auto.name().unwrap();
    assert!(
        name.contains("parameter"),
        "自动名称应包含 'parameter': {}",
        name
    );

    // 3. 重复名称应报错
    let result = graph.parameter(&[2, 2], Init::Zeros, "explicit_param");
    assert!(result.is_err(), "重复名称应该报错");
}

// ==================== 2. 值设置测试 ====================

/// 测试 Parameter 节点手动设置值（有效赋值与清除）
#[test]
fn test_node_parameter_manually_set_value() {
    let graph = Graph::new();
    let param = graph.parameter(&[2, 2], Init::Zeros, "param").unwrap();

    // 1. 有效赋值
    let new_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    param.set_value(&new_val).unwrap();
    let val = param.value().unwrap().unwrap();
    assert_abs_diff_eq!(val[[0, 0]], 1.0, epsilon = 1e-7);
    assert_abs_diff_eq!(val[[0, 1]], 2.0, epsilon = 1e-7);
    assert_abs_diff_eq!(val[[1, 0]], 3.0, epsilon = 1e-7);
    assert_abs_diff_eq!(val[[1, 1]], 4.0, epsilon = 1e-7);

    // 2. 再次赋值（覆盖）
    let override_val = Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    param.set_value(&override_val).unwrap();
    let val2 = param.value().unwrap().unwrap();
    assert_abs_diff_eq!(val2[[0, 0]], 10.0, epsilon = 1e-7);
    assert_abs_diff_eq!(val2[[1, 1]], 40.0, epsilon = 1e-7);

    // 3. 清除值（通过底层 NodeInner API）
    param.node().set_value(None).unwrap();
    assert!(param.node().value().is_none(), "清除后值应为 None");
}

/// 测试 Parameter 节点不支持动态 batch（形状完全固定）
///
/// Parameter 与 Input 不同：Input 第一维是动态的（batch），
/// 而 Parameter 所有维度都是固定的。
#[test]
fn test_node_parameter_shape_is_fixed() {
    let graph = Graph::new();

    // Parameter [2, 3] 的所有维度都应固定
    let param = graph.parameter(&[2, 3], Init::Zeros, "w").unwrap();
    let dyn_shape = param.dynamic_expected_shape();

    // 两个维度都不是动态的
    assert!(!dyn_shape.is_dynamic(0), "Parameter 行维度应固定");
    assert!(!dyn_shape.is_dynamic(1), "Parameter 列维度应固定");

    // 对比 Input 节点：第一维是动态的
    let input = graph.input(&Tensor::zeros(&[4, 3])).unwrap();
    let input_dyn = input.dynamic_expected_shape();
    assert!(input_dyn.is_dynamic(0), "Input batch 维度应动态");
    assert!(!input_dyn.is_dynamic(1), "Input 特征维度应固定");
}

/// 测试 Parameter 节点的预期形状（设置值/清除值后保持不变）
#[test]
fn test_node_parameter_expected_shape() {
    let graph = Graph::new();
    let param = graph.parameter(&[2, 3], Init::Zeros, "param").unwrap();

    // 1. 初始值形状
    let val = param.value().unwrap().unwrap();
    assert_eq!(val.shape(), &[2, 3]);
    assert_eq!(param.node().shape(), vec![2, 3]);

    // 2. 设置新值后，预期形状保持不变
    param.set_value(&Tensor::ones(&[2, 3])).unwrap();
    assert_eq!(param.node().shape(), vec![2, 3]);

    // 3. 清除值后，预期形状仍保持
    param.node().set_value(None).unwrap();
    assert_eq!(param.node().shape(), vec![2, 3], "预期形状应保持 [2, 3]");
}

// ==================== 3. 前向传播行为测试 ====================

/// 测试 Parameter 节点的前向传播行为
///
/// Parameter 创建时已有初始化值，forward 应静默成功。
#[test]
fn test_node_parameter_forward_propagation() {
    let graph = Graph::new();
    let param = graph.parameter(&[2, 2], Init::Ones, "param").unwrap();

    // 有初始化值，forward 应成功
    assert!(
        param.forward().is_ok(),
        "有值的 Parameter 节点 forward 应成功"
    );

    // 设置新值后，forward 仍成功
    param
        .set_value(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    assert!(
        param.forward().is_ok(),
        "设置新值后 Parameter forward 应成功"
    );
}

// ==================== 4. 反向传播+梯度正确性测试 ====================

/// 测试 Parameter 节点在计算图中有梯度
///
/// 构建 input * param -> mse_loss，验证 Parameter 反向传播后有梯度。
#[test]
fn test_node_parameter_backward_propagation() {
    let graph = Graph::new();

    // 构建计算图: input * param -> mse_loss(result, target)
    let input = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let param = graph.parameter(&[2, 2], Init::Zeros, "param").unwrap();
    param
        .set_value(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]))
        .unwrap();

    let mul = &input * &param; // element-wise multiply
    let loss = mul.mse_loss(&target).unwrap();

    // 初始时梯度为空
    assert!(param.grad().unwrap().is_none(), "初始梯度应为空");

    // 反向传播（backward 内部 ensure-forward）
    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // Parameter 应有梯度
    let grad = param.grad().unwrap();
    assert!(grad.is_some(), "Parameter 节点应该有梯度");

    // 清除梯度并验证
    graph.zero_grad().unwrap();
    assert!(param.grad().unwrap().is_none(), "清除后梯度应为空");
}

/// 测试 Parameter 节点的梯度值正确性
///
/// 简单计算图: `loss = mean((param - target)^2)`
/// `d_loss/d_param = 2 * (param - target) / n`
#[test]
fn test_node_parameter_gradient_correctness() {
    let graph = Graph::new();

    // param = [[1.0, 2.0]], target = [[0.0, 0.0]]
    let param = graph.parameter(&[1, 2], Init::Zeros, "param").unwrap();
    param
        .set_value(&Tensor::new(&[1.0, 2.0], &[1, 2]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.0, 0.0], &[1, 2]))
        .unwrap();

    // loss = mean((1-0)^2 + (2-0)^2) = mean(1 + 4) = 2.5
    let loss = param.mse_loss(&target).unwrap();

    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    // d_loss/d_param = 2 * (param - target) / 2 = (param - target) = [1.0, 2.0]
    let grad = param.grad().unwrap().expect("应有梯度");
    assert_eq!(grad.shape(), &[1, 2]);
    assert_abs_diff_eq!(grad[[0, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad[[0, 1]], 2.0, epsilon = 1e-5);
}

// ==================== 5. 动态形状测试 ====================

/// 测试 Parameter 节点的形状是固定的（所有维度非动态）
#[test]
fn test_parameter_dynamic_shape_fixed() {
    let graph = Graph::new();

    // 2D: FC 权重
    let param_2d = graph
        .parameter(&[16, 32], Init::Zeros, "fc_weight")
        .unwrap();
    let dyn_2d = param_2d.dynamic_expected_shape();
    assert!(!dyn_2d.is_dynamic(0), "第一维应固定");
    assert!(!dyn_2d.is_dynamic(1), "第二维应固定");
    assert_eq!(dyn_2d.dim(0), Some(16));
    assert_eq!(dyn_2d.dim(1), Some(32));

    // 4D: CNN 卷积核（通过底层 API 创建）
    let inner = graph.inner_rc();
    let node_4d = inner
        .borrow_mut()
        .create_parameter_node(&[32, 16, 3, 3], Some("conv_kernel"))
        .unwrap();
    let dyn_4d = node_4d.dynamic_expected_shape();
    assert!(!dyn_4d.is_dynamic(0), "C_out 应固定");
    assert!(!dyn_4d.is_dynamic(1), "C_in 应固定");
    assert!(!dyn_4d.is_dynamic(2), "kH 应固定");
    assert!(!dyn_4d.is_dynamic(3), "kW 应固定");
}

/// 测试 Parameter 节点可与不同 batch 的 Input 节点运算
///
/// Parameter 形状固定（如 [16, 32]），但 Input 可以改变 batch 大小。
/// MatMul: `[batch, 16] @ [16, 32]` -> `[batch, 32]`
#[test]
fn test_parameter_with_dynamic_batch_input() {
    let graph = Graph::new();

    let input = graph.input(&Tensor::ones(&[4, 16])).unwrap();
    let weight = graph.parameter(&[16, 32], Init::Ones, "weight").unwrap();
    let output = input.matmul(&weight).unwrap(); // [4,16] @ [16,32] -> [4,32]

    // 第一次 forward: batch=4
    output.forward().unwrap();
    let val1 = output.value().unwrap().unwrap();
    assert_eq!(val1.shape(), &[4, 32], "batch=4 时输出形状应为 [4, 32]");

    // 更新 Input 为 batch=8
    input.set_value(&Tensor::ones(&[8, 16])).unwrap();
    output.forward().unwrap();
    let val2 = output.value().unwrap().unwrap();
    assert_eq!(val2.shape(), &[8, 32], "batch=8 时输出形状应为 [8, 32]");

    // Parameter 形状始终不变
    let w_val = weight.value().unwrap().unwrap();
    assert_eq!(w_val.shape(), &[16, 32], "Parameter 形状应保持不变");
}

/// 测试 Parameter 梯度形状不随 batch 变化
///
/// 不论 Input 的 batch 大小如何变化，Parameter 的梯度形状始终与自身形状一致。
#[test]
fn test_parameter_gradient_with_dynamic_batch() {
    let graph = Graph::new_with_seed(42);
    graph.train();

    let input = graph.input(&Tensor::ones(&[2, 4])).unwrap();
    let weight = graph
        .parameter(
            &[4, 8],
            Init::Normal {
                mean: 0.0,
                std: 0.1,
            },
            "weight",
        )
        .unwrap();
    let output = input.matmul(&weight).unwrap();
    let target = graph.input(&Tensor::zeros(&[2, 8])).unwrap();
    let loss = output.mse_loss(&target).unwrap();

    // 第一次训练: batch=2
    graph.zero_grad().unwrap();
    loss.backward().unwrap();
    let grad1 = weight.grad().unwrap().expect("应有梯度");
    assert_eq!(grad1.shape(), &[4, 8], "梯度形状应与 Parameter 一致");

    // 更新为 batch=6
    input.set_value(&Tensor::ones(&[6, 4])).unwrap();
    target.set_value(&Tensor::zeros(&[6, 8])).unwrap();

    graph.zero_grad().unwrap();
    loss.backward().unwrap();
    let grad2 = weight.grad().unwrap().expect("应有梯度");
    assert_eq!(
        grad2.shape(),
        &[4, 8],
        "梯度形状应始终与 Parameter 一致（不随 batch 变化）"
    );
}

// ==================== 6. Create API 测试 ====================

#[test]
fn test_create_parameter_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 创建 Parameter 节点
    let param = inner
        .borrow_mut()
        .create_parameter_node(&[4, 8], Some("weight"))
        .unwrap();

    // 验证节点属性
    assert_eq!(param.shape(), vec![4, 8]);
    assert_eq!(param.name(), Some("weight"));
    assert!(param.is_leaf());
    assert!(param.parents().is_empty());

    // 验证有初始化值
    assert!(param.value().is_some());
}

#[test]
fn test_create_parameter_auto_name() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let param = inner
        .borrow_mut()
        .create_parameter_node(&[4, 8], None)
        .unwrap();

    let name = param.name().unwrap();
    assert!(
        name.contains("parameter"),
        "名称应包含 'parameter': {}",
        name
    );
}

#[test]
fn test_create_parameter_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak;
    {
        let param = inner
            .borrow_mut()
            .create_parameter_node(&[4, 8], None)
            .unwrap();
        weak = Rc::downgrade(&param);
        assert!(weak.upgrade().is_some());
    }
    // param 离开作用域，节点被释放
    assert!(weak.upgrade().is_none());
}

#[test]
fn test_create_parameter_seeded() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 使用相同种子创建两个参数
    let param1 = inner
        .borrow_mut()
        .create_parameter_node_seeded(&[4, 8], Some("w1"), 42)
        .unwrap();
    let param2 = inner
        .borrow_mut()
        .create_parameter_node_seeded(&[4, 8], Some("w2"), 42)
        .unwrap();

    // 验证值相同（相同种子）
    let v1 = param1.value().unwrap();
    let v2 = param2.value().unwrap();
    assert_eq!(v1.data_as_slice(), v2.data_as_slice());
}

#[test]
fn test_create_parameter_various_shapes() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // FC 权重 [in_features, out_features]
    let fc_weight = inner
        .borrow_mut()
        .create_parameter_node(&[784, 256], None)
        .unwrap();
    assert_eq!(fc_weight.shape(), vec![784, 256]);

    // CNN 卷积核 [C_out, C_in, kH, kW]
    let conv_kernel = inner
        .borrow_mut()
        .create_parameter_node(&[64, 3, 3, 3], None)
        .unwrap();
    assert_eq!(conv_kernel.shape(), vec![64, 3, 3, 3]);
}

#[test]
fn test_create_parameter_invalid_shape() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 1D 形状应该失败
    let result = inner.borrow_mut().create_parameter_node(&[10], None);
    assert!(result.is_err());

    // 5D 形状也应该失败
    let result = inner
        .borrow_mut()
        .create_parameter_node(&[1, 2, 3, 4, 5], None);
    assert!(result.is_err());
}
