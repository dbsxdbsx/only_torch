use approx::assert_abs_diff_eq;

use crate::nn::optimizer::{Optimizer, SGD};
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_node_sigmoid_creation() {
    let mut graph = Graph::new();

    // 1. 测试 Input 节点作为父节点
    {
        let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
        let sigmoid = graph
            .new_sigmoid_node(input, Some("sigmoid_with_input"))
            .unwrap();
        // 1.1 验证基本属性
        assert_eq!(graph.get_node_name(sigmoid).unwrap(), "sigmoid_with_input");
        assert_eq!(graph.get_node_parents(sigmoid).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(sigmoid).unwrap().len(), 0);
    }

    // 2. 测试 Parameter 节点作为父节点
    {
        let param = graph.new_parameter_node(&[2, 2], Some("param1")).unwrap();
        let sigmoid = graph
            .new_sigmoid_node(param, Some("sigmoid_with_param"))
            .unwrap();
        assert_eq!(graph.get_node_name(sigmoid).unwrap(), "sigmoid_with_param");
        assert_eq!(graph.get_node_parents(sigmoid).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(sigmoid).unwrap().len(), 0);
    }
}

#[test]
fn test_node_sigmoid_name_generation() {
    let mut graph = Graph::new();

    // 1. 测试节点显式命名
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let sigmoid = graph
        .new_sigmoid_node(input, Some("explicit_sigmoid"))
        .unwrap();
    assert_eq!(graph.get_node_name(sigmoid).unwrap(), "explicit_sigmoid");

    // 2. 测试节点自动命名
    let sigmoid2 = graph.new_sigmoid_node(input, None).unwrap();
    assert_eq!(graph.get_node_name(sigmoid2).unwrap(), "sigmoid_1");

    // 3. 测试节点名称重复
    let result = graph.new_sigmoid_node(input, Some("explicit_sigmoid"));
    assert_eq!(
        result,
        Err(GraphError::DuplicateNodeName(
            "节点explicit_sigmoid在图default_graph中重复".to_string()
        ))
    );
}

#[test]
fn test_node_sigmoid_manually_set_value() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let sigmoid = graph.new_sigmoid_node(input, Some("sigmoid")).unwrap();

    // 1. 测试直接设置 Sigmoid 节点的值（应该失败）
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(
        graph.set_node_value(sigmoid, Some(&test_value)),
        Err(GraphError::InvalidOperation(
            "节点[id=2, name=sigmoid, type=Sigmoid]的值只能通过前向传播计算得到，不能直接设置"
                .into()
        ))
    );

    // 2. 测试清除 Sigmoid 节点的值（也应该失败）
    assert_eq!(
        graph.set_node_value(sigmoid, None),
        Err(GraphError::InvalidOperation(
            "节点[id=2, name=sigmoid, type=Sigmoid]的值只能通过前向传播计算得到，不能直接设置"
                .into()
        ))
    );
}

#[test]
fn test_node_sigmoid_expected_shape() {
    let mut graph = Graph::new();

    // 1. 测试基本的 Sigmoid 节点预期形状
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let sigmoid = graph.new_sigmoid_node(input, Some("sigmoid")).unwrap();
    assert_eq!(
        graph.get_node_value_expected_shape(sigmoid).unwrap(),
        &[2, 2]
    );
    assert_eq!(graph.get_node_value_shape(sigmoid).unwrap(), None); // 实际值形状为 None（未计算）

    // 2. 测试前向传播后的形状
    let value = Tensor::zeros(&[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();
    graph.forward_node(sigmoid).unwrap();

    // 2.1 验证前向传播后的形状
    assert_eq!(
        graph.get_node_value_shape(sigmoid).unwrap().unwrap(),
        &[2, 2]
    ); // 实际值形状
    assert_eq!(
        graph.get_node_value_expected_shape(sigmoid).unwrap(),
        &[2, 2]
    ); // 预期形状保持不变

    // 2.2 测试父节点值在首次前向传播后，再次设置新值后的形状检查
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();

    // 2.2.1 验证预期形状和实际形状
    assert_eq!(
        graph.get_node_value_expected_shape(sigmoid).unwrap(),
        &[2, 2]
    );
    assert_eq!(
        graph.get_node_value_shape(sigmoid).unwrap().unwrap(),
        &[2, 2]
    ); // 虽然值已过期，但由于值仍然存在，所以形状不变
}

#[test]
fn test_node_sigmoid_forward_propagation() {
    // 1. 准备测试数据 (与 Python 测试 tests/calc_jacobi_by_pytorch/node_sigmoid.py 保持一致)
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    let expected = Tensor::new(
        &[
            0.622459352016449,
            0.2689414322376251,
            0.5,
            0.8807970285415649,
        ],
        &[2, 2],
    );

    // 2. 测试不同节点类型组合的前向传播
    let node_types = ["input", "parameter"];
    for parent_type in node_types {
        let mut graph = Graph::new();

        // 创建 parent 节点
        let parent = match parent_type {
            "input" => graph.new_input_node(&[2, 2], Some("input_1")).unwrap(),
            "parameter" => graph
                .new_parameter_node(&[2, 2], Some("parameter_1"))
                .unwrap(),
            _ => unreachable!(),
        };

        // Sigmoid 节点总是可训练的
        let sigmoid = graph.new_sigmoid_node(parent, Some("sigmoid")).unwrap();

        // 如果节点是 parameter，因创建时其值已隐式初始化过了，所以前向传播应成功
        if parent_type == "parameter" {
            graph.forward_node(sigmoid).unwrap();
        } else {
            // 如果是 input 节点，因创建时其值未初始化，所以前向传播应失败
            assert_eq!(
                graph.forward_node(sigmoid),
                Err(GraphError::InvalidOperation(format!(
                    "节点[id=1, name=input_1, type=Input]不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为0，而图的前向传播次数为1",
                )))
            );

            // 设置 input 节点的值
            graph.set_node_value(parent, Some(&value)).unwrap();

            // 设置值后前向传播应成功
            graph.forward_node(sigmoid).unwrap();
            let result = graph.get_node_value(sigmoid).unwrap().unwrap();

            // 只有当节点是 input 时才检查输出值
            if parent_type == "input" {
                assert_eq!(result, &expected);
            }
        }
    }
}

#[test]
fn test_node_sigmoid_backward_propagation() {
    let mut graph = Graph::new();

    // 1. 创建一个简单的 sigmoid 图：result = sigmoid(parent)
    let parent = graph.new_parameter_node(&[2, 2], Some("parent")).unwrap();
    let result = graph.new_sigmoid_node(parent, Some("result")).unwrap();

    // 2. 测试在前向传播之前进行反向传播（应该失败）
    assert_eq!(
        graph.backward_nodes(&[parent], result),
        Err(GraphError::ComputationError(format!(
            "反向传播：结果节点[id=2, name=result, type=Sigmoid]没有值"
        )))
    );

    // 3. 设置输入值 (与 Python 测试 tests/calc_jacobi_by_pytorch/node_sigmoid.py 保持一致)
    let parent_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(parent, Some(&parent_value)).unwrap();

    // 4. 反向传播前执行必要的前向传播
    graph.forward_node(result).unwrap();

    // 5. 反向传播
    // 5.1 sigmoid 节点 result 本身的雅可比矩阵至始至终都应为 None
    assert!(graph.get_node_jacobi(result).unwrap().is_none());

    // 5.2 对 parent 的反向传播（第 1 次，retain_graph=true 以便多次 backward）
    graph.backward_nodes_ex(&[parent], result, true).unwrap();
    let parent_jacobi = graph.get_node_jacobi(parent).unwrap().unwrap();

    // 验证雅可比矩阵（与 Python 输出一致）
    // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))，对角矩阵
    #[rustfmt::skip]
    let expected_jacobi = Tensor::new(
        &[
            0.23500370979309082, 0.0, 0.0, 0.0,
            0.0, 0.1966119408607483, 0.0, 0.0,
            0.0, 0.0, 0.25, 0.0,
            0.0, 0.0, 0.0, 0.10499362647533417,
        ],
        &[4, 4],
    );
    assert_eq!(parent_jacobi, &expected_jacobi);

    // 5.3 对 parent 的反向传播（第 2 次）- 梯度应该累积
    graph.backward_nodes_ex(&[parent], result, true).unwrap();
    let parent_jacobi_second = graph.get_node_jacobi(parent).unwrap().unwrap();
    assert_eq!(parent_jacobi_second, &(&expected_jacobi * 2.0));

    // 6. 清除雅可比矩阵并验证
    graph.clear_jacobi().unwrap();

    // 6.1 清除后，parent 和 result 的雅可比矩阵应该为 None
    assert!(graph.get_node_jacobi(parent).unwrap().is_none());
    assert!(graph.get_node_jacobi(result).unwrap().is_none());

    // 6.2 清除后再次反向传播 - 仍应正常工作
    // 6.2.1 sigmoid 节点 result 本身的雅可比矩阵至始至终都应为 None
    assert!(graph.get_node_jacobi(result).unwrap().is_none());

    // 6.2.2 对 parent 的反向传播（最后一次可以不保留图）
    graph.backward_nodes(&[parent], result).unwrap();
    let parent_jacobi_after_clear = graph.get_node_jacobi(parent).unwrap().unwrap();
    assert_eq!(parent_jacobi_after_clear, &expected_jacobi);
}

/// 简单的端到端验证测试
/// 验证 Sigmoid 在简单网络中的前向/反向传播功能
/// 网络结构: z -> sigmoid -> output
#[test]
fn test_node_sigmoid_simple_network() {
    let mut graph = Graph::new_with_seed(42);

    // 1. 构建最简单的网络: output = sigmoid(z)，z 是可训练参数
    let z = graph.new_parameter_node(&[1, 1], Some("z")).unwrap();
    let output = graph.new_sigmoid_node(z, Some("output")).unwrap();

    // 2. 设置 z 的初始值
    let z_value = Tensor::new(&[0.5], &[1, 1]);
    graph.set_node_value(z, Some(&z_value)).unwrap();

    // 3. 前向传播并验证 sigmoid(0.5) ≈ 0.6225
    graph.forward_node(output).unwrap();
    let output_val = graph.get_node_value(output).unwrap().unwrap()[[0, 0]];
    assert_abs_diff_eq!(output_val, 0.6225, epsilon = 0.01);

    // 4. 反向传播并验证雅可比矩阵形状
    graph.backward_nodes(&[z], output).unwrap();
    let z_jacobi = graph.get_node_jacobi(z).unwrap().unwrap();
    assert_eq!(z_jacobi.shape(), &[1, 1]);

    // 5. 验证优化器能正常工作
    let mut optimizer = SGD::new(&graph, 0.1).unwrap();
    optimizer.update(&mut graph).unwrap();
}

/// 验证 Sigmoid 在线性层后的功能
/// 网络结构: x -> MatMul(w) -> Add(b) -> Sigmoid -> output
#[test]
fn test_node_sigmoid_after_linear() {
    let mut graph = Graph::new_with_seed(42);

    // 1. 构建网络: output = sigmoid(w @ x + b)
    //    输入 x: 2x1，权重 w: 1x2，偏置 b: 1x1
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[1, 2], Some("w")).unwrap();
    let b = graph.new_parameter_node(&[1, 1], Some("b")).unwrap();

    let wx = graph.new_mat_mul_node(w, x, Some("wx")).unwrap(); // 1x2 @ 2x1 = 1x1
    let z = graph.new_add_node(&[wx, b], Some("z")).unwrap(); // 1x1 + 1x1 = 1x1
    let output = graph.new_sigmoid_node(z, Some("output")).unwrap();

    // 2. 设置输入并前向传播
    let input_tensor = Tensor::new(&[1.0, 0.5], &[2, 1]);
    graph.set_node_value(x, Some(&input_tensor)).unwrap();
    graph.forward_node(output).unwrap();

    // 3. 验证输出在 (0, 1) 范围内
    let output_val = graph.get_node_value(output).unwrap().unwrap()[[0, 0]];
    assert!(
        output_val > 0.0 && output_val < 1.0,
        "Sigmoid 输出应在 (0,1) 范围内，实际: {}",
        output_val
    );

    // 4. 反向传播并验证雅可比矩阵形状
    graph.backward_nodes(&[w, b], output).unwrap();
    let w_jacobi = graph.get_node_jacobi(w).unwrap().unwrap();
    let b_jacobi = graph.get_node_jacobi(b).unwrap().unwrap();
    assert_eq!(w_jacobi.shape(), &[1, 2]); // w: 1x2 展平后 2 个元素
    assert_eq!(b_jacobi.shape(), &[1, 1]); // b: 1x1

    // 5. 验证优化器能正常工作
    let mut optimizer = SGD::new(&graph, 0.1).unwrap();
    optimizer.update(&mut graph).unwrap();
}
