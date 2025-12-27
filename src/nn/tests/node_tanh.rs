use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_node_tanh_creation() {
    let mut graph = Graph::new();

    // 1. 测试Input节点作为父节点
    {
        let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
        let tanh = graph.new_tanh_node(input, Some("tanh_with_input")).unwrap();
        // 1.1 验证基本属性
        assert_eq!(graph.get_node_name(tanh).unwrap(), "tanh_with_input");
        assert_eq!(graph.get_node_parents(tanh).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(tanh).unwrap().len(), 0);
    }

    // 2. 测试Parameter节点作为父节点
    {
        let param = graph.new_parameter_node(&[2, 2], Some("param1")).unwrap();
        let tanh = graph.new_tanh_node(param, Some("tanh_with_param")).unwrap();
        assert_eq!(graph.get_node_name(tanh).unwrap(), "tanh_with_param");
        assert_eq!(graph.get_node_parents(tanh).unwrap().len(), 1);
        assert_eq!(graph.get_node_children(tanh).unwrap().len(), 0);
    }
}

#[test]
fn test_node_tanh_name_generation() {
    let mut graph = Graph::new();

    // 1. 测试节点显式命名
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let tanh = graph.new_tanh_node(input, Some("explicit_tanh")).unwrap();
    assert_eq!(graph.get_node_name(tanh).unwrap(), "explicit_tanh");

    // 2. 测试节点自动命名
    let tanh2 = graph.new_tanh_node(input, None).unwrap();
    assert_eq!(graph.get_node_name(tanh2).unwrap(), "tanh_1");

    // 3. 测试节点名称重复
    let result = graph.new_tanh_node(input, Some("explicit_tanh"));
    assert_eq!(
        result,
        Err(GraphError::DuplicateNodeName(
            "节点explicit_tanh在图default_graph中重复".to_string()
        ))
    );
}

#[test]
fn test_node_tanh_manually_set_value() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let tanh = graph.new_tanh_node(input, Some("tanh")).unwrap();

    // 1. 测试直接设置Tanh节点的值（应该失败）
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(
        graph.set_node_value(tanh, Some(&test_value)),
        Err(GraphError::InvalidOperation(
            "节点[id=2, name=tanh, type=Tanh]的值只能通过前向传播计算得到，不能直接设置".into()
        ))
    );

    // 2. 测试清除Tanh节点的值（也应该失败）
    assert_eq!(
        graph.set_node_value(tanh, None),
        Err(GraphError::InvalidOperation(
            "节点[id=2, name=tanh, type=Tanh]的值只能通过前向传播计算得到，不能直接设置".into()
        ))
    );
}

#[test]
fn test_node_tanh_expected_shape() {
    let mut graph = Graph::new();

    // 1. 测试基本的Tanh节点预期形状
    let input = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let tanh = graph.new_tanh_node(input, Some("tanh")).unwrap();
    assert_eq!(graph.get_node_value_expected_shape(tanh).unwrap(), &[2, 2]);
    assert_eq!(graph.get_node_value_shape(tanh).unwrap(), None); // 实际值形状为None（未计算）

    // 2. 测试前向传播后的形状
    let value = Tensor::zeros(&[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();
    graph.forward_node(tanh).unwrap();

    // 2.1 验证前向传播后的形状
    assert_eq!(graph.get_node_value_shape(tanh).unwrap().unwrap(), &[2, 2]); // 实际值形状
    assert_eq!(graph.get_node_value_expected_shape(tanh).unwrap(), &[2, 2]); // 预期形状保持不变

    // 2.2 测试父节点值在首次前向传播后，再次设置新值后的形状检查
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(input, Some(&value)).unwrap();

    // 2.2.1 验证预期形状和实际形状
    assert_eq!(graph.get_node_value_expected_shape(tanh).unwrap(), &[2, 2]);
    assert_eq!(graph.get_node_value_shape(tanh).unwrap().unwrap(), &[2, 2]); // 虽然值已过期，但由于值仍然存在，所以形状不变
}

#[test]
fn test_node_tanh_forward_propagation() {
    // 1. 准备测试数据 (与Python测试tests/python/calc_jacobi_by_pytorch/node_tanh.py保持一致)
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    let expected = Tensor::new(
        &[
            0.46211716532707214,
            -0.7615941762924194,
            0.0,
            0.9640275835990906,
        ],
        &[2, 2],
    );

    // 2. 测试不同节点类型组合的前向传播
    let node_types = ["input", "parameter"];
    for parent_type in node_types {
        let mut graph = Graph::new();

        // 创建parent节点
        let parent = match parent_type {
            "input" => graph.new_input_node(&[2, 2], Some("input_1")).unwrap(),
            "parameter" => graph
                .new_parameter_node(&[2, 2], Some("parameter_1"))
                .unwrap(),
            _ => unreachable!(),
        };

        // Tanh节点总是可训练的
        let tanh = graph.new_tanh_node(parent, Some("tanh")).unwrap();

        // 如果节点是parameter，因创建时其值已隐式初始化过了，所以前向传播应成功
        if parent_type == "parameter" {
            graph.forward_node(tanh).unwrap();
        } else {
            // 如果是input节点，因创建时其值未初始化，所以前向传播应失败
            assert_eq!(
                graph.forward_node(tanh),
                Err(GraphError::InvalidOperation(format!(
                    "节点[id=1, name=input_1, type=Input]不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为0，而图的前向传播次数为1",
                )))
            );

            // 设置input节点的值
            graph.set_node_value(parent, Some(&value)).unwrap();

            // 设置值后前向传播应成功
            graph.forward_node(tanh).unwrap();
            let result = graph.get_node_value(tanh).unwrap().unwrap();

            // 只有当节点是input时才检查输出值
            if parent_type == "input" {
                assert_eq!(result, &expected);
            }
        }
    }
}

#[test]
fn test_node_tanh_backward_propagation() {
    let mut graph = Graph::new();

    // 1. 创建一个简单的tanh图：result = tanh(parent)
    let parent = graph.new_parameter_node(&[2, 2], Some("parent")).unwrap();
    let result = graph.new_tanh_node(parent, Some("result")).unwrap();

    // 2. 测试在前向传播之前进行反向传播（应该失败）
    assert_eq!(
        graph.backward_nodes(&[parent], result),
        Err(GraphError::ComputationError(format!(
            "反向传播：结果节点[id=2, name=result, type=Tanh]没有值"
        )))
    );

    // 3. 设置输入值 (与Python测试tests/python/calc_jacobi_by_pytorch/node_tanh.py保持一致)
    let parent_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(parent, Some(&parent_value)).unwrap();

    // 4. 反向传播前执行必要的前向传播
    graph.forward_node(result).unwrap();

    // 5. 反向传播
    // 5.1 tanh节点result本身的雅可比矩阵至始至终都应为None
    assert!(graph.get_node_jacobi(result).unwrap().is_none());

    // 5.2 对parent的反向传播（第1次，retain_graph=true 以便多次 backward）
    graph.backward_nodes_ex(&[parent], result, true).unwrap();
    let parent_jacobi = graph.get_node_jacobi(parent).unwrap().unwrap();

    // 验证雅可比矩阵（与Python输出一致）
    // d(tanh(x))/dx = 1 - tanh²(x)，对角矩阵
    #[rustfmt::skip]
    let expected_jacobi = Tensor::new(
        &[
            0.7864477038383484, 0.0, 0.0, 0.0,
            0.0, 0.41997432708740234, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.07065081596374512,
        ],
        &[4, 4],
    );
    assert_eq!(parent_jacobi, &expected_jacobi);

    // 5.3 对parent的反向传播（第2次）- 梯度应该累积
    graph.backward_nodes_ex(&[parent], result, true).unwrap();
    let parent_jacobi_second = graph.get_node_jacobi(parent).unwrap().unwrap();
    assert_eq!(parent_jacobi_second, &(&expected_jacobi * 2.0));

    // 6. 清除雅可比矩阵并验证
    graph.clear_jacobi().unwrap();

    // 6.1 清除后，parent和result的雅可比矩阵应该为None
    assert!(graph.get_node_jacobi(parent).unwrap().is_none());
    assert!(graph.get_node_jacobi(result).unwrap().is_none());

    // 6.2 清除后再次反向传播 - 仍应正常工作
    // 6.2.1 tanh节点result本身的雅可比矩阵至始至终都应为None
    assert!(graph.get_node_jacobi(result).unwrap().is_none());

    // 6.2.2 对parent的反向传播（最后一次可以不保留图）
    graph.backward_nodes(&[parent], result).unwrap();
    let parent_jacobi_after_clear = graph.get_node_jacobi(parent).unwrap().unwrap();
    assert_eq!(parent_jacobi_after_clear, &expected_jacobi);
}
