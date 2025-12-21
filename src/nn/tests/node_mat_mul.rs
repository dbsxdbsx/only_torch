use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_node_mat_mul_creation() {
    let mut graph = Graph::new();

    // 1. 测试只有Input节点作为父节点
    {
        let input1 = graph.new_input_node(&[2, 3], Some("input1")).unwrap();
        let input2 = graph.new_input_node(&[3, 4], Some("input2")).unwrap();
        let mat_mul = graph
            .new_mat_mul_node(input1, input2, Some("mat_mul_with_inputs"))
            .unwrap();
        // 1.1 验证基本属性
        assert_eq!(graph.get_node_name(mat_mul).unwrap(), "mat_mul_with_inputs");
        assert_eq!(graph.get_node_parents(mat_mul).unwrap().len(), 2);
        assert_eq!(graph.get_node_children(mat_mul).unwrap().len(), 0);
    }

    // 2. 测试只有Parameter节点作为父节点
    {
        let param1 = graph.new_parameter_node(&[2, 3], Some("param1")).unwrap();
        let param2 = graph.new_parameter_node(&[3, 4], Some("param2")).unwrap();
        let mat_mul = graph
            .new_mat_mul_node(param1, param2, Some("mat_mul_with_params"))
            .unwrap();
        assert_eq!(graph.get_node_name(mat_mul).unwrap(), "mat_mul_with_params");
        assert_eq!(graph.get_node_parents(mat_mul).unwrap().len(), 2);
        assert_eq!(graph.get_node_children(mat_mul).unwrap().len(), 0);
    }

    // 3. 测试混合Input和Parameter节点作为父节点
    {
        let input = graph.new_input_node(&[2, 3], Some("input3")).unwrap();
        let param = graph.new_parameter_node(&[3, 4], Some("param3")).unwrap();
        let mat_mul = graph
            .new_mat_mul_node(input, param, Some("mat_mul_with_mixed"))
            .unwrap();
        assert_eq!(graph.get_node_name(mat_mul).unwrap(), "mat_mul_with_mixed");
        assert_eq!(graph.get_node_parents(mat_mul).unwrap().len(), 2);
        assert_eq!(graph.get_node_children(mat_mul).unwrap().len(), 0);
    }
}

#[test]
fn test_node_mat_mul_creation_with_inconsistent_shape() {
    let mut graph = Graph::new();

    // 1. 创建形状不匹配的父节点（不满足矩阵乘法规则）
    let input1 = graph.new_input_node(&[2, 3], Some("input1")).unwrap();
    let input2 = graph.new_input_node(&[2, 4], Some("input2")).unwrap();
    let param3 = graph.new_parameter_node(&[4, 3], Some("param3")).unwrap();

    // 2. 设置父节点的值
    let value1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let value2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let value3 = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[4, 3],
    );
    graph.set_node_value(input1, Some(&value1)).unwrap();
    graph.set_node_value(input2, Some(&value2)).unwrap();
    graph.set_node_value(param3, Some(&value3)).unwrap();

    // 3. 测试不同形状组合
    let nodes = [input1, input2, param3];
    for &left in nodes.iter() {
        for &right in nodes.iter() {
            let result = graph.new_mat_mul_node(left, right, None);

            let left_shape = graph.get_node_value(left).unwrap().unwrap().shape();
            let right_shape = graph.get_node_value(right).unwrap().unwrap().shape();

            if left_shape[1] == right_shape[0] {
                // 形状匹配,应该成功
                assert!(result.is_ok());
            } else {
                // 形状不匹配,应该失败
                assert_eq!(
                    result,
                    Err(GraphError::ShapeMismatch {
                        expected: vec![left_shape[0], right_shape[1]],
                        got: vec![left_shape[1], right_shape[0]],
                        message: format!(
                            "MatMul节点的2个父节点形状不兼容：父节点1的列数({})与父节点2的行数({})不相等。",
                            left_shape[1], right_shape[0]
                        ),
                    })
                );
            }
        }
    }
}

#[test]
fn test_node_mat_mul_name_generation() {
    // 1. 测试节点显式命名
    let mut graph = Graph::new();
    let input1 = graph.new_input_node(&[2, 3], Some("input1")).unwrap();
    let input2 = graph.new_input_node(&[3, 4], Some("input2")).unwrap();
    let mat_mul = graph
        .new_mat_mul_node(input1, input2, Some("explicit_mat_mul"))
        .unwrap();
    assert_eq!(graph.get_node_name(mat_mul).unwrap(), "explicit_mat_mul");

    // 2. 测试节点自动命名
    let mat_mul2 = graph.new_mat_mul_node(input1, input2, None).unwrap();
    assert_eq!(graph.get_node_name(mat_mul2).unwrap(), "mat_mul_1");

    // 3. 测试节点名称重复
    let result = graph.new_mat_mul_node(input1, input2, Some("explicit_mat_mul"));
    assert_eq!(
        result,
        Err(GraphError::DuplicateNodeName(
            "节点explicit_mat_mul在图default_graph中重复".to_string()
        ))
    );
}

#[test]
fn test_node_mat_mul_manually_set_value() {
    let mut graph = Graph::new();
    let input1 = graph.new_input_node(&[2, 3], Some("input1")).unwrap();
    let input2 = graph.new_input_node(&[3, 4], Some("input2")).unwrap();
    let mat_mul = graph
        .new_mat_mul_node(input1, input2, Some("mat_mul"))
        .unwrap();

    // 1. 测试直接设置MatMul节点的值（应该失败）
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    assert_eq!(
        graph.set_node_value(mat_mul, Some(&test_value)),
        Err(GraphError::InvalidOperation(
            "节点[id=3, name=mat_mul, type=MatMul]的值只能通过前向传播计算得到，不能直接设置"
                .into()
        ))
    );

    // 2. 测试清除MatMul节点的值（也应该失败）
    assert_eq!(
        graph.set_node_value(mat_mul, None),
        Err(GraphError::InvalidOperation(
            "节点[id=3, name=mat_mul, type=MatMul]的值只能通过前向传播计算得到，不能直接设置"
                .into()
        ))
    );
}

#[test]
fn test_node_mat_mul_expected_shape() {
    let mut graph = Graph::new();

    // 1. 测试基本的MatMul节点预期形状
    let input1 = graph.new_input_node(&[2, 3], Some("input1")).unwrap();
    let input2 = graph.new_input_node(&[3, 4], Some("input2")).unwrap();
    let mat_mul = graph
        .new_mat_mul_node(input1, input2, Some("mat_mul"))
        .unwrap();
    assert_eq!(
        graph.get_node_value_expected_shape(mat_mul).unwrap(),
        &[2, 4]
    );
    assert_eq!(graph.get_node_value_shape(mat_mul).unwrap(), None); // 实际值形状为None（未计算）

    // 2. 测试前向传播后的形状
    let value1 = Tensor::zeros(&[2, 3]);
    let value2 = Tensor::zeros(&[3, 4]);
    graph.set_node_value(input1, Some(&value1)).unwrap();
    graph.set_node_value(input2, Some(&value2)).unwrap();
    graph.forward_node(mat_mul).unwrap();

    // 2.1 验证前向传播后的形状
    assert_eq!(
        graph.get_node_value_shape(mat_mul).unwrap().unwrap(),
        &[2, 4]
    ); // 实际值形状
    assert_eq!(
        graph.get_node_value_expected_shape(mat_mul).unwrap(),
        &[2, 4]
    ); // 预期形状保持不变

    // 2.2 测试父节点值在首次前向传播后，再次设置新值后的形状检查
    let value1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let value2 = Tensor::new(
        &[
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[3, 4],
    );
    graph.set_node_value(input1, Some(&value1)).unwrap();
    graph.set_node_value(input2, Some(&value2)).unwrap();

    // 2.2.1 验证预期形状和实际形状
    assert_eq!(
        graph.get_node_value_expected_shape(mat_mul).unwrap(),
        &[2, 4]
    );
    assert_eq!(
        graph.get_node_value_shape(mat_mul).unwrap().unwrap(),
        &[2, 4]
    ); // 虽然值已过期，但由于值仍然存在，所以形状不变
}

#[test]
fn test_node_mat_mul_forward_propagation() {
    // 1. 准备测试数据 (与Python测试tests\calc_jacobi_by_pytorch\node_mat_mul.py保持一致)
    let value1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let value2 = Tensor::new(
        &[
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[3, 4],
    );
    let expected = value1.mat_mul(&value2); // 结果应该是[2, 4]的矩阵

    // 2. 测试不同节点类型组合的前向传播
    let node_types = ["input", "parameter"];
    for parent1_type in node_types {
        for parent2_type in node_types {
            let mut graph = Graph::new();

            // 创建parent1节点
            let parent1 = match parent1_type {
                "input" => graph.new_input_node(&[2, 3], Some("input_1")).unwrap(),
                "parameter" => graph
                    .new_parameter_node(&[2, 3], Some("parameter_1"))
                    .unwrap(),
                _ => unreachable!(),
            };

            // 创建parent2节点
            let parent2 = match parent2_type {
                "input" => graph.new_input_node(&[3, 4], Some("input_2")).unwrap(),
                "parameter" => graph
                    .new_parameter_node(&[3, 4], Some("parameter_2"))
                    .unwrap(),
                _ => unreachable!(),
            };

            // MatMul节点总是可训练的
            let mat_mul = graph
                .new_mat_mul_node(parent1, parent2, Some("mat_mul"))
                .unwrap();

            // 如果两个节点都是parameter，因创建时其值已隐式初始化过了，所以前向传播应成功
            if parent1_type == "parameter" && parent2_type == "parameter" {
                graph.forward_node(mat_mul).unwrap();
            } else {
                // 如果有input节点，因创建时其值未初始化，所以前向传播应失败
                if parent1_type == "input" {
                    assert_eq!(
                        graph.forward_node(mat_mul),
                        Err(GraphError::InvalidOperation(format!(
                            "节点[id=1, name=input_1, type=Input]不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为0，而图的前向传播次数为1",
                        )))
                    );
                } else if parent2_type == "input" {
                    assert_eq!(
                        graph.forward_node(mat_mul),
                        Err(GraphError::InvalidOperation(format!(
                            "节点[id=2, name=input_2, type=Input]不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为0，而图的前向传播次数为1",
                        )))
                    );
                }

                // 设置input节点的值
                if parent1_type == "input" {
                    graph.set_node_value(parent1, Some(&value1)).unwrap();
                }
                if parent2_type == "input" {
                    graph.set_node_value(parent2, Some(&value2)).unwrap();
                }

                // 设置值后前向传播应成功
                graph.forward_node(mat_mul).unwrap();
                let result = graph.get_node_value(mat_mul).unwrap().unwrap();

                // 只有当两个节点都是input时才检查输出值
                if parent1_type == "input" && parent2_type == "input" {
                    assert_eq!(result, &expected);
                }
            }
        }
    }
}

#[test]
fn test_node_mat_mul_backward_propagation() {
    let mut graph = Graph::new();

    // 1. 创建一个简单的矩阵乘法图：result = parent1 * parent2
    let parent1 = graph.new_parameter_node(&[2, 3], Some("parent1")).unwrap();
    let parent2 = graph.new_parameter_node(&[3, 4], Some("parent2")).unwrap();
    let result = graph
        .new_mat_mul_node(parent1, parent2, Some("result"))
        .unwrap();

    // 2. 测试在前向传播之前进行反向传播（应该失败）
    assert_eq!(
        graph.backward_nodes(&[parent1], result),
        Err(GraphError::ComputationError(format!(
            "反向传播：结果节点[id=3, name=result, type=MatMul]没有值"
        )))
    );

    // 3. 设置输入值 (与Python测试tests\calc_jacobi_by_pytorch\node_mat_mul.py保持一致)
    let parent1_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let parent2_value = Tensor::new(
        &[
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[3, 4],
    );
    graph.set_node_value(parent1, Some(&parent1_value)).unwrap();
    graph.set_node_value(parent2, Some(&parent2_value)).unwrap();

    // 4. 反向传播前执行必要的前向传播
    graph.forward_node(result).unwrap();

    // 5. 反向传播
    // 5.1 mat_mul节点result本身的雅可比矩阵至始至终都应为None
    assert!(graph.get_node_jacobi(result).unwrap().is_none());

    // 5.2 对parent1的反向传播（第1次）
    graph.backward_nodes(&[parent1], result).unwrap();
    let parent1_jacobi = graph.get_node_jacobi(parent1).unwrap().unwrap();
    let expected_jacobi_parent1 = Tensor::new(
        &[
            7.0, 11.0, 15.0, 0.0, 0.0, 0.0, 8.0, 12.0, 16.0, 0.0, 0.0, 0.0, 9.0, 13.0, 17.0, 0.0,
            0.0, 0.0, 10.0, 14.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 11.0, 15.0, 0.0, 0.0,
            0.0, 8.0, 12.0, 16.0, 0.0, 0.0, 0.0, 9.0, 13.0, 17.0, 0.0, 0.0, 0.0, 10.0, 14.0, 18.0,
        ],
        &[8, 6],
    );
    assert_eq!(parent1_jacobi, &expected_jacobi_parent1);

    // 5.3 对parent1的反向传播（第2次）- 梯度应该累积
    graph.backward_nodes(&[parent1], result).unwrap();
    let parent1_jacobi_second = graph.get_node_jacobi(parent1).unwrap().unwrap();
    assert_eq!(parent1_jacobi_second, &(&expected_jacobi_parent1 * 2.0));

    // 5.4 对parent2的反向传播（第1次）
    graph.backward_nodes(&[parent2], result).unwrap();
    let parent2_jacobi = graph.get_node_jacobi(parent2).unwrap().unwrap();
    let expected_jacobi_parent2 = Tensor::new(
        &[
            1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0,
            0.0, 5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0,
            0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0,
            0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0,
        ],
        &[8, 12],
    );
    assert_eq!(parent2_jacobi, &expected_jacobi_parent2);

    // 5.5 对parent2的反向传播（第2次）- 梯度应该累积
    graph.backward_nodes(&[parent2], result).unwrap();
    let parent2_jacobi_second = graph.get_node_jacobi(parent2).unwrap().unwrap();
    assert_eq!(parent2_jacobi_second, &(&expected_jacobi_parent2 * 2.0));

    // 6. 清除雅可比矩阵并验证
    graph.clear_jacobi().unwrap();

    // 6.1 清除后，parent1, parent2, result的雅可比矩阵应该为None
    assert!(graph.get_node_jacobi(parent1).unwrap().is_none());
    assert!(graph.get_node_jacobi(parent2).unwrap().is_none());
    assert!(graph.get_node_jacobi(result).unwrap().is_none());

    // 6.2 清除后再次反向传播 - 仍应正常工作
    // 6.2.1 mat_mul节点result本身的雅可比矩阵至始至终都应为None
    assert!(graph.get_node_jacobi(result).unwrap().is_none());

    // 6.2.2 对parent1的反向传播
    graph.backward_nodes(&[parent1], result).unwrap();
    let parent1_jacobi_after_clear = graph.get_node_jacobi(parent1).unwrap().unwrap();
    assert_eq!(parent1_jacobi_after_clear, &expected_jacobi_parent1);

    // 6.2.3 对parent2的反向传播
    graph.backward_nodes(&[parent2], result).unwrap();
    let parent2_jacobi_after_clear = graph.get_node_jacobi(parent2).unwrap().unwrap();
    assert_eq!(parent2_jacobi_after_clear, &expected_jacobi_parent2);
}
