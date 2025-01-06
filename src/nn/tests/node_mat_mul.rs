use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_new_node_mat_mul_with_inited_parents() {
    let mut graph = Graph::new();

    // 1. 测试基本构造（2个父节点）
    let var1 = graph
        .new_variable_node(&[2, 3], true, false, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[3, 4], true, false, Some("var2"))
        .unwrap();
    let mat_mul = graph
        .new_mat_mul_node(var1, var2, Some("mat_mul"), true)
        .unwrap();
    // 验证基本属性
    assert_eq!(graph.get_node_parents(mat_mul).unwrap().len(), 2);
    assert_eq!(graph.get_node_children(mat_mul).unwrap().len(), 0);
    assert_eq!(graph.get_node_name(mat_mul).unwrap(), "mat_mul");
    assert!(graph.is_node_trainable(mat_mul).unwrap());
}

#[test]
fn test_new_node_mat_mul_with_uninited_parents() {
    let mut graph = Graph::new();

    let var1 = graph
        .new_variable_node(&[2, 3], false, false, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[3, 4], false, false, Some("var2"))
        .unwrap();
    let mat_mul = graph
        .new_mat_mul_node(var1, var2, Some("mat_mul"), true)
        .unwrap();
    // 验证基本属性
    assert_eq!(graph.get_node_parents(mat_mul).unwrap().len(), 2);
    assert_eq!(graph.get_node_children(mat_mul).unwrap().len(), 0);
    assert_eq!(graph.get_node_name(mat_mul).unwrap(), "mat_mul");
    assert!(graph.is_node_trainable(mat_mul).unwrap());
}

#[test]
fn test_new_node_mat_mul_with_inconsistent_shape() {
    let mut graph = Graph::new();

    // 1. 创建形状不匹配的父节点（不满足矩阵乘法规则）
    let var1 = graph
        .new_variable_node(&[2, 3], true, true, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[2, 4], true, true, Some("var2"))
        .unwrap();
    let var3 = graph
        .new_variable_node(&[4, 3], true, true, Some("var3"))
        .unwrap();

    // 2. 设置父节点的值
    let value1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let value2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let value3 = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[4, 3],
    );
    graph.set_node_value(var1, Some(&value1)).unwrap();
    graph.set_node_value(var2, Some(&value2)).unwrap();
    graph.set_node_value(var3, Some(&value3)).unwrap();

    // 3. 测试不同形状组合
    let nodes = [var1, var2, var3];
    for &left in nodes.iter() {
        for &right in nodes.iter() {
            let result = graph.new_mat_mul_node(left, right, None, true);

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
fn test_node_mat_mul_trainable_flag() {
    let mut graph = Graph::new();

    // 1. 测试初始为可训练节点
    let var1 = graph
        .new_variable_node(&[2, 3], true, false, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[3, 4], true, false, Some("var2"))
        .unwrap();
    let mat_mul = graph
        .new_mat_mul_node(var1, var2, Some("mat_mul"), true)
        .unwrap();
    assert!(graph.is_node_trainable(mat_mul).unwrap());
    // 1.1 测试trainable标志的后期修改
    graph.set_node_trainable(mat_mul, false).unwrap();
    assert!(!graph.is_node_trainable(mat_mul).unwrap());
    graph.set_node_trainable(mat_mul, true).unwrap();
    assert!(graph.is_node_trainable(mat_mul).unwrap());

    // 2. 测试初始为不可训练节点
    let mat_mul_non_trainable = graph
        .new_mat_mul_node(var1, var2, Some("mat_mul_non_trainable"), false)
        .unwrap();
    assert!(!graph.is_node_trainable(mat_mul_non_trainable).unwrap());
    // 2.1 测试trainable标志的后期修改
    graph
        .set_node_trainable(mat_mul_non_trainable, true)
        .unwrap();
    assert!(graph.is_node_trainable(mat_mul_non_trainable).unwrap());
    graph
        .set_node_trainable(mat_mul_non_trainable, false)
        .unwrap();
    assert!(!graph.is_node_trainable(mat_mul_non_trainable).unwrap());
}

#[test]
fn test_node_mat_mul_name_generation() {
    // 1. 测试节点显式命名
    // 1.1 图默认命名+节点显式命名
    let mut graph = Graph::new();
    let var1 = graph
        .new_variable_node(&[2, 3], true, false, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[3, 4], true, false, Some("var2"))
        .unwrap();
    let mat_mul1 = graph
        .new_mat_mul_node(var1, var2, Some("explicit_mat_mul"), true)
        .unwrap();
    assert_eq!(graph.get_node_name(mat_mul1).unwrap(), "explicit_mat_mul");

    // 1.2 图显式命名+节点显式命名
    let mut graph_with_name = Graph::with_name("custom_graph");
    let var1 = graph_with_name
        .new_variable_node(&[2, 3], true, false, Some("var1"))
        .unwrap();
    let var2 = graph_with_name
        .new_variable_node(&[3, 4], true, false, Some("var2"))
        .unwrap();
    let mat_mul_named = graph_with_name
        .new_mat_mul_node(var1, var2, Some("explicit_mat_mul"), true)
        .unwrap();
    assert_eq!(
        graph_with_name.get_node_name(mat_mul_named).unwrap(),
        "explicit_mat_mul"
    );

    // 2. 测试节点自动命名
    // 2.1 图默认命名+节点默认命名
    let mat_mul2 = graph.new_mat_mul_node(var1, var2, None, true).unwrap();
    assert_eq!(graph.get_node_name(mat_mul2).unwrap(), "mat_mul_1");

    // 2.2 图显式命名+节点默认命名
    let mat_mul_custom = graph_with_name
        .new_mat_mul_node(var1, var2, None, true)
        .unwrap();
    assert_eq!(
        graph_with_name.get_node_name(mat_mul_custom).unwrap(),
        "mat_mul_1"
    );

    // 3. 测试重复名称的处理
    // 3.1 测试显式重复名称
    let duplicate_result = graph.new_mat_mul_node(var1, var2, Some("explicit_mat_mul"), true);
    assert_eq!(
        duplicate_result,
        Err(GraphError::DuplicateNodeName(
            "节点explicit_mat_mul在图default_graph中重复".to_string()
        ))
    );

    // 3.2 测试在不同图中可以使用相同名称
    let mut another_graph = Graph::with_name("another_graph");
    let var1 = another_graph
        .new_variable_node(&[2, 3], true, false, Some("var1"))
        .unwrap();
    let var2 = another_graph
        .new_variable_node(&[3, 4], true, false, Some("var2"))
        .unwrap();
    let mat_mul_another = another_graph
        .new_mat_mul_node(var1, var2, Some("explicit_mat_mul"), true)
        .unwrap();
    assert_eq!(
        another_graph.get_node_name(mat_mul_another).unwrap(),
        "explicit_mat_mul"
    );
}

#[test]
fn test_node_mat_mul_manually_set_value() {
    let mut graph = Graph::new();
    let var1 = graph
        .new_variable_node(&[2, 3], true, true, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[3, 4], true, true, Some("var2"))
        .unwrap();
    let mat_mul = graph
        .new_mat_mul_node(var1, var2, Some("mat_mul"), true)
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

    // 1. 创建测试节点
    let var1 = graph
        .new_variable_node(&[2, 3], true, true, Some("var1"))
        .unwrap();
    let var2 = graph
        .new_variable_node(&[3, 4], true, true, Some("var2"))
        .unwrap();
    let mat_mul = graph
        .new_mat_mul_node(var1, var2, Some("mat_mul"), true)
        .unwrap();

    // 2. 验证初始状态下的预期形状
    assert_eq!(
        graph.get_node_value_expected_shape(mat_mul).unwrap(),
        &[2, 4]
    ); // 预期形状为[2,4]
    assert_eq!(graph.get_node_value_shape(mat_mul).unwrap(), None); // 实际值形状为None

    // 3. 设置父节点的值并执行前向传播
    let value1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let value2 = Tensor::new(
        &[
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[3, 4],
    );
    graph.set_node_value(var1, Some(&value1)).unwrap();
    graph.set_node_value(var2, Some(&value2)).unwrap();
    graph.forward_node(mat_mul).unwrap();

    // 4. 验证前向传播后的形状
    assert_eq!(
        graph.get_node_value_expected_shape(mat_mul).unwrap(),
        &[2, 4]
    ); // 预期形状仍为[2,4]
    assert_eq!(
        graph.get_node_value_shape(mat_mul).unwrap().unwrap(),
        &[2, 4]
    ); // 实际值形状为[2,4]

    // 4.1 测试父节点值在首次前向传播后，再次设置新值后的形状检查
    let new_value1 = Tensor::new(&[7.0, 8.0, 9.0, 4.0, 5.0, 6.0], &[2, 3]);
    let new_value2 = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    );
    graph.set_node_value(var1, Some(&new_value1)).unwrap();
    graph.set_node_value(var2, Some(&new_value2)).unwrap();

    // 验证预期形状和实际形状
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
    // 准备测试数据 (与Python测试tests\calc_jacobi_by_pytorch\node_mat_mul.py保持一致)
    let value1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let value2 = Tensor::new(
        &[
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[3, 4],
    );
    let expected = value1.mat_mul(&value2); // 结果应该是[2,4]的矩阵

    // 测试不同初始化和可训练组合的前向传播
    for var1_trainable in [false, true] {
        for var2_trainable in [false, true] {
            for mat_mul_trainable in [false, true] {
                for var1_inited in [false, true] {
                    for var2_inited in [false, true] {
                        let mut graph = Graph::new();
                        let var1 = graph
                            .new_variable_node(&[2, 3], var1_inited, var1_trainable, None)
                            .unwrap();
                        let var2 = graph
                            .new_variable_node(&[3, 4], var2_inited, var2_trainable, None)
                            .unwrap();
                        let mat_mul = graph
                            .new_mat_mul_node(var1, var2, None, mat_mul_trainable)
                            .unwrap();

                        if var1_inited && var2_inited {
                            // 若两个父节点都初始化，前向传播应成功
                            graph.forward_node(mat_mul).unwrap();
                        } else {
                            // 若有未初始化的父节点，前向传播应失败
                            let check_output_value = !var1_inited && !var2_inited;
                            if !var1_inited {
                                assert_eq!(
                                    graph.forward_node(mat_mul),
                                    Err(GraphError::InvalidOperation(format!(
                                        "节点[id=1, name=variable_1, type=Variable]不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为0，而图的前向传播次数为1",
                                    )))
                                );
                            } else if !var2_inited {
                                assert_eq!(
                                    graph.forward_node(mat_mul),
                                    Err(GraphError::InvalidOperation(format!(
                                        "节点[id=2, name=variable_2, type=Variable]不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为0，而图的前向传播次数为1",
                                    )))
                                );
                            }

                            // 设置了未初始化父节点的值后, 此时前向传播应该成功
                            if !var1_inited {
                                graph.set_node_value(var1, Some(&value1)).unwrap();
                            }
                            if !var2_inited {
                                graph.set_node_value(var2, Some(&value2)).unwrap();
                            }

                            graph.forward_node(mat_mul).unwrap();
                            let result = graph.get_node_value(mat_mul).unwrap().unwrap();

                            if check_output_value {
                                assert_eq!(result, &expected);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_node_mat_mul_backward_propagation() {
    let mut graph = Graph::new();

    // 1. 创建一个简单的矩阵乘法图：z = x * y
    let x = graph
        .new_variable_node(&[2, 3], true, true, Some("x"))
        .unwrap();
    let y = graph
        .new_variable_node(&[3, 4], true, true, Some("y"))
        .unwrap();
    let z = graph.new_mat_mul_node(x, y, Some("z"), true).unwrap();

    // 2. 测试在前向传播之前进行反向传播（应该失败）
    assert_eq!(
        graph.backward_node(x, z),
        Err(GraphError::ComputationError(format!(
            "反向传播：结果节点[id=3, name=z, type=MatMul]没有值"
        )))
    );

    // 3. 设置输入值 (与Python测试tests\calc_jacobi_by_pytorch\node_mat_mul.py保持一致)
    let x_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let y_value = Tensor::new(
        &[
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[3, 4],
    );
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(y, Some(&y_value)).unwrap();

    // 4. 反向传播前执行必要的前向传播
    graph.forward_node(z).unwrap();

    // 5. 反向传播
    // 5.1 mat_mul节点z本身的雅可比矩阵至始至终都应为None
    assert!(graph.get_node_jacobi(z).unwrap().is_none());

    // 5.2 对x的反向传播（第一次）
    graph.backward_node(x, z).unwrap();
    let x_jacobi = graph.get_node_jacobi(x).unwrap().unwrap();
    // 验证对x的雅可比矩阵 [8, 6]
    let expected_jacobi_x = Tensor::new(
        &[
            7.0, 11.0, 15.0, 0.0, 0.0, 0.0, 8.0, 12.0, 16.0, 0.0, 0.0, 0.0, 9.0, 13.0, 17.0, 0.0,
            0.0, 0.0, 10.0, 14.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 11.0, 15.0, 0.0, 0.0,
            0.0, 8.0, 12.0, 16.0, 0.0, 0.0, 0.0, 9.0, 13.0, 17.0, 0.0, 0.0, 0.0, 10.0, 14.0, 18.0,
        ],
        &[8, 6],
    );
    assert_eq!(x_jacobi, &expected_jacobi_x);

    // 5.3 对x的反向传播（第二次）- 应该得到相同的结果
    graph.backward_node(x, z).unwrap();
    let x_jacobi_second = graph.get_node_jacobi(x).unwrap().unwrap();
    assert_eq!(x_jacobi_second, &expected_jacobi_x);

    // 5.4 对y的反向传播（第一次）
    graph.backward_node(y, z).unwrap();
    let y_jacobi = graph.get_node_jacobi(y).unwrap().unwrap();
    // 验证对y的雅可比矩阵 [8, 12]
    let expected_jacobi_y = Tensor::new(
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
    assert_eq!(y_jacobi, &expected_jacobi_y);

    // 5.5 对y的反向传播（第二次）- 应该得到相同的结果
    graph.backward_node(y, z).unwrap();
    let y_jacobi_second = graph.get_node_jacobi(y).unwrap().unwrap();
    assert_eq!(y_jacobi_second, &expected_jacobi_y);

    // 6. 清除雅可比矩阵并验证
    graph.clear_jacobi().unwrap();

    // 6.1 清除后，x,y,z的雅可比矩阵应该为None
    assert!(graph.get_node_jacobi(x).unwrap().is_none());
    assert!(graph.get_node_jacobi(y).unwrap().is_none());
    assert!(graph.get_node_jacobi(z).unwrap().is_none());

    // 6.2 清除后再次反向传播 - 仍应正常工作
    // 6.2.1 mat_mul节点z本身的雅可比矩阵至始至终都应为None
    assert!(graph.get_node_jacobi(z).unwrap().is_none());

    // 6.2.2 对x的反向传播
    graph.backward_node(x, z).unwrap();
    let x_jacobi_after_clear = graph.get_node_jacobi(x).unwrap().unwrap();
    assert_eq!(x_jacobi_after_clear, &expected_jacobi_x);

    // 6.2.3 对y的反向传播
    graph.backward_node(y, z).unwrap();
    let y_jacobi_after_clear = graph.get_node_jacobi(y).unwrap().unwrap();
    assert_eq!(y_jacobi_after_clear, &expected_jacobi_y);
}
