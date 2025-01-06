use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

#[test]
fn test_new_node_step_with_inited_parents() {
    let mut graph = Graph::new();

    // 测试基本构造（1个父节点）
    let var = graph
        .new_variable_node(&[2, 2], true, false, Some("var"))
        .unwrap();
    let step = graph.new_step_node(var, Some("step"), true).unwrap();
    // 验证基本属性
    assert_eq!(graph.get_node_parents(step).unwrap().len(), 1);
    assert_eq!(graph.get_node_children(step).unwrap().len(), 0);
    assert_eq!(graph.get_node_name(step).unwrap(), "step");
    assert!(graph.is_node_trainable(step).unwrap());
}

#[test]
fn test_new_node_step_with_uninited_parents() {
    let mut graph = Graph::new();

    let var = graph
        .new_variable_node(&[2, 2], false, false, Some("var"))
        .unwrap();
    let step = graph.new_step_node(var, Some("step"), true).unwrap();
    // 验证基本属性
    assert_eq!(graph.get_node_parents(step).unwrap().len(), 1);
    assert_eq!(graph.get_node_children(step).unwrap().len(), 0);
    assert_eq!(graph.get_node_name(step).unwrap(), "step");
    assert!(graph.is_node_trainable(step).unwrap());
}

#[test]
fn test_node_step_trainable_flag() {
    let mut graph = Graph::new();

    // 1. 测试初始为可训练节点
    let var = graph
        .new_variable_node(&[2, 2], true, false, Some("var"))
        .unwrap();
    let step = graph.new_step_node(var, Some("step"), true).unwrap();
    assert!(graph.is_node_trainable(step).unwrap());
    // 1.1 测试trainable标志的后期修改
    graph.set_node_trainable(step, false).unwrap();
    assert!(!graph.is_node_trainable(step).unwrap());
    graph.set_node_trainable(step, true).unwrap();
    assert!(graph.is_node_trainable(step).unwrap());

    // 2. 测试初始为不可训练节点
    let step_non_trainable = graph
        .new_step_node(var, Some("step_non_trainable"), false)
        .unwrap();
    assert!(!graph.is_node_trainable(step_non_trainable).unwrap());
    // 2.1 测试trainable标志的后期修改
    graph.set_node_trainable(step_non_trainable, true).unwrap();
    assert!(graph.is_node_trainable(step_non_trainable).unwrap());
    graph.set_node_trainable(step_non_trainable, false).unwrap();
    assert!(!graph.is_node_trainable(step_non_trainable).unwrap());
}

#[test]
fn test_node_step_name_generation() {
    // 1. 测试节点显式命名
    // 1.1 图默认命名+节点显式命名
    let mut graph = Graph::new();
    let var = graph
        .new_variable_node(&[2, 2], true, false, Some("var"))
        .unwrap();
    let step1 = graph
        .new_step_node(var, Some("explicit_step"), true)
        .unwrap();
    assert_eq!(graph.get_node_name(step1).unwrap(), "explicit_step");

    // 1.2 图显式命名+节点显式命名
    let mut graph_with_name = Graph::with_name("custom_graph");
    let var = graph_with_name
        .new_variable_node(&[2, 2], true, false, Some("var"))
        .unwrap();
    let step_named = graph_with_name
        .new_step_node(var, Some("explicit_step"), true)
        .unwrap();
    assert_eq!(
        graph_with_name.get_node_name(step_named).unwrap(),
        "explicit_step"
    );

    // 2. 测试节点自动命名
    // 2.1 图默认命名+节点默认命名
    let step2 = graph.new_step_node(var, None, true).unwrap();
    assert_eq!(graph.get_node_name(step2).unwrap(), "step_1");

    // 2.2 图显式命名+节点默认命名
    let step_custom = graph_with_name.new_step_node(var, None, true).unwrap();
    assert_eq!(
        graph_with_name.get_node_name(step_custom).unwrap(),
        "step_1"
    );

    // 3. 测试重复名称的处理
    // 3.1 测试显式重复名称
    let duplicate_result = graph.new_step_node(var, Some("explicit_step"), true);
    assert_eq!(
        duplicate_result,
        Err(GraphError::DuplicateNodeName(
            "节点explicit_step在图default_graph中重复".to_string()
        ))
    );

    // 3.2 测试在不同图中可以使用相同名称
    let mut another_graph = Graph::with_name("another_graph");
    let var = another_graph
        .new_variable_node(&[2, 2], true, false, Some("var"))
        .unwrap();
    let step_another = another_graph
        .new_step_node(var, Some("explicit_step"), true)
        .unwrap();
    assert_eq!(
        another_graph.get_node_name(step_another).unwrap(),
        "explicit_step"
    );
}

#[test]
fn test_node_step_manually_set_value() {
    let mut graph = Graph::new();
    let var = graph
        .new_variable_node(&[2, 2], true, true, Some("var"))
        .unwrap();
    let step = graph.new_step_node(var, Some("step"), true).unwrap();

    // 1. 测试直接设置Step节点的值（应该失败）
    let test_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(
        graph.set_node_value(step, Some(&test_value)),
        Err(GraphError::InvalidOperation(
            "节点[id=2, name=step, type=Step]的值只能通过前向传播计算得到，不能直接设置".into()
        ))
    );

    // 2. 测试清除Step节点的值（也应该失败）
    assert_eq!(
        graph.set_node_value(step, None),
        Err(GraphError::InvalidOperation(
            "节点[id=2, name=step, type=Step]的值只能通过前向传播计算得到，不能直接设置".into()
        ))
    );
}

#[test]
fn test_node_step_expected_shape() {
    let mut graph = Graph::new();

    // 1. 测试基本的Step节点预期形状
    let var = graph
        .new_variable_node(&[2, 2], false, false, Some("var"))
        .unwrap();
    let step = graph.new_step_node(var, Some("step"), true).unwrap();

    // 1.1 验证Step节点的预期形状（应该与父节点相同）
    assert_eq!(graph.get_node_value_expected_shape(step).unwrap(), &[2, 2]);
    assert_eq!(graph.get_node_value_shape(step).unwrap(), None); // 实际值形状为None（未计算）

    // 2. 测试前向传播后的形状
    let value = Tensor::zeros(&[2, 2]);
    graph.set_node_value(var, Some(&value)).unwrap();
    graph.forward_node(step).unwrap();

    // 2.1 验证前向传播后的形状
    assert_eq!(graph.get_node_value_shape(step).unwrap().unwrap(), &[2, 2]); // 实际值形状
    assert_eq!(graph.get_node_value_expected_shape(step).unwrap(), &[2, 2]); // 预期形状保持不变

    // 2.2 测试父节点值在首次前向传播后，再次设置新值后的形状检查
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(var, Some(&value)).unwrap();

    // 验证预期形状和实际形状
    assert_eq!(graph.get_node_value_expected_shape(step).unwrap(), &[2, 2]);
    assert_eq!(graph.get_node_value_shape(step).unwrap().unwrap(), &[2, 2]); // 虽然值已过期，但由于值仍然存在，所以形状不变
}

#[test]
fn test_node_step_forward_propagation() {
    // 准备测试数据 (与Python测试tests\calc_jacobi_by_pytorch\node_step.py保持一致)
    let value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    let expected = Tensor::new(&[1.0, 0.0, 1.0, 1.0], &[2, 2]);

    // 测试不同初始化和可训练组合的前向传播
    for var_trainable in [false, true] {
        for step_trainable in [false, true] {
            for var_inited in [false, true] {
                let mut graph = Graph::new();
                let var = graph
                    .new_variable_node(&[2, 2], var_inited, var_trainable, None)
                    .unwrap();
                let step = graph.new_step_node(var, None, step_trainable).unwrap();

                if var_inited {
                    // 若父节点已初始化，前向传播应成功
                    graph.forward_node(step).unwrap();
                } else {
                    // 若父节点未初始化，前向传播应失败
                    assert_eq!(
                        graph.forward_node(step),
                        Err(GraphError::InvalidOperation(format!(
                            "节点[id=1, name=variable_1, type=Variable]不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为0，而图的前向传播次数为1",
                        )))
                    );

                    // 设置了未初始化父节点的值后, 此时前向传播应该成功
                    graph.set_node_value(var, Some(&value)).unwrap();
                    graph.forward_node(step).unwrap();
                    let result = graph.get_node_value(step).unwrap().unwrap();
                    assert_eq!(result, &expected);
                }
            }
        }
    }
}

#[test]
fn test_node_step_backward_propagation() {
    let mut graph = Graph::new();

    // 1. 创建一个简单的阶跃图：y = step(x)
    let x = graph
        .new_variable_node(&[2, 2], true, true, Some("x"))
        .unwrap();
    let y = graph.new_step_node(x, Some("y"), true).unwrap();

    // 2. 测试在前向传播之前进行反向传播（应该失败）
    assert_eq!(
        graph.backward_node(x, y),
        Err(GraphError::ComputationError(format!(
            "反向传播：结果节点[id=2, name=y, type=Step]没有值"
        )))
    );

    // 3. 设置输入值 (与Python测试tests\calc_jacobi_by_pytorch\node_step.py保持一致)
    let x_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    graph.set_node_value(x, Some(&x_value)).unwrap();

    // 4. 反向传播前执行必要的前向传播
    graph.forward_node(y).unwrap();

    // 5. 反向传播
    // 5.1 step节点y本身的雅可比矩阵至始至终都应为None
    assert!(graph.get_node_jacobi(y).unwrap().is_none());

    // 5.2 对x的反向传播（第一次）
    graph.backward_node(x, y).unwrap();
    let x_jacobi = graph.get_node_jacobi(x).unwrap().unwrap();
    // 验证雅可比矩阵（与Python输出一致）
    let expected_jacobi = Tensor::new(
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        &[4, 4],
    );
    assert_eq!(x_jacobi, &expected_jacobi);

    // 5.3 对x的反向传播（第二次）- 应该得到相同的结果
    graph.backward_node(x, y).unwrap();
    let x_jacobi_second = graph.get_node_jacobi(x).unwrap().unwrap();
    assert_eq!(x_jacobi_second, &expected_jacobi);

    // 6. 清除雅可比矩阵并验证
    graph.clear_jacobi().unwrap();

    // 6.1 清除后，x,y的雅可比矩阵应该为None
    assert!(graph.get_node_jacobi(x).unwrap().is_none());
    assert!(graph.get_node_jacobi(y).unwrap().is_none());

    // 6.2 清除后再次反向传播 - 仍应正常工作
    // 6.2.1 step节点y本身的雅可比矩阵至始至终都应为None
    assert!(graph.get_node_jacobi(y).unwrap().is_none());

    // 6.2.2 对x的反向传播
    graph.backward_node(x, y).unwrap();
    let x_jacobi_after_clear = graph.get_node_jacobi(x).unwrap().unwrap();
    assert_eq!(x_jacobi_after_clear, &expected_jacobi);
}
