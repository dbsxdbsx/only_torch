use crate::nn::{Graph, GraphError, NodeId};
use crate::tensor::Tensor;

#[test]
fn test_graph_creation() {
    // 测试默认创建
    let graph = Graph::new();
    assert_eq!(graph.name(), "default_graph");
    assert_eq!(graph.nodes_count(), 0);

    // 测试指定名称创建
    let named_graph = Graph::with_name("custom_graph");
    assert_eq!(named_graph.name(), "custom_graph");
    assert_eq!(named_graph.nodes_count(), 0);
}

#[test]
fn test_graph_mode() {
    // 1. 默认创建模式
    // 默认应该是训练模式
    let mut graph = Graph::new();
    assert!(graph.is_train_mode());

    // 测试切换到评估模式
    graph.set_eval_mode();
    assert!(!graph.is_train_mode());

    // 测试切换回训练模式
    graph.set_train_mode();
    assert!(graph.is_train_mode());

    // 2. 测试指定名称创建
    // 默认应该是训练模式
    let mut named_graph = Graph::with_name("custom_graph");
    assert!(named_graph.is_train_mode());

    // 测试切换到评估模式
    named_graph.set_eval_mode();
    assert!(!named_graph.is_train_mode());

    // 测试切换回训练模式
    named_graph.set_train_mode();
    assert!(named_graph.is_train_mode());
}

#[test]
fn test_new_node_error_handling() {
    let mut graph = Graph::new();

    // 1. 测试节点未找到错误
    let invalid_id = NodeId(999);
    assert_eq!(
        graph.get_node_value(invalid_id),
        Err(GraphError::NodeNotFound(invalid_id))
    );

    // 2. 测试重复节点名称错误
    let _ = graph
        .new_parameter_node(&[2, 2], Some("duplicate"))
        .unwrap();
    assert_eq!(
        graph.new_parameter_node(&[2, 2], Some("duplicate")),
        Err(GraphError::DuplicateNodeName(format!(
            "节点duplicate在图default_graph中重复"
        )))
    );

    // 3. 测试形状不匹配导致的错误
    let param = graph.new_parameter_node(&[2, 2], None).unwrap();
    let wrong_shape = Tensor::new(&[1.0, 2.0], &[2, 1]);
    assert_eq!(
        graph.set_node_value(param, Some(&wrong_shape)),
        Err(GraphError::ShapeMismatch {
            expected: vec![2, 2],
            got: vec![2, 1],
            message: format!(
                "新张量的形状 [2, 1] 与节点 'parameter_1' 现有张量的形状 [2, 2] 不匹配。"
            )
        })
    );
}

#[test]
fn test_node_relationships() {
    let mut graph = Graph::new();

    // 1. 创建节点关系
    let input1 = graph.new_input_node(&[2, 2], Some("input1")).unwrap();
    let input2 = graph.new_input_node(&[2, 2], Some("input2")).unwrap();
    let add = graph.new_add_node(&[input1, input2], Some("add")).unwrap();

    // 2. 验证父子关系
    let parents = graph.get_node_parents(add).unwrap();
    assert_eq!(parents.len(), 2);
    assert!(parents.contains(&input1));
    assert!(parents.contains(&input2));

    let children1 = graph.get_node_children(input1).unwrap();
    let children2 = graph.get_node_children(input2).unwrap();
    assert_eq!(children1.len(), 1);
    assert_eq!(children2.len(), 1);
    assert!(children1.contains(&add));
    assert!(children2.contains(&add));
}

#[test]
fn test_node_jacobi() {
    let mut graph = Graph::new();

    // 1. 创建一个简单的计算图：y = wx + b
    let x = graph.new_input_node(&[3, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[1, 3], Some("w")).unwrap();
    let b = graph.new_parameter_node(&[1, 1], Some("b")).unwrap();
    let wx = graph.new_mat_mul_node(w, x, None).unwrap();
    let y = graph.new_add_node(&[wx, b], None).unwrap();

    // 2. 测试未计算时的雅可比矩阵获取
    // 2.1 输入节点不应该有雅可比矩阵
    assert_eq!(
        graph.get_node_jacobi(x),
        Err(GraphError::InvalidOperation(format!(
            "输入节点[id=1, name=x, type=Input]不应该有雅可比矩阵",
        )))
    );

    // 2.2 参数节点在反向传播前应该返回None
    assert_eq!(graph.get_node_jacobi(w).unwrap(), None);
    assert_eq!(graph.get_node_jacobi(b).unwrap(), None);

    // 2.3 计算节点在反向传播前应该返回None
    assert_eq!(graph.get_node_jacobi(wx).unwrap(), None);
    assert_eq!(graph.get_node_jacobi(y).unwrap(), None);

    // 3. 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    let w_value = Tensor::new(&[0.1, 0.2, 0.3], &[1, 3]);
    let b_value = Tensor::new(&[0.4], &[1, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(w, Some(&w_value)).unwrap();
    graph.set_node_value(b, Some(&b_value)).unwrap();

    // 4. 前向传播
    graph.forward_node(y).unwrap();

    // 5. 反向传播（只对w和b）
    graph.backward_nodes(&[w, b], y).unwrap();

    // 6. 验证雅可比矩阵
    // 6.1 输入节点仍然不应该有雅可比矩阵
    assert_eq!(
        graph.get_node_jacobi(x),
        Err(GraphError::InvalidOperation(format!(
            "输入节点[id=1, name=x, type=Input]不应该有雅可比矩阵",
        )))
    );

    // 6.2 验证w的雅可比矩阵（参与反向传播的参数节点）
    let w_jacobi = graph.get_node_jacobi(w).unwrap().unwrap();
    assert_eq!(w_jacobi.shape(), &[1, 3]); // 雅可比矩阵形状应该是[dy_rows * dy_cols, dw_rows * dw_cols]
    let w_jacobi_expected = x_value.transpose(); // 因为y=wx+b，所以dy/dw=x^T
    assert_eq!(w_jacobi, w_jacobi_expected);

    // 6.3 验证b的雅可比矩阵（参与反向传播的参数节点）
    let b_jacobi = graph.get_node_jacobi(b).unwrap().unwrap();
    assert_eq!(b_jacobi.shape(), &[1, 1]); // 雅可比矩阵形状应该是[dy_rows * dy_cols, db_rows * db_cols]
    let b_jacobi_expected = Tensor::new(&[1.0], &[1, 1]); // 因为y=wx+b，所以dy/db=1
    assert_eq!(b_jacobi, b_jacobi_expected);

    // 6.4 验证wx的雅可比矩阵（作为中间计算节点，也参与反向传播）
    let wx_jacobi = graph.get_node_jacobi(wx).unwrap().unwrap();
    assert_eq!(wx_jacobi.shape(), &[1, 1]); // 雅可比矩阵形状应该是[dy_rows * dy_cols, dwx_rows * dwx_cols]
    let wx_jacobi_expected = Tensor::new(&[1.0], &[1, 1]); // 因为y=wx+b，所以dy/d(wx)=1
    assert_eq!(wx_jacobi, wx_jacobi_expected);

    // 6.5 验证y的雅可比矩阵（作为结果节点）
    let y_jacobi = graph.get_node_jacobi(y).unwrap().unwrap();
    assert_eq!(y_jacobi.shape(), &[1, 1]); // 结果节点的雅可比矩阵应该是单位矩阵
    assert_eq!(y_jacobi, Tensor::new(&[1.0], &[1, 1]));
}

#[test]
fn test_node_grad() {
    let mut graph = Graph::new();

    // 1. 创建一个简单的计算图：y = wx
    let x = graph.new_input_node(&[3, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[1, 3], Some("w")).unwrap();
    let y = graph.new_mat_mul_node(w, x, None).unwrap();

    // 2. 设置输入值并进行前向和反向传播
    let x_value = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    let w_value = Tensor::new(&[0.1, 0.2, 0.3], &[1, 3]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(w, Some(&w_value)).unwrap();
    graph.forward_node(y).unwrap();
    graph.backward_nodes(&[w], y).unwrap();

    // 3. 验证梯度是否等于雅可比矩阵的转置并重塑
    let w_jacobi = graph.get_node_jacobi(w).unwrap().unwrap();
    let w_grad = graph.get_node_grad(w).unwrap().unwrap();
    assert_eq!(w_grad, w_jacobi.transpose().reshape(w_value.shape()));
}
