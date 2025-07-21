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

#[test]
fn test_forward_with_partial_forward_propagation() {
    let mut graph = Graph::new();

    // 1. 创建计算图：z = x + y, w = x + y (两个节点都依赖同样的add操作)
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let y = graph.new_input_node(&[2, 1], Some("y")).unwrap();
    let add1 = graph.new_add_node(&[x, y], Some("add1")).unwrap();
    let add2 = graph.new_add_node(&[x, y], Some("add2")).unwrap();

    // 2. 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let y_value = Tensor::new(&[0.5, 1.5], &[2, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(y, Some(&y_value)).unwrap();

    // 3. 第一次前向传播add1
    graph.forward_node(add1).unwrap();
    let first_pass_id = graph.last_forward_pass_id();

    // 验证所有节点的pass_id都是第一次的pass_id
    assert_eq!(
        graph.get_node(x).unwrap().last_forward_pass_id(),
        first_pass_id
    );
    assert_eq!(
        graph.get_node(y).unwrap().last_forward_pass_id(),
        first_pass_id
    );
    assert_eq!(
        graph.get_node(add1).unwrap().last_forward_pass_id(),
        first_pass_id
    );

    // add2还没有被计算，所以其pass_id应该为0
    assert_eq!(graph.get_node(add2).unwrap().last_forward_pass_id(), 0);

    // 4. 创建一个更复杂的图来测试重复计算避免：
    // 创建一个菱形依赖：final = add1 + add2，其中add1和add2都依赖x和y
    let final_add = graph.new_add_node(&[add1, add2], Some("final")).unwrap();

    // 5. 前向传播final节点，这会触发对add1和add2的计算
    // 在这个过程中，x和y应该只被计算一次（重复计算避免）
    graph.forward_node(final_add).unwrap();
    let second_pass_id = graph.last_forward_pass_id();
    assert_eq!(second_pass_id, first_pass_id + 1);

    // 验证所有节点都被更新到新的pass_id
    assert_eq!(
        graph.get_node(x).unwrap().last_forward_pass_id(),
        second_pass_id
    );
    assert_eq!(
        graph.get_node(y).unwrap().last_forward_pass_id(),
        second_pass_id
    );
    assert_eq!(
        graph.get_node(add1).unwrap().last_forward_pass_id(),
        second_pass_id
    );
    assert_eq!(
        graph.get_node(add2).unwrap().last_forward_pass_id(),
        second_pass_id
    );
    assert_eq!(
        graph.get_node(final_add).unwrap().last_forward_pass_id(),
        second_pass_id
    );

    // 6. 再次前向传播final节点，验证所有节点都会重新计算
    graph.forward_node(final_add).unwrap();
    let third_pass_id = graph.last_forward_pass_id();
    assert_eq!(third_pass_id, second_pass_id + 1);

    // 验证所有节点的pass_id都更新为新的pass_id
    assert_eq!(
        graph.get_node(x).unwrap().last_forward_pass_id(),
        third_pass_id
    );
    assert_eq!(
        graph.get_node(y).unwrap().last_forward_pass_id(),
        third_pass_id
    );
    assert_eq!(
        graph.get_node(add1).unwrap().last_forward_pass_id(),
        third_pass_id
    );
    assert_eq!(
        graph.get_node(add2).unwrap().last_forward_pass_id(),
        third_pass_id
    );
    assert_eq!(
        graph.get_node(final_add).unwrap().last_forward_pass_id(),
        third_pass_id
    );
}

#[test]
fn test_continuous_backward_jacobi_accumulation() {
    let mut graph = Graph::new();

    // 创建简单的计算图：y = x + b
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let b = graph.new_parameter_node(&[2, 1], Some("b")).unwrap();
    let y = graph.new_add_node(&[x, b], Some("y")).unwrap();

    // 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let b_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(b, Some(&b_value)).unwrap();

    // 前向传播
    graph.forward_node(y).unwrap();

    // 验证初始状态：没有雅可比矩阵
    assert_eq!(graph.get_node_jacobi(b).unwrap(), None);

    // 第1次反向传播
    graph.backward_nodes(&[b], y).unwrap();
    let first_jacobi = graph.get_node_jacobi(b).unwrap().unwrap().clone();

    // 对于 y = x + b，dy/db = I（单位矩阵）
    let expected_single = Tensor::eyes(2);
    assert_eq!(first_jacobi, expected_single);

    // 第2次反向传播（连续）- 应该累积梯度
    graph.backward_nodes(&[b], y).unwrap();
    let second_jacobi = graph.get_node_jacobi(b).unwrap().unwrap().clone();

    // 累积后应该是2倍的单位矩阵
    let expected_accumulated = expected_single.clone() + expected_single.clone();
    assert_eq!(second_jacobi, expected_accumulated);

    // 第3次反向传播 - 继续累积
    graph.backward_nodes(&[b], y).unwrap();
    let third_jacobi = graph.get_node_jacobi(b).unwrap().unwrap().clone();

    // 累积后应该是3倍的单位矩阵
    let expected_triple =
        expected_single.clone() + expected_single.clone() + expected_single.clone();
    assert_eq!(third_jacobi, expected_triple);

    // 测试clear_jacobi功能
    graph.clear_jacobi().unwrap();
    assert_eq!(graph.get_node_jacobi(b).unwrap(), None);

    // 清除后再次反向传播，应该重新开始
    graph.backward_nodes(&[b], y).unwrap();
    let after_clear_jacobi = graph.get_node_jacobi(b).unwrap().unwrap().clone();
    assert_eq!(after_clear_jacobi, expected_single);

    // 测试set_value不会自动清除雅可比矩阵的行为（需要手动清除）
    // 先进行反向传播，确保有雅可比矩阵
    graph.backward_nodes(&[b], y).unwrap();
    let jacobi_before_set = graph.get_node_jacobi(b).unwrap().unwrap().clone();

    // 设置新的参数值，雅可比矩阵应该仍然存在（不自动清除）
    let new_b_value = Tensor::new(&[0.3, 0.4], &[2, 1]);
    graph.set_node_value(b, Some(&new_b_value)).unwrap();
    let jacobi_after_set = graph.get_node_jacobi(b).unwrap().unwrap().clone();
    assert_eq!(jacobi_after_set, jacobi_before_set); // 雅可比矩阵应该保持不变
}

#[test]
fn test_backward_without_any_forward() {
    let mut graph = Graph::new();

    // 创建简单的线性模型：y = wx + b
    let x = graph.new_input_node(&[3, 1], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[1, 3], Some("w")).unwrap();
    let b = graph.new_parameter_node(&[1, 1], Some("b")).unwrap();
    let wx = graph.new_mat_mul_node(w, x, Some("wx")).unwrap();
    let y = graph.new_add_node(&[wx, b], Some("y")).unwrap();

    // 设置输入值，但不进行任何前向传播
    let x_value = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();

    // 验证：所有节点的前向传播ID都是0（没有前向传播）
    assert_eq!(graph.get_node(x).unwrap().last_forward_pass_id(), 0);
    assert_eq!(graph.get_node(w).unwrap().last_forward_pass_id(), 0);
    assert_eq!(graph.get_node(b).unwrap().last_forward_pass_id(), 0);
    assert_eq!(graph.get_node(wx).unwrap().last_forward_pass_id(), 0);
    assert_eq!(graph.get_node(y).unwrap().last_forward_pass_id(), 0);

    // 验证：结果节点y没有值（因为没有前向传播）
    assert!(graph.get_node_value(y).unwrap().is_none());

    // 关键测试：在没有任何前向传播的情况下尝试反向传播
    // 这应该失败，因为结果节点y没有值
    assert_eq!(
        graph.backward_nodes(&[w], y),
        Err(GraphError::ComputationError(
            "反向传播：结果节点[id=5, name=y, type=Add]没有值".to_string()
        ))
    );

    // 同样，对参数b的反向传播也应该失败
    assert_eq!(
        graph.backward_nodes(&[b], y),
        Err(GraphError::ComputationError(
            "反向传播：结果节点[id=5, name=y, type=Add]没有值".to_string()
        ))
    );

    // 验证：反向传播失败后，参数节点仍然没有雅可比矩阵
    assert!(graph.get_node_jacobi(w).unwrap().is_none());
    assert!(graph.get_node_jacobi(b).unwrap().is_none());

    // 验证：图的反向传播ID没有增加（因为反向传播失败了）
    assert_eq!(graph.last_backward_pass_id(), 0);
}

#[test]
fn test_backward_with_partial_forward_propagation() {
    let mut graph = Graph::new();

    // 创建一个计算图：z = (x + a) + (y + b)，其中一个参数节点有多个子节点，但只有部分子节点参与了前向传播
    // 结构：
    //   a -> left_add -> z (参与前向传播)
    //   a -> new_add (不参与前向传播)
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let y = graph.new_input_node(&[2, 1], Some("y")).unwrap();
    let a = graph.new_parameter_node(&[2, 1], Some("a")).unwrap();
    let b = graph.new_parameter_node(&[2, 1], Some("b")).unwrap();
    let c = graph.new_parameter_node(&[2, 1], Some("c")).unwrap();

    let left_add = graph.new_add_node(&[x, a], Some("left_add")).unwrap();
    let right_add = graph.new_add_node(&[y, b], Some("right_add")).unwrap();
    let z = graph
        .new_add_node(&[left_add, right_add], Some("z"))
        .unwrap();

    // 创建一个不参与主计算路径的分支
    let new_add = graph.new_add_node(&[a, c], Some("new_add")).unwrap();

    // 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let y_value = Tensor::new(&[3.0, 4.0], &[2, 1]);
    let a_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    let b_value = Tensor::new(&[0.3, 0.4], &[2, 1]);
    let c_value = Tensor::new(&[0.5, 0.6], &[2, 1]);

    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(y, Some(&y_value)).unwrap();
    graph.set_node_value(a, Some(&a_value)).unwrap();
    graph.set_node_value(b, Some(&b_value)).unwrap();
    graph.set_node_value(c, Some(&c_value)).unwrap();

    // 只对主路径进行前向传播，不对new_add分支进行前向传播
    graph.forward_node(z).unwrap();

    // 验证：主路径已前向传播，new_add分支没有
    assert_eq!(graph.get_node(left_add).unwrap().last_forward_pass_id(), 1);
    assert_eq!(graph.get_node(right_add).unwrap().last_forward_pass_id(), 1);
    assert_eq!(graph.get_node(z).unwrap().last_forward_pass_id(), 1);
    assert_eq!(graph.get_node(new_add).unwrap().last_forward_pass_id(), 0);

    // 关键测试：对参数a进行反向传播
    // 即使a有一个子节点(new_add)没有前向传播，反向传播也应该成功
    // 并且只考虑已前向传播的子节点(left_add)
    graph.backward_nodes(&[a], z).unwrap();

    // 验证反向传播成功
    let a_jacobi = graph.get_node_jacobi(a).unwrap().unwrap();
    // 对于 z = (x + a) + (y + b)，dz/da = I（单位矩阵）
    assert_eq!(a_jacobi, &Tensor::eyes(2));

    // 验证new_add分支确实没有参与反向传播
    assert_eq!(graph.get_node(new_add).unwrap().last_forward_pass_id(), 0);
    assert!(graph.get_node_jacobi(new_add).unwrap().is_none());

    // 这个测试证明了：即使参数节点有未前向传播的子节点分支，
    // 反向传播仍然能够正常工作，只考虑已前向传播的路径
}

#[test]
fn test_forward_pass_id_increment() {
    let mut graph = Graph::new();

    // 1. 创建简单的计算图：y = x + b
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let b = graph.new_parameter_node(&[2, 1], Some("b")).unwrap();
    let y = graph.new_add_node(&[x, b], Some("y")).unwrap();

    // 2. 初始状态：pass_id应该为0
    assert_eq!(graph.last_forward_pass_id(), 0);

    // 3. 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let b_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(b, Some(&b_value)).unwrap();

    // 4. 第一次前向传播
    graph.forward_node(y).unwrap();
    assert_eq!(graph.last_forward_pass_id(), 1);

    // 5. 第二次前向传播
    graph.forward_node(y).unwrap();
    assert_eq!(graph.last_forward_pass_id(), 2);

    // 6. 第三次前向传播
    graph.forward_node(y).unwrap();
    assert_eq!(graph.last_forward_pass_id(), 3);
}

#[test]
fn test_backward_pass_id_increment() {
    let mut graph = Graph::new();

    // 1. 创建简单的计算图：y = x + b
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let b = graph.new_parameter_node(&[2, 1], Some("b")).unwrap();
    let y = graph.new_add_node(&[x, b], Some("y")).unwrap();

    // 2. 初始状态：pass_id应该为0
    assert_eq!(graph.last_backward_pass_id(), 0);

    // 3. 设置输入值并前向传播
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let b_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(b, Some(&b_value)).unwrap();
    graph.forward_node(y).unwrap();

    // 4. 第一次反向传播
    graph.backward_nodes(&[b], y).unwrap();
    assert_eq!(graph.last_backward_pass_id(), 1);

    // 5. 第二次反向传播
    graph.backward_nodes(&[b], y).unwrap();
    assert_eq!(graph.last_backward_pass_id(), 2);

    // 6. 第三次反向传播
    graph.backward_nodes(&[b], y).unwrap();
    assert_eq!(graph.last_backward_pass_id(), 3);
}

#[test]
fn test_node_pass_id_synchronization() {
    let mut graph = Graph::new();

    // 1. 创建计算图：z = (x + y) * w
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let y = graph.new_input_node(&[2, 1], Some("y")).unwrap();
    let w = graph.new_parameter_node(&[1, 2], Some("w")).unwrap();
    let add = graph.new_add_node(&[x, y], Some("add")).unwrap();
    let z = graph.new_mat_mul_node(w, add, Some("z")).unwrap();

    // 2. 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let y_value = Tensor::new(&[0.5, 1.5], &[2, 1]);
    let w_value = Tensor::new(&[0.1, 0.2], &[1, 2]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(y, Some(&y_value)).unwrap();
    graph.set_node_value(w, Some(&w_value)).unwrap();

    // 3. 前向传播并验证节点pass_id同步
    graph.forward_node(z).unwrap();
    let graph_forward_pass_id = graph.last_forward_pass_id();

    // 验证所有参与计算的节点的前向pass_id都与图的pass_id一致
    assert_eq!(
        graph.get_node(x).unwrap().last_forward_pass_id(),
        graph_forward_pass_id
    );
    assert_eq!(
        graph.get_node(y).unwrap().last_forward_pass_id(),
        graph_forward_pass_id
    );
    assert_eq!(
        graph.get_node(w).unwrap().last_forward_pass_id(),
        graph_forward_pass_id
    );
    assert_eq!(
        graph.get_node(add).unwrap().last_forward_pass_id(),
        graph_forward_pass_id
    );
    assert_eq!(
        graph.get_node(z).unwrap().last_forward_pass_id(),
        graph_forward_pass_id
    );

    // 4. 反向传播并验证节点pass_id同步
    graph.backward_nodes(&[w, add], z).unwrap();
    let graph_backward_pass_id = graph.last_backward_pass_id();

    // 验证参与反向传播的节点的反向pass_id都与图的pass_id一致
    assert_eq!(
        graph.get_node(w).unwrap().last_backward_pass_id(),
        graph_backward_pass_id
    );
    assert_eq!(
        graph.get_node(add).unwrap().last_backward_pass_id(),
        graph_backward_pass_id
    );
    assert_eq!(
        graph.get_node(z).unwrap().last_backward_pass_id(),
        graph_backward_pass_id
    );

    // 输入节点不参与反向传播，所以其反向pass_id应该仍为0
    assert_eq!(graph.get_node(x).unwrap().last_backward_pass_id(), 0);
    assert_eq!(graph.get_node(y).unwrap().last_backward_pass_id(), 0);
}

#[test]
fn test_pass_id_rollback_on_error() {
    let mut graph = Graph::new();

    // 1. 创建计算图：y = x + b
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let b = graph.new_parameter_node(&[2, 1], Some("b")).unwrap();
    let y = graph.new_add_node(&[x, b], Some("y")).unwrap();

    // 2. 设置b的值，但故意不设置x的值
    let b_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    graph.set_node_value(b, Some(&b_value)).unwrap();

    // 3. 记录初始pass_id
    let initial_forward_pass_id = graph.last_forward_pass_id();
    let initial_backward_pass_id = graph.last_backward_pass_id();
    assert_eq!(initial_forward_pass_id, 0);
    assert_eq!(initial_backward_pass_id, 0);

    // 4. 尝试前向传播，应该失败（因为x没有值）
    let forward_result = graph.forward_node(y);
    assert!(forward_result.is_err());

    // 验证前向传播失败后pass_id被正确回滚
    assert_eq!(graph.last_forward_pass_id(), initial_forward_pass_id);

    // 5. 设置x的值，使前向传播能够成功
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();

    // 6. 现在前向传播应该成功
    graph.forward_node(y).unwrap();
    assert_eq!(graph.last_forward_pass_id(), 1);

    // 7. 尝试对输入节点进行反向传播，应该失败
    let backward_result = graph.backward_nodes(&[x], y);
    assert!(backward_result.is_err());

    // 验证反向传播失败后pass_id被正确回滚
    assert_eq!(graph.last_backward_pass_id(), initial_backward_pass_id);

    // 8. 对参数节点进行反向传播，应该成功
    graph.backward_nodes(&[b], y).unwrap();
    assert_eq!(graph.last_backward_pass_id(), 1);
}

// #[test]
// fn test_outdated_node_skipping_in_backward() {
//     let mut graph = Graph::new();

//     // 1. 创建复杂的计算图：result = (a + b) * c + d
//     let a = graph.new_input_node(&[2, 1], Some("a")).unwrap();
//     let b = graph.new_input_node(&[2, 1], Some("b")).unwrap();
//     let c = graph.new_parameter_node(&[1, 2], Some("c")).unwrap();
//     let d = graph.new_parameter_node(&[1, 1], Some("d")).unwrap();
//     let add_ab = graph.new_add_node(&[a, b], Some("add_ab")).unwrap();
//     let mul_c = graph.new_mat_mul_node(c, add_ab, Some("mul_c")).unwrap();
//     let result = graph.new_add_node(&[mul_c, d], Some("result")).unwrap();

//     // 2. 设置初始值并进行第一次前向传播
//     let a_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
//     let b_value = Tensor::new(&[0.5, 1.5], &[2, 1]);
//     let c_value = Tensor::new(&[0.1, 0.2], &[1, 2]);
//     let d_value = Tensor::new(&[0.3], &[1, 1]);

//     graph.set_node_value(a, Some(&a_value)).unwrap();
//     graph.set_node_value(b, Some(&b_value)).unwrap();
//     graph.set_node_value(c, Some(&c_value)).unwrap();
//     graph.set_node_value(d, Some(&d_value)).unwrap();

//     // 第一次前向传播
//     graph.forward_node(result).unwrap();
//     let first_forward_pass_id = graph.last_forward_pass_id();

//     // 3. 修改a的值但不重新前向传播，使add_ab和mul_c的值过时
//     let a_new_value = Tensor::new(&[3.0, 4.0], &[2, 1]);
//     graph.set_node_value(a, Some(&a_new_value)).unwrap();

//     // 验证a的前向pass_id被重置，但其他节点保持原来的pass_id
//     assert_eq!(
//         graph.get_node(a).unwrap().last_forward_pass_id(),
//         first_forward_pass_id
//     );
//     assert_eq!(
//         graph.get_node(add_ab).unwrap().last_forward_pass_id(),
//         first_forward_pass_id
//     );
//     assert_eq!(
//         graph.get_node(mul_c).unwrap().last_forward_pass_id(),
//         first_forward_pass_id
//     );

//     // 4. 修改d的值，这会更新d的前向pass_id但不影响其他节点
//     let d_new_value = Tensor::new(&[0.5], &[1, 1]);
//     graph.set_node_value(d, Some(&d_new_value)).unwrap();

//     // 验证修改值后，所有节点的pass_id保持不变（因为还没有新的前向传播）
//     assert_eq!(
//         graph.get_node(a).unwrap().last_forward_pass_id(),
//         first_forward_pass_id
//     );
//     assert_eq!(
//         graph.get_node(add_ab).unwrap().last_forward_pass_id(),
//         first_forward_pass_id
//     );
//     assert_eq!(
//         graph.get_node(mul_c).unwrap().last_forward_pass_id(),
//         first_forward_pass_id
//     );
//     assert_eq!(
//         graph.get_node(d).unwrap().last_forward_pass_id(),
//         first_forward_pass_id
//     );

//     // 5. 现在重新前向传播result，这会重新计算所有节点
//     graph.forward_node(result).unwrap();
//     let second_forward_pass_id = graph.last_forward_pass_id();
//     assert_eq!(second_forward_pass_id, first_forward_pass_id + 1);

//     // 验证所有节点现在都有最新的前向pass-id
//     assert_eq!(
//         graph.get_node(a).unwrap().last_forward_pass_id(),
//         second_forward_pass_id
//     );
//     assert_eq!(
//         graph.get_node(add_ab).unwrap().last_forward_pass_id(),
//         second_forward_pass_id
//     );
//     assert_eq!(
//         graph.get_node(mul_c).unwrap().last_forward_pass_id(),
//         second_forward_pass_id
//     );
//     assert_eq!(
//         graph.get_node(d).unwrap().last_forward_pass_id(),
//         second_forward_pass_id
//     );
//     assert_eq!(
//         graph.get_node(result).unwrap().last_forward_pass_id(),
//         second_forward_pass_id
//     );

//     // 6. 进行反向传播
//     graph.backward_nodes(&[c, d], result).unwrap();

//     // 验证反向传播成功
//     let backward_pass_id = graph.last_backward_pass_id();
//     assert_eq!(
//         graph.get_node(c).unwrap().last_backward_pass_id(),
//         backward_pass_id
//     );
//     assert_eq!(
//         graph.get_node(d).unwrap().last_backward_pass_id(),
//         backward_pass_id
//     );
//     assert_eq!(
//         graph.get_node(result).unwrap().last_backward_pass_id(),
//         backward_pass_id
//     );
// }
