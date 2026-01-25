use crate::nn::GraphInner;
use crate::tensor::Tensor;

#[test]
fn test_forward_with_partial_forward_propagation() {
    let mut graph = GraphInner::new();

    // 1. 创建计算图：z = x + y, w = x + y (两个节点都依赖同样的add操作)
    let x = graph.new_basic_input_node(&[2, 1], Some("x")).unwrap();
    let y = graph.new_basic_input_node(&[2, 1], Some("y")).unwrap();
    let add1 = graph.new_add_node(&[x, y], Some("add1")).unwrap();
    let add2 = graph.new_add_node(&[x, y], Some("add2")).unwrap();

    // 2. 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let y_value = Tensor::new(&[0.5, 1.5], &[2, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(y, Some(&y_value)).unwrap();

    // 3. 第1次前向传播add1
    graph.forward(add1).unwrap();
    let first_pass_id = graph.last_forward_pass_id();

    // 验证所有节点的pass_id都是第1次的pass_id
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
    graph.forward(final_add).unwrap();
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
    graph.forward(final_add).unwrap();
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
fn test_forward_pass_id_increment() {
    let mut graph = GraphInner::new();

    // 1. 创建简单的计算图：y = x + b
    let x = graph.new_basic_input_node(&[2, 1], Some("x")).unwrap();
    let b = graph.new_parameter_node(&[2, 1], Some("b")).unwrap();
    let y = graph.new_add_node(&[x, b], Some("y")).unwrap();

    // 2. 初始状态：pass_id应该为0
    assert_eq!(graph.last_forward_pass_id(), 0);

    // 3. 设置输入值
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let b_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.set_node_value(b, Some(&b_value)).unwrap();

    // 4. 第1次前向传播
    graph.forward(y).unwrap();
    assert_eq!(graph.last_forward_pass_id(), 1);

    // 5. 第2次前向传播
    graph.forward(y).unwrap();
    assert_eq!(graph.last_forward_pass_id(), 2);

    // 6. 第3次前向传播
    graph.forward(y).unwrap();
    assert_eq!(graph.last_forward_pass_id(), 3);
}

#[test]
fn test_pass_id_rollback_on_forward_error() {
    let mut graph = GraphInner::new();

    // 1. 创建计算图：y = x + b
    let x = graph.new_basic_input_node(&[2, 1], Some("x")).unwrap();
    let b = graph.new_parameter_node(&[2, 1], Some("b")).unwrap();
    let y = graph.new_add_node(&[x, b], Some("y")).unwrap();

    // 2. 设置b的值，但故意不设置x的值
    let b_value = Tensor::new(&[0.1, 0.2], &[2, 1]);
    graph.set_node_value(b, Some(&b_value)).unwrap();

    // 3. 记录初始pass_id
    let initial_forward_pass_id = graph.last_forward_pass_id();
    assert_eq!(initial_forward_pass_id, 0);

    // 4. 尝试前向传播，应该失败（因为x没有值）
    let forward_result = graph.forward(y);
    assert!(forward_result.is_err());

    // 验证前向传播失败后pass_id被正确回滚
    assert_eq!(graph.last_forward_pass_id(), initial_forward_pass_id);

    // 5. 设置x的值，使前向传播能够成功
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();

    // 6. 现在前向传播应该成功
    graph.forward(y).unwrap();
    assert_eq!(graph.last_forward_pass_id(), 1);
}
