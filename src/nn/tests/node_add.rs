use super::*;
use crate::nn::nodes::Add;

#[test]
fn test_new_for_node_add() {
    // 1.构造2个父节点
    let a = Variable::new(&[2, 3], false, false, None);
    let b = Variable::new(&[2, 3], false, false, None);
    let node = Add::new(&vec![a.as_node_enum(), b.as_node_enum()], None);
    assert_eq!(node.parents().len(), 2);
    assert_eq!(node.children().len(), 0);
    assert_eq!(node.name(), "<default>_add_1");
    assert!(!node.is_inited());
    assert!(node.is_trainable());
    // 2.构造3个父节点
    let c = Variable::new(&[2, 3], false, false, None);
    let node = Add::new(
        &vec![a.as_node_enum(), b.as_node_enum(), c.as_node_enum()],
        None,
    );
    assert_eq!(node.parents().len(), 3);
    assert_eq!(node.children().len(), 0);
    assert_eq!(node.name(), "<default>_add_2");
    assert!(!node.is_inited());
    assert!(node.is_trainable());
}

#[test]
fn test_new_panic_for_node_add() {
    // 1.因父节点数量过少导致报错
    let a = Variable::new(&[2, 3], false, false, None);
    assert_panic!(
        Add::new(&vec![a.as_node_enum()], None),
        "Add节点至少需要2个父节点"
    );
    // 2.因父节点形状不符合加法规则导致报错
    let b = Variable::new(&[2, 4], false, false, None);
    assert_panic!(
        Add::new(&vec![a.as_node_enum(), b.as_node_enum()], None),
        "形状不一致且两个张量没有一个是标量，故无法相加：第一个张量的形状为[2, 3]，第二个张量的形状为[2, 4]"
    );
}

#[test]
fn test_calc_value_for_node_add() {
    // 1.构造2个父节点
    let mut a = Variable::new(&[2, 3], false, false, None);
    let mut b = Variable::new(&[2, 3], false, false, None);
    let mut add_node = Add::new(&vec![a.as_node_enum(), b.as_node_enum()], None);
    // 2.计算前校验
    assert!(!add_node.is_inited());
    // 3.在add节点后赋值2个var节点
    let a_value = Tensor::normal(0.0, 1.0, &[2, 3]);
    let b_value = Tensor::normal(0.0, 1.0, &[2, 3]);
    a.set_value(&a_value);
    assert!(a.is_inited());
    b.set_value(&b_value);
    assert!(b.is_inited());
    // 4.计算后校验
    add_node.forward();
    assert!(add_node.is_inited());
    assert_eq!(add_node.value(), a_value + b_value);
    // 5.1 再次设置父节点值并检查
    let new_a_value = Tensor::normal(1.0, 0.5, &[2, 3]);
    let new_b_value = Tensor::normal(-1.0, 0.5, &[2, 3]);
    a.set_value(&new_a_value);
    b.set_value(&new_b_value);
    // 5.2 重新计算Add节点的值
    add_node.forward();
    // 5.3 检查Add节点的值是否正确更新
    assert!(add_node.is_inited());
    assert_eq!(add_node.value(), new_a_value + new_b_value);
}
