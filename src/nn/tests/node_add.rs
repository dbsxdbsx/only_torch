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
    assert_eq!(node.name(), "<default>_add");
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
    assert_eq!(node.name(), "<default>_add");
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
    // NOTE: 以下部分暂时不需要，因为Variabel节点的形状已经在构造时校验过了，只可能是2阶张量，所以加法结果也不会出现2阶以外情况
    // // 3.因计算结果不是2阶张量导致报错
    // let c = Variable::new(&[2, 3, 4], false, false, None);
    // let d = Variable::new(&[2, 3, 4], false, false, None);
    // assert_panic!(
    //     Add::new(&vec![c.as_node_enum(), d.as_node_enum()], None),
    //     "经Add节点计算的值必须是2阶张量, 但结果却是`3`"
    // );
}

#[test]
fn test_calc_value_for_node_add() {
    // 1.构造2个父节点
    let mut a = Variable::new(&[2, 3], false, false, None);
    let mut b = Variable::new(&[2, 3], false, false, None);
    let mut node = Add::new(&vec![a.as_node_enum(), b.as_node_enum()], None);
    // 2.计算前校验
    assert!(!node.is_inited());
    // 3.计算
    let a_value = Tensor::normal(0.0, 1.0, &[2, 3]);
    let b_value = Tensor::normal(0.0, 1.0, &[2, 3]);
    // TODO：任何保证转入Add等节点的父节点变量是个可变引用呢(这样Variable节点才能动态更新其值)？
    a.set_value(&a_value);
    b.set_value(&b_value);
    node.forward();
    // 4.计算后校验
    assert!(node.is_inited());
    assert_eq!(node.value(), a_value + b_value);
}
