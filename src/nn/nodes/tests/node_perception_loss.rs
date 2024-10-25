use super::*;
use crate::nn::nodes::PerceptionLoss;

#[test]
fn test_new_for_node_perception_loss() {
    // 1.构造父节点
    let x = Variable::new(&[2, 2], false, false, None);
    let node = PerceptionLoss::new(&vec![x.as_node_enum()], None);
    assert_eq!(node.parents_names().len(), 1);
    assert_eq!(node.children().len(), 0);
    assert_eq!(node.name(), "<default>_perception_loss_1");
    assert!(!node.is_inited());
    assert!(node.is_trainable());
}

#[test]
fn test_new_panic_for_node_perception_loss() {
    // 1.因父节点数量不等于1导致报错
    let x = Variable::new(&[2, 2], false, false, None);
    let y = Variable::new(&[2, 2], false, false, None);
    assert_panic!(
        PerceptionLoss::new(&vec![x.as_node_enum(), y.as_node_enum()], None),
        "PerceptionLoss节点只能有1个父节点"
    );
}

#[test]
fn test_calc_value_for_node_perception_loss() {
    // 1.构造父节点
    let mut x = Variable::new(&[2, 2], false, false, None);
    let mut loss_node = PerceptionLoss::new(&vec![x.as_node_enum()], None);

    // 2.计算前校验
    assert!(!loss_node.is_inited());

    // 3.设置父节点值
    let x_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    x.set_value(&x_value);
    assert!(x.is_inited());

    // 4.计算后校验
    loss_node.forward();
    assert!(loss_node.is_inited());
    let expected = Tensor::new(&[0.0, 1.0, 0.0, 0.0], &[2, 2]);
    assert_eq!(loss_node.value(), &expected);

    // 5.1 再次设置父节点值并检查
    let new_x_value = Tensor::new(&[-0.5, 0.0, 1.0, -2.0], &[2, 2]);
    x.set_value(&new_x_value);
    // 5.2 重新计算Loss节点的值
    loss_node.forward();
    // 5.3 检查Loss节点的值是否正确更新
    let new_expected = Tensor::new(&[0.5, 0.0, 0.0, 2.0], &[2, 2]);
    assert_eq!(loss_node.value(), &new_expected);
}

#[test]
fn test_jacobi_for_node_perception_loss() {
    // 1.构造父节点
    let mut x = Variable::new(&[2, 2], false, false, None);
    let mut loss_node = PerceptionLoss::new(&vec![x.as_node_enum()], None);

    // 2.设置父节点值
    let x_value = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[2, 2]);
    x.set_value(&x_value);

    // 3.计算Loss节点的值
    loss_node.forward();

    // 4.计算并检查雅可比矩阵
    let jacobi = loss_node.calc_jacobi_to_a_parent(&x.as_node_enum());

    // 5.验证雅可比矩阵
    // 对于PerceptionLoss，当输入x >= 0时，导数为0；当x < 0时，导数为-1
    let expected_jacobi = Tensor::new(
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        &[4, 4],
    );
    assert_eq!(jacobi, expected_jacobi);
}
