use super::*;
use crate::nn::nodes::MatMul;

#[test]
fn test_new_for_node_mat_mul() {
    // 1.构造2个父节点
    let a = Variable::new(&[2, 3], false, false, None);
    let b = Variable::new(&[3, 4], false, false, None);
    let node = MatMul::new(&vec![a.as_node_enum(), b.as_node_enum()], None);
    assert_eq!(node.parents_ids().len(), 2);
    assert_eq!(node.children_ids().len(), 0);
    assert_eq!(node.name(), "<default>_mat_mul_1");
    assert!(!node.is_inited());
    assert!(node.is_trainable());
}

#[test]
fn test_new_panic_for_node_mat_mul() {
    // 1.因父节点数量不等于2导致报错
    let a = Variable::new(&[2, 3], false, false, None);
    assert_panic!(
        MatMul::new(&vec![a.as_node_enum()], None),
        "MatMul节点需恰好2个父节点"
    );

    // 2.因父节点形状不符合矩阵乘法规则导致报错
    let b = Variable::new(&[2, 4], false, false, None);
    assert_panic!(
        MatMul::new(&vec![a.as_node_enum(), b.as_node_enum()], None),
        "前一个张量的列数必须等于后一个张量的行数"
    );
}

#[test]
fn test_calc_value_for_node_mat_mul() {
    // 1.构造2个父节点
    let mut a = Variable::new(&[2, 3], false, false, None);
    let mut b = Variable::new(&[3, 4], false, false, None);
    let mut mat_mul_node = MatMul::new(&vec![a.as_node_enum(), b.as_node_enum()], None);

    // 2.计算前校验
    assert!(!mat_mul_node.is_inited());

    // 3.在mat_mul节点后赋值2个var节点
    let a_value = Tensor::normal(0.0, 1.0, &[2, 3]);
    let b_value = Tensor::normal(0.0, 1.0, &[3, 4]);
    a.set_value(&a_value);
    assert!(a.is_inited());
    b.set_value(&b_value);
    assert!(b.is_inited());

    // 4.计算后校验
    mat_mul_node.forward();
    assert!(mat_mul_node.is_inited());
    assert_eq!(mat_mul_node.value(), a_value.mat_mul(&b_value));

    // 5.1 再次设置父节点值并检查
    let new_a_value = Tensor::normal(1.0, 0.5, &[2, 3]);
    let new_b_value = Tensor::normal(-1.0, 0.5, &[3, 4]);
    a.set_value(&new_a_value);
    b.set_value(&new_b_value);
    // 5.2 重新计算MatMul节点的值
    mat_mul_node.forward();
    // 5.3 检查MatMul节点的值是否正确更新
    assert!(mat_mul_node.is_inited());
    assert_eq!(mat_mul_node.value(), new_a_value.mat_mul(&new_b_value));
}

#[test]
fn test_jacobi_for_node_mat_mul() {
    // 1. 构造两个父节点 - 使用与Python示例相同的维度
    let mut a = Variable::new(&[2, 3], false, false, None);
    let mut b = Variable::new(&[3, 4], false, false, None);
    let mut mat_mul_node = MatMul::new(&vec![a.as_node_enum(), b.as_node_enum()], None);

    // 2. 为父节点设置值
    let a_value = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b_value = Tensor::new(
        &[
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        &[3, 4],
    );
    a.set_value(&a_value);
    b.set_value(&b_value);

    // 3. 计算MatMul节点的值
    mat_mul_node.forward();

    // 4. 计算雅可比矩阵
    let jacobi_a = mat_mul_node.calc_jacobi_to_a_parent(&a.as_node_enum());
    let jacobi_b = mat_mul_node.calc_jacobi_to_a_parent(&b.as_node_enum());

    // 5. 验证雅可比矩阵
    // 对a的雅可比矩阵 [8, 6]
    let expected_jacobi_a = Tensor::new(
        &[
            7.0, 11.0, 15.0, 0.0, 0.0, 0.0, 8.0, 12.0, 16.0, 0.0, 0.0, 0.0, 9.0, 13.0, 17.0, 0.0,
            0.0, 0.0, 10.0, 14.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 11.0, 15.0, 0.0, 0.0,
            0.0, 8.0, 12.0, 16.0, 0.0, 0.0, 0.0, 9.0, 13.0, 17.0, 0.0, 0.0, 0.0, 10.0, 14.0, 18.0,
        ],
        &[8, 6],
    );
    assert_eq!(jacobi_a, expected_jacobi_a);

    // 对b的雅可比矩阵 [8, 12]
    let expected_jacobi_b = Tensor::new(
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
    assert_eq!(jacobi_b, expected_jacobi_b);
}
