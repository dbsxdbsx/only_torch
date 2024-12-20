use super::*;

#[test]
fn test_new_for_node_variable() {
    let shape = [2, 3];
    let init = false;
    let trainable = false;
    let name = Some("test_variable");
    let variable = Variable::new(&shape, init, trainable, name);

    assert_eq!(variable.value().unwrap().shape(), &shape);
    assert!(!variable.is_inited());
    assert!(!variable.is_trainable());
    assert_eq!(variable.name(), "test_variable");
}

#[test]
fn test_new_panic_for_node_variable() {
    let init = false;
    let trainable = false;
    let name = Some("test_variable");
    // 因阶数过低导致报错
    let shape = [2];
    assert_panic!(
        Variable::new(&shape, init, trainable, name),
        "Variable节点必须是2阶张量, 但得到的形状却是`1`"
    );
    // 因阶数过高导致报错
    let shape = [2, 3, 4];
    assert_panic!(
        Variable::new(&shape, init, trainable, name),
        "Variable节点必须是2阶张量, 但得到的形状却是`3`"
    );
}

#[test]
fn test_set_value_for_node_variable() {
    let shape = [2, 3];
    let init = false;
    let trainable = false;
    let name = Some("test_variable");

    let mut variable = Variable::new(&shape, init, trainable, name);
    let new_value = Tensor::normal(0.0, 1.0, &shape);
    // 赋值前
    assert!(!variable.is_inited());
    variable.set_value(Some(&new_value));
    // 赋值后
    assert!(variable.is_inited());
    assert_eq!(variable.value().unwrap().shape(), &shape);
}

#[test]
fn test_set_value_panic_for_node_variable() {
    let shape = [2, 3];
    let init = false;
    let trainable = false;
    let name = Some("test_variable");

    let mut variable = Variable::new(&shape, init, trainable, name);
    let new_value = Tensor::normal(0.0, 1.0, &[2, 4]);
    assert_panic!(variable.set_value(Some(&new_value))); // 形状不符导致报错
}

#[test]
fn test_forward_for_node_variable() {
    let shape = [2, 3];
    let init = false;
    let trainable = false;
    let name = Some("test_variable");

    let mut variable = Variable::new(&shape, init, trainable, name);
    let new_value = Tensor::normal(0.0, 1.0, &shape);
    variable.set_value(Some(&new_value));
    variable.calc_value_by_parents(&[]);
    assert_eq!(variable.value().unwrap().shape(), &shape);
}

#[test]
fn test_forward_panic_for_node_variable() {
    let shape = [2, 3];
    let init = false;
    let trainable = false;
    let name = Some("test_variable");

    let mut variable = Variable::new(&shape, init, trainable, name);
    assert_panic!(variable.calc_value_by_parents(&[])); // 未初始化的节点不能前向传播
}
