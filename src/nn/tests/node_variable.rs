use crate::nn::nodes::{Add, NodeEnum, TraitForNode, Variable};
use crate::tensor::Tensor;

#[test]
fn test_variable_new() {
    let shape = [2, 3];
    let init = false;
    let trainable = true;
    let name = Some("test_variable");

    let variable = Variable::new(&shape, init, trainable, name);

    assert_eq!(variable.shape(), &shape);
    assert!(!variable.is_inited());
    assert!(variable.is_trainable());
    assert_eq!(variable.name(), "test_variable");
    // 这里可以添加更多的断言来验证`value`是否正确初始化
}

// #[test]
// fn test_variable_set_value() {
//     let shape = [2, 3];
//     let init = false;
//     let trainable = false;
//     let name = Some("test_variable");

//     let mut variable = Variable::new(&shape, init, trainable, name);
//     let new_value = Tensor::normal(0.0, 1.0, &shape);
//     variable.set_value(&new_value);

//     assert_eq!(variable.value, new_value);
//     // 这里可以添加更多的断言来验证`value`是否被正确设置
// }
