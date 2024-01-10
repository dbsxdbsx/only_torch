use crate::tensor::Tensor;
use crate::utils::traits::node::Node;

crate::node! {
pub struct Variable {
    name: Option<String>,   // 节点名称
    value: Option<Tensor>,  // 本节点的值
    jacobi: Option<Tensor>, // 结果节点对本节点的雅可比矩阵
    trainable: bool,
    parents: Vec<Box<dyn Node>>,  // 父节点列表
    children: Vec<Box<dyn Node>>, // 子节点列表
    shape: Vec<usize>,
}
}

impl Variable {
    pub fn new(shape: &[usize], init: bool, trainable: bool, name: Option<&str>) -> Self {
        assert!(shape.len() == 2);
        Variable {
            name: name.map(|n| n.to_string()),
            value: if init {
                Some(Tensor::new_normal(0.0, 0.001, shape))
            } else {
                None
            },
            trainable,
            shape: shape.into(),
            jacobi: None,
            parents: vec![],
            children: vec![],
        }
    }
}
