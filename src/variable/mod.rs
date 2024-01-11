use crate::tensor::Tensor;
use crate::utils::traits::node::Node;

crate::node! {
pub struct Variable {}
}

impl Variable {
    pub fn new(shape: &[usize], init: bool, trainable: bool, name: Option<&str>) -> Self {
        assert!(shape.len() == 2);
        Variable {
            name: name.map(|n| n.to_string()),
            value: if init {
                Tensor::normal(0.0, 0.001, shape)
            } else {
                Tensor::empty(shape)
            },
            trainable,
            children: vec![],
        }
    }
}
