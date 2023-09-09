use crate::tensor::Tensor;
use crate::utils::traits::node::Node;
/* #[macro_use]
node! { */
struct Variable {
    name: Option<String>,   // 节点名称
    value: Option<Tensor>,  // 本节点的值
    jacobi: Option<Tensor>, // 结果节点对本节点的雅可比矩阵
    trainable: bool,
    parents: Vec<Box<dyn Node>>,  // 父节点列表
    children: Vec<Box<dyn Node>>, // 子节点列表
    shape: Vec<usize>,
}
// }

impl Variable {
    fn new(shape: &[usize], init: bool, trainable: bool, name: Option<&str>) -> Self {
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

    fn set_value(&mut self, value: &Tensor) {
        assert_eq!(value.shape(), self.shape);

        // 本节点的值被改变，重置所有下游节点的值
        self.reset_value(true);
        self.value = Some(value.clone());
    }
}

impl Node for Variable {
    fn gen_node_name(&mut self) {
        if self.name.is_none() {
            let name = std::any::type_name::<Variable>();
            self.name = Some(name.into());
        }
    }

    fn get_parents(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.parents
    }

    fn get_children(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.children
    }

    fn get_value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn compute(&mut self) {
        todo!()
    }

    fn get_jacobi(&self, parent: &dyn Node) -> Tensor {
        todo!()
    }

    fn backward(&self, result: &dyn Node) -> Tensor {
        todo!()
    }

    fn clear_jacobi(&mut self) {
        self.jacobi = None
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn reset_value(&mut self, recursive: bool) {
        self.value = None;
        if recursive {
            for child in self.get_children() {
                child.reset_value(true);
            }
        }
    }
}
