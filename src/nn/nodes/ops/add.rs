use crate::nn::nodes::{NodeEnum, TraitForNode};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Add {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓node basic fields↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    name: String,               // 节点名称
    value: Tensor,              // 本节点的值, 若"is_uninited()"则表示未初始化
    parents_names: Vec<String>, // 父节点名称列表
    children: Vec<NodeEnum>,    // 子节点
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑node basic fields↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
    jacobi: Tensor,
}

impl Add {
    pub fn new(parents: &[NodeEnum], name: Option<&str>) -> Self {
        // 1.构造前必要的校验
        // 1.1 既然是加法，那么肯定至少有2个父节点
        assert!(parents.len() >= 2, "Add节点至少需2个父节点");
        // 1.2 parents的形状需要符合矩阵加法的规则
        let mut test_tensor = Tensor::default();
        for parent in parents {
            // NOTE:即使父节点值未初始化，只要值的形状符合运算规则，就不会报错
            test_tensor += parent.value();
        }
        // 1.3 计算结果必须是2阶张量
        assert!(
            test_tensor.shape().len() == 2,
            "经Add节点计算的值必须是2阶张量, 但结果却是`{:?}`",
            test_tensor.dimension()
        );

        // 2.构造本节点
        let mut v = Self {
            name: name.unwrap_or_default().into(),
            parents_names: parents.iter().map(|p| p.name().to_string()).collect(),
            children: vec![],
            value: Tensor::uninited(test_tensor.shape()),
            jacobi: Tensor::default(),
        };

        // 3.注册并返回
        v.register(true);
        v
    }
}

impl TraitForNode for Add {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓固定trait实现↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

    fn parents_names(&self) -> &[String] {
        &self.parents_names
    }

    fn name_prefix(&self) -> &str {
        "<default>_add"
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn set_name(&mut self, name: &str) {
        self.name = name.into();
    }

    fn children(&self) -> &[NodeEnum] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut Vec<NodeEnum> {
        &mut self.children
    }

    fn value(&self) -> &Tensor {
        &self.value
    }

    fn value_mut(&mut self) -> &mut Tensor {
        &mut self.value
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑固定trait实现↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    fn as_node_enum(&self) -> NodeEnum {
        NodeEnum::Add(self.clone())
    }

    fn jacobi(&self) -> &Tensor {
        &self.jacobi
    }

    fn jacobi_mut(&mut self) -> &mut Tensor {
        &mut self.jacobi
    }

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓梯度核心↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

    fn calc_value(&mut self) {
        let mut temp_value = Tensor::zeros(self.parents()[0].borrow_mut().value().shape());
        for parent in self.parents() {
            temp_value += parent.borrow().value();
        }
        self.value = temp_value;
    }

    fn calc_jacobi_to_a_parent(&self, parent: &NodeEnum) -> Tensor {
        self.check_parent("Add", parent);
        Tensor::eyes(self.value().size()) // 矩阵之和对其中任一个矩阵的雅可比矩阵是单位矩阵
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑梯度核心↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
