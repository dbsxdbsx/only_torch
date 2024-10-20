use std::cell::RefCell;
use std::rc::Rc;

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
        assert!(parents.len() >= 2, "Add节点至少需要2个父节点");
        // 1.2 parents的形状需要复合张量加法的规则
        let mut test_tensor = Tensor::default();
        for parent in parents {
            test_tensor += parent.value(); // NOTE:即使父节点值未初始化，只要值的形状符合运算规则，就不会报错
        }
        // 1.3 必须是2阶张量
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
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓部分固定↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    fn parents(&self) -> Vec<Rc<RefCell<NodeEnum>>> {
        crate::nn::graph::convert_parents(&self.parents_names)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑部分固定↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓固定trait实现↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    #[doc = " 获取节点名称前缀"]
    fn name_prefix(&self) -> &str {
        "<default>_add"
    }
    #[doc = r" 获取节点名称（任何节点都必须有个不为空的节点名）"]
    fn name(&self) -> &str {
        &self.name
    }
    #[doc = " 设置节点名称"]
    fn set_name(&mut self, name: &str) {
        self.name = name.into();
    }
    fn children(&self) -> &[NodeEnum] {
        &self.children
    }
    fn children_mut(&mut self) -> &mut Vec<NodeEnum> {
        &mut self.children
    }
    // TODO: 冗余的field方法，后期需要删除
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
    // TODO: 冗余的field方法，后期需要删除
    #[doc = r" 返回(不同于`set_jacobi`，这里仅返回但不计算)结果节点对本节点的雅可比矩阵"]
    fn jacobi(&self) -> &Tensor {
        &self.jacobi
    }
    #[doc = r" 设置结果节点对本节点的雅可比矩阵"]
    fn jacobi_mut(&mut self) -> &mut Tensor {
        &mut self.jacobi
    }

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓梯度核心↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    #[doc = r" 根据父节点的值计算本节点的值(每个使用本trait的节点类都需要实现这个方法)"]
    fn calc_value(&mut self) {
        let mut temp_value = Tensor::zero(self.parents()[0].borrow_mut().value().shape());
        for parent in self.parents() {
            temp_value += parent.borrow_mut().value();
        }
        self.value = temp_value;
    }
    #[doc = r" 计算并返回本节点对某个父节点的雅可比矩阵（需手动实现）"]
    fn calc_jacobi_to_a_parent(&self, _parent: &NodeEnum) -> Tensor {
        Tensor::eye(self.dimension()) // 矩阵之和对其中任一个矩阵的雅可比矩阵是单位矩阵
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑梯度核心↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
