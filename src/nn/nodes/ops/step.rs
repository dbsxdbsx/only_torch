use crate::nn::nodes::{NodeEnum, TraitForNode};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// 阶跃算子
/// 若输入大于等于0时输出1，输入小于0时输出0
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Step {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓node basic fields↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    name: String,               // 节点名称
    value: Tensor,              // 本节点的值, 若"is_uninited()"则表示未初始化
    parents_names: Vec<String>, // 父节点名称列表
    children: Vec<NodeEnum>,    // 子节点
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑node basic fields↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
    jacobi: Tensor,
}

impl Step {
    pub fn new(parents: &[NodeEnum], name: Option<&str>) -> Self {
        // 1.构造前必要的校验
        // 1.1 阶跃算子只能有1个父节点
        assert!(parents.len() == 1, "Step节点只能有1个父节点");
        // NOTE：这里不用检查父节点的形状必须是矩阵，因为父节点在构造时已经检查过了

        // 2.构造本节点
        let mut v = Self {
            name: name.unwrap_or_default().into(),
            parents_names: parents.iter().map(|p| p.name().to_string()).collect(),
            children: vec![],
            value: Tensor::uninited(parents[0].value().shape()),
            jacobi: Tensor::default(),
        };

        // 3.注册并返回
        v.register(true);
        v
    }
}

impl TraitForNode for Step {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓固定trait实现↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    fn parents_names(&self) -> &[String] {
        &self.parents_names
    }
    fn name_prefix(&self) -> &str {
        "<default>_step"
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
        NodeEnum::Step(self.clone())
    }

    fn jacobi(&self) -> &Tensor {
        &self.jacobi
    }
    fn jacobi_mut(&mut self) -> &mut Tensor {
        &mut self.jacobi
    }

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓梯度核心↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    fn calc_value(&mut self) {
        let parents = self.parents();
        let parent = parents[0].borrow();
        self.value = parent.value().where_greater_equal_than(0.0, 1.0, 0.0);
    }
    fn calc_jacobi_to_a_parent(&self, parent: &NodeEnum) -> Tensor {
        self.check_parent("Step", parent);
        Tensor::zeros(&self.value().shape()) // Step函数的导数在所有点都是0
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑梯度核心↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
