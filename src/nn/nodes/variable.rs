use crate::nn::graph::update_node_in_default_graph;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

use super::{NodeEnum, TraitForNode};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Variable {
    trainable: bool, // 是否可训练
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓node basic fields↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    name: String,  // 节点名称
    value: Tensor, // 本节点的值, 若“!is_inited()”则表示未初始化
    children: Vec<NodeEnum>, // 子节点列表

                   /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑node basic fields↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}

impl Variable {
    pub fn new(shape: &[usize], init: bool, trainable: bool, name: Option<&str>) -> Self {
        // 1.构造前必要的校验
        assert!(
            shape.len() == 2,
            "Variable节点必须是2阶张量, 但传入的形状却是`{:?}`",
            shape.len()
        );

        // 2.构造本节点
        let mut v = Self {
            name: name.unwrap_or_default().into(),
            value: if init {
                Tensor::normal(0.0, 0.001, shape)
            } else {
                Tensor::uninited(shape)
            },
            trainable,
            children: vec![],
        };

        // 3.注册并返回
        v.register(false);
        v
    }
    /// Variable节点特有的方法
    /// 因其可以创建未初始化的实例，所以需要个供外部实例设置值的方法
    /// 设置后，本节点的值就不再是“未初始化”的了，且所有下游节点的值会被重置
    pub fn set_value(&mut self, value: &Tensor) {
        assert_eq!(value.shape(), self.value().shape());
        // 本节点的值被改变，重置本节点及所有下游节点的值
        self.reset_value(true);
        // 重置后，再设置本节点的值
        self.value = value.clone();
        // 更新默认图中的节点
        update_node_in_default_graph(&self.as_node_enum());
    }
}

impl TraitForNode for Variable {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓固定trait实现↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

    fn parents_names(&self) -> &[String] {
        unreachable!("Variable节点无需父节点");
    }

    fn name_prefix(&self) -> &str {
        "<default>_variable"
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
    fn is_trainable(&self) -> bool {
        self.trainable
    }

    fn as_node_enum(&self) -> NodeEnum {
        NodeEnum::Variable(self.clone())
    }

    fn jacobi(&self) -> &Tensor {
        unreachable!()
    }

    fn jacobi_mut(&mut self) -> &mut Tensor {
        unreachable!()
    }

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓梯度核心↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

    fn calc_value(&mut self) {
        unreachable!()
    }

    fn calc_jacobi_to_a_parent(&self, _parent: &NodeEnum) -> Tensor {
        unreachable!()
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑梯度核心↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
