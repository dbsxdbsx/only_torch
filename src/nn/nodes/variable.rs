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
                   // TODO: need? #[serde(default)]
                   /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑node basic fields↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}

impl Variable {
    pub fn new(shape: &[usize], init: bool, trainable: bool, name: Option<&str>) -> Self {
        // 1.构造前必要的校验
        assert!(shape.len() == 2);

        // 2.构造本节点
        let mut v = Variable {
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
        assert_eq!(value.shape(), self.shape());
        // 本节点的值被改变，重置所有下游节点的值
        self.reset_value(true);
        self.value = value.clone();
    }
}

impl TraitForNode for Variable {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓部分固定↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    fn gen_name(&mut self) {
        self.name = "<default>_variable".into();
    }
    fn parents(&self) -> &Vec<NodeEnum> {
        unreachable!("Variable节点无需父节点");
    }
    fn parents_mut(&mut self) -> &mut Vec<NodeEnum> {
        unreachable!("Variable节点无需父节点");
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑部分固定↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓固定trait实现↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    #[doc = r" 获取节点名称（任何节点都必须有个不为空的节点名）"]
    fn name(&self) -> &str {
        &self.name
    }
    fn children(&self) -> &Vec<NodeEnum> {
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
    fn is_trainable(&self) -> bool {
        self.trainable
    }

    fn as_node_enum(&self) -> NodeEnum {
        NodeEnum::Variable(self.clone())
    }

    // TODO: 冗余的field方法，后期需要删除
    #[doc = r" 返回(不同于`set_jacobi`，这里仅返回但不计算)结果节点对本节点的雅可比矩阵"]
    fn jacobi(&self) -> &Tensor {
        unreachable!()
    }
    #[doc = r" 设置结果节点对本节点的雅可比矩阵"]
    fn jacobi_mut(&mut self) -> &mut Tensor {
        unreachable!()
    }

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓梯度核心↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    #[doc = r" 根据父节点的值计算本节点的值(每个使用本trait的节点类都需要实现这个方法)"]
    fn calc_value(&mut self) {
        unreachable!()
    }
    #[doc = r" 计算并返回本节点对某个父节点的雅可比矩阵（需手动实现）"]
    fn calc_jacobi_to_a_parent(&self, _parent: &NodeEnum) -> Tensor {
        unreachable!()
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑梯度核心↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
