use serde::{Deserialize, Serialize};

use crate::nn::nodes::{NodeEnum, TraitForNode};
pub use crate::tensor::Tensor;
use crate::utils::add_node_to_default_graph;

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Add {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓node basic fields↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    name: Option<String>, // 节点名称
    value: Tensor,        // 本节点的值, 若“is_uninited()”则表示未初始化
    // TODO: need? #[serde(default)]
    parents: Option<Vec<NodeEnum>>, // 父节点列表，有些节点不需要父节点，如“Variable”, 所以用Option, todo: 非用Option不可？
    children: Vec<NodeEnum>,        // 子节点
                                    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑node basic fields↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}

impl Add {
    pub fn new(parents: &[NodeEnum], name: Option<&str>) -> Self {
        assert!(parents.len() >= 2); // 既然是加法，那么肯定至少有两个父节点

        // 检查父节点的形状是否一致
        let shape = parents[0].shape();
        for p in parents.iter().skip(1) {
            assert_eq!(p.shape(), shape);
        }
        let v = Add {
            name: name.map(|n| n.to_string()),
            parents: Some(parents.to_vec()),
            children: vec![],
            ..Default::default()
        };

        // 将本节点添加到默认计算图中
        add_node_to_default_graph(&v.as_node_enum());

        v
    }
}

impl TraitForNode for Add {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓固定trait实现↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    fn gen_node_name(&mut self) {
        if self.name.is_none() {
            let name = std::any::type_name::<Self>();
            self.name = Some(name.into());
        }
    }
    fn parents(&self) -> &[NodeEnum] {
        self.parents
            .as_ref()
            .expect("parents字段未初始化")
            .as_slice()
    }
    fn parents_mut(&mut self) -> &mut [NodeEnum] {
        self.parents
            .as_mut()
            .expect("parents字段未初始化")
            .as_mut_slice()
    }
    fn children(&self) -> &[NodeEnum] {
        &self.children
    }
    fn children_mut(&mut self) -> &mut [NodeEnum] {
        self.children.as_mut_slice()
    }
    fn value(&self) -> &Tensor {
        &self.value
    }
    fn set_value(&mut self, value: &Tensor) {
        assert_eq!(value.shape(), self.shape());
        // 本节点的值被改变，重置所有下游节点的值
        self.reset_value(true);
        self.value = value.clone();
    }
    fn reset_value(&mut self, recursive: bool) {
        self.value = Tensor::uninited(self.shape());
        if recursive {
            for child in self.children.iter_mut() {
                child.reset_value(true);
            }
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑固定trait实现↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
    fn as_node_enum(&self) -> NodeEnum {
        NodeEnum::Add(self.clone())
    }

    #[doc = r" 根据父节点的值计算本节点的值(每个使用本trait的节点类都需要实现这个方法)"]
    fn compute(&mut self) {
        todo!()
    }

    #[doc = r" 返回结果节点对本节点的雅可比矩阵"]
    fn jacobi(&self) -> &Tensor {
        todo!()
    }

    #[doc = r" 设置结果节点对本节点的雅可比矩阵"]
    fn set_jacobi(&mut self, _jacobi: Tensor) {
        todo!()
    }

    #[doc = r" 计算并返回本节点对某个父节点的雅可比矩阵（需手动实现）"]
    fn parent_jacobi(&self, _parent: &NodeEnum) -> Tensor {
        todo!()
    }
}
