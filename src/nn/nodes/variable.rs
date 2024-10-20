use std::cell::RefCell;
use std::rc::Rc;

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
        assert_eq!(value.shape(), self.shape());
        // 本节点的值被改变，重置本节点及所有下游节点的值
        self.reset_value(true);
        // 重置后，再设置本节点的值
        self.value = value.clone();
        // 更新默认图中的节点
        update_node_in_default_graph(&self.as_node_enum());
    }
}

impl TraitForNode for Variable {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓部分固定↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    fn parents(&self) -> Vec<Rc<RefCell<NodeEnum>>> {
        unreachable!("Variable节点无需父节点");
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑部分固定↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓固定trait实现↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    #[doc = " 获取节点名称前缀"]
    fn name_prefix(&self) -> &str {
        "<default>_variable"
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

    #[doc = " 注册本节点，干3件事："]
    #[doc = " 1.（可选）将本节点添加到父节点的子节点列表中；"]
    #[doc = " 2.将本节点添加到默认计算图中；"]
    #[doc = " 3.若节点名称为空，生成默认节点名称"]
    #[doc = " TODO：是否需要检查图中节点名称重复？"]
    fn register(&mut self, add_to_parents_children: bool) {
        if add_to_parents_children {
            self.add_the_node_to_children_of_parent_nodes();
        }
        if self.name().is_empty() {
            let gen_name = self.gen_name();
            self.set_name(&gen_name);
        }
        crate::nn::graph::add_node_to_default_graph(&self.as_node_enum());
    }

    #[doc = " 将本节点添加到父节点的子节点列表中"]
    fn add_the_node_to_children_of_parent_nodes(&mut self) {
        let node_enum = self.as_node_enum();
        for parent in self.parents() {
            parent.borrow_mut().children_mut().push(node_enum.clone());
        }
    }

    #[doc = " 是否节点值已初始化"]
    fn is_inited(&self) -> bool {
        self.value().is_inited()
    }

    #[doc = " 生成节点名称，如果用户初始化时未指定，则根据节点类型生成类似于\"MatMul:3\"的节点名，"]
    #[doc = " 如果指定了name_scope，则生成类似\"Hidden/MatMul:3\"的节点名"]
    fn gen_name(&mut self) -> String {
        crate::nn::graph::generate_unique_name(self.name_prefix())
    }

    #[doc = " 返回本节点值（张量）的形状（节点未初始化也能获取）"]
    fn shape(&self) -> &[usize] {
        self.value().shape()
    }

    #[doc = " 返回本节点值（张量）的维度（阶数）（节点未初始化也能获取）"]
    fn dimension(&self) -> usize {
        self.value().dimension()
    }

    #[doc = " 返回本节点值（张量）的元素个数（节点未初始化也能获取）"]
    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    #[doc = " 返回本节点值（张量）是否是标量"]
    fn is_scalar_node(&self) -> bool {
        self.value().is_scalar()
    }

    #[doc = " 重置本节点的值(设置为未初始化)，可选择是否递归重置所有下游节点"]
    fn reset_value(&mut self, recursive: bool) {
        *self.value_mut() = Tensor::uninited(self.shape());
        crate::nn::graph::update_node_in_default_graph(&self.as_node_enum());
        if recursive {
            for child in self.children_mut().iter_mut() {
                child.reset_value(true);
            }
        }
    }

    #[doc = " 清空结果节点对本节点的雅可比矩阵"]
    fn clear_jacobi(&mut self) {
        *self.jacobi_mut() = Tensor::uninited(&[]);
    }

    #[doc = " 前向传播计算本节点的值（若父节点的值未被计算，则递归调用父节点的forward方法）"]
    fn forward(&mut self) {
        if let NodeEnum::Variable(_) = self.as_node_enum() {
            assert!(
                self.is_inited(),
                "Variable节点在`forward`前必须已初始化其值"
            );
        } else {
            for node in self.parents() {
                if !node.borrow().is_inited() {
                    node.borrow_mut().forward();
                }
            }
            self.calc_value();
        }
    }

    #[doc = " 反向传播，计算结果节点对本节点的雅可比矩阵"]
    fn backward(&mut self, result_node: &NodeEnum) -> &Tensor {
        if self.jacobi().is_inited() {
            return self.jacobi();
        }
        *self.jacobi_mut() = Tensor::zero(&[result_node.dimension(), self.dimension()]);
        let parent_node: NodeEnum = self.as_node_enum();
        let mut tmp_jacobis = Vec::new();
        for child in self.children_mut() {
            assert!(child.is_inited());
            let jacobi_1 = child.calc_jacobi_to_a_parent(&parent_node);
            let jacobi_2 = child.backward(result_node);
            tmp_jacobis.push(jacobi_1 * jacobi_2);
        }
        for tmp_jacobi in tmp_jacobis {
            *self.jacobi_mut() = self.jacobi() + tmp_jacobi;
        }
        self.jacobi()
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑梯度核心↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
