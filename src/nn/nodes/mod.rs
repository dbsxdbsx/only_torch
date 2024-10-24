mod ops;
mod variable;

use std::cell::RefCell;
use std::rc::Rc;

pub use self::ops::*;
pub use variable::Variable;

// ----------------------以下是节点相关的基本特性、接口、宏----------------------
/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓节点（Node）特性↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
use crate::tensor::Tensor;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[enum_dispatch]
#[derive(Debug, Clone, Serialize, Deserialize)]
// TODO: change the name to `Node`
pub enum NodeEnum {
    Variable,
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓算子↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    Add,
    MatMul,
    Step,
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑算子↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
impl NodeEnum {
    /// 比较两个NodeEnum实例是否在内存中完全相同
    pub fn eq_memory(&self, other: &NodeEnum) -> bool {
        std::ptr::eq(self, other)
    }
}

// 实现Node trait的方法
#[enum_dispatch(NodeEnum)]
pub trait TraitForNode {
    /// 注册本节点，干3件事：
    /// 1.（可选）将本节点添加到父节点的子节点列表中；
    /// 2.若节点名称为空，生成默认节点名称
    /// 3.将本节点添加到默认计算图中；
    /// TODO：是否需要检查图中节点名称重复？
    fn register(&mut self, add_to_parents_children: bool) {
        // 1.将本节点添加到父节点的子节点列表中
        if add_to_parents_children {
            self.add_the_node_to_children_of_parent_nodes();
        }
        // 2.若节点名称为空，生成默认节点名称
        if self.name().is_empty() {
            let gen_name = self.gen_name();
            self.set_name(&gen_name);
        }
        // 3.将本节点添加到默认计算图中
        super::graph::add_node_to_default_graph(&self.as_node_enum());
    }
    /// 将本节点添加到父节点的子节点列表中
    fn add_the_node_to_children_of_parent_nodes(&mut self) {
        let node_enum = self.as_node_enum();
        for parent in self.parents() {
            parent.borrow_mut().children_mut().push(node_enum.clone());
        }
    }
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓基本↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 是否节点值已初始化
    fn is_inited(&self) -> bool {
        self.value().is_inited()
    }
    /// 获取节点名称（任何节点都必须有个不为空的节点名）
    fn name(&self) -> &str;
    /// 获取节点名称前缀
    fn name_prefix(&self) -> &str;
    /// 生成节点名称，若用户初始化时未指定，则根据节点类型生成类似于"MatMul:3"的节点名，
    /// `若指定了name_scope，则生成类似"Hidden/MatMul:3"的节点名`
    fn gen_name(&mut self) -> String {
        super::graph::generate_unique_name(self.name_prefix())
    }
    /// 设置节点名称
    fn set_name(&mut self, name: &str);

    /// 检查输入的父节点是否是本节点的父节点(之一)
    fn check_parent(&self, node_type_name: &str, parent: &NodeEnum) {
        assert!(
            self.parents_names().contains(&parent.name().to_string()),
            "输入的父节点'{}'不是{}节点的父节点",
            parent.name(),
            node_type_name
        );
    }

    /// 获取本节点的父节点（有些是不需要的，比如“Variable”）
    fn parents(&self) -> Vec<Rc<RefCell<NodeEnum>>> {
        crate::nn::graph::convert_parents(self.parents_names())
    }
    /// 获取本节点的所有父节点名称
    fn parents_names(&self) -> &[String];
    /// 获取本节点的子节点
    fn children(&self) -> &[NodeEnum];
    fn children_mut(&mut self) -> &mut Vec<NodeEnum>;

    /// 获取本节点的实际值（张量）
    fn value(&self) -> &Tensor;
    /// 设置本节点的实际值（张量）
    fn value_mut(&mut self) -> &mut Tensor;
    /// 重置本节点的值(设置为未初始化)，可选择是否递归重置所有下游节点
    fn reset_value(&mut self, recursive: bool) {
        *self.value_mut() = Tensor::uninited(self.value().shape());
        super::graph::update_node_in_default_graph(&self.as_node_enum());
        if recursive {
            for child in self.children_mut().iter_mut() {
                child.reset_value(true);
            }
        }
    }
    // TODO: need?
    fn as_node_enum(&self) -> NodeEnum;
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑基本↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓梯度相关↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    fn is_trainable(&self) -> bool {
        true // 默认可训练, 除非特殊指定（比如`Variable`）
    }
    // 不管何种节点，以下2个是计算梯度的核心方法
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓需手动实现↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 根据父节点的值计算本节点的值（需手动实现）
    fn calc_value(&mut self);
    /// 计算并返回本节点对某个父节点的雅可比矩阵
    fn calc_jacobi_to_a_parent(&self, parent: &NodeEnum) -> Tensor;
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑需手动实现↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    // TODO: 下面的是rust因使用trait所特有的，后期通过宏，直接调用trait_field就不用这些个方法了
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓rust特有的↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 返回结果节点对本节点的雅可比矩阵的不可变引用
    fn jacobi(&self) -> &Tensor;
    /// 返回结果节点对本节点的雅可比矩阵的可变引用
    fn jacobi_mut(&mut self) -> &mut Tensor;
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑rust特有的↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /// 清空结果节点对本节点的雅可比矩阵
    fn clear_jacobi(&mut self) {
        *self.jacobi_mut() = Tensor::uninited(&[]);
    }

    /// 前向传播计算本节点的值（若父节点的值未被计算，则递归调用父节点的forward方法）
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
    /// 反向传播，计算结果节点对本节点的雅可比矩阵
    /// NOTE: 这里的逻辑参考了https://github.com/zc911/MatrixSlow/blob/a6db0d38802004449941e6644e609a2455b26327/matrixslow/core/node.py#L83
    fn backward(&mut self, result_node: &NodeEnum) -> &Tensor {
        if !self.jacobi().is_inited() {
            if self.as_node_enum().eq_memory(&result_node.as_node_enum()) {
                *self.jacobi_mut() = Tensor::eyes(self.value().size());
            } else {
                *self.jacobi_mut() =
                    Tensor::zeros(&[result_node.value().size(), self.value().size()]);

                // 对每个子节点进行反向传播
                for child in self.children_mut() {
                    if child.is_inited() {
                        let child_backward = child.backward(result_node);
                        // TODO:
                        // let child_jacobi = child.calc_jacobi_to_a_parent(&self.as_node_enum());
                        // *self.jacobi_mut() = self.jacobi() + child_backward * child_jacobi;
                    }
                }
            }
        }

        self.jacobi()
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑梯度相关↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}

// ----------------------以上是节点（Node）特性、接口、宏----------------------
