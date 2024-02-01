use std::any::Any;

use crate::tensor::Tensor;
use crate::variable::Variable;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[enum_dispatch]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeEnum {
    Variable(Variable),
    // Other variants
}

// 实现Node trait的方法
#[enum_dispatch(NodeEnum)]
pub trait Node: Any {
    /// 生成节点名称，如果用户初始化时未指定，则根据节点类型生成类似于"MatMul:3"的节点名，
    /// 如果指定了name_scope，则生成类似"Hidden/MatMul:3"的节点名
    fn gen_node_name(&mut self);
    /// 返回本节点值（张量）的形状
    fn shape(&self) -> &[usize] {
        self.value().shape()
    }
    /// 返回本节点值（张量）的维度（阶数）
    fn dimension(&self) -> usize {
        self.value().dimension()
    }
    /// 返回本节点值（张量）的元素个数
    fn len(&self) -> usize {
        self.shape().iter().product()
    }
    /// 获取本节点的父节点（有些是不需要的，比如“Variable”）
    fn parents(&self) -> &[NodeEnum];
    fn parents_mut(&mut self) -> &mut [NodeEnum];
    /// 获取本节点的子节点
    fn children(&self) -> &[NodeEnum];
    fn children_mut(&mut self) -> &mut [NodeEnum];
    /// 设置本节点的实际值（张量）
    fn set_value(&mut self, value: &Tensor);
    /// 获取本节点的实际值（张量）
    fn value(&self) -> &Tensor;
    /// 重置本节点的值，并递归重置本节点的下游节点的值
    fn reset_value(&mut self, recursive: bool);
    //
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    //
    fn as_node_enum(&self) -> NodeEnum;
}

// 使用宏来简化结构体的定义
#[macro_export]
macro_rules! node {
    (pub struct $struct_name:ident {
        $($body:tt)*
    }) => {
        use crate::utils::traits::node::NodeEnum; // 添加这行
        $crate::node!(@impl pub struct $struct_name { $($body)* });
    };
    (struct $struct_name:ident {
        $($body:tt)*
    }) => {
        $crate::node!(@impl struct $struct_name { $($body)* });
    };
    (@impl $vis:vis struct $struct_name:ident {
        $($user_field_vis:vis $user_field_name:ident : $user_field_type:ty),*
        $(,)?
    }) => {
        paste::paste! {
            use serde::{Serialize, Deserialize};

            #[derive(Debug, Clone, Serialize, Deserialize, Default)]
            $vis struct $struct_name {
                name: Option<String>, // 节点名称
                value: Tensor, // 本节点的值, 若“is_uninited()”则表示未初始化
                trainable: bool, // 是否可训练
                children: Vec<NodeEnum>, // 子节点列表
                #[serde(default)]
                parents: Option<Vec<NodeEnum>>, // 父节点列表，有些节点不需要父节点，如“Variable”, 所以用Option
                // 以下是自定义的字段
                $($user_field_vis $user_field_name : $user_field_type,)*
            }

            impl Node for [<$struct_name>] {
                fn gen_node_name(&mut self) {
                    if self.name.is_none() {
                        let name = std::any::type_name::<Self>();
                        self.name = Some(name.into());
                    }
                }
                fn parents(&self) -> &[NodeEnum] {
                    self.parents.as_ref().expect("parents字段未初始化").as_slice()
                }
                fn parents_mut(&mut self) -> &mut [NodeEnum] {
                    self.parents.as_mut().expect("parents字段未初始化").as_mut_slice()
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
                    self.value = Tensor::empty(self.shape());
                    if recursive {
                        for child in self.children.iter_mut() {
                            child.reset_value(true);
                        }
                    }
                }
                // Any的方法
                fn as_any(&self) -> &dyn std::any::Any {
                    self
                }
                fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                    self
                }
                //
                fn as_node_enum(&self) -> NodeEnum {
                    NodeEnum::$struct_name(self.clone())
                }
            }
        }
    };
}

pub trait Gradient: Node {
    /// 返回结果节点对本节点的雅可比矩阵
    fn jacobi(&self) -> &Tensor;
    /// 设置结果节点对本节点的雅可比矩阵
    fn set_jacobi(&mut self, jacobi: Tensor);
    /// 清空结果节点对本节点的雅可比矩阵
    fn clear_jacobi(&mut self) {
        self.set_jacobi(Tensor::empty(&[0, 0]));
    }
    /// 计算并返回本节点对某个父节点的雅可比矩阵（需手动实现）
    fn parent_jacobi(&self, parent: &NodeEnum) -> Tensor;
    /// 前向传播计算本节点的值，若父节点的值未被计算，则递归调用父节点的forward方法
    fn forward(&mut self) {
        for node in self.parents_mut() {
            // 这里暗示了能取到父节点的node一定是实现了Gradient的类型
            if node.value().is_uninited() {
                node.as_any_mut()
                    .downcast_mut::<Box<dyn Gradient>>()
                    .unwrap()
                    .forward();
            }
        }
        self.compute();
    }
    /// 反向传播，计算结果节点对本节点的雅可比矩阵
    fn backward(&mut self, result: &NodeEnum) -> &Tensor {
        // fn backward(&mut self, result: &mut NodeEnum) -> &Tensor {
        if !self.jacobi().is_uninited() {
            return &self.jacobi();
        }
        // TODO: 真的需要这一block吗？ 对自身
        // if std::ptr::eq(self as *const _, result as *const NodeEnum as *const _) {
        // if std::ptr::eq(self as *const _, result as *const _) {
        //     self.set_jacobi(Tensor::eye(self.dimension()));
        //     return &self.jacobi();
        // }
        // 对其它节点
        self.set_jacobi(Tensor::zero(&[result.dimension(), self.dimension()]));
        let parent_node: NodeEnum = self.as_node_enum();
        let mut temp_jacobis = Vec::new();
        for child in self.children_mut() {
            if !child.value().is_uninited() {
                // jacobi_1
                let jacobi_1 = child
                    .as_any()
                    .downcast_ref::<Box<dyn Gradient>>()
                    .unwrap()
                    .parent_jacobi(&parent_node);
                // jacobi_2
                let jacobi_2 = child
                    .as_any_mut()
                    .downcast_mut::<Box<dyn Gradient>>()
                    .unwrap()
                    .backward(result);

                temp_jacobis.push(jacobi_1 * jacobi_2);
            }
        }
        // 最终获得结果节点对本节点的雅可比矩阵
        for jacobi in temp_jacobis {
            self.set_jacobi(self.jacobi() + jacobi);
        }

        self.jacobi()
    }
    /// 根据父节点的值计算本节点的值(每个使用本trait的节点类都需要实现这个方法)
    fn compute(&mut self);
}
