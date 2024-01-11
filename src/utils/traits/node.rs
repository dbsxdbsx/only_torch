use std::any::Any;

use crate::tensor::Tensor;

pub trait Node: Any {
    // TODO: 可否基于反射去做
    /// 生成节点名称，如果用户初始化时未指定，则根据节点类型生成类似于"MatMul:3"的节点名，
    /// 如果指定了name_scope，则生成类似"Hidden/MatMul:3"的节点名
    fn gen_node_name(&mut self);
    /// 获取本节点的子节点
    fn get_children(&mut self) -> &mut [Box<dyn Node>];
    /// 设置本节点的实际值（张量）
    fn set_value(&mut self, value: &Tensor);
    /// 获取本节点的实际值（张量）
    fn value(&self) -> &Tensor;
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
    /// 重置本节点的值，并递归重置本节点的下游节点的值
    fn reset_value(&mut self, recursive: bool);
    //
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait Gradient: Node {
    /// 获取本节点的父节点
    /// 虽然看似这个方法应当在Node中定义，但不需要求梯度的节点类是没“父节点”概念的，如：Variable
    fn get_parents(&mut self) -> &mut [Box<dyn Node>]; // TODO: 是否该返回 &mut [Box<dyn Gradient>]？
    /// 前向传播计算本节点的值，若父节点的值未被计算，则递归调用父节点的forward方法
    fn forward(&mut self) {
        for node in self.get_parents() {
            if node.value().is_empty() {
                node.as_any_mut()
                    .downcast_mut::<Box<dyn Gradient>>()
                    .unwrap()
                    .forward();
            }
        }
        self.compute();
    }
    /// 根据父节点的值计算本节点的值(每个使用本trait的节点类都需要实现这个方法)
    fn compute(&mut self);
    /// （若没算过）计算并非返回本节点对某个父节点的雅可比矩阵（每个使用本trait的节点类都需要实现这个方法）
    fn get_jacobi(&mut self, parent: &dyn Node) -> &Tensor;
    /// 反向传播，计算结果节点对本节点的雅可比矩阵
    fn backward(&mut self, result: &mut dyn Node) -> &Tensor;
    /// 清空结果节点对本节点的雅可比矩阵
    fn clear_jacobi(&mut self);
}

#[macro_export]
macro_rules! node {
    (pub struct $struct_name:ident { $($body:tt)* }) => {
        $crate::node!( @impl pub struct $struct_name { $($body)* });
    };
    (struct $struct_name:ident { $($body:tt)* }) => {
        $crate::node!( @impl struct $struct_name { $($body)* });
    };
    (@impl $vis:vis struct $struct_name:ident { $($user_field_vis:vis $user_field_name:ident : $user_field_type:ty),* $(,)? }) => {
        paste::paste! {
            $vis struct $struct_name {
                name: Option<String>, // 节点名称
                value: Tensor, // TODO: 本节点的值, 可否直接结合Tensor::empty()使用Tensor类型？
                // value: Option<Tensor>, // TODO: 本节点的值, 可否直接结合Tensor::empty()使用Tensor类型？
                trainable: bool, // 是否可训练
                children: Vec<Box<dyn Node>>, // 子节点列表
                // shape: Vec<usize>, // TODO: 节点值（即张量）的形状, 直接通过value.unwrap().shape()方法获取如何？
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
                fn get_children(&mut self) -> &mut [Box<dyn Node>] {
                    &mut self.children
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
                        for child in self.get_children() {
                            child.reset_value(true);
                        }
                    }
                }
                //
                fn as_any(&self) -> &dyn std::any::Any {
                    self
                }
                fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                    self
                }
            }
        }
    };
}

#[macro_export]
macro_rules! gradient {
    (pub struct $struct_name:ident { $($body:tt)* }) => {
        $crate::gradient!(@node pub struct $struct_name { $($body)* });
        $crate::gradient!(@impl pub struct $struct_name);
    };
    (struct $struct_name:ident { $($body:tt)* }) => {
        $crate::gradient!(@node struct $struct_name { $($body)* });
        $crate::gradient!(@impl struct $struct_name);
    };
    (@node $vis:vis struct $struct_name:ident { $($body:tt)* }) => {
        $crate::node!(struct $struct_name {
            jacobi: Tensor, // 结果节点对本节点的雅可比矩阵（可通过is_empty()判断其是否已计算过）
            parents: Vec<Box<dyn Node>>, // 父节点列表
            $($body)* });
    };
    (@impl $vis:vis struct $struct_name:ident) => {
        paste::paste! {
            impl Gradient for [<$struct_name>] {
                fn get_parents(&mut self) -> &mut [Box<dyn Node>] {
                    &mut self.parents
                }
                fn compute(&mut self) {
                    // 用户需要在这里实现具体的计算逻辑
                    todo!()
                }
                fn get_jacobi(&mut self, _parent: &dyn Node) -> &Tensor {
                    // 用户需要在这里实现具体的雅可比矩阵计算逻辑
                    todo!()
                }
                fn backward(&mut self, result: &mut dyn Node) -> &Tensor {
                    todo!()
                    // if self.jacobi.is_empty() {
                    //     if std::ptr::eq(self, result) {
                    //     // if std::ptr::eq(self as *const _, result as *const _) {
                    //         self.jacobi = Tensor::eye(self.dimension());
                    //     } else {
                    //         self.jacobi = Tensor::zero(&[result.dimension(), self.dimension()]);
                    //         for child in self.get_children() {
                    //             if !child.value().is_empty() {
                    //                 let c = child.as_any_mut()
                    //                 .downcast_mut::<Box<dyn Gradient>>()
                    //                 .unwrap();
                    //                 self.jacobi += c.backward(result) * c.get_jacobi(self);
                    //             }
                    //         }
                    //     }
                    // }
                    // &self.jacobi
                }
                fn clear_jacobi(&mut self) {
                    self.jacobi = Tensor::empty(&[]); // NOTE: jacobi的初始形状是未知的
                }
            }
        }
    };
}
