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
    fn get_value(&self) -> Option<&Tensor>;
    /// 返回本节点值的形状
    fn shape(&self) -> &[usize];
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
    /// 虽然看似这个方法应当在Node中定义，但不需要求梯度的节点类是不需要有“父节点”的概念的，如：Variable
    fn get_parents(&mut self) -> &mut [Box<dyn Node>];
    /// 前向传播计算本节点的值，若父节点的值未被计算，则递归调用父节点的forward方法
    fn forward(&mut self) {
        for node in self.get_parents() {
            if node.get_value().is_none() {
                if let Some(gradient_node) = node.as_any_mut().downcast_mut::<Box<dyn Gradient>>() {
                    gradient_node.forward();
                }
            }
        }
        self.compute()
    }

    /// 抽象方法，根据父节点的值计算本节点的值
    fn compute(&mut self);

    /// 抽象方法，计算本节点对某个父节点的雅可比矩阵
    fn get_jacobi(&self, parent: &dyn Node) -> Tensor;

    /// 反向传播，计算结果节点对本节点的雅可比矩阵
    fn backward(&self, result: &dyn Node) -> Tensor;

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
                value: Option<Tensor>, // 本节点的值
                trainable: bool, // 是否可训练
                children: Vec<Box<dyn Node>>, // 子节点列表
                shape: Vec<usize>, // 节点值（即张量）的形状
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
                fn get_value(&self) -> Option<&Tensor> {
                    self.value.as_ref()
                }
                fn set_value(&mut self, value: &Tensor) {
                    assert_eq!(value.shape(), self.shape);
                    // 本节点的值被改变，重置所有下游节点的值
                    self.reset_value(true);
                    self.value = Some(value.clone());
                }
                fn reset_value(&mut self, recursive: bool) {
                    self.value = None;
                    if recursive {
                        for child in self.get_children() {
                            child.reset_value(true);
                        }
                    }
                }
                fn shape(&self) -> &[usize] {
                    &self.shape
                }
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
