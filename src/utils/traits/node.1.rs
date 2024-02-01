use crate::tensor::Tensor;
use std::any::Any;

pub trait Node: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓节点类的基本方法↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
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
    fn get_parents(&mut self) -> &mut [Box<dyn Node>];
    /// 获取本节点的子节点
    fn get_children(&mut self) -> &mut [Box<dyn Node>];
    /// 设置本节点的实际值（张量）
    fn set_value(&mut self, value: &Tensor);
    /// 获取本节点的实际值（张量）
    fn value(&self) -> &Tensor;
    /// 重置本节点的值，并递归重置本节点的下游节点的值
    fn reset_value(&mut self, recursive: bool);
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑节点类的基本方法↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓梯度相关方法↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 前向传播计算本节点的值，若父节点的值未被计算，则递归调用父节点的forward方法
    fn forward(&mut self) {
        for node in self.get_parents() {
            if node.value().is_uninited() {
                // node.as_any_mut().downcast_mut::<Box<dyn Gradient>>().unwrap()
                node.forward();
            }
        }
        self.compute();
    }
    /// 反向传播，计算结果节点对本节点的雅可比矩阵
    // fn backward(&mut self, result: &mut dyn Node) -> &Tensor;
    fn backward(&mut self, result: &mut Box<dyn Node>) -> &Tensor;

    /// 根据父节点的值计算本节点的值（需手动实现）
    fn compute(&mut self);
    ///（若没算过）计算并非返回本节点对某个父节点的雅可比矩阵（需手动实现）
    fn get_jacobi(&mut self, parent: &dyn Node) -> &Tensor;
    /// 清空结果节点对本节点的雅可比矩阵
    fn clear_jacobi(&mut self);
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑梯度相关方法↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}

// 使用宏来简化结构体的定义
#[macro_export]
macro_rules! node {
    (pub struct $struct_name:ident {
        $($body:tt)*
    }) => {
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
            $vis struct $struct_name {
                name: Option<String>, // 节点名称
                value: Tensor, // 本节点的值, 若“is_uninited()”则表示未初始化
                trainable: bool, // 是否可训练
                children: Vec<Box<dyn Node>>, // 子节点列表
                parents: Option<Vec<Box<dyn Node>>>, // 父节点列表，有些节点不需要父节点，如“Variable”, 所以用Option
                jacobi: Tensor, // 结果节点对本节点的雅可比矩阵
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
                fn get_parents(&mut self) -> &mut [Box<dyn Node>] {
                    &mut self.parents.expect("节点类中未实现get_parents方法")
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
                fn backward(&mut self, result: &mut Box<dyn Node>) -> &Tensor {
                    if !self.jacobi.is_uninited() {
                        return &self.jacobi;
                    }
                    // 对自身
                    // let raw_var = self as *const (dyn Node + 'static);
                    // let raw_result = Box::into_raw(Box::from_raw(result as *mut (dyn Node + 'static))) as *const (dyn Node + 'static);
                    // if std::ptr::eq(raw_var, raw_result) {
                    //     self.jacobi = Tensor::eye(self.dimension());
                    //     return &self.jacobi;
                    // }
                    // if let Some(var) = self.as_any().downcast_ref::<Box<dyn Node>>() {
                    //     if std::ptr::eq(var as *const _, result.as_ref() as *const _) {
                    //         self.jacobi = Tensor::eye(self.dimension());
                    //         return &self.jacobi;
                    //     }
                    // }
                    if std::ptr::eq(self, result) {
                    }
                    // if std::ptr::eq(self as *const _, result.as_ref() as *const _) {
                    //     self.jacobi = Tensor::eye(self.dimension());
                    //     return &self.jacobi;
                    // }
                    // 对其它节点
                    self.jacobi = Tensor::zero(&[result.dimension(), self.dimension()]);
                    for child in self.get_children() {
                        if !child.value().is_uninited() {
                            // let c = child.as_any_mut()
                            // .downcast_mut::<Box<dyn Gradient>>()
                            // .unwrap();
                            // self.jacobi += c.backward(result) * c.get_jacobi(self);
                            self.jacobi += child.backward(result) * child.get_jacobi(self);
                        }
                    }
                    &self.jacobi
                }
                fn compute(&mut self) {
                    panic!("默认实现: 请在节点类中实现具体的compute方法")
                }
                fn get_jacobi(&mut self, _parent: &dyn Node) -> &Tensor {
                    panic!("默认实现: 请在节点类中实现具体的get_jacobi方法")
                }
                fn clear_jacobi(&mut self) {
                    self.jacobi = Tensor::empty(&[]); // NOTE: jacobi的初始形状是未知的
                }
            }
        }
    };
}
