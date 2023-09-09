use crate::tensor::Tensor;

pub trait NodeTrait {
    /// 获取本节点的父节点
    fn get_parents<T: NodeTrait>(&self) -> Vec<&T>;

    /// 获取本节点的子节点
    fn get_children<T: NodeTrait>(&self) -> Vec<&T>;

    /// 生成节点名称，如果用户不指定，则根据节点类型生成类似于"MatMul:3"的节点名，
    /// 如果指定了name_scope，则生成类似"Hidden/MatMul:3"的节点名
    fn get_node_name(&self);

    /// 前向传播计算本节点的值，若父节点的值未被计算，则递归调用父节点的forward方法
    fn forward(&mut self);

    /// 抽象方法，根据父节点的值计算本节点的值
    fn compute(&mut self);

    /// 抽象方法，计算本节点对某个父节点的雅可比矩阵
    fn get_jacobi<T: NodeTrait>(&self, parent: &T) -> Tensor;

    /// 反向传播，计算结果节点对本节点的雅可比矩阵
    fn backward<T: NodeTrait>(&self, result: &T) -> Tensor;

    /// 清空结果节点对本节点的雅可比矩阵
    fn clear_jacobi(&mut self);

    /// 返回本节点的值展平成向量后的维数
    fn dimension(&self) -> usize;

    /// 返回本节点的值作为矩阵的形状：（行数，列数）
    fn shape(&self) -> (usize, usize);

    /// 重置本节点的值，并递归重置本节点的下游节点的值
    fn reset_value(&mut self, recursive: bool);
    // NOTE: 所有实现本trait的struct都应按如下实现reset_value方法：
    // fn reset_value(&mut self, recursive: bool) {
    //     self.value = None;
    //     if recursive {
    //         for child in self.get_children() {
    //             child.reset_value(true);
    //         }
    //     }
    // }
}
