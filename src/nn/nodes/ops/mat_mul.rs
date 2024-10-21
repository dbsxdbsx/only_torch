use crate::nn::nodes::{NodeEnum, TraitForNode};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatMul {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓node basic fields↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    name: String,               // 节点名称
    value: Tensor,              // 本节点的值, 若"is_uninited()"则表示未初始化
    parents_names: Vec<String>, // 父节点名称列表
    children: Vec<NodeEnum>,    // 子节点
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑node basic fields↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
    jacobi: Tensor,
}

impl MatMul {
    pub fn new(parents: &[NodeEnum], name: Option<&str>) -> Self {
        // 1.构造前必要的校验
        // 1.1 矩阵乘法需要恰好两个父节点
        assert!(parents.len() == 2, "MatMul节点需恰好2个父节点");
        // 1.2 parents的形状需要符合矩阵乘法的规则
        let mut test_tensor = Tensor::eyes(parents[0].value().shape()[0]);
        for parent in parents {
            // NOTE: 即使父节点值未初始化，只要值的形状符合运算规则，就不会报错
            test_tensor = test_tensor.mat_mul(parent.value());
        }
        // 1.3 计算结果必须是2阶张量
        assert!(
            test_tensor.shape().len() == 2,
            "经MatMul节点计算的值必须是2阶张量, 但结果却是`{:?}`",
            test_tensor.dimension()
        );

        // 2.构造本节点
        let mut v = Self {
            name: name.unwrap_or_default().into(),
            parents_names: parents.iter().map(|p| p.name().to_string()).collect(),
            children: vec![],
            value: Tensor::uninited(test_tensor.shape()),
            jacobi: Tensor::default(),
        };

        // 3.注册并返回
        v.register(true);
        v
    }
}

impl TraitForNode for MatMul {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓固定trait实现↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    #[doc = " 获取本节点的所有父节点名称"]
    fn parents_names(&self) -> &[String] {
        &self.parents_names
    }
    #[doc = " 获取节点名称前缀"]
    fn name_prefix(&self) -> &str {
        "<default>_mat_mul"
    }
    #[doc = r" 获取节点名称（任何节点都必须有个不为空的节点名）"]
    fn name(&self) -> &str {
        &self.name
    }
    #[doc = " 设置节点名称"]
    fn set_name(&mut self, name: &str) {
        self.name = name.into();
    }
    #[doc = " 获取本节点的子节点"]
    fn children(&self) -> &[NodeEnum] {
        &self.children
    }
    #[doc = " 获取本节点的子节点"]
    fn children_mut(&mut self) -> &mut Vec<NodeEnum> {
        &mut self.children
    }
    #[doc = " 获取本节点的实际值（张量）"]
    fn value(&self) -> &Tensor {
        &self.value
    }
    #[doc = " 设置本节点的实际值（张量）"]
    fn value_mut(&mut self) -> &mut Tensor {
        &mut self.value
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑固定trait实现↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    fn as_node_enum(&self) -> NodeEnum {
        NodeEnum::MatMul(self.clone())
    }

    #[doc = r" 返回结果节点对本节点的雅可比矩阵的不可变引用"]
    fn jacobi(&self) -> &Tensor {
        &self.jacobi
    }
    #[doc = r" 返回结果节点对本节点的雅可比矩阵的可变引用"]
    fn jacobi_mut(&mut self) -> &mut Tensor {
        &mut self.jacobi
    }

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓梯度核心↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    #[doc = r" 根据父节点的值计算本节点的值（需手动实现）"]
    fn calc_value(&mut self) {
        let parents = self.parents();
        let parent1 = parents[0].borrow();
        let parent2 = parents[1].borrow();
        self.value = parent1.value().mat_mul(parent2.value());
    }
    #[doc = r" 计算并返回本节点对某个父节点的雅可比矩阵（需手动实现）"]
    /// NOTE: 这里的逻辑取巧参考了：https://github.com/zc911/MatrixSlow/blob/a6db0d38802004449941e6644e609a2455b26327/matrixslow/ops/ops.py#L61
    fn calc_jacobi_to_a_parent(&self, parent: &NodeEnum) -> Tensor {
        // TODO: delete
        Tensor::default()
        // let parents = self.parents();
        // let parent1 = parents[0].borrow();
        // let parent2 = parents[1].borrow();

        // if parent.name() == parent1.name() {
        //     // 对第一个父节点的雅可比矩阵
        //     Tensor::fill_diagonal_with_tensor(
        //         &Tensor::zeros(&[self.dimension(), parent.dimension()]),
        //         &parent2.value().transpose(),
        //     )
        // } else if parent.name() == parent2.name() {
        //     // 对第二个父节点的雅可比矩阵
        //     let jacobi = Tensor::fill_diagonal_with_tensor(
        //         &Tensor::zeros(&[self.dimension(), parent.dimension()]),
        //         parent1.value(),
        //     );
        //     let row_sort = Tensor::arange(self.dimension())
        //         .reshape(&self.shape().iter().rev().cloned().collect::<Vec<_>>())
        //         .transpose()
        //         .flatten();
        //     let col_sort = Tensor::arange(parent.dimension())
        //         .reshape(&parent.shape().iter().rev().cloned().collect::<Vec<_>>())
        //         .transpose()
        //         .flatten();
        //     jacobi.index_select(&row_sort, 0).index_select(&col_sort, 1)
        // } else {
        //     panic!(
        //         "MatMul节点的父节点名称必须匹配`{}`或`{}`，但传入的父节点名称是`{}`",
        //         parent1.name(),
        //         parent2.name(),
        //         parent.name()
        //     );
        // }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑梯度核心↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
