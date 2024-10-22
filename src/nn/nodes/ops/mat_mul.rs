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

    fn parents_names(&self) -> &[String] {
        &self.parents_names
    }

    fn name_prefix(&self) -> &str {
        "<default>_mat_mul"
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn set_name(&mut self, name: &str) {
        self.name = name.into();
    }

    fn children(&self) -> &[NodeEnum] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut Vec<NodeEnum> {
        &mut self.children
    }

    fn value(&self) -> &Tensor {
        &self.value
    }

    fn value_mut(&mut self) -> &mut Tensor {
        &mut self.value
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑固定trait实现↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    fn as_node_enum(&self) -> NodeEnum {
        NodeEnum::MatMul(self.clone())
    }

    fn jacobi(&self) -> &Tensor {
        &self.jacobi
    }

    fn jacobi_mut(&mut self) -> &mut Tensor {
        &mut self.jacobi
    }

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓梯度核心↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

    fn calc_value(&mut self) {
        let parents = self.parents();
        let parent1 = parents[0].borrow();
        let parent2 = parents[1].borrow();
        self.value = parent1.value().mat_mul(parent2.value());
    }

    /// NOTE: 这里的逻辑本想取巧参考：https://github.com/zc911/MatrixSlow/blob/a6db0d38802004449941e6644e609a2455b26327/matrixslow/ops/ops.py#L61
    /// 但发现太难懂了，所以还是用最原始的实现吧
    fn calc_jacobi_to_a_parent(&self, parent: &NodeEnum) -> Tensor {
        let parents = self.parents();
        let parent_a = parents[0].borrow();
        let parent_b = parents[1].borrow();

        let parent_a_shape = parent_a.value().shape();
        let parent_b_shape = parent_b.value().shape();
        let m = parent_a_shape[0];
        let n = parent_a_shape[1];
        let p = parent_b_shape[1];

        if parent.name() == parent_a.name() {
            // dC/dA
            let mut jacobi = Tensor::zeros(&[m * p, m * n]);
            for i in 0..m {
                for j in 0..p {
                    for k in 0..n {
                        jacobi[[i * p + j, i * n + k]] = parent_b.value()[[k, j]];
                    }
                }
            }
            jacobi
        } else if parent.name() == parent_b.name() {
            // dC/dB
            let mut jacobi = Tensor::zeros(&[m * p, n * p]);
            for i in 0..m {
                for j in 0..p {
                    for k in 0..n {
                        jacobi[[i * p + j, k * p + j]] = parent_a.value()[[i, k]];
                    }
                }
            }
            jacobi
        } else {
            panic!("输入的parent不是该节点的父节点之一");
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑梯度核心↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
