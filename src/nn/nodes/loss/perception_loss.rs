use crate::nn::nodes::{NodeEnum, TraitForNode};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerceptionLoss {
    name: String,
    value: Tensor,
    parents_names: Vec<String>,
    children: Vec<NodeEnum>,
    jacobi: Tensor,
}

impl PerceptionLoss {
    pub fn new(parents: &[NodeEnum], name: Option<&str>) -> Self {
        assert!(parents.len() == 1, "PerceptionLoss节点只能有1个父节点");

        let mut v = Self {
            name: name.unwrap_or_default().into(),
            parents_names: parents.iter().map(|p| p.name().to_string()).collect(),
            children: vec![],
            value: Tensor::uninited(parents[0].value().shape()),
            jacobi: Tensor::default(),
        };

        v.register(true);
        v
    }
}

impl TraitForNode for PerceptionLoss {
    fn parents_names(&self) -> &[String] {
        &self.parents_names
    }

    fn name_prefix(&self) -> &str {
        "<default>_perception_loss"
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

    fn as_node_enum(&self) -> NodeEnum {
        NodeEnum::PerceptionLoss(self.clone())
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
        let parent = parents[0].borrow();
        self.value = parent.value().where_with_f32(|x| x >= 0.0, |_| 0.0, |x| -x);
    }

    fn calc_jacobi_to_a_parent(&self, parent: &NodeEnum) -> Tensor {
        self.check_parent("PerceptionLoss", parent);
        let diag = parent.value().where_with_f32(|x| x >= 0.0, |_| 0.0, |_| -1.0);
        // TODO: let flatten_view = diag.flatten_view();
        let flatten = diag.flatten();
        Tensor::diag(&flatten)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑梯度核心↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
