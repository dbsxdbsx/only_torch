use super::*;
use crate::nn::nodes::{Add, NodeEnum, Variable};
use crate::tensor::Tensor;

// #[test]
// fn test_add_node_success() {
//     let parent1 = Variable::new(
//         Tensor::new(&[1.0, 2.0, 3.0], &[3]),
//         true,
//         true,
//         "parent1".into(),
//     );
//     let parent2 = Variable::new(Tensor::new(&[4.0, 5.0, 6.0], &[3]));
//     let parents = vec![parent1, parent2];
//     let add_node = Add::new(&parents, Some("add_node"));
//     assert_eq!(add_node.get_name(), "add_node");
// }

// #[test]
// #[should_panic(expected = "TensorError::IncompatibleShape")]
// fn test_add_node_panic() {
//     let parent1 = NodeEnum::Add(Add::new(
//         &vec![NodeEnum::Add(Add::default())],
//         Some("parent1"),
//     ));
//     let parent2 = NodeEnum::Add(Add::new(
//         &vec![NodeEnum::Add(Add::default())],
//         Some("parent2"),
//     ));
//     let parent3 = NodeEnum::Add(Add::new(
//         &vec![NodeEnum::Add(Add::default())],
//         Some("parent3"),
//     ));
//     let parents = vec![parent1, parent2, parent3];
//     let _ = Add::new(&parents, Some("add_node"));
// }
