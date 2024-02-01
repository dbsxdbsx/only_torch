// use crate::node;
// use crate::tensor::Tensor;
// use crate::utils::traits::node::Node;
// use crate::variable::Variable;

// #[test]
// fn test_variable() {
//     let mut v = Variable::new(&[2, 3], true, true, None);
//     v.forward();
//     println!("{:?}", v);
//     println!("{:?}", v.value());
//     println!("{:?}", v.shape());
//     println!("{:?}", v.dimension());
//     println!("{:?}", v.len());
//     println!("{:?}", v.get_parents());
//     println!("{:?}", v.get_children());
//     println!("{:?}", v.trainable);
//     println!("{:?}", v.name);
//     println!("{:?}", v.jacobi);
//     println!("{:?}", v);
// }
