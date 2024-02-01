// use crate::gradient;
// use crate::tensor::Tensor;
// use crate::utils::traits::node::{Gradient, Node};

// gradient! {
//     pub  struct  PublicStruct {
//         pub public_field1:bool,
//         private_field1: Vec<Box<dyn Node>>,
//         pub public_field2: i32, // 故意写的注释
//         private_field2: Vec<Box<dyn Node>>,  // 故意写的注释
//     }
// }

// gradient! {
// struct   PrivateStruct {
//        pub public_field1:bool,
//        pub public_field2: i32, // 故意写的注释
//        private_field1: Vec<Box<dyn Node>>,
//        private_field2: Vec<Box<dyn Node>>,  // 故意写的注释
//    }
// }

// #[test]
// fn test_gradient_macro() {
//     // 公有类
//     let v1 = PublicStruct {
//         name: None,
//         value: Tensor::empty(&[]),
//         children: vec![],
//         trainable: true,
//         jacobi: Tensor::empty(&[]),
//         parents: vec![],
//         //
//         private_field1: vec![],
//         private_field2: vec![],
//         public_field1: true,
//         public_field2: 1,
//     };
//     v1.len(); // 测试trait的默认方法
//     v1.shape(); // 测试trait的宏展开方法
//                 // 私有类
//     let v2 = PrivateStruct {
//         name: None,
//         value: Tensor::eye(3),
//         children: vec![],
//         trainable: false,
//         jacobi: Tensor::empty(&[]),
//         parents: vec![],
//         //
//         private_field1: vec![],
//         private_field2: vec![],
//         public_field1: true,
//         public_field2: 1,
//     };
//     v2.len(); // 测试trait的默认方法
//     v2.shape(); // 测试trait的宏展开方法
// }
