// use super::*;
// use crate::nn::graph::{Graph, GraphError};
// use crate::nn::nodes::node_handle::NodeId;
// use crate::nn::nodes::Add;
// use crate::tensor::Tensor;

// #[test]
// fn test_new_for_node_add() {
//     let mut graph = Graph::with_name("test_graph").unwrap();

//     // 1.测试基本构造
//     let a = graph.variable(&[2, 3], false, false, Some("a"));
//     let b = graph.variable(&[2, 3], false, false, Some("b"));
//     let node_id = graph.add(&[a, b], None);

//     // 验证基本属性
//     assert_eq!(graph.get_node(node_id).unwrap().parents().len(), 2);
//     assert_eq!(graph.get_node(node_id).unwrap().children().len(), 0);
//     assert_eq!(graph.get_node(node_id).unwrap().name(), "test_graph_add_1");
//     assert!(!graph.get_node(node_id).unwrap().is_inited());
//     assert!(graph.get_node(node_id).unwrap().is_trainable());

//     // 2.测试多父节点
//     let c = graph.variable(&[2, 3], false, false, Some("c"));
//     let node_multi = graph.add(&[a, b, c], None);
//     assert_eq!(graph.get_node(node_multi).unwrap().parents().len(), 3);
//     assert_eq!(graph.get_node(node_multi).unwrap().children().len(), 0);
//     assert_eq!(
//         graph.get_node(node_multi).unwrap().name(),
//         "test_graph_add_2"
//     );
//     assert!(!graph.get_node(node_multi).unwrap().is_inited());
//     assert!(graph.get_node(node_multi).unwrap().is_trainable());
// }

// #[test]
// fn test_duplicate_graph_name() {
//     // 1. 创建第一个图
//     let _graph1 = Graph::with_name("test_graph").unwrap();

//     // 2. 尝试创建同名图，应该失败
//     assert!(matches!(
//         Graph::with_name("test_graph"),
//         Err(GraphError::DuplicateName(_))
//     ));
// }

// #[test]
// fn test_calc_value_for_node_add() {
//     let mut graph = Graph::new();

//     // 1.构造2个父节点
//     let a = graph.variable(&[2, 3], false, false, Some("a"));
//     let b = graph.variable(&[2, 3], false, false, Some("b"));
//     let add_node = graph.add(&[a, b], None);

//     // 2.计算前校验
//     assert!(!graph.get_node(add_node).unwrap().is_inited());
//     assert_eq!(
//         graph.get_node(add_node).unwrap().name(),
//         "default_graph_add_1"
//     );

//     // 3.在add节点后赋值2个var节点
//     let a_value = Tensor::normal(0.0, 1.0, &[2, 3]);
//     let b_value = Tensor::normal(0.0, 1.0, &[2, 3]);
//     graph
//         .get_node_mut(a)
//         .unwrap()
//         .set_value(Some(&a_value))
//         .unwrap();
//     assert!(graph.get_node(a).unwrap().is_inited());
//     graph
//         .get_node_mut(b)
//         .unwrap()
//         .set_value(Some(&b_value))
//         .unwrap();
//     assert!(graph.get_node(b).unwrap().is_inited());

//     // 4.计算后校验
//     graph.forward_node(add_node).unwrap();
//     assert!(graph.get_node(add_node).unwrap().is_inited());
//     assert_eq!(
//         graph.get_node_value(add_node).unwrap(),
//         &(a_value + b_value)
//     );

//     // 5.1 再次设置父节点值并检查
//     let new_a_value = Tensor::normal(1.0, 0.5, &[2, 3]);
//     let new_b_value = Tensor::normal(-1.0, 0.5, &[2, 3]);
//     graph
//         .get_node_mut(a)
//         .unwrap()
//         .set_value(Some(&new_a_value))
//         .unwrap();
//     graph
//         .get_node_mut(b)
//         .unwrap()
//         .set_value(Some(&new_b_value))
//         .unwrap();
//     // 5.2 重新计算Add节点的值
//     graph.forward_node(add_node).unwrap();
//     // 5.3 检查Add节点的值是否正确更新
//     assert!(graph.get_node(add_node).unwrap().is_inited());
//     assert_eq!(
//         graph.get_node_value(add_node).unwrap(),
//         &(new_a_value + new_b_value)
//     );
// }

// #[test]
// fn test_jacobi_for_node_add_with_2_parents() {
//     let mut graph = Graph::new();

//     // 1. 构造两个父节点
//     let a = graph.variable(&[2, 3], false, false, None);
//     let b = graph.variable(&[2, 3], false, false, None);
//     let add_node = graph.add(&[a, b], None);

//     // 2. 为父节点设��值
//     let a_value = Tensor::normal(0.0, 1.0, &[2, 3]);
//     let b_value = Tensor::normal(0.0, 1.0, &[2, 3]);
//     graph
//         .get_node_mut(a)
//         .unwrap()
//         .set_value(Some(&a_value))
//         .unwrap();
//     graph
//         .get_node_mut(b)
//         .unwrap()
//         .set_value(Some(&b_value))
//         .unwrap();

//     // 3. 计算Add节点的值
//     graph.forward_node(add_node).unwrap();

//     // 4. 计算并检查雅可比矩阵
//     let jacobi_a = graph
//         .get_node(add_node)
//         .unwrap()
//         .calc_jacobi_to_a_parent(a)
//         .unwrap();
//     let jacobi_b = graph
//         .get_node(add_node)
//         .unwrap()
//         .calc_jacobi_to_a_parent(b)
//         .unwrap();

//     // 5. 验证雅可比矩阵
//     // 对于加法操作，雅可比矩阵应该是单位矩阵
//     assert_eq!(jacobi_a, Tensor::eyes(6));
//     assert_eq!(jacobi_b, Tensor::eyes(6));
// }

// #[test]
// fn test_jacobi_for_node_add_with_3_parents() {
//     let mut graph = Graph::new();

//     // 1. 构造三个父节点
//     let a = graph.variable(&[2, 3], false, false, None);
//     let b = graph.variable(&[2, 3], false, false, None);
//     let c = graph.variable(&[2, 3], false, false, None);

//     // 2. 为父节点设置值
//     let a_value = Tensor::normal(0.0, 1.0, &[2, 3]);
//     let b_value = Tensor::normal(0.0, 1.0, &[2, 3]);
//     let c_value = Tensor::normal(0.0, 1.0, &[2, 3]);
//     graph
//         .get_node_mut(a)
//         .unwrap()
//         .set_value(Some(&a_value))
//         .unwrap();
//     graph
//         .get_node_mut(b)
//         .unwrap()
//         .set_value(Some(&b_value))
//         .unwrap();
//     graph
//         .get_node_mut(c)
//         .unwrap()
//         .set_value(Some(&c_value))
//         .unwrap();

//     // 3. 创建Add节点并计算值
//     let add_node = graph.add(&[a, b, c], None);
//     graph.forward_node(add_node).unwrap();

//     // 4. 计算并验证对三个父节点的雅可比矩阵
//     let jacobi_a = graph
//         .get_node(add_node)
//         .unwrap()
//         .calc_jacobi_to_a_parent(a)
//         .unwrap();
//     let jacobi_b = graph
//         .get_node(add_node)
//         .unwrap()
//         .calc_jacobi_to_a_parent(b)
//         .unwrap();
//     let jacobi_c = graph
//         .get_node(add_node)
//         .unwrap()
//         .calc_jacobi_to_a_parent(c)
//         .unwrap();

//     // 5. 验证雅可比矩阵
//     assert_eq!(jacobi_b, Tensor::eyes(6));
//     assert_eq!(jacobi_a, Tensor::eyes(6));
//     assert_eq!(jacobi_c, Tensor::eyes(6));
// }

// #[test]
// fn test_add_node_name() {
//     let mut graph = Graph::with_name("test_graph").unwrap();

//     // 1. 测试默认命名
//     let a = graph.variable(&[2, 3], false, false, Some("a")).unwrap();
//     let b = graph.variable(&[2, 3], false, false, Some("b")).unwrap();
//     let node_id = graph.add(&[a, b], None).unwrap();

//     // 验证节点名称
//     assert_eq!(graph.get_node(node_id).unwrap().name(), "test_graph_add_1");

//     // 2. 测试多个父节点的命名
//     let c = graph.variable(&[2, 3], false, false, Some("c")).unwrap();
//     let node_multi = graph.add(&[a, b, c], None).unwrap();

//     // 验证节点名称
//     assert_eq!(
//         graph.get_node(node_multi).unwrap().name(),
//         "test_graph_add_2"
//     );
// }

// #[test]
// fn test_add_node_value() {
//     let mut graph = Graph::with_name("test_graph").unwrap();

//     // 1. 创建测试数据
//     let a = graph.variable(&[2, 3], false, false, Some("a")).unwrap();
//     let b = graph.variable(&[2, 3], false, false, Some("b")).unwrap();
//     let add_node = graph.add(&[a, b], None).unwrap();

//     // 2. 设置输入值
//     let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//     let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
//     let a_value = Tensor::new(&[2, 3], &a_data);
//     let b_value = Tensor::new(&[2, 3], &b_data);
//     graph
//         .get_node_mut(a)
//         .unwrap()
//         .set_value(Some(&a_value))
//         .unwrap();
//     graph
//         .get_node_mut(b)
//         .unwrap()
//         .set_value(Some(&b_value))
//         .unwrap();

//     // 3. 前向传播
//     graph.forward_node(add_node).unwrap();

//     // 4. 验证结果
//     let expected_data = vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
//     let expected = Tensor::new(&[2, 3], &expected_data);
//     assert_eq!(graph.get_node_value(add_node).unwrap(), &expected);
// }

// #[test]
// fn test_add_node_jacobi() {
//     let mut graph = Graph::with_name("test_graph").unwrap();

//     // 1. 创建测试数据
//     let a = graph.variable(&[2, 3], false, false, None).unwrap();
//     let b = graph.variable(&[2, 3], false, false, None).unwrap();
//     let add_node = graph.add(&[a, b], None).unwrap();

//     // 2. 设置输入值
//     let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//     let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
//     let a_value = Tensor::new(&[2, 3], &a_data);
//     let b_value = Tensor::new(&[2, 3], &b_data);
//     graph
//         .get_node_mut(a)
//         .unwrap()
//         .set_value(Some(&a_value))
//         .unwrap();
//     graph
//         .get_node_mut(b)
//         .unwrap()
//         .set_value(Some(&b_value))
//         .unwrap();

//     // 3. 前向传播
//     graph.forward_node(add_node).unwrap();

//     // 4. 反向传播
//     graph.backward_node(a, add_node).unwrap();
//     graph.backward_node(b, add_node).unwrap();

//     // 5. 验证结果
//     let expected = Tensor::eyes(6);
//     assert_eq!(graph.get_node(a).unwrap().jacobi().unwrap(), &expected);
//     assert_eq!(graph.get_node(b).unwrap().jacobi().unwrap(), &expected);
// }

// #[test]
// fn test_add_node_multi_parents() {
//     let mut graph = Graph::with_name("test_graph").unwrap();

//     // 1. 创建测试数据
//     let a = graph.variable(&[2, 3], false, false, None).unwrap();
//     let b = graph.variable(&[2, 3], false, false, None).unwrap();
//     let c = graph.variable(&[2, 3], false, false, None).unwrap();

//     // 2. 设置输入值
//     let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//     let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
//     let c_data = vec![13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
//     let a_value = Tensor::new(&[2, 3], &a_data);
//     let b_value = Tensor::new(&[2, 3], &b_data);
//     let c_value = Tensor::new(&[2, 3], &c_data);
//     graph
//         .get_node_mut(a)
//         .unwrap()
//         .set_value(Some(&a_value))
//         .unwrap();
//     graph
//         .get_node_mut(b)
//         .unwrap()
//         .set_value(Some(&b_value))
//         .unwrap();
//     graph
//         .get_node_mut(c)
//         .unwrap()
//         .set_value(Some(&c_value))
//         .unwrap();

//     // 3. 创建Add节点并前向传播
//     let add_node = graph.add(&[a, b, c], None).unwrap();
//     graph.forward_node(add_node).unwrap();

//     // 4. 验证结果
//     let expected_data = vec![21.0, 24.0, 27.0, 30.0, 33.0, 36.0];
//     let expected = Tensor::new(&[2, 3], &expected_data);
//     assert_eq!(graph.get_node_value(add_node).unwrap(), &expected);
// }
