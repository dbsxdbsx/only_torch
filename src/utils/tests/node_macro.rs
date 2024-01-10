use crate::node;
use crate::tensor::Tensor;
use crate::utils::traits::node::Node;

node! {
    pub  struct  PublicStruct {
        name: Option<String>,   // 节点名称
        pub value: Option<Tensor>,  // 本节点的值
        pub parents: Vec<Box<dyn Node>>,  // 父节点列表
        children: Vec<Box<dyn Node>>, // 子节点列表
        shape: Vec<usize>,
    }
}
node! {
 struct   PrivateStruct {
        name: Option<String>,   // 节点名称
        value: Option<Tensor>,  // 本节点的值
        parents: Vec<Box<dyn Node>>,  // 父节点列表
        pub children: Vec<Box<dyn Node>>, // 子节点列表
        pub shape: Vec<usize>,
    }
}

#[test]
fn test_node_macro() {
    // 公有类
    let v1 = PublicStruct {
        name: None,
        value: None,
        parents: vec![],
        children: vec![],
        shape: vec![],
    };
    v1.len(); // 测试trait的默认方法
    v1.shape(); // 测试trait的宏展开方法
                // 私有类
    let v2 = PrivateStruct {
        name: None,
        value: None,
        parents: vec![],
        children: vec![],
        shape: vec![],
    };
    v2.len(); // 测试trait的默认方法
    v2.shape(); // 测试trait的宏展开方法
}
