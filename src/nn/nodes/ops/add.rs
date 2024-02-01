#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Add {
    //↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓node basic fields↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    name: Option<String>,    // 节点名称
    value: Tensor,           // 本节点的值, 若“is_empty()”则表示未初始化
    trainable: bool,         // 是否可训练
    children: Vec<NodeEnum>, // 子节点列表
    // TODO: need? #[serde(default)]
    parents: Option<Vec<NodeEnum>>, // 父节点列表，有些节点不需要父节点，如“Variable”, 所以用Option
                                    //↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑node basic fields↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
}

impl Add {
    pub fn new(shape: &[usize], init: bool, trainable: bool, name: Option<&str>) -> Self {
        // assert!(shape.len() >= 2);
        // let v = Add {
        //     name: name.map(|n| n.to_string()),
        //     value: if init {
        //         Tensor::normal(0.0, 0.001, shape)
        //     } else {
        //         Tensor::empty(shape)
        //     },
        //     trainable,
        //     children: vec![],
        //     ..Default::default()
        // };
        // // 将本节点添加到默认计算图中
        // add_node_to_default_graph(&v.as_node_enum());

        v
    }
}
