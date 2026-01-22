/*
 * @Author       : 老董
 * @Date         : 2025-12-27
 * @Description  : 图描述符（Graph Descriptor）
 *                 统一的中间表示（IR），用于序列化、可视化和调试输出
 */

use serde::{Deserialize, Serialize};

/// 图的可序列化描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDescriptor {
    /// 格式版本（用于向后兼容）
    pub version: String,
    /// 图名称
    pub name: String,
    /// 所有节点描述
    pub nodes: Vec<NodeDescriptor>,
    /// 参数文件路径（相对于 JSON 文件），仅在保存完整模型时使用
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params_file: Option<String>,
}

/// 节点描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDescriptor {
    /// 节点 ID
    pub id: u64,
    /// 节点名称
    pub name: String,
    /// 节点类型
    pub node_type: NodeTypeDescriptor,
    /// 输出形状（固定形状，用于参数计算等）
    pub output_shape: Vec<usize>,
    /// 动态形状（支持动态维度，None 表示动态）
    ///
    /// 例如：`[None, Some(128)]` 表示 `[?, 128]`，batch 维度动态
    /// 如果为 None，表示该节点不支持动态维度（等同于固定形状）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_shape: Option<Vec<Option<usize>>>,
    /// 父节点 ID 列表（定义拓扑）
    pub parents: Vec<u64>,
    /// 参数数量（仅 Parameter 类型有意义）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param_count: Option<usize>,
}

/// 节点类型描述（包含类型特定参数）
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NodeTypeDescriptor {
    Input,
    Parameter,
    State,           // 时间状态节点（RNN 隐藏状态等）
    Identity,        // 恒等映射（用于 detach 等）
    GradientRouter,  // 梯度路由器（ModelState 内部使用，可视化时特殊样式）
    Add,
    Divide,
    Subtract,
    MatMul,
    Multiply,
    Sigmoid,
    Softmax,
    Tanh,
    LeakyReLU {
        alpha: f32,
    },
    Sign,
    SoftPlus,
    Step,
    Reshape {
        target_shape: Vec<usize>,
    },
    Flatten,
    Conv2d {
        stride: (usize, usize),
        padding: (usize, usize),
    },
    MaxPool2d {
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },
    AvgPool2d {
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },
    /// 张量索引选择（RNN 展开式设计用）
    Select {
        axis: usize,
        index: usize,
    },
    MSELoss,
    SoftmaxCrossEntropy,
    /// 动态零张量（RNN 初始隐藏状态）
    ZerosLike,
}

impl GraphDescriptor {
    /// 创建新的图描述符
    pub fn new(name: &str) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            name: name.to_string(),
            nodes: Vec::new(),
            params_file: None,
        }
    }

    /// 添加节点描述
    pub fn add_node(&mut self, node: NodeDescriptor) {
        self.nodes.push(node);
    }

    /// 获取总参数量
    pub fn total_params(&self) -> usize {
        self.nodes.iter().filter_map(|n| n.param_count).sum()
    }

    /// 转换为 JSON 字符串
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// 从 JSON 字符串解析
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl NodeDescriptor {
    /// 创建新的节点描述
    ///
    /// # 参数
    /// - `id`: 节点 ID
    /// - `name`: 节点名称
    /// - `node_type`: 节点类型描述
    /// - `output_shape`: 固定输出形状
    /// - `dynamic_shape`: 动态形状（None 表示固定形状，Some([None, Some(128)]) 表示 [?, 128]）
    /// - `parents`: 父节点 ID 列表
    pub fn new(
        id: u64,
        name: &str,
        node_type: NodeTypeDescriptor,
        output_shape: Vec<usize>,
        dynamic_shape: Option<Vec<Option<usize>>>,
        parents: Vec<u64>,
    ) -> Self {
        let param_count = if matches!(node_type, NodeTypeDescriptor::Parameter) {
            Some(output_shape.iter().product())
        } else {
            None
        };

        Self {
            id,
            name: name.to_string(),
            node_type,
            output_shape,
            dynamic_shape,
            parents,
            param_count,
        }
    }

    /// 获取用于显示的形状字符串
    ///
    /// 如果有动态形状，使用 `?` 表示动态维度
    /// 例如：`[?, 128]` 表示 batch 维度动态
    pub fn display_shape(&self) -> String {
        if let Some(dyn_shape) = &self.dynamic_shape {
            let dims: Vec<String> = dyn_shape
                .iter()
                .map(|d| match d {
                    Some(n) => n.to_string(),
                    None => "?".to_string(),
                })
                .collect();
            format!("[{}]", dims.join(", "))
        } else {
            format!("{:?}", self.output_shape)
        }
    }

    /// 检查节点是否支持动态 batch
    pub fn has_dynamic_batch(&self) -> bool {
        self.dynamic_shape
            .as_ref()
            .is_some_and(|ds| ds.first().is_some_and(|d| d.is_none()))
    }
}
