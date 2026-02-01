mod input;
mod loss;
mod ops;
mod parameter;
mod state;

pub(in crate::nn) use input::InputVariant;
pub use loss::DEFAULT_HUBER_DELTA;
pub use loss::Reduction;
pub(in crate::nn) use loss::{BCE, Huber, MAE, MSE, SoftmaxCrossEntropy};
pub use ops::DEFAULT_DROPOUT_P;
pub(in crate::nn) use ops::*;
pub(crate) use parameter::Parameter;
pub(in crate::nn) use state::State;

use enum_dispatch::enum_dispatch;
use strum::{EnumCount, VariantNames};

/// 统一定义 NodeType 枚举和元数据
///
/// 添加新节点时，只需在此宏调用中添加一项，枚举和元数据自动同步。
/// `debug.rs` 会自动读取 `NODE_METADATA` 常量，无需额外维护。
macro_rules! define_node_types {
    (
        $(
            $(#[$attr:meta])*
            $variant:ident($inner:ty) {
                category: $cat:literal,
                description: $desc:literal,
                var_method: $var:expr $(,)?
            }
        ),* $(,)?
    ) => {
        #[enum_dispatch]
        #[derive(Clone, EnumCount, VariantNames)]
        pub(in crate::nn) enum NodeType {
            $(
                $(#[$attr])*
                $variant($inner),
            )*
        }

        /// 节点元数据（由 define_node_types! 宏自动生成）
        ///
        /// 格式：(类别, 描述, Var 方法)
        /// 顺序与 `NodeType::VARIANTS` 完全一致
        pub const NODE_METADATA: &[(&str, &str, Option<&str>)] = &[
            $( ($cat, $desc, $var), )*
        ];
    };
}

// ============================================================================
// 节点类型定义（枚举变体 + 元数据）
//
// 添加新节点时，在对应分类下添加一项即可，格式：
//   VariantName(InnerType) {
//       category: "分类名",
//       description: "简要描述",
//       var_method: Some("方法名") 或 None,
//   },
// ============================================================================
define_node_types! {
    // ==================== 输入/参数/状态 ====================
    Input(InputVariant) {
        category: "输入",
        description: "外部数据输入（Data/Target/Smart/RecurrentOutput）",
        var_method: None,
    },
    Parameter(Parameter) {
        category: "参数",
        description: "可学习参数（weight/bias）",
        var_method: None,
    },
    State(State) {
        category: "状态",
        description: "时间状态节点（RNN 隐藏状态）",
        var_method: None,
    },

    // ==================== 算术运算 ====================
    Add(Add) {
        category: "算术",
        description: "逐元素加法（支持广播）",
        var_method: Some("+ 运算符"),
    },
    Subtract(Subtract) {
        category: "算术",
        description: "逐元素减法（支持广播）",
        var_method: Some("- 运算符"),
    },
    Multiply(Multiply) {
        category: "算术",
        description: "逐元素乘法（支持广播）",
        var_method: Some("* 运算符"),
    },
    Divide(Divide) {
        category: "算术",
        description: "逐元素除法（支持广播）",
        var_method: Some("/ 运算符"),
    },

    // ==================== 矩阵/卷积运算 ====================
    MatMul(MatMul) {
        category: "矩阵/卷积",
        description: "矩阵乘法",
        var_method: Some("matmul()"),
    },
    Conv2d(Conv2d) {
        category: "矩阵/卷积",
        description: "2D 卷积",
        var_method: None,
    },
    MaxPool2d(MaxPool2d) {
        category: "矩阵/卷积",
        description: "2D 最大池化",
        var_method: None,
    },
    AvgPool2d(AvgPool2d) {
        category: "矩阵/卷积",
        description: "2D 平均池化",
        var_method: None,
    },

    // ==================== 形状变换 ====================
    Reshape(Reshape) {
        category: "形状",
        description: "张量变形",
        var_method: Some("reshape()"),
    },
    Flatten(Flatten) {
        category: "形状",
        description: "展平（保留 batch 维）",
        var_method: Some("flatten()"),
    },
    Select(Select) {
        category: "形状",
        description: "固定索引选择（RNN 时间步）",
        var_method: Some("select()"),
    },
    Gather(Gather) {
        category: "形状",
        description: "动态索引收集（强化学习）",
        var_method: Some("gather()"),
    },
    Stack(Stack) {
        category: "形状",
        description: "张量堆叠/拼接",
        var_method: Some("Var::stack()"),
    },

    // ==================== 比较/归约 ====================
    Maximum(Maximum) {
        category: "归约",
        description: "逐元素取最大值",
        var_method: None,
    },
    Minimum(Minimum) {
        category: "归约",
        description: "逐元素取最小值",
        var_method: None,
    },
    Amax(Amax) {
        category: "归约",
        description: "沿轴取最大值",
        var_method: None,
    },
    Amin(Amin) {
        category: "归约",
        description: "沿轴取最小值",
        var_method: None,
    },
    Sum(Sum) {
        category: "归约",
        description: "归约求和",
        var_method: Some("sum() / sum_axis()"),
    },
    Mean(Mean) {
        category: "归约",
        description: "归约求均值",
        var_method: Some("mean() / mean_axis()"),
    },

    // ==================== 激活函数 ====================
    Sigmoid(Sigmoid) {
        category: "激活",
        description: "Sigmoid 激活",
        var_method: Some("sigmoid()"),
    },
    Tanh(Tanh) {
        category: "激活",
        description: "Tanh 激活",
        var_method: Some("tanh()"),
    },
    LeakyReLU(LeakyReLU) {
        category: "激活",
        description: "LeakyReLU 激活（含 ReLU）",
        var_method: Some("leaky_relu() / relu()"),
    },
    Softmax(Softmax) {
        category: "激活",
        description: "Softmax 归一化",
        var_method: Some("softmax()"),
    },
    LogSoftmax(LogSoftmax) {
        category: "激活",
        description: "数值稳定的 log(softmax)",
        var_method: Some("log_softmax()"),
    },
    SoftPlus(SoftPlus) {
        category: "激活",
        description: "SoftPlus 激活（平滑 ReLU）",
        var_method: Some("softplus()"),
    },
    Step(Step) {
        category: "激活",
        description: "阶跃函数",
        var_method: Some("step()"),
    },
    Sign(Sign) {
        category: "激活",
        description: "符号函数",
        var_method: Some("sign()"),
    },
    Abs(Abs) {
        category: "激活",
        description: "绝对值",
        var_method: Some("abs()"),
    },
    Ln(Ln) {
        category: "激活",
        description: "自然对数",
        var_method: Some("ln()"),
    },

    // ==================== 损失函数 ====================
    MSE(MSE) {
        category: "损失",
        description: "均方误差损失",
        var_method: Some("mse_loss()"),
    },
    MAE(MAE) {
        category: "损失",
        description: "平均绝对误差损失",
        var_method: Some("mae_loss()"),
    },
    BCE(BCE) {
        category: "损失",
        description: "二元交叉熵损失",
        var_method: Some("bce_loss()"),
    },
    Huber(Huber) {
        category: "损失",
        description: "Huber 损失（强化学习）",
        var_method: Some("huber_loss()"),
    },
    SoftmaxCrossEntropy(SoftmaxCrossEntropy) {
        category: "损失",
        description: "Softmax + 交叉熵损失",
        var_method: Some("cross_entropy()"),
    },

    // ==================== 辅助节点 ====================
    Identity(Identity) {
        category: "辅助",
        description: "恒等映射（用于 detach）",
        var_method: Some("detach_node()"),
    },
    Dropout(Dropout) {
        category: "辅助",
        description: "随机丢弃（正则化）",
        var_method: Some("dropout()"),
    },
    ZerosLike(ZerosLike) {
        category: "辅助",
        description: "动态零张量（RNN 初始状态）",
        var_method: None,
    },
}

use super::{GraphError, NodeHandle, NodeId};
use crate::nn::format_node_display;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use std::any::type_name;

#[enum_dispatch(NodeType)]
pub(in crate::nn::nodes) trait TraitNode {
    fn id(&self) -> NodeId;

    fn set_id(&mut self, id: NodeId);

    fn name(&self) -> &str;

    fn set_name(&mut self, name: &str);

    fn get_type_name(&self) -> &'static str {
        type_name::<Self>().split("::").last().unwrap_or("Unknown")
    }

    fn display_node(&self) -> String {
        format_node_display(self.id(), self.name(), self.get_type_name())
    }

    // 根据父节点的值计算本节点的值（注意：由于该接口只在Graph中使用，所以实现时不用关心父节点的值是否已被计算，所有父节点的值可以已预先被计算过了）
    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError>;

    fn value(&self) -> Option<&Tensor>;

    fn set_value(&mut self, _value: Option<&Tensor>) -> Result<(), GraphError> {
        Err(GraphError::InvalidOperation(format!(
            "{}的值只能通过前向传播计算得到，不能直接设置",
            self.display_node()
        )))
    }

    /// 清除节点的值（用于释放内存）
    ///
    /// 与 `set_value(None)` 不同，此方法专门用于内存管理，
    /// 对于不允许直接设置值的节点（如运算节点）也能正常清除。
    fn clear_value(&mut self) -> Result<(), GraphError>;

    /// 强制设置节点的值（绕过类型检查）
    ///
    /// ⚠️ 仅供内部使用（如 BPTT 快照恢复）。
    /// 普通用户应使用 `set_value`，它会检查节点类型。
    fn set_value_unchecked(&mut self, value: Option<&Tensor>);

    // ========== 梯度（VJP 模式）==========

    /// 计算本节点对父节点的梯度（VJP 模式）
    ///
    /// # 参数
    /// - `target_parent`: 目标父节点
    /// - `upstream_grad`: 从下游传来的梯度，shape 与本节点 value 相同
    /// - `assistant_parent`: 辅助父节点（用于双父节点如 `MatMul`）
    ///
    /// # 返回
    /// 对 `target_parent` 的梯度，shape 与 `target_parent.value` 相同
    ///
    /// # 默认实现
    /// 返回错误，需要各节点自行实现
    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let _ = (target_parent, upstream_grad, assistant_parent);
        Err(GraphError::InvalidOperation(format!(
            "{}尚未实现 calc_grad_to_parent",
            self.display_node()
        )))
    }

    /// 获取节点的梯度
    fn grad(&self) -> Option<&Tensor> {
        None // 默认不支持，需要各节点实现
    }

    /// 设置节点的梯度
    fn set_grad(&mut self, _grad: Option<&Tensor>) -> Result<(), GraphError> {
        Err(GraphError::InvalidOperation(format!(
            "{}尚未实现 set_grad",
            self.display_node()
        )))
    }

    /// 清除节点的梯度
    fn clear_grad(&mut self) -> Result<(), GraphError> {
        self.set_grad(None)
    }

    // ========== 通用方法 ==========

    fn is_inited(&self) -> bool {
        self.value().is_some()
    }

    /// 返回节点的预期输出形状（固定形状）
    ///
    /// 这个形状在节点创建时就已确定，存储在节点中。
    /// 对于支持动态维度的节点，应同时实现 `dynamic_expected_shape`。
    fn value_expected_shape(&self) -> &[usize];

    /// 返回节点的动态形状
    ///
    /// 默认实现基于 `value_expected_shape` 创建固定形状。
    /// 支持动态维度的节点（如 GradientRouter、State）应覆盖此方法。
    ///
    /// # 返回
    /// - 普通节点：固定形状 `[32, 128]`
    /// - 动态节点：带动态维度 `[?, 128]`
    fn dynamic_expected_shape(&self) -> DynamicShape {
        DynamicShape::fixed(self.value_expected_shape())
    }

    /// 检查此节点是否支持动态 batch
    ///
    /// 默认返回 false。GradientRouter 和 State 应返回 true。
    fn supports_dynamic_batch(&self) -> bool {
        false
    }

    // ========== 训练模式 ==========

    /// 设置训练模式
    ///
    /// 仅 Dropout、BatchNorm 等训练/评估行为不同的节点需要实现。
    /// 默认空实现，大多数节点不需要关心训练模式。
    ///
    /// # 参数
    /// - `is_training`: 是否处于训练模式
    fn set_training_mode(&mut self, _is_training: bool) {
        // 默认空实现
    }
}
